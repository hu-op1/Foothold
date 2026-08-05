"""Benchmark the Gated Delta Rule scan kernel and the causal depthwise conv1d.

Qwen3.5-style hybrid models replace most full-attention layers with a Gated
DeltaNet (linear attention) block.  Its two distinctive kernels are:

  - the gated delta rule scan (chunked for prefill, recurrent for decode):
    per-token cost 4*kd*vd*nvh MACs, state read+write 2*nvh*kd*vd*dt bytes
    (the model repeats Q/K to the value-head count before the kernel, see
    modeling_qwen3_5.py:521-523, so cost scales with nvh not nhk);
  - a causal depthwise conv1d (kernel k over conv_dim channels).

Neither kernel is covered by the matmul / flash-attention / elementwise
benches, so without this file their roofline params fall back to
flash-attention fits.

Prefers the FLA fused kernels (fla.ops.gated_delta_rule, causal-conv1d)
when installed; falls back to the torch reference implementations vendored
below (same code paths as modeling_qwen3_5.py when FLA is absent).
"""

import os
import torch
import torch.nn.functional as F
from itertools import product
from tqdm import tqdm
from bench.utils import (warmup, benchmark, auto_warmup_iters, check_memory,
                         load_completed_keys, append_csv_row)


class KernelFaultError(RuntimeError):
    """A CUDA kernel fault (e.g. illegal memory access) poisoned the context.

    The faulting combo is recorded as OOM in the CSV before this is raised, so
    a rerun with overwrite=false resumes past it.  main.py auto-restarts the
    process on this exception instead of forcing the user to rerun by hand.
    """

# ── Optional fused kernels ──────────────────────────────────────────────
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    _HAS_FLA = True
except (ImportError, ModuleNotFoundError, OSError):
    chunk_gated_delta_rule = fused_recurrent_gated_delta_rule = None
    _HAS_FLA = False
    import sys
    print(
        "[gateddelta] WARNING: fla (flash-linear-attention) 未安装，"
        "Gated Delta Rule 回退至 torch 参考实现测量（慢核）——"
        "该数据反映当前真实执行路径；装上 fla 后重跑 bench 才能得到融合核带宽。",
        file=sys.stderr,
    )

try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except (ImportError, ModuleNotFoundError, OSError):
    causal_conv1d_fn = None
    _HAS_CAUSAL_CONV1D = False
    import sys
    print(
        "[gateddelta] WARNING: causal_conv1d 未安装，"
        "conv1d 回退至 F.conv1d 测量（与建模回落路径一致）。",
        file=sys.stderr,
    )


# ── Torch reference implementations (vendored from modeling_qwen3_5.py) ─

def _l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    # GVA: repeat Q/K to the value-head count so every head lines up — the
    # fused kernels and modeling_qwen3_5.py repeat Q/K before the recurrence.
    nvh = value.shape[1]
    if key.shape[1] != nvh:
        rep = nvh // key.shape[1]
        query = query.repeat_interleave(rep, dim=1)
        key = key.repeat_interleave(rep, dim=1)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    # GVA: repeat Q/K to the value-head count so every head lines up (see
    # torch_chunk_gated_delta_rule).
    nvh = value.shape[1]
    if key.shape[1] != nvh:
        rep = nvh // key.shape[1]
        query = query.repeat_interleave(rep, dim=1)
        key = key.repeat_interleave(rep, dim=1)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    # `activation` is unused — kept for API compatibility with
    # causal_conv1d_fn (flash-linear-attention). The benchmark only
    # measures the identity case (activation=None).
    del activation
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    return out.to(hidden_states.dtype)


# ── Analytical flops/bytes (MUST match sim/gateddelta cost formulas) ────
# NOTE: these scale with BOTH s and batch b.  sim/graph.py builds these per
# request (2*macs*num_new etc.) and sums over the scheduled batch, so a
# benchmark tensor of shape (b, s, ...) carries b*s tokens' worth of work.

def scan_flops(nvh, kd, vd, s, b):
    """Per-token 4*kd*vd*nvh MACs (delta-rule state ops), times b*s tokens."""
    return 4 * nvh * kd * vd * s * b * 2


def scan_bytes(nvh, kd, vd, dt_bytes, s, chunk_size, b):
    """State read+write per token; chunked prefill amortizes over chunk_size."""
    state = 2 * nvh * kd * vd * dt_bytes
    return state * (s / chunk_size if s > 1 else s) * b


def conv_flops(C, k, s, b):
    return 2 * C * k * s * b


def conv_bytes(C, k, dt_bytes, s, b):
    return 2 * k * C * dt_bytes * s * b


# ── Benchmark driver ────────────────────────────────────────────────────

DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}

GD_FIELDS = ["op_name", "dtype", "b", "nhk", "nvh", "kd", "vd", "s_q",
             "time_ms", "flops", "bytes"]
GD_KEY_FIELDS = ["op_name", "dtype", "b", "nvh", "kd", "vd", "s_q"]


def _dtype_list(config):
    raw = config["dtype"]
    return raw if isinstance(raw, list) else [raw]


def bench_gateddelta(config, output_path="results/gateddelta.csv"):
    dtypes = _dtype_list(config)
    warmup_cfg = config.get("warmup", config.get("warmup_iters", 10))
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_min_time = config.get("min_time_ms", 200)
    bench_max_iters = config.get("max_iters", 10000)
    bench_calib = config.get("calib_iters", 20)
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    gd_cfg = config["gateddelta"]
    shapes = gd_cfg["shapes"]
    batch_raw = gd_cfg.get("batch", 1)
    batch_list = batch_raw if isinstance(batch_raw, list) else [batch_raw]
    s_q_list = gd_cfg.get("s_q", [1, 128, 1024, 4096, 16384])
    conv_channels = gd_cfg.get("conv_channels", [8192])
    conv_kernel = gd_cfg.get("conv_kernel", 4)
    chunk_size = gd_cfg.get("chunk_size", 64)

    print(f"GatedDelta backend: "
          f"{'fla (fused)' if _HAS_FLA else 'torch reference (fallback)'}")
    print(f"Conv1d backend: "
          f"{'causal_conv1d (fused)' if _HAS_CAUSAL_CONV1D else 'F.conv1d (fallback)'}")
    print(f"GatedDelta dtypes: {dtypes}, shapes: {shapes}, "
          f"chunk_size: {chunk_size}")

    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    done_keys = load_completed_keys(output_path, GD_KEY_FIELDS)

    results = []
    new_count = 0
    skip_count = 0

    def _wu(fn):
        if warmup_cfg == "auto":
            w = auto_warmup_iters(fn, bench_min_time, bench_max_iters,
                                  bench_calib, warmup_ratio)
        else:
            w = int(warmup_cfg)
        warmup(fn, w)

    def _append(row):
        nonlocal new_count
        results.append(row)
        append_csv_row(output_path, GD_FIELDS, row)
        new_count += 1

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        if dt_name in ("float8_e4m3fn", "float8_e5m2"):
            print(f"\n  [skip] float8 GatedDelta: no native float8 kernels")
            continue

        # ── gated delta rule scan ──
        for shape in shapes:
            nhk, nvh, kd, vd = (shape["nhk"], shape["nvh"],
                                shape["kd"], shape["vd"])
            for b_val, s_q in tqdm(
                list(product(batch_list, s_q_list)),
                desc=f"GatedDelta {dt_name} nhk={nhk} nvh={nvh} kd={kd} vd={vd}",
            ):
                key = ("gated_delta_rule", dt_name, b_val, nvh, kd, vd, s_q)
                if key in done_keys:
                    skip_count += 1
                    continue

                # fla fused chunk kernel peak HBM footprint: q/k/v inputs plus
                # l2norm copies of q/k, the WY tensors w/u, h, v_new, o, and
                # the solved (B,T,HV,chunk_size) intra-chunk matrix A — each
                # input-scale.  Per token-head that is
                # (5*kd + 5*vd + chunk_size + 2)*dt_bytes (see
                # fla/ops/gated_delta_rule/chunk_fwd.py:chunk_fwd_o et al.).
                # The old "inputs * 2" heuristic let b=256/s_q=1024 through at
                # ~13 GiB but the kernel really needs ~22 GiB -> OOM.  The
                # torch reference casts to fp32 and materializes per-chunk
                # attention, so it keeps the conservative ~6x input heuristic.
                if _HAS_FLA:
                    act_gb = (b_val * s_q * nvh
                              * (5 * kd + 5 * vd + chunk_size + 2) * dt_bytes
                              + b_val * nvh * kd * vd * dt_bytes) / (1024 ** 3)
                else:
                    act_gb = (b_val * s_q * nvh * (2 * kd + vd) * dt_bytes * 6
                              + b_val * nvh * kd * vd * 8) / (1024 ** 3)
                def _record_skip():
                    """Persist an OOM/unsupported row so resume skips this combo."""
                    row = {
                        "op_name": "gated_delta_rule", "dtype": dt_name,
                        "b": b_val, "nhk": nhk, "nvh": nvh, "kd": kd, "vd": vd,
                        "s_q": s_q, "time_ms": "OOM", "flops": 0, "bytes": 0,
                    }
                    results.append(row)
                    append_csv_row(output_path, GD_FIELDS, row)
                    done_keys.add(key)

                if not check_memory(act_gb, max_mem):
                    _record_skip()
                    continue

                q = torch.randn(b_val, s_q, nvh, kd, dtype=dtype, device=device)
                k = torch.randn(b_val, s_q, nvh, kd, dtype=dtype, device=device)
                v = torch.randn(b_val, s_q, nvh, vd, dtype=dtype, device=device)
                g = torch.randn(b_val, s_q, nvh, dtype=dtype, device=device)
                beta = torch.randn(b_val, s_q, nvh, dtype=dtype, device=device)
                state = torch.zeros(b_val, nvh, kd, vd, dtype=dtype, device=device)

                if s_q == 1:
                    if _HAS_FLA:
                        def scan_fn(q=q, k=k, v=v, g=g, beta=beta, st=state):
                            return fused_recurrent_gated_delta_rule(
                                q, k, v, g, beta, initial_state=st,
                                output_final_state=True,
                                use_qk_l2norm_in_kernel=True,
                            )
                    else:
                        def scan_fn(q=q, k=k, v=v, g=g, beta=beta, st=state):
                            return torch_recurrent_gated_delta_rule(
                                q, k, v, g, beta, initial_state=st,
                                output_final_state=True,
                                use_qk_l2norm_in_kernel=True,
                            )
                else:
                    if _HAS_FLA:
                        def scan_fn(q=q, k=k, v=v, g=g, beta=beta):
                            return chunk_gated_delta_rule(
                                q, k, v, g, beta, initial_state=None,
                                output_final_state=True,
                                use_qk_l2norm_in_kernel=True,
                            )
                    else:
                        def scan_fn(q=q, k=k, v=v, g=g, beta=beta):
                            return torch_chunk_gated_delta_rule(
                                q, k, v, g, beta, chunk_size=chunk_size,
                                initial_state=None, output_final_state=True,
                                use_qk_l2norm_in_kernel=True,
                            )

                try:
                    _wu(scan_fn)
                    ms = benchmark(scan_fn, min_time_ms=bench_min_time,
                                   max_iters=bench_max_iters,
                                   calib_iters=bench_calib)
                except torch.cuda.OutOfMemoryError:
                    # OOM is recoverable: release the tensors and move on.
                    del q, k, v, g, beta, state
                    torch.cuda.empty_cache()
                    _record_skip()
                    continue
                except RuntimeError as exc:
                    # A kernel fault (illegal memory access) poisons the CUDA
                    # context — every later call would fail too.  Record the
                    # combo first so a rerun resumes past it, then signal main
                    # to restart the process.  Non-CUDA RuntimeErrors (e.g. a
                    # shape bug in the reference) are re-raised to surface
                    # loudly instead of being mistaken for a kernel fault.
                    if "cuda" not in str(exc).lower():
                        raise
                    _record_skip()
                    raise KernelFaultError(
                        f"gated_delta_rule {dt_name} b={b_val} nvh={nvh} "
                        f"kd={kd} vd={vd} s_q={s_q}: {exc}"
                    ) from exc
                _append({
                    "op_name": "gated_delta_rule", "dtype": dt_name,
                    "b": b_val, "nhk": nhk, "nvh": nvh, "kd": kd, "vd": vd,
                    "s_q": s_q, "time_ms": f"{ms:.6f}",
                    "flops": scan_flops(nvh, kd, vd, s_q, b_val),
                    "bytes": scan_bytes(nvh, kd, vd, dt_bytes, s_q, chunk_size, b_val),
                })
                del q, k, v, g, beta, state

        # ── causal depthwise conv1d ──
        for C in conv_channels:
            for b_val, s_q in tqdm(
                list(product(batch_list, s_q_list)),
                desc=f"Conv1d {dt_name} C={C} k={conv_kernel}",
            ):
                key = ("conv1d", dt_name, b_val, C, conv_kernel, 1, s_q)
                if key in done_keys:
                    skip_count += 1
                    continue

                act_gb = (b_val * C * s_q * dt_bytes * 3) / (1024 ** 3)
                if not check_memory(act_gb, max_mem):
                    row = {
                        "op_name": "conv1d", "dtype": dt_name,
                        "b": b_val, "nhk": C, "nvh": conv_kernel,
                        "kd": C, "vd": 1, "s_q": s_q,
                        "time_ms": "OOM", "flops": 0, "bytes": 0,
                    }
                    results.append(row)
                    append_csv_row(output_path, GD_FIELDS, row)
                    done_keys.add(key)
                    continue

                x = torch.randn(b_val, C, s_q, dtype=dtype, device=device)
                weight = torch.randn(C, conv_kernel, dtype=dtype, device=device)

                if _HAS_CAUSAL_CONV1D:
                    def conv_fn(x=x, w=weight):
                        return causal_conv1d_fn(x, w, bias=None,
                                                activation=None)
                else:
                    def conv_fn(x=x, w=weight):
                        return torch_causal_conv1d_fn(x, w, bias=None,
                                                      activation=None)

                def _conv_skip():
                    row = {
                        "op_name": "conv1d", "dtype": dt_name,
                        "b": b_val, "nhk": C, "nvh": conv_kernel,
                        "kd": C, "vd": 1, "s_q": s_q,
                        "time_ms": "OOM", "flops": 0, "bytes": 0,
                    }
                    results.append(row)
                    append_csv_row(output_path, GD_FIELDS, row)
                    done_keys.add(key)

                try:
                    _wu(conv_fn)
                    ms = benchmark(conv_fn, min_time_ms=bench_min_time,
                                   max_iters=bench_max_iters,
                                   calib_iters=bench_calib)
                except torch.cuda.OutOfMemoryError:
                    del x, weight
                    torch.cuda.empty_cache()
                    _conv_skip()
                    continue
                except RuntimeError as exc:
                    if "cuda" not in str(exc).lower():
                        raise
                    _conv_skip()
                    raise KernelFaultError(
                        f"conv1d {dt_name} C={C} b={b_val} s_q={s_q}: {exc}"
                    ) from exc
                _append({
                    "op_name": "conv1d", "dtype": dt_name,
                    "b": b_val, "nhk": C, "nvh": conv_kernel,
                    "kd": C, "vd": 1, "s_q": s_q,
                    "time_ms": f"{ms:.6f}",
                    "flops": conv_flops(C, conv_kernel, s_q, b_val),
                    "bytes": conv_bytes(C, conv_kernel, dt_bytes, s_q, b_val),
                })
                del x, weight

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif output_path and new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")
    return results
