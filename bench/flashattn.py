"""Benchmark FlashAttention (via torch SDPA) over a (s_q, s_kv) grid.

FlashAttention is the dominant kernel in LLM inference attention layers.
Its hardware efficiency differs from matmul due to tiling, causal masking,
and SRAM-aware scheduling.  This benchmark provides measured data so the
roofline fit can produce FA-specific (F_peak, B_peak, p) parameters instead
of reusing matmul-fitted values.

Prefers Dao-AILab's flash_attn package when available (Linux + CUDA).
Falls back to torch.nn.functional.scaled_dot_product_attention on other
platforms (Windows, CPU).
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import (warmup, benchmark, auto_warmup_iters, check_memory,
                         load_completed_keys, append_csv_row)
from bench.gateddelta import KernelFaultError

# flash_attn is Linux + CUDA only; fall back to torch SDPA on Windows / CPU.
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ModuleNotFoundError:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False
    import sys
    print("[flash_attn] flash_attn 未安装，回退至 torch SDPA（Windows/CPU 兼容）", file=sys.stderr)


# Bytes per element for each dtype.
DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}

# Fixed CSV column order (for incremental append).
FA_FIELDS = ["op_name", "dtype", "b", "nh", "nh_kv", "hd",
             "s_q", "s_kv", "time_ms", "flops", "bytes"]
FA_KEY_FIELDS = ["op_name", "dtype", "b", "s_q", "s_kv"]


def _dtype_list(config):
    """Resolve dtype field (scalar for backward compat, list for multi-precision)."""
    raw = config["dtype"]
    return raw if isinstance(raw, list) else [raw]


def _fa_bytes(b, nh, s_q, nh_kv, s_kv, hd, dt_bytes):
    """Analytical bytes moved by FlashAttention (no S×S HBM round-trip).

    Q read + K read + V read + O write.
    """
    return b * hd * dt_bytes * (nh * s_q + nh_kv * s_kv + nh_kv * s_kv + nh * s_q)


def _fa_flops(b, nh, s_q, s_kv, hd):
    """Analytical FLOPs for attention: 2·s_q·s_kv·hd per head, times 2 for mul+add."""
    return 4 * b * nh * s_q * s_kv * hd


def bench_flashattn(config, output_path="results/flashattn.csv"):
    dtypes = _dtype_list(config)
    warmup_cfg = config.get("warmup", config.get("warmup_iters", 10))
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_min_time = config.get("min_time_ms", 200)
    bench_max_iters = config.get("max_iters", 10000)
    bench_calib = config.get("calib_iters", 20)
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    fa_cfg = config["flashattn"]
    batch_raw = fa_cfg.get("batch", 1)
    batch_list = batch_raw if isinstance(batch_raw, list) else [batch_raw]
    nh = fa_cfg.get("num_heads", 32)
    nh_kv = fa_cfg.get("num_kv_heads", 8)
    hd = fa_cfg.get("head_dim", 128)

    print(f"FlashAttn backend: {'flash_attn (native)' if _HAS_FLASH_ATTN else 'torch SDPA (fallback)'}")
    print(f"FlashAttn dtypes: {dtypes}")
    print(f"FlashAttn batch sizes: {batch_list}")

    # ── overwrite: delete existing CSV so all combos re-run ──
    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    # ── resume: load already-completed combos ──
    done_keys = load_completed_keys(output_path, FA_KEY_FIELDS)

    results = []
    new_count = 0
    skip_count = 0

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        # float8 FlashAttention: neither flash_attn_func nor torch SDPA
        # accept float8 inputs.  Skip for now; float8 attention via
        # dedicated Hopper kernels (cuDNN/cuBLAS) is a different codepath.
        is_float8 = dt_name in ("float8_e4m3fn", "float8_e5m2")
        if is_float8:
            print(f"\n  [skip] float8 FlashAttn: not supported for float8")
            continue

        # Skip s_q > s_kv combos: degenerate under flash_attn's causal offset
        # convention (offset = s_kv - s_q < 0 → most queries see no keys → NaN
        # output, kernel does ~1/16 of the work) and never a real serving shape
        # (KV always contains the query tokens, so s_kv >= s_q).
        combos = [(b_val, sq, skv) for b_val in batch_list
                  for sq in fa_cfg["s_q"] for skv in fa_cfg["s_kv"]
                  if sq <= skv]
        for b_val, s_q, s_kv in tqdm(combos, desc=f"FlashAttn {dt_name}"):
            key = ("flashattn", dt_name, b_val, s_q, s_kv)
            if key in done_keys:
                skip_count += 1
                continue

            # Memory check
            act_bytes = b_val * (nh * s_q + 2 * nh_kv * s_kv + nh * s_q) * hd * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                row = {
                    "op_name": "flashattn", "dtype": dt_name,
                    "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                    "s_q": s_q, "s_kv": s_kv,
                    "time_ms": "OOM", "flops": 0, "bytes": 0,
                }
                results.append(row)
                append_csv_row(output_path, FA_FIELDS, row)
                done_keys.add(key)
                continue

            # Create tensors in (batch, seqlen, nheads, hd) — flash_attn format.
            # SDPA expects (batch, nheads, seqlen, hd); we transpose on the fly.
            q = torch.randn(b_val, s_q, nh, hd, dtype=dtype, device=device)
            k = torch.randn(b_val, s_kv, nh_kv, hd, dtype=dtype, device=device)
            v = torch.randn(b_val, s_kv, nh_kv, hd, dtype=dtype, device=device)

            if _HAS_FLASH_ATTN:
                def fa_fn(q=q, k=k, v=v):
                    flash_attn_func(q, k, v, causal=True)
            else:
                # SDPA fallback: (N, S, H, D) → (N, H, S, D).
                def fa_fn(q=q, k=k, v=v):
                    F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        is_causal=True,
                    )

            try:
                if warmup_cfg == "auto":
                    wu = auto_warmup_iters(fa_fn, bench_min_time, bench_max_iters,
                                           bench_calib, warmup_ratio)
                else:
                    wu = int(warmup_cfg)
                warmup(fa_fn, wu)
                ms = benchmark(fa_fn, min_time_ms=bench_min_time, max_iters=bench_max_iters,
                                calib_iters=bench_calib)
            except torch.cuda.OutOfMemoryError:
                # The flash-attn workspace can exceed the analytical estimate;
                # record the combo as OOM and move on (resume skips it).
                del fa_fn, q, k, v
                torch.cuda.empty_cache()
                row = {
                    "op_name": "flashattn", "dtype": dt_name,
                    "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                    "s_q": s_q, "s_kv": s_kv,
                    "time_ms": "OOM", "flops": 0, "bytes": 0,
                }
                results.append(row)
                append_csv_row(output_path, FA_FIELDS, row)
                done_keys.add(key)
                continue
            except RuntimeError as exc:
                # A kernel fault (illegal memory access) poisons the CUDA
                # context — record the combo first so a rerun resumes past it,
                # then signal main.py to restart the process.
                if "cuda" not in str(exc).lower():
                    raise
                del fa_fn, q, k, v
                row = {
                    "op_name": "flashattn", "dtype": dt_name,
                    "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                    "s_q": s_q, "s_kv": s_kv,
                    "time_ms": "OOM", "flops": 0, "bytes": 0,
                }
                results.append(row)
                append_csv_row(output_path, FA_FIELDS, row)
                done_keys.add(key)
                raise KernelFaultError(
                    f"flashattn {dt_name} b={b_val} s_q={s_q} s_kv={s_kv}: {exc}"
                ) from exc

            row = {
                "op_name": "flashattn", "dtype": dt_name,
                "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                "s_q": s_q, "s_kv": s_kv,
                "time_ms": f"{ms:.6f}",
                "flops": _fa_flops(b_val, nh, s_q, s_kv, hd),
                "bytes": _fa_bytes(b_val, nh, s_q, nh_kv, s_kv, hd, dt_bytes),
            }
            results.append(row)
            append_csv_row(output_path, FA_FIELDS, row)
            done_keys.add(key)
            new_count += 1

            del fa_fn, q, k, v

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif output_path and new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")
    return results
