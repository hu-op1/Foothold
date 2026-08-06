"""Benchmark individual GPU ops under CUDA Graph replay.

CUDA Graph captures a sequence of GPU operations and replays them with
a single host-side launch.  This eliminates per-kernel CPU→GPU dispatch
overhead (~5–10 μs each).  For a full forward pass with ~100 kernels per
layer, this saves ~0.5–1.0 ms per step — critical for decode (small-batch)
performance.

This benchmark measures each op **under graph replay** so the roofline
fit can produce CUDA-Graph-specific parameters.  The fitted params
reflect the GPU's true hardware limits (F_peak, B_peak) without kernel-
launch overhead baked in, and can be used directly by the simulator
when CUDA Graph is active.

Covers:
  - matmul    → cudagraph_matmul.csv
  - elementwise ops → cudagraph_elementwise.csv
  - FlashAttention  → cudagraph_flashattn.csv
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import (warmup, benchmark, auto_warmup_iters, check_memory,
                         supports_float8_matmul, make_float8_tensor,
                         load_completed_keys, append_csv_row, CudaTimer)

# flash_attn is Linux + CUDA only; fall back to torch SDPA on Windows / CPU.
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ModuleNotFoundError:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False
    print("[flash_attn] flash_attn 未安装，回退至 torch SDPA（Windows/CPU 兼容）", file=sys.stderr)

DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}

# ── CSV schemas ──────────────────────────────────────────────────────────

CG_MATMUL_FIELDS = ["op_name", "dtype", "M", "K", "N",
                    "time_ms", "flops", "bytes"]
CG_MATMUL_KEY_FIELDS = ["op_name", "dtype", "M", "K", "N"]

CG_ELEM_FIELDS = ["op_name", "dtype", "operator", "N",
                  "time_ms", "bytes"]
CG_ELEM_KEY_FIELDS = ["op_name", "dtype", "operator", "N"]

CG_FA_FIELDS = ["op_name", "dtype", "b", "nh", "nh_kv", "hd",
                "s_q", "s_kv", "time_ms", "flops", "bytes"]
CG_FA_KEY_FIELDS = ["op_name", "dtype", "b", "s_q", "s_kv"]


def _dtype_list(config):
    raw = config["dtype"]
    return raw if isinstance(raw, list) else [raw]


# ── CUDA Graph benchmark helpers ────────────────────────────────────────

def _cuda_graph_benchmark(fn_setup, fn_forward, warmup_iters, bench_cfg):
    """Benchmark a function under CUDA Graph replay.

    Args:
        fn_setup: callable that returns (args, kwargs) to pass to fn_forward.
                  Must return the SAME tensors on every call (no re-allocation).
        fn_forward: the op to capture, e.g. ``lambda a, w: torch.mm(a, w)``.
        warmup_iters: number of graph-replay warmup iterations.
        bench_cfg: dict with min_time_ms, max_iters, calib_iters keys.

    Returns:
        Average time per replay in milliseconds (GPU time via CUDA events).
    """
    min_time_ms = bench_cfg.get("min_time_ms", 200)
    max_iters = bench_cfg.get("max_iters", 10000)
    calib_iters = bench_cfg.get("calib_iters", 20)

    # ── Allocate persistent tensors ──
    args, kwargs = fn_setup()

    # ── Eager warmup (needed before graph capture to trigger CUDA JIT / autotune) ──
    for _ in range(max(5, warmup_iters // 2)):
        fn_forward(*args, **kwargs)

    # ── Capture CUDA Graph ──
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn_forward(*args, **kwargs)

    # ── Warmup graph replay ──
    for _ in range(max(5, warmup_iters // 2)):
        g.replay()

    # ── Benchmark graph replay ──
    timer = CudaTimer()
    total = 0.0
    iters = 0

    calib = min(calib_iters, max_iters)
    for _ in range(calib):
        with timer:
            g.replay()
        total += timer.elapsed_ms
    iters = calib

    if total >= min_time_ms or iters >= max_iters:
        return total / iters

    per_iter = total / iters if iters > 0 else 0.001
    remaining = min_time_ms - total
    extra = min(int(remaining / per_iter) + 1, max_iters - iters)
    for _ in range(extra):
        with timer:
            g.replay()
        total += timer.elapsed_ms
    iters += extra

    del g
    return total / iters


# ── Matmul under CUDA Graph ─────────────────────────────────────────────

def bench_cudagraph_matmul(config, output_path="results/cudagraph_matmul.csv"):
    dtypes = _dtype_list(config)
    warmup_cfg = config.get("warmup", config.get("warmup_iters", 10))
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_cfg = {
        "min_time_ms": config.get("min_time_ms", 200),
        "max_iters": config.get("max_iters", 10000),
        "calib_iters": config.get("calib_iters", 20),
    }
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    cg_cfg = config.get("cudagraph", {})
    grid = cg_cfg.get("matmul", config["matmul"])  # reuse matmul grid if not specified

    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    done_keys = load_completed_keys(output_path, CG_MATMUL_KEY_FIELDS)
    new_count = 0
    skip_count = 0

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        is_float8 = dt_name in ("float8_e4m3fn", "float8_e5m2")
        if is_float8 and not supports_float8_matmul():
            print(f"\n  [skip] float8 matmul requires sm ≥ 8.9 (Hopper+)")
            continue

        from itertools import product
        combos = list(product(grid["M"], grid["K"], grid["N"]))
        for M, K, N in tqdm(combos, desc=f"CUDA Graph Matmul {dt_name}"):
            key = ("cudagraph_matmul", dt_name, M, K, N)
            if key in done_keys:
                skip_count += 1
                continue

            act_bytes = (M * K + K * N + M * N) * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                row = {"op_name": "cudagraph_matmul", "dtype": dt_name,
                       "M": M, "K": K, "N": N,
                       "time_ms": "OOM", "flops": 2 * M * K * N, "bytes": act_bytes}
                append_csv_row(output_path, CG_MATMUL_FIELDS, row)
                done_keys.add(key)
                continue

            # Persistent tensors — CUDA Graph requires stable memory addresses
            if is_float8:
                a = make_float8_tensor(M, K, device=device)
                w = make_float8_tensor(K, N, device=device)
                scale = torch.tensor(1.0, device=device)

                def setup():
                    return (a, w, scale), {}

                def forward(a_t, w_t, sc):
                    torch._scaled_mm(a_t, w_t, scale_a=sc, scale_b=sc,
                                     out_dtype=torch.float16)
            else:
                a = torch.randn(M, K, dtype=dtype, device=device)
                w = torch.randn(K, N, dtype=dtype, device=device)

                def setup():
                    return (a, w), {}

                def forward(a_t, w_t):
                    torch.mm(a_t, w_t)

            try:
                if warmup_cfg == "auto":
                    wu = auto_warmup_iters(lambda: forward(*setup()[0], **setup()[1]),
                                           bench_cfg["min_time_ms"],
                                           bench_cfg["max_iters"],
                                           bench_cfg["calib_iters"],
                                           warmup_ratio)
                else:
                    wu = int(warmup_cfg)

                ms = _cuda_graph_benchmark(setup, forward, wu, bench_cfg)
            except torch.cuda.OutOfMemoryError:
                # Graph capture holds all tensors resident; a scratch
                # allocation beyond the estimate OOMs here.  Record and move on.
                del setup, a, w
                torch.cuda.empty_cache()
                row = {"op_name": "cudagraph_matmul", "dtype": dt_name,
                       "M": M, "K": K, "N": N,
                       "time_ms": "OOM", "flops": 2 * M * K * N, "bytes": act_bytes}
                append_csv_row(output_path, CG_MATMUL_FIELDS, row)
                done_keys.add(key)
                continue

            row = {"op_name": "cudagraph_matmul", "dtype": dt_name,
                   "M": M, "K": K, "N": N,
                   "time_ms": f"{ms:.6f}",
                   "flops": 2 * M * K * N, "bytes": act_bytes}
            append_csv_row(output_path, CG_MATMUL_FIELDS, row)
            done_keys.add(key)
            new_count += 1

            del setup, a, w

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")


# ── Elementwise ops under CUDA Graph ────────────────────────────────────

def bench_cudagraph_elementwise(config, output_path="results/cudagraph_elementwise.csv"):
    dtypes = _dtype_list(config)
    warmup_cfg = config.get("warmup", config.get("warmup_iters", 10))
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_cfg = {
        "min_time_ms": config.get("min_time_ms", 200),
        "max_iters": config.get("max_iters", 10000),
        "calib_iters": config.get("calib_iters", 20),
    }
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    cg_cfg = config.get("cudagraph", {})
    grid = cg_cfg.get("elementwise", config["elementwise"])

    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    done_keys = load_completed_keys(output_path, CG_ELEM_KEY_FIELDS)
    new_count = 0
    skip_count = 0

    operators = grid.get("operators", ["residual_add", "rmsnorm", "swiglu", "rope"])

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        for op_name in operators:
            for N in tqdm(grid["N"], desc=f"CUDA Graph Elem {dt_name}/{op_name}"):
                key = ("cudagraph_elem", dt_name, op_name, N)
                if key in done_keys:
                    skip_count += 1
                    continue

                act_bytes = N * 2 * dt_bytes  # input + output
                act_gb = act_bytes / (1024 ** 3)
                if not check_memory(act_gb, max_mem):
                    row = {"op_name": "cudagraph_elem", "dtype": dt_name,
                           "operator": op_name, "N": N,
                           "time_ms": "OOM", "bytes": act_bytes}
                    append_csv_row(output_path, CG_ELEM_FIELDS, row)
                    done_keys.add(key)
                    continue

                # Build persistent tensors and forward function
                if op_name == "residual_add":
                    x = torch.randn(N, dtype=dtype, device=device)
                    y = torch.randn(N, dtype=dtype, device=device)

                    def setup():
                        return (x, y), {}

                    def forward(x_t, y_t):
                        x_t.add_(y_t)

                elif op_name == "rmsnorm":
                    hd_val = 4096
                    n_tokens = max(1, N // hd_val)
                    x = torch.randn(n_tokens, hd_val, dtype=dtype, device=device)
                    w_norm = torch.randn(hd_val, dtype=dtype, device=device)

                    def setup():
                        return (x, w_norm), {}

                    def forward(x_t, w_t):
                        rstd = x_t.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
                        x_t.mul_(rstd).mul_(w_t)

                elif op_name == "swiglu":
                    hd_val = 4096
                    n_tokens = max(1, N // hd_val)
                    gate = torch.randn(n_tokens, hd_val, dtype=dtype, device=device)
                    up = torch.randn(n_tokens, hd_val, dtype=dtype, device=device)

                    def setup():
                        return (gate, up), {}

                    def forward(g_t, u_t):
                        torch.nn.functional.silu(g_t, inplace=True)
                        g_t.mul_(u_t)

                elif op_name == "rope":
                    hd_val = 128
                    nh_val = 32
                    n_tokens = max(1, N // (nh_val * hd_val))
                    q = torch.randn(n_tokens, nh_val, hd_val, dtype=dtype, device=device)
                    cos = torch.randn(1, 1, hd_val, device=device)
                    sin = torch.randn(1, 1, hd_val, device=device)

                    def setup():
                        return (q, cos, sin), {}

                    def forward(q_t, cos_t, sin_t):
                        q2 = q_t.reshape(n_tokens, nh_val, hd_val)
                        q_r = q2 * cos_t
                        half_hd = hd_val // 2
                        q_rot = torch.empty_like(q2)
                        q_rot[..., :half_hd] = -q2[..., half_hd:]
                        q_rot[..., half_hd:] = q2[..., :half_hd]
                        q_r.add_(q_rot * sin_t)
                        q_t.copy_(q_r)

                else:
                    continue

                try:
                    if warmup_cfg == "auto":
                        wu = auto_warmup_iters(lambda: forward(*setup()[0], **setup()[1]),
                                               bench_cfg["min_time_ms"],
                                               bench_cfg["max_iters"],
                                               bench_cfg["calib_iters"],
                                               warmup_ratio)
                    else:
                        wu = int(warmup_cfg)

                    ms = _cuda_graph_benchmark(setup, forward, wu, bench_cfg)
                except torch.cuda.OutOfMemoryError:
                    del setup
                    torch.cuda.empty_cache()
                    row = {"op_name": "cudagraph_elem", "dtype": dt_name,
                           "operator": op_name, "N": N,
                           "time_ms": "OOM", "bytes": act_bytes}
                    append_csv_row(output_path, CG_ELEM_FIELDS, row)
                    done_keys.add(key)
                    continue

                row = {"op_name": "cudagraph_elem", "dtype": dt_name,
                       "operator": op_name, "N": N,
                       "time_ms": f"{ms:.6f}", "bytes": act_bytes}
                append_csv_row(output_path, CG_ELEM_FIELDS, row)
                done_keys.add(key)
                new_count += 1

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")


# ── FlashAttention under CUDA Graph ─────────────────────────────────────

def bench_cudagraph_flashattn(config, output_path="results/cudagraph_flashattn.csv"):
    dtypes = _dtype_list(config)
    warmup_cfg = config.get("warmup", config.get("warmup_iters", 10))
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_cfg = {
        "min_time_ms": config.get("min_time_ms", 200),
        "max_iters": config.get("max_iters", 10000),
        "calib_iters": config.get("calib_iters", 20),
    }
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    fa_cfg = config.get("flashattn", {})
    cg_cfg = config.get("cudagraph", {})
    grid = cg_cfg.get("flashattn", fa_cfg)

    batch_raw = grid.get("batch", 1)
    batch_list = batch_raw if isinstance(batch_raw, list) else [batch_raw]
    nh = grid.get("num_heads", 32)
    nh_kv = grid.get("num_kv_heads", 8)
    hd = grid.get("head_dim", 128)

    print("CUDA Graph FlashAttn backend: flash_attn (native)")

    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    done_keys = load_completed_keys(output_path, CG_FA_KEY_FIELDS)
    new_count = 0
    skip_count = 0

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        is_float8 = dt_name in ("float8_e4m3fn", "float8_e5m2")
        if is_float8:
            print(f"\n  [skip] float8 FlashAttn: flash_attn does not support float8")
            continue

        combos = [(b_val, sq, skv) for b_val in batch_list
                  for sq in grid["s_q"] for skv in grid["s_kv"]]

        for b_val, s_q, s_kv in tqdm(combos, desc=f"CUDA Graph FA {dt_name}"):
            key = ("cudagraph_flashattn", dt_name, b_val, s_q, s_kv)
            if key in done_keys:
                skip_count += 1
                continue

            act_bytes = b_val * (nh * s_q + 2 * nh_kv * s_kv + nh * s_q) * hd * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                row = {"op_name": "cudagraph_flashattn", "dtype": dt_name,
                       "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                       "s_q": s_q, "s_kv": s_kv,
                       "time_ms": "OOM", "flops": 0, "bytes": 0}
                append_csv_row(output_path, CG_FA_FIELDS, row)
                done_keys.add(key)
                continue

            q = torch.randn(b_val, s_q, nh, hd, dtype=dtype, device=device)
            k = torch.randn(b_val, s_kv, nh_kv, hd, dtype=dtype, device=device)
            v = torch.randn(b_val, s_kv, nh_kv, hd, dtype=dtype, device=device)

            def setup():
                return (q, k, v), {}

            if _HAS_FLASH_ATTN:
                def forward(q_t, k_t, v_t):
                    flash_attn_func(q_t, k_t, v_t, causal=True)
            else:
                def forward(q_t, k_t, v_t):
                    F.scaled_dot_product_attention(
                        q_t.transpose(1, 2), k_t.transpose(1, 2), v_t.transpose(1, 2),
                        is_causal=True,
                    )

            try:
                if warmup_cfg == "auto":
                    wu = auto_warmup_iters(lambda: forward(*setup()[0], **setup()[1]),
                                           bench_cfg["min_time_ms"],
                                           bench_cfg["max_iters"],
                                           bench_cfg["calib_iters"],
                                           warmup_ratio)
                else:
                    wu = int(warmup_cfg)

                ms = _cuda_graph_benchmark(setup, forward, wu, bench_cfg)
            except torch.cuda.OutOfMemoryError:
                del setup, q, k, v
                torch.cuda.empty_cache()
                row = {"op_name": "cudagraph_flashattn", "dtype": dt_name,
                       "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                       "s_q": s_q, "s_kv": s_kv,
                       "time_ms": "OOM", "flops": 0, "bytes": 0}
                append_csv_row(output_path, CG_FA_FIELDS, row)
                done_keys.add(key)
                continue

            _fa_flops = 4 * b_val * nh * s_q * s_kv * hd
            _fa_bytes = b_val * hd * dt_bytes * (nh * s_q + nh_kv * s_kv + nh_kv * s_kv + nh * s_q)

            row = {"op_name": "cudagraph_flashattn", "dtype": dt_name,
                   "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                   "s_q": s_q, "s_kv": s_kv,
                   "time_ms": f"{ms:.6f}",
                   "flops": _fa_flops, "bytes": _fa_bytes}
            append_csv_row(output_path, CG_FA_FIELDS, row)
            done_keys.add(key)
            new_count += 1

            del setup, q, k, v

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")


# ── Top-level entry ─────────────────────────────────────────────────────

def bench_cudagraph_all(config, out_dir="results"):
    """Run all CUDA Graph benchmarks. Output goes to out_dir/<gpu>/cudagraph_*.csv."""
    os.makedirs(out_dir, exist_ok=True)
    gpu = config.get("gpu", "unknown")

    matmul_path = os.path.join(out_dir, "cudagraph_matmul.csv")
    elem_path = os.path.join(out_dir, "cudagraph_elementwise.csv")
    fa_path = os.path.join(out_dir, "cudagraph_flashattn.csv")

    print("\n[CUDA Graph Matmul]")
    bench_cudagraph_matmul(config, output_path=matmul_path)
    torch.cuda.empty_cache()

    print("\n[CUDA Graph Elementwise]")
    bench_cudagraph_elementwise(config, output_path=elem_path)
    torch.cuda.empty_cache()

    if config.get("flashattn"):
        print("\n[CUDA Graph FlashAttn]")
        bench_cudagraph_flashattn(config, output_path=fa_path)
        torch.cuda.empty_cache()

    return matmul_path, elem_path, fa_path
