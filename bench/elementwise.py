"""Benchmark representative elementwise ops to validate memory bandwidth model.

All elementwise ops are deep in the memory-bound regime (AI < 1 FLOP/byte).
We measure 3 representatives with different arithmetic complexity:
  - residual_add: pure add (lightest)
  - rmsnorm: reduction + division + sqrt
  - softmax: multi-pass with exp (heaviest)

If their effective bandwidth agrees, we can reuse the same B_peak for all
elementwise ops without per-operator bench/fit.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory


DTYPE_BYTES = 2

# Bytes-per-element factors (reads + write, times per pass)
BYTES_FACTORS = {
    "residual_add": 3,   # read A + read B + write C
    "rmsnorm": 4,        # read + write, with rsqrt reduction
    "softmax": 6,        # multi-pass: max/exp + sum + normalize
}


def _check_oom(N, max_mem):
    """Conservative: 3 * N * dtype_size for worst-case intermediate buffers."""
    act_gb = (3 * N * DTYPE_BYTES) / (1024 ** 3)
    return not check_memory(act_gb, max_mem)


def bench_elementwise(config, output_path="results/elementwise.xlsx"):
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    grid = config["elementwise"]
    results = []

    for N in tqdm(grid["N"], desc="Elementwise"):
        oom = _check_oom(N, max_mem)
        if oom:
            for op in grid["operators"]:
                results.append({
                    "op_name": op,
                    "N": N,
                    "time_ms": "OOM",
                    "flops": 0,
                    "bytes": BYTES_FACTORS[op] * N * DTYPE_BYTES,
                })
            continue

        x = torch.randn(N, dtype=dtype, device=device)
        y = torch.randn(N, dtype=dtype, device=device)

        # --- residual_add ---
        def add_fn(x=x, y=y):
            x + y

        warmup(add_fn, warmup_iters)
        ms = benchmark(add_fn, bench_iters)
        results.append({
            "op_name": "residual_add",
            "N": N,
            "time_ms": f"{ms:.6f}",
            "flops": N,
            "bytes": 3 * N * DTYPE_BYTES,
        })

        # --- rmsnorm ---
        w = torch.ones(N, dtype=dtype, device=device)

        def rms_fn(x=x, w=w):
            F.rms_norm(x, (N,), w, 1e-5)

        warmup(rms_fn, warmup_iters)
        ms = benchmark(rms_fn, bench_iters)
        results.append({
            "op_name": "rmsnorm",
            "N": N,
            "time_ms": f"{ms:.6f}",
            "flops": 4 * N,
            "bytes": 4 * N * DTYPE_BYTES,
        })

        # --- softmax ---
        def soft_fn(x=x):
            F.softmax(x, dim=0)

        warmup(soft_fn, warmup_iters)
        ms = benchmark(soft_fn, bench_iters)
        results.append({
            "op_name": "softmax",
            "N": N,
            "time_ms": f"{ms:.6f}",
            "flops": 5 * N,
            "bytes": 6 * N * DTYPE_BYTES,
        })

        del x, y, w

    if output_path:
        save_xlsx(results, output_path)
    return results
