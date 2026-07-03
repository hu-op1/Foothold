"""Benchmark torch.mm over a shape grid to characterize GPU roofline.

Covers both memory-bound (small M) and compute-bound (large M) regimes.
"""

import torch
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory


# Bytes per element for each dtype.
# fp16/bf16 = 2, fp8 variants = 1.
# int8/int4 weight-only quant is not benchmarked via torch.mm — they use
# dedicated dequant+compute codepaths (torch._int_mm, custom CUDA kernels).
DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}


def _dtype_list(config):
    """Resolve dtype field (scalar for backward compat, list for multi-precision)."""
    raw = config["dtype"]
    return raw if isinstance(raw, list) else [raw]


def bench_matmul(config, output_path="results/matmul.xlsx"):
    dtypes = _dtype_list(config)
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    from itertools import product

    grid = config["matmul"]
    results = []

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)
        combos = list(product(grid["M"], grid["K"], grid["N"]))
        for M, K, N in tqdm(combos, desc=f"Matmul {dt_name}"):
            act_bytes = (M * K + K * N + M * N) * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                results.append({
                    "op_name": "matmul",
                    "dtype": dt_name,
                    "M": M, "K": K, "N": N,
                    "time_ms": "OOM",
                    "flops": 2 * M * K * N,
                    "bytes": act_bytes,
                })
                continue

            a = torch.randn(M, K, dtype=dtype, device=device)
            w = torch.randn(K, N, dtype=dtype, device=device)

            def mm(a=a, w=w):
                torch.mm(a, w)

            warmup(mm, warmup_iters)
            avg_ms = benchmark(mm, bench_iters)

            results.append({
                "op_name": "matmul",
                "dtype": dt_name,
                "M": M, "K": K, "N": N,
                "time_ms": f"{avg_ms:.6f}",
                "flops": 2 * M * K * N,
                "bytes": act_bytes,
            })

            del a, w

    if output_path:
        save_xlsx(results, output_path)
    return results
