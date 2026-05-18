"""Benchmark torch.mm over a shape grid to characterize GPU roofline.

Covers both memory-bound (small M) and compute-bound (large M) regimes.
"""

import torch
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory


DTYPE_BYTES = 2  # fp16


def bytes_matmul(M, K, N):
    """Bytes moved for [M,K] × [K,N] GEMM (input + weight + output)."""
    return (M * K + K * N + M * N) * DTYPE_BYTES


def bench_matmul(config, output_path="results/matmul.xlsx"):
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    from itertools import product

    grid = config["matmul"]
    results = []
    combos = list(product(grid["M"], grid["K"], grid["N"]))

    for M, K, N in tqdm(combos, desc="Matmul"):
        act_gb = bytes_matmul(M, K, N) / (1024 ** 3)
        oom = not check_memory(act_gb, max_mem)
        if oom:
            results.append({
                "op_name": "matmul",
                "M": M, "K": K, "N": N,
                "time_ms": "OOM",
                "flops": 2 * M * K * N,
                "bytes": bytes_matmul(M, K, N),
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
            "M": M, "K": K, "N": N,
            "time_ms": f"{avg_ms:.6f}",
            "flops": 2 * M * K * N,
            "bytes": bytes_matmul(M, K, N),
        })

        del a, w

    if output_path:
        save_xlsx(results, output_path)
    return results
