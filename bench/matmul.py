"""Benchmark torch.mm over a shape grid to characterize GPU roofline.

Covers both memory-bound (small M) and compute-bound (large M) regimes.
"""

import torch
from tqdm import tqdm
from bench.utils import (warmup, benchmark, save_csv, check_memory,
                         supports_float8_matmul, make_float8_tensor)


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


def bench_matmul(config, output_path="results/matmul.csv"):
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

        # float8 matmul requires torch._scaled_mm (Hopper+ / sm ≥ 8.9).
        # torch.mm does not accept float8 inputs on any GPU.
        is_float8 = dt_name in ("float8_e4m3fn", "float8_e5m2")
        if is_float8 and not supports_float8_matmul():
            print(f"\n  [skip] float8 matmul requires sm ≥ 8.9 (Hopper+), "
                  f"this GPU is sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
            continue

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

            if is_float8:
                a = make_float8_tensor(M, K, device=device)
                w = make_float8_tensor(K, N, device=device)
                scale = torch.tensor(1.0, device=device)

                def mm(a=a, w=w, scale=scale):
                    torch._scaled_mm(a, w, scale_a=scale, scale_b=scale,
                                     out_dtype=torch.float16)
            else:
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
        save_csv(results, output_path)
    return results
