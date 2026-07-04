"""Benchmark torch.mm over a shape grid to characterize GPU roofline.

Covers both memory-bound (small M) and compute-bound (large M) regimes.
"""

import torch
from tqdm import tqdm
from bench.utils import (warmup, benchmark, check_memory,
                         supports_float8_matmul, make_float8_tensor,
                         load_completed_keys, append_csv_row)


# Bytes per element for each dtype.
# fp16/bf16 = 2, fp8 variants = 1.
# int8/int4 weight-only quant is not benchmarked via torch.mm — they use
# dedicated dequant+compute codepaths (torch._int_mm, custom CUDA kernels).
DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}

# Fixed CSV column order (for incremental append).
MATMUL_FIELDS = ["op_name", "dtype", "M", "K", "N", "time_ms", "flops", "bytes"]
MATMUL_KEY_FIELDS = ["op_name", "dtype", "M", "K", "N"]


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

    # ── resume: load already-completed combos ──
    done_keys = load_completed_keys(output_path, MATMUL_KEY_FIELDS)

    results = []
    new_count = 0
    skip_count = 0

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
            key = ("matmul", dt_name, M, K, N)
            if key in done_keys:
                skip_count += 1
                continue

            act_bytes = (M * K + K * N + M * N) * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                row = {
                    "op_name": "matmul", "dtype": dt_name,
                    "M": M, "K": K, "N": N,
                    "time_ms": "OOM",
                    "flops": 2 * M * K * N, "bytes": act_bytes,
                }
                results.append(row)
                append_csv_row(output_path, MATMUL_FIELDS, row)
                done_keys.add(key)
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

            row = {
                "op_name": "matmul", "dtype": dt_name,
                "M": M, "K": K, "N": N,
                "time_ms": f"{avg_ms:.6f}",
                "flops": 2 * M * K * N, "bytes": act_bytes,
            }
            results.append(row)
            append_csv_row(output_path, MATMUL_FIELDS, row)
            done_keys.add(key)
            new_count += 1

            del a, w

    total = len(done_keys) + skip_count  # done_keys now includes all combos
    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif output_path and new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")
    return results
