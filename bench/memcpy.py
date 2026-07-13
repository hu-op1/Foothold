"""Benchmark GPU↔CPU memory copy bandwidth (D2H / H2D).

Produces a byte-size → transfer-time lookup table for the simulator.
"""

import os
import torch
from tqdm import tqdm

from bench.utils import (
    CudaTimer, warmup, benchmark, auto_warmup_iters,
    check_memory, load_completed_keys, append_csv_row,
)

MEMCPY_FIELDS = ["op_name", "direction", "dtype", "bytes", "time_ms"]
MEMCPY_KEY_FIELDS = ["op_name", "direction", "dtype", "bytes"]


def bench_memcpy(config, output_path="results/memcpy.csv"):
    dtypes = config.get("dtype", ["float16"])
    if isinstance(dtypes, str):
        dtypes = [dtypes]
    warmup_cfg = config.get("warmup", "auto")
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_min_time = config.get("min_time_ms", 200)
    bench_max_iters = config.get("max_iters", 10000)
    bench_calib = config.get("calib_iters", 20)
    max_mem = config["max_memory_gb"]

    memcpy_cfg = config.get("memcpy")
    if not memcpy_cfg:
        print("  [skip] No memcpy config found")
        return []

    byte_sizes = sorted(memcpy_cfg["bytes"])

    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    done_keys = load_completed_keys(output_path, MEMCPY_KEY_FIELDS)

    device = torch.device("cuda")

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = dtype.itemsize if hasattr(dtype, "itemsize") else 2

        pbar = tqdm(byte_sizes, desc=f"memcpy {dt_name}")
        for nbytes in pbar:
            n_elems = max(1, nbytes // dt_bytes)

            for direction in ("d2h", "h2d"):
                key = ("memcpy", direction, dt_name, nbytes)
                if key in done_keys:
                    continue

                act_gb = (n_elems * dt_bytes * 2) / (1024 ** 3)
                if not check_memory(act_gb, max_mem):
                    row = {"op_name": "memcpy", "direction": direction,
                           "dtype": dt_name, "bytes": nbytes,
                           "time_ms": "OOM"}
                    append_csv_row(output_path, MEMCPY_FIELDS, row)
                    done_keys.add(key)
                    continue

                x_gpu = torch.randn(n_elems, dtype=dtype, device=device)
                x_cpu = torch.empty(n_elems, dtype=dtype, pin_memory=True)

                if direction == "d2h":
                    def fn(x_gpu=x_gpu, x_cpu=x_cpu):
                        x_cpu.copy_(x_gpu, non_blocking=True)
                        torch.cuda.synchronize()
                else:
                    def fn(x_gpu=x_gpu, x_cpu=x_cpu):
                        x_gpu.copy_(x_cpu, non_blocking=True)
                        torch.cuda.synchronize()

                if warmup_cfg == "auto":
                    wu = auto_warmup_iters(fn, bench_min_time, bench_max_iters,
                                           bench_calib, warmup_ratio)
                else:
                    wu = int(warmup_cfg)
                warmup(fn, wu)

                ms = benchmark(fn, min_time_ms=bench_min_time,
                               max_iters=bench_max_iters, calib_iters=bench_calib)

                row = {"op_name": "memcpy", "direction": direction,
                       "dtype": dt_name, "bytes": nbytes,
                       "time_ms": f"{ms:.6f}"}
                append_csv_row(output_path, MEMCPY_FIELDS, row)
                done_keys.add(key)

                del x_gpu, x_cpu

    return []
