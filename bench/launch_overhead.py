"""Benchmark CPU→GPU kernel launch overhead.

CUDA events (CudaTimer) measure GPU-side execution time — they cannot see
the CPU-side dispatch overhead (~5-10 µs per kernel launch).  For decode
steps where total GPU time is only ~100-500 µs, this overhead can be 10-30%
of the step time.  For prefill steps (GPU time ~10-100 ms), it's 1-3%.

This benchmark uses CPU wall-clock timing vs GPU event timing to isolate
the pure CPU→GPU dispatch overhead that is invisible to CUDA events.

Method: slope analysis
  - Run a trivial kernel (GPU time ≈ 0) at varying batch sizes N
  - CPU wall-clock = N × (launch_overhead + gpu_time_per_kernel)
  - GPU event time = N × gpu_time_per_kernel
  - Linear regression on CPU time → slope = launch_overhead + gpu_per_kernel
  - launch_overhead = slope(s_cpu) - slope(s_gpu)

We also benchmark a "realistic" small matmul (M=1, K=4096, N=4096) to
validate the overhead estimate in a more representative setting.
"""

import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from bench.utils import CudaTimer

# CSV schema
FIELDS = ["op_name", "dtype", "n_launches", "cpu_time_ms", "gpu_time_ms"]
KEY_FIELDS = ["op_name", "dtype", "n_launches"]

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _trivial_kernel(x: torch.Tensor):
    """Minimal GPU work: ~1 FLOP, ~0 GPU time — dominated by launch overhead."""
    return x.add_(0.001)


def _small_matmul_kernel(a: torch.Tensor, w: torch.Tensor):
    """Representative decode matmul: M=1, K=4096, N=4096."""
    return torch.mm(a, w)


def _measure_overhead_slope(fn_setup, fn_kernel, n_values, warmup=5, trials=3):
    """Measure per-kernel launch overhead via slope analysis.

    For each N in *n_values*, runs N back-to-back kernel launches.  Measures
    both CPU wall-clock time (includes launch queuing + GPU time) and GPU
    event time (pure GPU execution).  Linear regression on both yields the
    per-launch overhead as slope_cpu - slope_gpu.

    Args:
        fn_setup: callable() → args tuple, allocates persistent tensors.
        fn_kernel: callable(*args) → result, the kernel to benchmark.
        n_values: list of int, batch sizes for the slope analysis.
        warmup: number of warmup runs.
        trials: number of measurement trials (median taken).

    Returns:
        dict with keys: overhead_us (float), slope_cpu_us, slope_gpu_us,
        r2_cpu, r2_gpu, all_times (list of (N, cpu_ms, gpu_ms) tuples).
    """
    args = fn_setup()

    # Warmup
    for _ in range(warmup):
        for _ in range(max(n_values)):
            fn_kernel(*args)
    torch.cuda.synchronize()

    all_times = []

    for N in n_values:
        cpu_times = []
        gpu_times = []

        for _ in range(trials):
            # ── CPU wall-clock ──
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                fn_kernel(*args)
            torch.cuda.synchronize()
            cpu_time_s = time.perf_counter() - t0

            # ── GPU event time ──
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(N):
                fn_kernel(*args)
            end.record()
            torch.cuda.synchronize()
            gpu_time_s = start.elapsed_time(end) / 1000.0

            cpu_times.append(cpu_time_s * 1000)  # → ms
            gpu_times.append(gpu_time_s * 1000)  # → ms

        all_times.append((N, float(np.median(cpu_times)), float(np.median(gpu_times))))

    # ── Linear regression: time = intercept + N × slope ──
    xs = np.array([t[0] for t in all_times], dtype=np.float64)
    cpu_ys = np.array([t[1] for t in all_times], dtype=np.float64)
    gpu_ys = np.array([t[2] for t in all_times], dtype=np.float64)

    # slope only (force intercept=0: launch overhead ∝ N)
    # Using polyfit deg=1 for slope; r2 from correlation
    slope_cpu = float(np.sum(xs * cpu_ys) / np.sum(xs * xs))
    slope_gpu = float(np.sum(xs * gpu_ys) / np.sum(xs * xs))

    pred_cpu = slope_cpu * xs
    pred_gpu = slope_gpu * xs
    ss_res_cpu = float(np.sum((cpu_ys - pred_cpu) ** 2))
    ss_tot_cpu = float(np.sum((cpu_ys - np.mean(cpu_ys)) ** 2))
    ss_res_gpu = float(np.sum((gpu_ys - pred_gpu) ** 2))
    ss_tot_gpu = float(np.sum((gpu_ys - np.mean(gpu_ys)) ** 2))
    r2_cpu = 1 - ss_res_cpu / ss_tot_cpu if ss_tot_cpu > 0 else 0
    r2_gpu = 1 - ss_res_gpu / ss_tot_gpu if ss_tot_gpu > 0 else 0

    # Overhead = CPU slope - GPU slope (both in ms per launch)
    overhead_ms = slope_cpu - slope_gpu
    overhead_us = max(overhead_ms * 1000, 0.0)

    return {
        "overhead_us": overhead_us,
        "slope_cpu_us": slope_cpu * 1000,
        "slope_gpu_us": slope_gpu * 1000,
        "r2_cpu": r2_cpu,
        "r2_gpu": r2_gpu,
        "all_times": all_times,
    }


def bench_launch_overhead(cfg, output_path=None):
    """Run kernel launch overhead benchmarks for all configured dtypes.

    Writes results to *output_path* CSV (appends if exists, overwrites
    duplicate keys).  Returns list of result dicts.

    Config keys used (under ``launch_overhead`` in bench.yaml):
        n_values: list of int — kernel launch counts for slope analysis.
        trials: int — measurement trials per N (median taken).
        warmup: int — warmup iterations.

    If ``launch_overhead`` section is absent from config, sensible defaults
    are used.
    """
    lo_cfg = cfg.get("launch_overhead", {})
    n_values = lo_cfg.get("n_values", [10, 50, 100, 200, 500, 1000])
    trials = lo_cfg.get("trials", 5)
    warmup_iters = lo_cfg.get("warmup", 10)
    dtypes = cfg.get("dtype", ["float16"])
    if isinstance(dtypes, str):
        dtypes = [dtypes]

    # Load existing keys for checkpoint/resume
    completed = set()
    if output_path and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = tuple(row.get(k, "") for k in KEY_FIELDS)
                completed.add(key)
    else:
        completed = set()

    results = []
    need_header = not completed

    for dtype in tqdm(dtypes, desc="Launch overhead"):
        if dtype not in DTYPE_MAP:
            print(f"  Skipping dtype {dtype} (not supported)")
            continue
        torch_dtype = DTYPE_MAP[dtype]

        # ── Trivial kernel (add_) ──
        trivial_key = ("launch_trivial", dtype, "slope")
        if trivial_key not in completed:
            x = torch.randn(1, device="cuda", dtype=torch_dtype)
            info = _measure_overhead_slope(
                lambda: (x,),
                lambda t: _trivial_kernel(t),
                n_values, warmup=warmup_iters, trials=trials,
            )
            print(f"  [{dtype}] trivial add_: overhead={info['overhead_us']:.1f} µs  "
                  f"cpu_slope={info['slope_cpu_us']:.1f} µs  "
                  f"gpu_slope={info['slope_gpu_us']:.1f} µs  "
                  f"R²_cpu={info['r2_cpu']:.4f}")

            for N, cpu_ms, gpu_ms in info["all_times"]:
                row = {"op_name": "launch_trivial", "dtype": dtype,
                       "n_launches": N, "cpu_time_ms": cpu_ms,
                       "gpu_time_ms": gpu_ms}
                results.append(row)
                if output_path:
                    _write_row(output_path, row, need_header, FIELDS)
                    need_header = False

            # Store fit result as a special row with n_launches=0
            fit_row = {"op_name": "launch_trivial", "dtype": dtype,
                       "n_launches": 0, "cpu_time_ms": info["overhead_us"] / 1000,
                       "gpu_time_ms": info["slope_gpu_us"] / 1000}
            results.append(fit_row)
            if output_path:
                _write_row(output_path, fit_row, need_header, FIELDS)
                need_header = False

        # ── Small matmul (M=1, K=4096, N=4096) — realistic decode kernel ──
        matmul_key = ("launch_matmul_small", dtype, "slope")
        if matmul_key not in completed:
            K, N = 4096, 4096
            a = torch.randn(1, K, device="cuda", dtype=torch_dtype)
            w = torch.randn(K, N, device="cuda", dtype=torch_dtype)
            info = _measure_overhead_slope(
                lambda: (a, w),
                lambda a_, w_: _small_matmul_kernel(a_, w_),
                n_values, warmup=warmup_iters, trials=trials,
            )
            print(f"  [{dtype}] small matmul (M=1,K={K},N={N}): "
                  f"overhead={info['overhead_us']:.1f} µs  "
                  f"cpu_slope={info['slope_cpu_us']:.1f} µs  "
                  f"gpu_slope={info['slope_gpu_us']:.2f} µs  "
                  f"R²_cpu={info['r2_cpu']:.4f}")

            for N_val, cpu_ms, gpu_ms in info["all_times"]:
                row = {"op_name": "launch_matmul_small", "dtype": dtype,
                       "n_launches": N_val, "cpu_time_ms": cpu_ms,
                       "gpu_time_ms": gpu_ms}
                results.append(row)
                if output_path:
                    _write_row(output_path, row, need_header, FIELDS)
                    need_header = False

            fit_row = {"op_name": "launch_matmul_small", "dtype": dtype,
                       "n_launches": 0, "cpu_time_ms": info["overhead_us"] / 1000,
                       "gpu_time_ms": info["slope_gpu_us"] / 1000}
            results.append(fit_row)
            if output_path:
                _write_row(output_path, fit_row, need_header, FIELDS)
                need_header = False

    return results


def _write_row(path, row, write_header, fields):
    """Append a single row to CSV, writing header if *write_header* is True."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    # Quick smoke test
    import yaml
    cfg = yaml.safe_load(open("config/bench.yaml", encoding="utf-8"))
    bench_launch_overhead(cfg, output_path="bench/results/test/launch_overhead.csv")
