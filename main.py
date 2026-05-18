#!/usr/bin/env python3
"""GPU operator benchmark suite — characterize hardware for throughput prediction.

Usage:
    uv run python main.py                        # run benchmarks
    uv run python main.py --config my_conf.yaml  # custom config
    uv run python main.py --fit results/         # fit existing results
"""

import argparse
import os
import sys
import time

import torch
import yaml
from tqdm import tqdm

from bench.matmul import bench_matmul
from bench.elementwise import bench_elementwise


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_benchmarks(args):
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    cfg = load_config(args.config)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GiB)")
    print(f"Config: {args.config}")
    print(f"dtype: {cfg['dtype']}, warmup={cfg['warmup_iters']}, "
          f"iters={cfg['bench_iters']}")
    matmul_combos = len(cfg["matmul"]["M"]) * len(cfg["matmul"]["K"]) * len(cfg["matmul"]["N"])
    elem_combos = len(cfg["elementwise"]["N"]) * len(cfg["elementwise"]["operators"])
    print(f"Matmul combos: {matmul_combos}, Elementwise combos: {elem_combos}")
    print("=" * 70)

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    matmul_xlsx = os.path.join(out_dir, "matmul.xlsx")
    elem_xlsx = os.path.join(out_dir, "elementwise.xlsx")

    t0 = time.perf_counter()

    tqdm.write("\n[Matmul]")
    bench_matmul(cfg, output_path=matmul_xlsx)
    torch.cuda.empty_cache()

    tqdm.write("\n[Elementwise]")
    bench_elementwise(cfg, output_path=elem_xlsx)
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output files: {matmul_xlsx}, {elem_xlsx}")


def run_fit(results_dir):
    from fit import load_results, fit_all

    matmul_xlsx = os.path.join(results_dir, "matmul.xlsx")
    elem_xlsx = os.path.join(results_dir, "elementwise.xlsx")

    results = []
    for path in [matmul_xlsx, elem_xlsx]:
        if os.path.exists(path):
            results.extend(load_results(path))
        else:
            print(f"Warning: not found: {path}")

    if not results:
        print("No valid results to fit.")
        return

    print(f"Loaded {len(results)} rows from {results_dir}")
    fit_all(results)


def main():
    parser = argparse.ArgumentParser(description="GPU Roofline Benchmark Suite")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", default="results",
        help="Directory to save output xlsx files",
    )
    parser.add_argument(
        "--fit", default=None, metavar="DIR",
        help="Fit existing results in DIR (skips benchmarks)",
    )
    args = parser.parse_args()

    if args.fit:
        run_fit(args.fit)
    else:
        run_benchmarks(args)


if __name__ == "__main__":
    main()
