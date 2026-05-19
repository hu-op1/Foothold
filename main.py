#!/usr/bin/env python3
"""GPU roofline benchmark suite — benchmark, fit, predict.

Usage:
    uv run python main.py                                    # run benchmarks
    uv run python main.py --fit results/ --save fitted.json  # fit + save
    uv run python main.py --predict --model "Llama-2-7B"     # predict one
    uv run python main.py --predict --all                    # predict all
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
    with open(path, encoding="utf-8") as f:
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


def run_fit(args):
    from fit import load_results, fit_all, save_fitted_params

    results_dir = args.fit
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
    params = fit_all(results)

    if args.save:
        save_fitted_params(params, args.save)


def run_predict(args):
    from perf_predict.predict import load_model_specs, load_hw_params, predict, print_one, print_all

    specs = load_model_specs()
    models = specs.get("models", [])

    if args.list:
        print("Available models:")
        for m in models:
            na = m.get("attn_layers", m["num_layers"])
            print(f"  {m['name']:<20}  h={m['hidden_dim']}, nh={m['num_heads']}, "
                  f"nl={m['num_layers']}, attn_layers={na}")
        return

    params_path = args.params or "perf_predict/fitted_params.json"
    try:
        hw_params = load_hw_params(params_path)
    except FileNotFoundError:
        print(f"Fitted params not found: {params_path}")
        print("Run: uv run python main.py --fit results/ --save <path>")
        return

    if args.all:
        print_all(models, args.batch, args.input_len, args.output_len, hw_params)
    elif args.model:
        model = next((m for m in models if m["name"] == args.model), None)
        if not model:
            print(f"Model not found: {args.model}")
            return
        r = predict(model, args.batch, args.input_len, args.output_len, hw_params)
        print_one(args.model, model, r, args.batch, args.input_len, args.output_len)
    else:
        print("Specify --model <name> or --all")


def main():
    parser = argparse.ArgumentParser(description="GPU Roofline Benchmark Suite")
    # ── bench ──
    parser.add_argument("--config", default="config/default.yaml", help="[bench] Path to YAML config file")
    parser.add_argument("--output", default="results", help="[bench] Directory to save output xlsx files")
    # ── fit ──
    parser.add_argument("--fit", default=None, metavar="DIR", help="[fit] Fit existing results in DIR")
    parser.add_argument("--save", default=None, metavar="PATH", help="[fit] Save fitted params JSON")
    # ── predict ──
    parser.add_argument("--predict", action="store_true", help="[predict] Predict inference throughput")
    parser.add_argument("--model", type=str, help="[predict] Model name")
    parser.add_argument("--all", action="store_true", help="[predict] Predict all models")
    parser.add_argument("--list", action="store_true", help="[predict] List available models")
    parser.add_argument("--params", type=str, help="[predict] Path to fitted_params.json")
    parser.add_argument("--batch", type=int, default=1, help="[predict] Batch size")
    parser.add_argument("--input-len", type=int, default=2048, help="[predict] Prompt length (tokens)")
    parser.add_argument("--output-len", type=int, default=512, help="[predict] Generation length (tokens)")
    args = parser.parse_args()

    if args.fit:
        run_fit(args)
    elif args.predict:
        run_predict(args)
    else:
        run_benchmarks(args)


if __name__ == "__main__":
    main()
