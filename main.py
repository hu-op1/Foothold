#!/usr/bin/env python3
"""GPU roofline benchmark suite — benchmark, fit, predict.

All paths auto-derived from config's `gpu` field. Override with CLI flags.

Usage:
    uv run python main.py --bench                            # run benchmarks → bench/results/<gpu>/
    uv run python main.py --fit                              # fit and save → fit/results/<gpu>.json
    uv run python main.py --predict                          # predict from config/predict.yaml
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

BENCH_RESULTS = os.path.join("bench", "results")
FIT_RESULTS = os.path.join("fit", "results")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def bench_results_dir(cfg):
    gpu = cfg.get("gpu")
    if not gpu:
        sys.exit("Config is missing `gpu` field. Add e.g. `gpu: \"3090\"` to your config.")
    return os.path.join(BENCH_RESULTS, gpu)

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

    out_dir = bench_results_dir(cfg)
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

    cfg = load_config(args.config)

    bench_dir = args.fit or bench_results_dir(cfg)
    matmul_xlsx = os.path.join(bench_dir, "matmul.xlsx")
    elem_xlsx = os.path.join(bench_dir, "elementwise.xlsx")

    results = []
    for path in [matmul_xlsx, elem_xlsx]:
        if os.path.exists(path):
            results.extend(load_results(path))
        else:
            print(f"Warning: not found: {path}")

    if not results:
        print("No valid results to fit.")
        return

    print(f"Loaded {len(results)} rows from {bench_dir}")
    params = fit_all(results)

    gpu = cfg.get("gpu", "unknown")
    os.makedirs(FIT_RESULTS, exist_ok=True)
    save_path = os.path.join(FIT_RESULTS, f"{gpu}.json")
    save_fitted_params(params, save_path)


def run_predict(args):
    from perf_predict.predict import load_model_specs, load_hw_params, predict, print_one, print_all

    cfg = load_config(args.predict_config)
    gpu = cfg.get("gpu", "unknown")
    model_sel = cfg.get("model", "all")
    batch = cfg.get("batch", 1)
    input_len = cfg.get("input_len", 2048)
    output_len = cfg.get("output_len", 512)

    specs = load_model_specs()
    models = specs.get("models", [])

    if model_sel == "list":
        print("Available models:")
        for m in models:
            na = m.get("attn_layers", m["num_layers"])
            print(f"  {m['name']:<20}  h={m['hidden_dim']}, nh={m['num_heads']}, "
                  f"nl={m['num_layers']}, attn_layers={na}")
        return

    params_path = cfg.get("params") or os.path.join(FIT_RESULTS, f"{gpu}.json")
    if not os.path.exists(params_path):
        params_path = "perf_predict/fitted_params.json"

    try:
        hw_params = load_hw_params(params_path)
    except FileNotFoundError:
        print(f"Fitted params not found: {params_path}")
        print("Run: uv run python main.py --fit")
        return

    if model_sel == "all":
        print_all(models, batch, input_len, output_len, hw_params)
    else:
        model = next((m for m in models if m["name"] == model_sel), None)
        if not model:
            print(f"Model not found: {model_sel}")
            return
        r = predict(model, batch, input_len, output_len, hw_params)
        print_one(model_sel, model, r, batch, input_len, output_len)


def run_pd_sim(args):
    from pd_sim.config import load_config as load_pd_config
    from pd_sim.trace import load_trace
    from pd_sim.engine import SimulationEngine
    from pd_sim.strategy import search
    from pd_sim.report import print_comparison_table, export_json
    from perf_predict.predict import load_model_specs, load_hw_params

    cfg = load_pd_config(args.pd_config)
    gpu = cfg.get("gpu", "unknown")

    # Load model specs
    specs = load_model_specs()
    models = specs.get("models", [])
    model_sel = cfg.get("model", "all")
    if model_sel == "all":
        model = models[0]
    else:
        model = next((m for m in models if m["name"] == model_sel), None)
        if not model:
            print(f"Model not found: {model_sel}")
            return

    # Load hardware params
    hw_path = cfg.get("params") or os.path.join(FIT_RESULTS, f"{gpu}.json")
    if not os.path.exists(hw_path):
        hw_path = "perf_predict/fitted_params.json"
    hw = load_hw_params(hw_path)

    # Load trace
    trace_path = cfg["trace"]["path"]
    max_reqs = cfg["trace"].get("max_requests")
    trace_fmt = cfg["trace"].get("format", "sharegpt")
    requests = load_trace(trace_path, fmt=trace_fmt, max_requests=max_reqs)
    print(f"Loaded {len(requests)} requests from {trace_path}")

    # Run strategy search
    mode = cfg["strategy"]["mode"]
    print(f"Strategy mode: {mode}")
    print(f"Model: {model['name']}, GPU: {gpu}")

    engine = SimulationEngine(cfg, model, hw)
    results = search(engine, requests, cfg)

    # Report
    print_comparison_table(results, cfg)
    export_json(results, os.path.join("pd_sim", "output", "results.json"))


def main():
    parser = argparse.ArgumentParser(description="GPU Roofline Benchmark Suite")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file")
    # ── mode ──
    parser.add_argument("--bench", action="store_true", help="Run GPU kernel benchmarks → bench/results/<gpu>/")
    parser.add_argument("--fit", default=None, nargs="?", const="", metavar="DIR", help="Fit roofline model → fit/results/<gpu>.json")
    parser.add_argument("--predict", action="store_true", help="Predict inference throughput (config/predict.yaml)")
    parser.add_argument("--predict-config", default="config/predict.yaml", help="Path to predict config")
    parser.add_argument("--pd-sim", action="store_true",
                        help="Run PD disaggregation simulation (config/pd_sim.yaml)")
    parser.add_argument("--pd-config", default="config/pd_sim.yaml",
                        help="Path to pd_sim config")
    args = parser.parse_args()

    fit_mode = args.fit is not None
    if args.fit == "":
        args.fit = None

    if fit_mode:
        run_fit(args)
    elif args.predict:
        run_predict(args)
    elif args.bench:
        run_benchmarks(args)
    elif args.pd_sim:
        run_pd_sim(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
