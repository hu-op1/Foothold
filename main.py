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
    from pd_sim.report import export_xlsx
    from perf_predict.predict import load_model_specs, load_hw_params

    # Load model specs first (needed for config defaults)
    specs = load_model_specs()
    models = specs.get("models", [])

    cfg = load_pd_config(args.pd_config)
    gpu = cfg.get("gpu", "unknown")

    model_sel = cfg.get("model")
    if not model_sel:
        print("Config is missing `model` field. Available models:")
        for m in models:
            print(f"  {m['name']}")
        return
    model = next((m for m in models if m["name"] == model_sel), None)
    if not model:
        print(f"Model '{model_sel}' not found in model_specs.yaml")
        return

    # Re-load config with model-aware KV cache default
    cfg = load_pd_config(args.pd_config, model_spec=model)

    # Load hardware params
    hw_path = cfg.get("params") or os.path.join(FIT_RESULTS, f"{gpu}.json")
    if not os.path.exists(hw_path):
        hw_path = "perf_predict/fitted_params.json"
    hw = load_hw_params(hw_path)

    # Load trace
    trace_path = cfg["trace"]["path"]
    max_reqs = cfg["trace"].get("max_requests")
    requests = load_trace(trace_path, max_requests=max_reqs)
    print(f"Loaded {len(requests)} requests from {trace_path}")

    # Run strategy search
    mode = cfg["strategy"]["mode"]
    print(f"Strategy mode: {mode}")
    print(f"Model: {model['name']}, GPU: {gpu}")

    engine = SimulationEngine(cfg, model, hw)
    results = search(engine, requests, cfg)

    # Export results to xlsx
    export_xlsx(results, os.path.join("pd_sim", "output", "results.xlsx"))


def main():
    parser = argparse.ArgumentParser(
        description="GPU Roofline Benchmark Suite — benchmark, fit, predict, PD sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  uv run python main.py --bench\n"
               "  uv run python main.py --fit\n"
               "  uv run python main.py --predict\n"
               "  uv run python main.py --pd-sim",
    )

    subs = parser.add_subparsers(dest="mode", title="modes")
    subs.required = False

    # ── bench ──
    bench_p = subs.add_parser("bench", help="Run GPU kernel benchmarks")
    bench_p.add_argument("--config", default="config/default.yaml",
                         help="Benchmark config (default: config/default.yaml)")

    # ── fit ──
    fit_p = subs.add_parser("fit", help="Fit roofline model from benchmark data")
    fit_p.add_argument("--config", default="config/default.yaml",
                       help="Benchmark config for GPU name (default: config/default.yaml)")
    fit_p.add_argument("--dir", default=None, metavar="DIR",
                       help="Override bench results directory")

    # ── predict ──
    pred_p = subs.add_parser("predict", help="Predict inference throughput")
    pred_p.add_argument("--config", default="config/predict.yaml",
                        help="Predict config (default: config/predict.yaml)")

    # ── pd-sim ──
    pd_p = subs.add_parser("pd-sim", help="Run PD disaggregation simulation")
    pd_p.add_argument("--config", default="config/pd_sim.yaml",
                      help="Simulation config (default: config/pd_sim.yaml)")

    # Backward-compat: support --bench / --fit / --predict / --pd-sim flags too
    parser.add_argument("--bench", action="store_true", dest="_flag_bench",
                        help=argparse.SUPPRESS)
    parser.add_argument("--fit", default=None, nargs="?", const="", metavar="DIR",
                        dest="_flag_fit", help=argparse.SUPPRESS)
    parser.add_argument("--predict", action="store_true", dest="_flag_predict",
                        help=argparse.SUPPRESS)
    parser.add_argument("--pd-sim", action="store_true", dest="_flag_pd_sim",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", default="config/default.yaml", dest="_flag_config",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Route based on subparser mode, fall back to flags for backward compat
    mode = args.mode
    if mode is None:
        # Check backward-compat flags
        if getattr(args, "_flag_bench", False):
            mode = "bench"
        elif getattr(args, "_flag_fit", None) is not None:
            mode = "fit"
        elif getattr(args, "_flag_predict", False):
            mode = "predict"
        elif getattr(args, "_flag_pd_sim", False):
            mode = "pd-sim"

    if mode == "bench":
        args.config = getattr(args, "config", "config/default.yaml")
        run_benchmarks(args)
    elif mode == "fit":
        fit_dir = getattr(args, "dir", None)
        args.fit = fit_dir if fit_dir else getattr(args, "_flag_fit", "")
        args.config = getattr(args, "config", "config/default.yaml")
        run_fit(args)
    elif mode == "predict":
        args.predict_config = getattr(args, "config", "config/predict.yaml")
        run_predict(args)
    elif mode == "pd-sim":
        args.pd_config = getattr(args, "config", "config/pd_sim.yaml")
        run_pd_sim(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
