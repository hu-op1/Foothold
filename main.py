#!/usr/bin/env python3
"""GPU roofline benchmark suite — benchmark, fit, strategy search.

Usage:
    uv run python main.py bench                   # GPU kernel microbenchmarks
    uv run python main.py fit                     # Roofline model fitting
    uv run python main.py search                  # PD disaggregation strategy search

Single simulation with time-series output:
    uv run python main.py sim -o out/my_run/
"""

import argparse
import json
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


def run_search(args):
    from sim.config import load_config as load_pd_config
    from sim.trace import load_trace
    from sim.engine import SimulationEngine
    from sim.strategy import search
    from sim.report import export_xlsx
    from sim.config import load_model_spec

    cfg = load_pd_config(args.config)
    gpu = cfg.get("gpu", "unknown")

    model_sel = cfg.get("model")
    if not model_sel:
        print("Config is missing `model` field.")
        return
    model = load_model_spec(model_sel)
    if not model:
        print(f"Model '{model_sel}' could not be loaded via AutoConfig")
        return

    # Re-load config with model-aware KV cache default
    cfg = load_pd_config(args.config, model_spec=model)

    # Load hardware params
    hw_path = cfg.get("params") or os.path.join(FIT_RESULTS, f"{gpu}.json")
    with open(hw_path, encoding="utf-8") as f:
        hw = json.load(f)

    # Load trace
    trace_path = cfg["trace"]["path"]
    max_reqs = cfg["trace"].get("max_requests")
    trace_format = cfg["trace"].get("format", "sharegpt")
    requests = load_trace(trace_path, max_requests=max_reqs, format=trace_format)
    print(f"Loaded {len(requests)} requests from {trace_path} (format={trace_format})")

    # Run strategy search
    mode = cfg["strategy"]["mode"]
    print(f"Strategy mode: {mode}")
    print(f"Model: {model['name']}, GPU: {gpu}")

    engine = SimulationEngine(cfg, model, hw)
    results = search(engine, requests, cfg)

    # Export results to xlsx
    out_path = str(cfg.get("output", "sim/output/results.xlsx"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_xlsx(results, out_path)


def run_sim(args):
    """Run a single simulation with time-series output for LLMServingSim comparison."""
    from sim.config import load_config as load_pd_config
    from sim.trace import load_trace
    from sim.run_single import run_single
    from sim.config import load_model_spec

    cfg = load_pd_config(args.config)
    gpu = cfg.get("gpu", "unknown")

    model_sel = cfg.get("model")
    if not model_sel:
        print("Config is missing `model` field.")
        return
    model = load_model_spec(model_sel)
    if not model:
        print(f"Model '{model_sel}' could not be loaded via AutoConfig")
        return

    cfg = load_pd_config(args.config, model_spec=model)

    hw_path = cfg.get("params") or os.path.join(FIT_RESULTS, f"{gpu}.json")
    with open(hw_path, encoding="utf-8") as f:
        hw = json.load(f)

    max_reqs = getattr(args, "max_requests", None) or cfg["trace"].get("max_requests")
    requests = load_trace(cfg["trace"]["path"], max_requests=max_reqs,
                          format=cfg["trace"].get("format", "sharegpt"))
    print(f"Loaded {len(requests)} requests from {cfg['trace']['path']}")

    output_dir = args.output_dir
    if not output_dir:
        import time as _time
        output_dir = f"sim/output/{_time.strftime('%Y%m%d-%H%M%S')}"

    print(f"Model: {model['name']}  GPU: {gpu}  Output: {output_dir}")

    run_single(cfg, model, hw, requests,
               output_dir=output_dir,
               tick_seconds=args.tick_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Roofline Benchmark Suite — benchmark, fit, search, sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  uv run python main.py bench\n"
               "  uv run python main.py fit\n"
               "  uv run python main.py search\n"
               "  uv run python main.py sim -o out/run1",
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

    # ── search ──
    search_p = subs.add_parser("search", help="Grid search for optimal PD strategy")
    search_p.add_argument("--config", default="config/search.yaml",
                      help="Simulation config (default: config/search.yaml)")

    # ── sim ──
    sim_p = subs.add_parser("sim", help="Run a single simulation with time-series output")
    sim_p.add_argument("--config", default="config/sim.yaml")
    sim_p.add_argument("-o", "--output-dir", default=None)
    sim_p.add_argument("--tick-seconds", type=float, default=0.5)
    sim_p.add_argument("--max-requests", type=int, default=None)

    # Backward-compat: support --bench / --fit / --search flags too
    parser.add_argument("--bench", action="store_true", dest="_flag_bench",
                        help=argparse.SUPPRESS)
    parser.add_argument("--fit", default=None, nargs="?", const="", metavar="DIR",
                        dest="_flag_fit", help=argparse.SUPPRESS)
    parser.add_argument("--search", action="store_true", dest="_flag_search",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", default="config/default.yaml", dest="_flag_config",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Route based on subparser mode, fall back to flags for backward compat
    mode = args.mode
    if mode is None:
        if getattr(args, "_flag_bench", False):
            mode = "bench"
        elif getattr(args, "_flag_fit", None) is not None:
            mode = "fit"
        elif getattr(args, "_flag_search", False):
            mode = "search"

    if mode == "bench":
        args.config = getattr(args, "config", "config/default.yaml")
        run_benchmarks(args)
    elif mode == "fit":
        fit_dir = getattr(args, "dir", None)
        args.fit = fit_dir if fit_dir else getattr(args, "_flag_fit", "")
        args.config = getattr(args, "config", "config/default.yaml")
        run_fit(args)
    elif mode == "search":
        args.config = getattr(args, "config", "config/search.yaml")
        run_search(args)
    elif mode == "sim":
        args.config = getattr(args, "config", "config/sim.yaml")
        run_sim(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
