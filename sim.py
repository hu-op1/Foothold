#!/usr/bin/env python3
"""Run a single PD-disaggregation simulation with full time-series recording.

Thin CLI wrapper — delegates to :func:`pd_sim.run_single.run_single`.

Usage::

    uv run python sim.py --config config/pd_sim.yaml -o out/my_run/
    uv run python sim.py --config config/pd_sim.yaml --max-requests 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time


def main():
    parser = argparse.ArgumentParser(
        description="Run a single PD-disaggregation simulation with time-series output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  uv run python sim.py --config config/pd_sim.yaml\n"
               "  uv run python sim.py --config config/pd_sim.yaml -o out/run1",
    )
    parser.add_argument("--config", default="config/sim.yaml")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--tick", type=float, default=0.5, dest="tick_seconds")
    parser.add_argument("--max-requests", type=int, default=None)

    args = parser.parse_args()

    # Deferred imports — keep --help fast
    from pd_sim.config import load_config as load_pd_config
    from pd_sim.trace import load_trace
    from pd_sim.run_single import run_single
    from perf_predict.predict import load_model_specs, load_hw_params

    FIT_RESULTS = os.path.join("fit", "results")

    specs = load_model_specs()
    models = specs.get("models", [])

    cfg = load_pd_config(args.config)
    gpu = cfg.get("gpu", "unknown")

    model_sel = cfg.get("model")
    if not model_sel:
        print("Config is missing `model` field. Available models:")
        for m in models:
            print(f"  {m['name']}")
        sys.exit(1)
    model = next((m for m in models if m["name"] == model_sel), None)
    if not model:
        print(f"Model '{model_sel}' not found.")
        sys.exit(1)

    cfg = load_pd_config(args.config, model_spec=model)

    hw_path = cfg.get("params") or os.path.join(FIT_RESULTS, f"{gpu}.json")
    if not os.path.exists(hw_path):
        hw_path = "perf_predict/fitted_params.json"
    hw = load_hw_params(hw_path)

    max_reqs = args.max_requests or cfg["trace"].get("max_requests")
    requests = load_trace(cfg["trace"]["path"], max_requests=max_reqs,
                          format=cfg["trace"].get("format", "sharegpt"))
    print(f"Loaded {len(requests)} requests from {cfg['trace']['path']}")

    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"pd_sim/output/{_time.strftime('%Y%m%d-%H%M%S')}"

    print(f"Model: {model['name']}  GPU: {gpu}  Output: {output_dir}")

    run_single(cfg, model, hw, requests,
               output_dir=output_dir,
               tick_seconds=args.tick_seconds)


if __name__ == "__main__":
    main()
