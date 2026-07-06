"""Run a single simulation with full time-series recording.

Produces the three artifacts that :file:`LLMServingSim/bench/validate.sh`
expects for side-by-side comparison::

    <output_dir>/meta.json
    <output_dir>/requests.jsonl
    <output_dir>/timeseries.csv

Also writes ``results.csv`` alongside for convenience.

Reads from ``config/sim.yaml`` — scalar strategy values, no search grid.
"""

from __future__ import annotations

import json
import os
import time

from sim.engine import SimulationEngine
from sim.recorder import SimRecorder
from sim.report import export_csv


def run_single(
    cfg: dict,
    model_spec: dict,
    hw_params: dict,
    requests: list,
    *,
    output_dir: str,
    tick_seconds: float = 0.5,
    extra_meta: dict | None = None,
) -> str:
    """Run one simulation configuration and write all artifacts to *output_dir*.

    Returns the path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    sim_cfg = cfg.get("simulation", cfg)
    strat = cfg.get("strategy", {})
    mode = strat.get("mode", "colocated")
    max_tokens = sim_cfg.get("max_num_batched_tokens", 8192)
    threshold = sim_cfg.get("long_prefill_token_threshold", 1024)
    slo = cfg.get("slo", {})
    gpu_name = cfg.get("gpu", "unknown")
    trace_path = cfg.get("trace", {}).get("path", "")

    # Build a clean per-run config
    run_cfg = json.loads(json.dumps(cfg))
    run_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
    run_cfg["simulation"]["long_prefill_token_threshold"] = threshold

    # ── Run ──────────────────────────────────────────────────────────
    recorder = SimRecorder(tick_seconds=tick_seconds)
    recorder.start()

    engine = SimulationEngine(run_cfg, model_spec, hw_params)

    t0 = time.perf_counter()

    tp_size = pp_size = 1
    dp = 0
    pd_ratio = (1, 1)
    d_tp = d_pp = 1

    if mode == "colocated":
        tp_size = strat.get("tp_size", 1)
        pp_size = strat.get("pp_size", 1)
        total_gpus = strat.get("total_gpus", 1)
        dp = total_gpus // (tp_size * pp_size)
        metrics = engine.run(list(requests), mode="colocated",
                             tp_size=tp_size, dp=dp, pp_size=pp_size,
                             recorder=recorder)
    else:
        pd_ratio = strat.get("pd_ratio", [1, 1])
        p_tp = strat.get("p_tp_size", 1)
        p_pp = strat.get("p_pp_size", 1)
        d_tp = strat.get("d_tp_size", 1)
        d_pp = strat.get("d_pp_size", 1)
        tp_size, pp_size = p_tp, p_pp
        total_gpus = p_tp * p_pp * pd_ratio[0] + d_tp * d_pp * pd_ratio[1]
        metrics = engine.run(list(requests), mode="disaggregated",
                             pd_ratio=pd_ratio, tp_size=p_tp,
                             d_tp_size=d_tp,
                             pp_size=p_pp, d_pp_size=d_pp,
                             recorder=recorder)

    elapsed = time.perf_counter() - t0
    recorder.finish()

    # ── Write artifacts ──────────────────────────────────────────────
    slo_info = metrics.slo_compliance(slo.get("p90_ttft_ms", 400),
                                      slo.get("p90_tpot_ms", 100))

    meta_extra = extra_meta or {}
    meta_extra.update({
        "mode": mode,
        "gpu": gpu_name,
        "total_gpus": total_gpus,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "max_batched_tokens": max_tokens,
        "prefill_threshold": threshold,
        "throughput_tok_s": metrics.throughput(),
        "ttft_mean_ms": metrics.mean_ttft() * 1000,
        "tpot_mean_ms": metrics.mean_tpot() * 1000,
        "slo_pass": slo_info["slo_pass"],
        "elapsed_s": round(elapsed, 3),
    })
    if mode == "disaggregated":
        meta_extra["pd_ratio"] = list(pd_ratio)
        meta_extra["d_tp_size"] = d_tp
        meta_extra["d_pp_size"] = d_pp
    if mode == "colocated":
        meta_extra["dp"] = dp

    recorder.write(output_dir,
                   model=model_spec.get("name", ""),
                   trace_path=trace_path,
                   extra_meta=meta_extra)

    # Also write the standard CSV for convenience
    total_t = metrics.total_time
    csv_result = [{
        "label": f"{mode} (batch={max_tokens}, thr={threshold})",
        "metrics_raw": {
            "input_throughput": metrics.input_throughput(),
            "output_throughput": metrics.throughput(),
            "total_throughput": metrics.total_throughput(),
            "mean_ttft_ms": metrics.mean_ttft() * 1000,
            "p50_ttft_ms": metrics.p50_ttft() * 1000,
            "p90_ttft_ms": metrics.p90_ttft() * 1000,
            "p99_ttft_ms": metrics.p99_ttft() * 1000,
            "mean_tpot_ms": metrics.mean_tpot() * 1000,
            "p50_tpot_ms": metrics.p50_tpot() * 1000,
            "p90_tpot_ms": metrics.p90_tpot() * 1000,
            "p99_tpot_ms": metrics.p99_tpot() * 1000,
            "p50_ms": metrics.p50_latency() * 1000,
            "p90_ms": metrics.p90_latency() * 1000,
            "p95_ms": metrics.p95_latency() * 1000,
            "p99_ms": metrics.p99_latency() * 1000,
            "num_requests": metrics.num_requests,
            "total_input_tokens": metrics.total_input_tokens,
            "total_output_tokens": metrics.total_output_tokens,
            "total_time_s": total_t,
            "cache_hit_rate": metrics.cache_hit_rate
            if metrics.cache_hit_rate is not None else 0.0,
        },
        "slo_pass": slo_info["slo_pass"],
        "elapsed": elapsed,
    }]
    # Add time breakdown percentages
    breakdown = metrics.time_breakdown_pct()
    csv_result[0]["metrics_raw"].update(breakdown)
    export_csv(csv_result, os.path.join(output_dir, "results.csv"))

    print()
    width = 60
    print("━" * width)
    print(f"  Simulation Complete ({elapsed:.1f}s)".center(width - 2))
    print("━" * width)
    print(f"  {'Throughput':<22} {metrics.throughput():>10.1f} tok/s")
    print(f"  {'Cache hit rate':<22} {metrics.cache_hit_rate*100 if metrics.cache_hit_rate else 0:>10.1f}%")
    print(f"  {'SLO pass':<22} {'YES' if slo_info['slo_pass'] else 'NO':>10}")
    print("  " + "─" * (width - 4))
    print(f"  {'TTFT mean':<22} {metrics.mean_ttft() * 1000:>10.1f} ms")
    print(f"  {'TTFT p50 / p90 / p99':<22} {metrics.p50_ttft() * 1000:>6.1f} / {metrics.p90_ttft() * 1000:>5.1f} / {metrics.p99_ttft() * 1000:>5.1f} ms")
    print("  " + "─" * (width - 4))
    print(f"  {'TPOT mean':<22} {metrics.mean_tpot() * 1000:>10.1f} ms")
    print(f"  {'TPOT p50 / p90 / p99':<22} {metrics.p50_tpot() * 1000:>6.1f} / {metrics.p90_tpot() * 1000:>5.1f} / {metrics.p99_tpot() * 1000:>5.1f} ms")
    print("  " + "─" * (width - 4))
    print(f"  {'Latency p50 / p90 / p99':<22} {metrics.p50_latency() * 1000:>6.1f} / {metrics.p90_latency() * 1000:>5.1f} / {metrics.p99_latency() * 1000:>5.1f} ms")
    print("  " + "─" * (width - 4))
    print(f"  {'Requests':<22} {metrics.num_requests:>10}")
    print(f"  {'Input / output tokens':<22} {metrics.total_input_tokens:>8,} / {metrics.total_output_tokens:,}")
    print(f"  {'Wall / sim time':<22} {elapsed:>8.1f}s / {metrics.total_time:>5.1f}s")
    print("━" * width)
    print(f"  Output: {output_dir}")
    print(f"  {'meta.json':>9}  {'requests.jsonl':>17}  "
          f"{'timeseries.csv':>17}  {'results.csv':>16}")

    return output_dir
