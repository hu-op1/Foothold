"""Run a single simulation with full time-series recording.

Produces the three artifacts that :file:`LLMServingSim/bench/validate.sh`
expects for side-by-side comparison::

    <output_dir>/meta.json
    <output_dir>/requests.jsonl
    <output_dir>/timeseries.csv

Also writes ``results.xlsx`` alongside for convenience.

Reads from ``config/sim.yaml`` — scalar strategy values, no search grid.
"""

from __future__ import annotations

import json
import os
import time

from sim.engine import SimulationEngine
from sim.recorder import SimRecorder
from sim.report import export_xlsx


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
    score = metrics.score(slo.get("ttft_ms", 400), slo.get("tpot_ms", 100),
                          slo.get("p99_latency_ms", 4000))
    slo_info = metrics.slo_compliance(slo.get("ttft_ms", 400),
                                      slo.get("tpot_ms", 100),
                                      slo.get("p99_latency_ms", 4000))

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
        "slo_score": slo_info["score"],
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

    # Also write the standard XLSX for convenience
    total_t = metrics.total_time
    xlsx_result = [{
        "label": f"{mode} (batch={max_tokens}, thr={threshold})",
        "metrics_raw": {
            "throughput": metrics.throughput(),
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
        "score": score,
        "elapsed": elapsed,
    }]
    # Add time breakdown percentages
    breakdown = metrics.time_breakdown_pct()
    xlsx_result[0]["metrics_raw"].update(breakdown)
    export_xlsx(xlsx_result, os.path.join(output_dir, "results.xlsx"))

    print(f"\nSimulation complete ({elapsed:.1f}s)")
    print(f"  Throughput: {metrics.throughput():.1f} tok/s")
    print(f"  TTFT mean:  {metrics.mean_ttft() * 1000:.1f} ms")
    print(f"  TPOT mean:  {metrics.mean_tpot() * 1000:.1f} ms")
    print(f"  Requests:   {metrics.num_requests}")
    print(f"  Output:     {output_dir}")
    print(f"  meta.json  requests.jsonl  timeseries.csv  results.xlsx")

    return output_dir
