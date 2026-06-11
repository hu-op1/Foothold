"""Terminal table and JSON output for simulation results."""

import json
import os


def print_comparison_table(results: list[dict], cfg: dict) -> None:
    """Print a comparison table of strategy results.

    Args:
        results: list of {label, metrics: MetricsCollector, score}
        cfg: simulation config.
    """
    slo = cfg["slo"]

    print()
    print("=" * 90)
    print("  PD Strategy Comparison")
    trace_path = cfg["trace"]["path"]
    model_name = cfg.get("model", "unknown")
    gpu = cfg.get("gpu", "unknown")
    total_gpus = cfg["strategy"].get("total_gpus", "?")
    n_reqs = results[0]["metrics"].num_requests if results else 0
    print(f"  Trace: {trace_path} ({n_reqs} requests)")
    print(f"  Model: {model_name}, GPU: {gpu} x{total_gpus}")
    print("=" * 90)

    header = f"  {'Strategy':<20} | {'Thrpt(tok/s)':>12} | {'TTFT(ms)':>9} | {'TPOT(ms)':>9} | {'P99(ms)':>9} | {'SLO%':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best = max(results, key=lambda r: r["score"])
    for entry in results:
        m = entry["metrics"]
        marker = " ★" if entry is best else ""
        slo_info = m.slo_compliance(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])
        print(
            f"  {entry['label']:<20} | {m.throughput():>12.0f} | "
            f"{m.mean_ttft()*1000:>8.1f} | {m.mean_tpot()*1000:>8.1f} | "
            f"{m.p99_latency()*1000:>8.1f} | "
            f"{slo_info['score']*100:>5.1f}%{marker}"
        )

    if best:
        print()
        print(f"  ★ Optimal: {best['label']}")

    print()


def export_json(results: list[dict], path: str) -> None:
    """Export results to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    output = []
    for entry in results:
        m = entry["metrics"]
        output.append({
            "label": entry["label"],
            "throughput_tok_s": m.throughput(),
            "mean_ttft_ms": m.mean_ttft() * 1000,
            "mean_tpot_ms": m.mean_tpot() * 1000,
            "p50_latency_ms": m.p50_latency() * 1000,
            "p95_latency_ms": m.p95_latency() * 1000,
            "p99_latency_ms": m.p99_latency() * 1000,
            "num_requests": m.num_requests,
            "total_output_tokens": m.total_output_tokens,
            "total_time_s": m.total_time,
            "score": entry["score"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results exported to: {path}")
