"""Terminal table and JSON output for simulation results."""

import json
import os


def print_comparison_table(results: list[dict], cfg: dict) -> None:
    """Print a comparison table of strategy results.

    Args:
        results: list of {label, metrics_raw, score}
        cfg: simulation config.
    """
    trace_path = cfg["trace"]["path"]
    gpu = cfg.get("gpu", "unknown")
    total_gpus = cfg["strategy"].get("total_gpus", "?")

    print()
    print("=" * 90)
    print("  PD Strategy Comparison")
    if results:
        n_reqs = results[0]["metrics_raw"]["num_requests"]
        print(f"  Trace: {trace_path} ({n_reqs} requests)")
    print(f"  GPU: {gpu} x{total_gpus}")
    print("=" * 90)

    header = f"  {'Strategy':<32} | {'Thrpt(tok/s)':>12} | {'TTFT(ms)':>9} | {'TPOT(ms)':>9} | {'P99(ms)':>9} | {'SLO%':>6} | {'Time':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best = max(results, key=lambda r: r["score"])
    for entry in results:
        m = entry["metrics_raw"]
        marker = " ★" if entry is best else ""
        print(
            f"  {entry['label']:<32} | {m['throughput']:>12.0f} | "
            f"{m['mean_ttft_ms']:>8.1f} | {m['mean_tpot_ms']:>8.1f} | "
            f"{m['p99_ms']:>8.1f} | "
            f"{entry['slo_score']*100:>5.1f}% | "
            f"{entry['elapsed']:>4.1f}s{marker}"
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
        m = entry["metrics_raw"]
        output.append({
            "label": entry["label"],
            "throughput_tok_s": m["throughput"],
            "mean_ttft_ms": m["mean_ttft_ms"],
            "mean_tpot_ms": m["mean_tpot_ms"],
            "p50_latency_ms": m["p50_ms"],
            "p95_latency_ms": m["p95_ms"],
            "p99_latency_ms": m["p99_ms"],
            "num_requests": m["num_requests"],
            "total_output_tokens": m["total_output_tokens"],
            "total_time_s": m["total_time_s"],
            "score": entry["score"],
            "elapsed_s": entry["elapsed"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results exported to: {path}")
