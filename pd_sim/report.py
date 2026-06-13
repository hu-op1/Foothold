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
    print("=" * 120)
    print("  PD Strategy Comparison")
    if results:
        n_reqs = results[0]["metrics_raw"]["num_requests"]
        print(f"  Trace: {trace_path} ({n_reqs} requests)")
    print(f"  GPU: {gpu} x{total_gpus}")
    print("=" * 120)

    header = (
        f"  {'Strategy':<32} | {'Thrpt':>7} | {'InThrpt':>8} | {'TotalThrpt':>11} | "
        f"{'TTFTmean':>8} | {'TTFTp99':>8} | {'TPOTmean':>9} | {'TPOTp99':>9} | "
        f"{'P99lat':>8} | {'TotTime':>8} | {'SLO%':>6} | {'Time':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    best = max(results, key=lambda r: r["score"]) if results else None
    for entry in results:
        m = entry["metrics_raw"]
        marker = " ★" if entry is best else ""
        tt_s = m.get("total_time_s", 0)
        print(
            f"  {entry['label']:<32} | "
            f"{m['throughput']:>7.0f} | "
            f"{m['input_throughput']:>8.0f} | "
            f"{m['total_throughput']:>11.0f} | "
            f"{m['mean_ttft_ms']:>8.0f} | "
            f"{m['p99_ttft_ms']:>8.0f} | "
            f"{m['mean_tpot_ms']:>9.1f} | "
            f"{m['p99_tpot_ms']:>9.1f} | "
            f"{m['p99_ms']:>8.0f} | "
            f"{tt_s:>8.0f} | "
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
            "input_throughput_tok_s": m["input_throughput"],
            "output_throughput_tok_s": m["output_throughput"],
            "total_throughput_tok_s": m["total_throughput"],
            "ttft_ms": {
                "mean": m["mean_ttft_ms"],
                "p50": m["p50_ttft_ms"],
                "p90": m["p90_ttft_ms"],
                "p99": m["p99_ttft_ms"],
            },
            "tpot_ms": {
                "mean": m["mean_tpot_ms"],
                "p50": m["p50_tpot_ms"],
                "p90": m["p90_tpot_ms"],
                "p99": m["p99_tpot_ms"],
            },
            "total_latency_ms": {
                "p50": m["p50_ms"],
                "p90": m["p90_ms"],
                "p95": m["p95_ms"],
                "p99": m["p99_ms"],
            },
            "num_requests": m["num_requests"],
            "total_input_tokens": m["total_input_tokens"],
            "total_output_tokens": m["total_output_tokens"],
            "total_time_s": m["total_time_s"],
            "score": entry["score"],
            "elapsed_s": entry["elapsed"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results exported to: {path}")
