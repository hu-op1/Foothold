"""Terminal table and CSV output for simulation results."""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


_LABEL_RE = re.compile(r"^(.*?) \(batch=(\d+), thr=(\d+)\)$")


def _parse_label(label: str) -> tuple[str, int, int]:
    """Parse a label into (strategy_type, batch, threshold).

    >>> _parse_label("Colo TP1 DP4 (batch=256, thr=256)")
    ("Colo TP1 DP4", 256, 256)
    >>> _parse_label("Disagg 1P(TP1×1):3D (batch=512, thr=1024)")
    ("Disagg 1P(TP1×1):3D", 512, 1024)
    """
    m = _LABEL_RE.match(label)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return label, 0, 0


def _write_scalability_csv(results: list[dict], base_path: str) -> None:
    """Write scalability summary as a separate CSV alongside main results."""
    scalability = None
    for r in results:
        if "_scalability" in r:
            scalability = r["_scalability"]
            break

    if not scalability:
        return

    path = os.path.splitext(base_path)[0] + "_scalability.csv"
    headers = [
        "total_gpus",
        "colo_label",
        "colo_throughput_tok_s",
        "colo_ttft_p99_ms",
        "colo_tpot_p99_ms",
        "colo_p99_latency_ms",
        "colo_slo_pass",
        "disagg_label",
        "disagg_throughput_tok_s",
        "disagg_ttft_p99_ms",
        "disagg_tpot_p99_ms",
        "disagg_p99_latency_ms",
        "disagg_slo_pass",
        "winner",
    ]

    def _m(r, key):
        return r["metrics_raw"].get(key, 0) if r else 0

    rows = []
    for s in scalability:
        colo = s["best_colo"]
        disagg = s["best_disagg"]
        ct = _m(colo, "throughput")
        dt = _m(disagg, "throughput")
        winner = "Colocated" if ct >= dt else "Disaggregated"
        rows.append({
            "total_gpus": s["total_gpus"],
            "colo_label": colo["label"] if colo else "—",
            "colo_throughput_tok_s": ct,
            "colo_ttft_p99_ms": _m(colo, "p99_ttft_ms"),
            "colo_tpot_p99_ms": _m(colo, "p99_tpot_ms"),
            "colo_p99_latency_ms": _m(colo, "p99_ms"),
            "colo_slo_pass": colo.get("slo_pass", False) if colo else False,
            "disagg_label": disagg["label"] if disagg else "—",
            "disagg_throughput_tok_s": dt,
            "disagg_ttft_p99_ms": _m(disagg, "p99_ttft_ms"),
            "disagg_tpot_p99_ms": _m(disagg, "p99_tpot_ms"),
            "disagg_p99_latency_ms": _m(disagg, "p99_ms"),
            "disagg_slo_pass": disagg.get("slo_pass", False) if disagg else False,
            "winner": winner,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scalability summary saved to: {path}")


def _plot_scalability(results: list[dict], csv_path: str) -> None:
    """Generate a scalability plot PNG alongside the csv."""
    scalability = None
    for r in results:
        if "_scalability" in r:
            scalability = r["_scalability"]
            break
    if not scalability or len(scalability) < 2:
        return

    gpus = [s["total_gpus"] for s in scalability]
    colo_tp = [s["best_colo"]["metrics_raw"]["throughput"] if s["best_colo"] else 0
               for s in scalability]
    disagg_tp = [s["best_disagg"]["metrics_raw"]["throughput"] if s["best_disagg"] else 0
                 for s in scalability]

    # If all zero (SLO too strict), fall back to raw best throughput from all results
    print("  [fallback] no SLO-compliant results, falling back to raw best throughput")
    if max(colo_tp) == 0 and max(disagg_tp) == 0:
        for i, g in enumerate(gpus):
            colo_best = max(
                (r for r in results if r.get("total_gpus") == g
                 and r.get("mode_label") == "colocated"),
                key=lambda r: r["metrics_raw"]["throughput"], default=None)
            disagg_best = max(
                (r for r in results if r.get("total_gpus") == g
                 and r.get("mode_label") == "disaggregated"),
                key=lambda r: r["metrics_raw"]["throughput"], default=None)
            colo_tp[i] = colo_best["metrics_raw"]["throughput"] if colo_best else 0
            disagg_tp[i] = disagg_best["metrics_raw"]["throughput"] if disagg_best else 0
        print("(SLO too strict — plotting raw best throughput per GPU count)")

    # Per-GPU efficiency
    colo_eff = [colo_tp[i] / gpus[i] if gpus[i] > 0 else 0
                for i in range(len(gpus))]
    disagg_eff = [disagg_tp[i] / gpus[i] if gpus[i] > 0 else 0
                  for i in range(len(gpus))]

    # Ideal linear scaling (from first GPU count with nonzero throughput)
    ideal = None
    for i, g in enumerate(gpus):
        if colo_tp[i] > 0 or disagg_tp[i] > 0:
            base_tp = max(colo_tp[i], disagg_tp[i])
            ideal = [base_tp * (gg / g) for gg in gpus]
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("PD Disaggregation Scalability", fontsize=14, fontweight="bold", y=1.01)

    # ── Left: Throughput ──
    ax1.plot(gpus, colo_tp, "o-", color="#4472C4", linewidth=2.2, markersize=8,
             label="Colocated (best)")
    ax1.plot(gpus, disagg_tp, "s--", color="#C00000", linewidth=2.2, markersize=8,
             label="Disaggregated (best)")
    if ideal:
        ax1.plot(gpus, ideal, ":", color="gray", linewidth=1.5, alpha=0.7,
                 label=f"Ideal linear (×{gpus[0]} GPU baseline)")

    # Mark winner at each GPU count
    for i, g in enumerate(gpus):
        if colo_tp[i] == 0 and disagg_tp[i] == 0:
            continue
        winner = "C" if colo_tp[i] >= disagg_tp[i] else "D"
        y = max(colo_tp[i], disagg_tp[i])
        ax1.annotate(winner, (g, y), textcoords="offset points", xytext=(0, 10),
                     fontsize=10, fontweight="bold", ha="center",
                     color="#4472C4" if winner == "C" else "#C00000")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("GPU Count (log₂)", fontsize=12)
    ax1.set_ylabel("Throughput (tok/s)", fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(gpus)
    ax1.set_xticklabels([str(g) for g in gpus])
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    # ── Right: Per-GPU Efficiency ──
    ax2.plot(gpus, colo_eff, "o-", color="#4472C4", linewidth=2.2, markersize=8,
             label="Colocated")
    ax2.plot(gpus, disagg_eff, "s--", color="#C00000", linewidth=2.2, markersize=8,
             label="Disaggregated")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("GPU Count (log₂)", fontsize=12)
    ax2.set_ylabel("Throughput per GPU (tok/s/GPU)", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(gpus)
    ax2.set_xticklabels([str(g) for g in gpus])
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    plt.tight_layout()

    png_path = os.path.splitext(csv_path)[0] + "_scalability.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Scalability plot saved to: {png_path}")


# Fixed CSV column order — shared between incremental append and final export.
SEARCH_FIELDNAMES = [
    "strategy_type",
    "batch",
    "thr",
    "input_throughput_tok_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "tpot_mean_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
    "latency_p50_ms", "latency_p90_ms", "latency_p95_ms", "latency_p99_ms",
    "num_requests",
    "total_input_tokens",
    "total_output_tokens",
    "total_time_s",
    "cache_hit_rate",
    "attn_proj_pct", "ffn_proj_pct",
    "attn_prefill_pct", "attn_decode_pct",
    "fused_add_norm_pct", "swiglu_pct", "rope_pct",
    "lm_head_pct",
    "all_reduce_pct", "inter_stage_comm_pct",
    "kv_transfer_pct", "swap_pct",
    "slo_pass",
    "elapsed_s",
]


def flatten_result(entry: dict) -> dict:
    """Flatten a search result dict into a CSV row dict."""
    m = entry["metrics_raw"]
    strategy_type, batch, thr = _parse_label(entry["label"])
    return {
        "strategy_type": strategy_type,
        "batch": batch,
        "thr": thr,
        "input_throughput_tok_s": m["input_throughput"],
        "output_throughput_tok_s": m["output_throughput"],
        "total_throughput_tok_s": m["total_throughput"],
        "ttft_mean_ms": m["mean_ttft_ms"], "ttft_p50_ms": m["p50_ttft_ms"],
        "ttft_p90_ms": m["p90_ttft_ms"], "ttft_p99_ms": m["p99_ttft_ms"],
        "tpot_mean_ms": m["mean_tpot_ms"], "tpot_p50_ms": m["p50_tpot_ms"],
        "tpot_p90_ms": m["p90_tpot_ms"], "tpot_p99_ms": m["p99_tpot_ms"],
        "latency_p50_ms": m["p50_ms"], "latency_p90_ms": m["p90_ms"],
        "latency_p95_ms": m["p95_ms"], "latency_p99_ms": m["p99_ms"],
        "num_requests": m["num_requests"],
        "total_input_tokens": m["total_input_tokens"],
        "total_output_tokens": m["total_output_tokens"],
        "total_time_s": m["total_time_s"],
        "cache_hit_rate": m.get("cache_hit_rate", 0.0),
        "attn_proj_pct": m.get("attn_proj_pct", 0.0),
        "ffn_proj_pct": m.get("ffn_proj_pct", 0.0),
        "attn_prefill_pct": m.get("attn_prefill_pct", 0.0),
        "attn_decode_pct": m.get("attn_decode_pct", 0.0),
        "fused_add_norm_pct": m.get("fused_add_norm_pct", 0.0),
        "swiglu_pct": m.get("swiglu_pct", 0.0),
        "rope_pct": m.get("rope_pct", 0.0),
        "lm_head_pct": m.get("lm_head_pct", 0.0),
        "all_reduce_pct": m.get("all_reduce_pct", 0.0),
        "inter_stage_comm_pct": m.get("inter_stage_comm_pct", 0.0),
        "kv_transfer_pct": m.get("kv_transfer_pct", 0.0),
        "swap_pct": m.get("swap_pct", 0.0),
        "slo_pass": entry.get("slo_pass", False),
        "elapsed_s": entry["elapsed"],
    }


def export_csv(results: list[dict], path: str) -> None:
    """Export results to CSV (overwrites), plus scalability artifacts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rows = [flatten_result(r) for r in results]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results exported to: {path}")

    # ── Scalability summary (separate CSV, if GPU sweep was run) ──
    _write_scalability_csv(results, path)

    # ── Scalability plot ──
    _plot_scalability(results, path)
