"""Compare sim output against LLMServingSim / vLLM output.

Conditionally includes each data source based on config:
  - Foothold sim: always (from ``sim_dir``)
  - LLMServingSim: if ``sim_csv`` and ``sim_log`` are both set
  - vLLM: if ``vllm_dir`` is set

Colors are read from ``config.validate.yaml`` → ``colors`` section (hex strings).
"""

from __future__ import annotations

from pathlib import Path

from validate.plot import (
    load_requests,
    load_timeseries,
    compute_latencies,
    load_sim_csv,
    load_sim_log,
    sim_latencies,
    plot_throughput,
    plot_requests,
    plot_latency_cdfs,
    write_summary,
)


_LINESTYLES = {"foothold": "-", "llmservingsim": "--", "vllm": ":"}


def _build_ts_dataset(ts_rows: list[dict], label: str, color: str, linestyle: str) -> dict | None:
    if not ts_rows:
        return None
    return {
        "t": [r["t"] for r in ts_rows],
        "prompt": [r["prompt_throughput"] for r in ts_rows],
        "gen": [r["gen_throughput"] for r in ts_rows],
        "running": [r["running"] for r in ts_rows],
        "waiting": [r["waiting"] for r in ts_rows],
        "label": label,
        "color": color,
        "linestyle": linestyle,
    }


def _build_latency_dataset(ttft, tpot, lat, label: str, color: str, linestyle: str) -> dict:
    return {
        "ttft": ttft,
        "tpot": tpot,
        "lat": lat,
        "label": label,
        "color": color,
        "linestyle": linestyle,
    }


def _get_colors(config: dict) -> dict:
    c = config.get("colors", {}) or {}
    return {
        "foothold": c.get("foothold", "#1f77b4"),
        "llmservingsim": c.get("llmservingsim", "#800080"),
        "vllm": c.get("vllm", "#2ca02c"),
    }


def run_compare(config: dict, *, vllm_dir_override: str | None = None) -> None:
    sim_dir = Path(config.get("sim_dir") or "")
    sim_csv = config.get("sim_csv")
    sim_log = config.get("sim_log")
    vllm_dir = vllm_dir_override or config.get("vllm_dir")
    if vllm_dir:
        vllm_dir = Path(vllm_dir)
    compare_dir = Path(config.get("compare_dir") or "") or sim_dir
    title = config.get("title", "sim vs vLLM")
    prefix = config.get("prefix", "")

    if not sim_dir.is_dir():
        print(f"Sim dir not found: {sim_dir}")
        return

    compare_dir.mkdir(parents=True, exist_ok=True)

    colors = _get_colors(config)

    throughput_datasets = []
    requests_datasets = []
    latency_datasets = []

    # ── Foothold sim (always) ──────────────────────────────────────────
    print(f"Loading foothold sim from {sim_dir} ...")
    sim_reqs = load_requests(sim_dir / "requests.jsonl")
    sim_ts = load_timeseries(sim_dir / "timeseries.csv")
    sim_ttft, sim_tpot, sim_lat = compute_latencies(sim_reqs)
    print(f"  {len(sim_reqs)} requests, {len(sim_ts)} timeseries rows")

    ts_ds = _build_ts_dataset(sim_ts, "foothold-sim", colors["foothold"], _LINESTYLES["foothold"])
    if ts_ds:
        throughput_datasets.append(ts_ds)
        requests_datasets.append(ts_ds)
    latency_datasets.append(_build_latency_dataset(sim_ttft, sim_tpot, sim_lat, "foothold-sim", colors["foothold"], _LINESTYLES["foothold"]))

    # ── LLMServingSim (optional) ───────────────────────────────────────
    if sim_csv and sim_log:
        sim_csv_p = Path(sim_csv)
        sim_log_p = Path(sim_log)
        if sim_csv_p.exists() and sim_log_p.exists():
            print(f"Loading LLMServingSim data from {sim_csv_p.parent} ...")
            lssim_reqs = load_sim_csv(sim_csv_p)
            lssim_ts = load_sim_log(sim_log_p)
            lssim_ttft, lssim_tpot, lssim_lat = sim_latencies(lssim_reqs)
            print(f"  {len(lssim_reqs)} requests, {len(lssim_ts)} timeseries rows")

            ts_ds = _build_ts_dataset(lssim_ts, "LLMServingSim", colors["llmservingsim"], _LINESTYLES["llmservingsim"])
            if ts_ds:
                throughput_datasets.append(ts_ds)
                requests_datasets.append(ts_ds)
            latency_datasets.append(_build_latency_dataset(lssim_ttft, lssim_tpot, lssim_lat, "LLMServingSim", colors["llmservingsim"], _LINESTYLES["llmservingsim"]))

    # ── vLLM (optional) ────────────────────────────────────────────────
    if vllm_dir and vllm_dir.is_dir():
        print(f"Loading vLLM data from {vllm_dir} ...")
        vllm_reqs = load_requests(vllm_dir / "requests.jsonl")
        vllm_ts = load_timeseries(vllm_dir / "timeseries.csv")
        vllm_ttft, vllm_tpot, vllm_lat = compute_latencies(vllm_reqs)
        print(f"  {len(vllm_reqs)} requests, {len(vllm_ts)} timeseries rows")

        ts_ds = _build_ts_dataset(vllm_ts, "vLLM", colors["vllm"], _LINESTYLES["vllm"])
        if ts_ds:
            throughput_datasets.append(ts_ds)
            requests_datasets.append(ts_ds)
        latency_datasets.append(_build_latency_dataset(vllm_ttft, vllm_tpot, vllm_lat, "vLLM", colors["vllm"], _LINESTYLES["vllm"]))

    # ── Plot ───────────────────────────────────────────────────────────
    print("Generating charts ...")
    if throughput_datasets:
        plot_throughput(compare_dir, prefix, throughput_datasets, title)
    if requests_datasets:
        plot_requests(compare_dir, prefix, requests_datasets, title)
    if latency_datasets:
        plot_latency_cdfs(compare_dir, prefix, latency_datasets, title)
        write_summary(compare_dir, prefix, latency_datasets, throughput_datasets, requests_datasets)

    print(f"Done → {compare_dir}")
