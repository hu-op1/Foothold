"""Validate CLI — subcommand dispatch for validate mode.

Usage:
    uv run python main.py --validate                  # visualize sim output only
    uv run python main.py --validate --compare        # compare (sim + any configured data)
    uv run python main.py --validate --vllm           # send trace to vLLM
    uv run python main.py --validate --vllm --compare # send to vLLM then compare
"""

from __future__ import annotations

from pathlib import Path

from validate.plot import (
    load_requests,
    load_timeseries,
    compute_latencies,
    plot_throughput,
    plot_requests,
    plot_latency_cdfs,
    write_summary,
)
from validate.send import run_send, run_send_embedded
from validate.compare import run_compare


def _run_visualize(config: dict) -> None:
    """Visualize a single sim run (no external comparison)."""
    sim_dir = Path(config.get("sim_dir") or "")
    if not sim_dir.is_dir():
        print(f"Sim dir not found: {sim_dir}")
        return

    title = config.get("title", "foothold sim")
    prefix = config.get("prefix", "")

    print(f"Loading foothold sim from {sim_dir} ...")
    sim_reqs = load_requests(sim_dir / "requests.jsonl")
    sim_ts = load_timeseries(sim_dir / "timeseries.csv")
    sim_ttft, sim_tpot, sim_lat = compute_latencies(sim_reqs)
    print(f"  {len(sim_reqs)} requests, {len(sim_ts)} timeseries rows")

    style = {"label": "foothold-sim", "color": "C0", "linestyle": "-"}

    if sim_ts:
        ds_ts = {
            "t": [r["t"] for r in sim_ts],
            "prompt": [r["prompt_throughput"] for r in sim_ts],
            "gen": [r["gen_throughput"] for r in sim_ts],
        } | style
        ds_rq = {
            "t": [r["t"] for r in sim_ts],
            "running": [r["running"] for r in sim_ts],
            "waiting": [r["waiting"] for r in sim_ts],
        } | style
        plot_throughput(sim_dir, prefix, [ds_ts], title)
        plot_requests(sim_dir, prefix, [ds_rq], title)

    ds_lat = {"ttft": sim_ttft, "tpot": sim_tpot, "lat": sim_lat, "label": "foothold-sim",
              "color": "C0", "linestyle": "-"}
    plot_latency_cdfs(sim_dir, prefix, [ds_lat], title)
    write_summary(sim_dir, prefix, [ds_lat])

    print(f"Done → {sim_dir}")


def dispatch(config: dict, *, vllm: bool = False, compare: bool = False) -> None:
    vllm_cfg = config.get("vllm", {})
    embedded = vllm_cfg.get("embedded", False)

    if vllm and compare:
        if embedded:
            run_send_embedded(config)
        else:
            run_send(config)
        vllm_dir = vllm_cfg.get("output_dir", "")
        run_compare(config, vllm_dir_override=vllm_dir)
    elif vllm:
        if embedded:
            run_send_embedded(config)
        else:
            run_send(config)
    elif compare:
        run_compare(config)
    else:
        _run_visualize(config)
