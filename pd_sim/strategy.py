"""Grid search over colocated and disaggregated strategy parameters."""

import json

from pd_sim.engine import SimulationEngine


def search(engine: SimulationEngine, requests: list, cfg: dict) -> list[dict]:
    """Run grid search over strategy configurations.

    Returns list of {label, metrics, score} sorted by score descending.
    """
    mode = cfg["strategy"]["mode"]
    search_cfg = cfg["strategy"]["search"]
    slo = cfg["slo"]

    results = []

    if mode in ("colocated", "auto"):
        for chunk_size in search_cfg.get("chunk_sizes", [512]):
            for max_tokens in search_cfg.get("max_batched_tokens", [8192]):
                label = f"Colo (chunk={chunk_size}, batch={max_tokens})"

                local_cfg = _deep_copy_config(cfg)
                local_cfg["simulation"]["long_prefill_token_threshold"] = chunk_size
                local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                engine.cfg = local_cfg

                metrics = engine.run(list(requests), mode="colocated", chunk_size=chunk_size)
                score = metrics.score(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])
                results.append({"label": label, "metrics": metrics, "score": score})

    if mode in ("disaggregated", "auto"):
        for pd_ratio in search_cfg.get("pd_ratios", [[1, 1]]):
            p, d = pd_ratio if isinstance(pd_ratio, (list, tuple)) else (pd_ratio, 1)
            label = f"Disagg ({p}P:{d}D)"

            metrics = engine.run(list(requests), mode="disaggregated", pd_ratio=(p, d))
            score = metrics.score(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])
            results.append({"label": label, "metrics": metrics, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _deep_copy_config(cfg: dict) -> dict:
    """Simple deep copy via JSON round-trip (sufficient for our configs)."""
    return json.loads(json.dumps(cfg))
