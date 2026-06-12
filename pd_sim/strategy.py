"""Grid search over colocated and disaggregated strategy parameters."""

import json
import time

from pd_sim.engine import SimulationEngine


def search(engine: SimulationEngine, requests: list, cfg: dict) -> list[dict]:
    """Run grid search over strategy configurations.

    Returns list of {label, metrics, score} sorted by score descending.
    """
    mode = cfg["strategy"]["mode"]
    search_cfg = cfg["strategy"]["search"]
    slo = cfg["slo"]

    # Count total candidates for progress display
    total = 0
    if mode in ("colocated", "auto"):
        total += len(search_cfg.get("chunk_sizes", [512])) * len(search_cfg.get("max_batched_tokens", [8192]))
    if mode in ("disaggregated", "auto"):
        total += len(search_cfg.get("pd_ratios", [[1, 1]]))
    print(f"\n  Evaluating {total} strategies...\n")

    results = []
    n = 0

    if mode in ("colocated", "auto"):
        for chunk_size in search_cfg.get("chunk_sizes", [512]):
            for max_tokens in search_cfg.get("max_batched_tokens", [8192]):
                label = f"Colo (chunk={chunk_size}, batch={max_tokens})"
                n += 1

                local_cfg = _deep_copy_config(cfg)
                local_cfg["simulation"]["long_prefill_token_threshold"] = chunk_size
                local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                engine.cfg = local_cfg

                t0 = time.perf_counter()
                metrics = engine.run(list(requests), mode="colocated", chunk_size=chunk_size)
                elapsed = time.perf_counter() - t0
                score = metrics.score(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])
                results.append({"label": label, "metrics": metrics, "score": score})

                _print_result(n, total, label, metrics, score, elapsed)

    if mode in ("disaggregated", "auto"):
        for pd_ratio in search_cfg.get("pd_ratios", [[1, 1]]):
            p, d = pd_ratio if isinstance(pd_ratio, (list, tuple)) else (pd_ratio, 1)
            label = f"Disagg ({p}P:{d}D)"
            n += 1

            t0 = time.perf_counter()
            metrics = engine.run(list(requests), mode="disaggregated", pd_ratio=(p, d))
            elapsed = time.perf_counter() - t0
            score = metrics.score(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])
            results.append({"label": label, "metrics": metrics, "score": score})

            _print_result(n, total, label, metrics, score, elapsed)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _print_result(n, total, label, metrics, score, elapsed):
    """Print a single search result row."""
    print(f"  [{n}/{total}] {label:<32} "
          f"thrpt={metrics.throughput():>8.0f} tok/s  "
          f"TTFT={metrics.mean_ttft()*1000:>7.1f}ms  "
          f"TPOT={metrics.mean_tpot()*1000:>7.1f}ms  "
          f"P99={metrics.p99_latency()*1000:>7.1f}ms  "
          f"score={score:.1f}  "
          f"({elapsed:.2f}s)")


def _deep_copy_config(cfg: dict) -> dict:
    """Simple deep copy via JSON round-trip (sufficient for our configs)."""
    return json.loads(json.dumps(cfg))
