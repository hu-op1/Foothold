"""Grid search over colocated and disaggregated strategy parameters."""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from pd_sim.engine import SimulationEngine


def search(engine: SimulationEngine, requests: list, cfg: dict) -> list[dict]:
    """Run grid search over strategy configurations.

    Returns list of {label, metrics, score} sorted by score descending.
    """
    mode = cfg["strategy"]["mode"]
    search_cfg = cfg["strategy"]["search"]
    slo = cfg["slo"]
    workers = cfg["strategy"].get("workers", 1)

    # Build task list
    tasks: list[dict] = []

    if mode in ("colocated", "auto"):
        for chunk_size in search_cfg.get("chunk_sizes", [512]):
            for max_tokens in search_cfg.get("max_batched_tokens", [8192]):
                label = f"Colo (chunk={chunk_size}, batch={max_tokens})"
                local_cfg = _deep_copy_config(cfg)
                local_cfg["simulation"]["long_prefill_token_threshold"] = chunk_size
                local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                tasks.append({
                    "label": label,
                    "cfg": local_cfg,
                    "mode": "colocated",
                    "chunk_size": chunk_size,
                })

    if mode in ("disaggregated", "auto"):
        for pd_ratio in search_cfg.get("pd_ratios", [[1, 1]]):
            p, d = pd_ratio if isinstance(pd_ratio, (list, tuple)) else (pd_ratio, 1)
            label = f"Disagg ({p}P:{d}D)"
            local_cfg = _deep_copy_config(cfg)
            tasks.append({
                "label": label,
                "cfg": local_cfg,
                "mode": "disaggregated",
                "pd_ratio": (p, d),
            })

    total = len(tasks)
    print(f"\n  Evaluating {total} strategies (workers={workers})...\n")

    # Serialize shared data for workers
    model_spec = engine.model
    hw_params = engine.hw

    results = []

    if workers <= 1:
        # Sequential (original behavior)
        for i, t in enumerate(tasks, 1):
            r = _run_one(t, requests, model_spec, hw_params, slo)
            results.append(r)
            _print_result(i, total, r)
    else:
        # Parallel
        future_to_idx = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, t in enumerate(tasks):
                fut = pool.submit(_run_one, t, requests, model_spec, hw_params, slo)
                future_to_idx[fut] = (i + 1, t["label"])

            n = 0
            for fut in as_completed(future_to_idx):
                idx, label = future_to_idx[fut]
                r = fut.result()
                results.append(r)
                n += 1
                _print_result(n, total, r)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _run_one(task: dict, requests: list, model_spec: dict, hw_params: dict,
             slo: dict) -> dict:
    """Run a single strategy task. Top-level function for picklability."""
    t0 = time.perf_counter()

    engine = SimulationEngine(task["cfg"], model_spec, hw_params)
    mode = task["mode"]

    if mode == "colocated":
        metrics = engine.run(list(requests), mode=mode, chunk_size=task.get("chunk_size"))
    else:
        metrics = engine.run(list(requests), mode=mode, pd_ratio=task.get("pd_ratio"))

    elapsed = time.perf_counter() - t0
    score = metrics.score(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])

    slo_info = metrics.slo_compliance(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])

    return {
        "label": task["label"],
        "metrics_raw": {
            "throughput": metrics.throughput(),
            "mean_ttft_ms": metrics.mean_ttft() * 1000,
            "mean_tpot_ms": metrics.mean_tpot() * 1000,
            "p50_ms": metrics.p50_latency() * 1000,
            "p95_ms": metrics.p95_latency() * 1000,
            "p99_ms": metrics.p99_latency() * 1000,
            "num_requests": metrics.num_requests,
            "total_output_tokens": metrics.total_output_tokens,
            "total_time_s": metrics.total_time,
        },
        "score": score,
        "elapsed": elapsed,
        "slo_score": slo_info["score"],
    }


def _print_result(n, total, entry):
    """Print a single search result row."""
    m = entry["metrics_raw"]
    label = entry["label"]
    score = entry["score"]
    elapsed = entry["elapsed"]
    print(f"  [{n}/{total}] {label:<32} "
          f"thrpt={m['throughput']:>8.0f} tok/s  "
          f"TTFT={m['mean_ttft_ms']:>7.1f}ms  "
          f"TPOT={m['mean_tpot_ms']:>7.1f}ms  "
          f"P99={m['p99_ms']:>7.1f}ms  "
          f"score={score:.1f}  "
          f"({elapsed:.2f}s)")


def _deep_copy_config(cfg: dict) -> dict:
    """Simple deep copy via JSON round-trip."""
    return json.loads(json.dumps(cfg))
