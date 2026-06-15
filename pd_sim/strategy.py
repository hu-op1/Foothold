"""Grid search over colocated and disaggregated strategy parameters."""

import json
import time

from tqdm import tqdm

from pd_sim.engine import SimulationEngine
from pd_sim.config import valid_tp_sizes, total_vram_gb


def search(engine: SimulationEngine, requests: list, cfg: dict) -> list[dict]:
    """Run grid search over strategy configurations.

    Cartesian product: tp_sizes × max_batched_tokens × prefill_thresholds
                       × (colocated + each P:D ratio)

    Returns list of {label, metrics_raw, score, elapsed, slo_score}
    sorted by score descending.
    """
    mode = cfg["strategy"]["mode"]
    search_cfg = cfg["strategy"]["search"]
    slo = cfg["slo"]
    total_gpus = cfg["strategy"]["total_gpus"]
    model_spec = engine.model
    gpu_name = cfg.get("gpu", "3090")
    kv_cache_gb = cfg["simulation"]["kv_cache_memory_gb"]
    max_model_len = model_spec.get("max_model_len", 8192)
    max_num_seqs = cfg["simulation"].get("max_num_seqs", 256)

    max_tokens_list = search_cfg.get("max_batched_tokens", [8192])
    thresholds_list = search_cfg.get("prefill_thresholds", [1024])
    pd_ratios = search_cfg.get("pd_ratios", [[1, 1]])
    tp_sizes = search_cfg.get("tp_sizes", [1])
    d_tp_sizes = search_cfg.get("decode_tp_sizes") or tp_sizes
    hw_params = engine.hw

    # Pre-compute valid TP sizes for this model+GPU
    valid_tps = valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, total_gpus,
                               max_model_len, max_num_seqs)
    tp_sizes = [t for t in tp_sizes if t in valid_tps]
    d_tp_sizes = [t for t in d_tp_sizes if t in valid_tps]
    if not tp_sizes:
        tp_sizes = [1]
    if not d_tp_sizes:
        d_tp_sizes = [1]

    tasks: list[dict] = []

    for tp in tp_sizes:
        for max_tokens in max_tokens_list:
            for threshold in thresholds_list:
                if threshold > max_tokens:
                    continue

                # Colocated: TP=tp, DP = total_gpus / tp independent instances
                if mode in ("colocated", "auto") and total_gpus % tp == 0:
                    dp = total_gpus // tp
                    label = f"Colo TP{tp} DP{dp} (batch={max_tokens}, thr={threshold})"
                    local_cfg = _deep_copy_config(cfg)
                    local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                    local_cfg["simulation"]["long_prefill_token_threshold"] = threshold
                    tasks.append({
                        "label": label, "cfg": local_cfg, "mode": "colocated",
                        "tp": tp, "dp": dp,
                    })

                # Disaggregated: P side TP=tp, D side TP=d_tp
                if mode in ("disaggregated", "auto"):
                    for pd_ratio in pd_ratios:
                        p, d = pd_ratio if isinstance(pd_ratio, (list, tuple)) else (pd_ratio, 1)
                        if p % tp != 0:
                            continue
                        dp_p = p // tp
                        for d_tp in d_tp_sizes:
                            if d % d_tp != 0:
                                continue
                            dp_d = d // d_tp
                            label = (f"Disagg {p}P(TP{tp}×{dp_p}):{d}D(TP{d_tp}×{dp_d}) "
                                     f"(batch={max_tokens}, thr={threshold})")
                            local_cfg = _deep_copy_config(cfg)
                            local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                            local_cfg["simulation"]["long_prefill_token_threshold"] = threshold
                            tasks.append({
                                "label": label, "cfg": local_cfg,
                                "mode": "disaggregated", "pd_ratio": (p, d),
                                "tp": tp, "dp_p": dp_p,
                                "d_tp": d_tp, "dp_d": dp_d,
                            })

    total = len(tasks)
    print(f"\n  Model={model_spec['name']} ({model_spec['total_params_b']/1e9:.1f}B params)")
    print(f"  GPU={gpu_name} x{total_gpus} ({total_vram_gb(gpu_name)}GB each)")
    print(f"  Valid TP sizes: {tp_sizes}")
    print(f"  Evaluating {total} strategies...\n")

    results = []
    for t in tqdm(tasks, desc="  Searching", unit="strat",
                  bar_format="{desc}: {percentage:3.0f}% |{bar}| "
                             "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        r = _run_one(t, requests, model_spec, hw_params, slo)
        results.append(r)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _run_one(task: dict, requests: list, model_spec: dict, hw_params: dict,
             slo: dict) -> dict:
    """Run a single strategy task. Top-level function for picklability."""
    t0 = time.perf_counter()

    engine = SimulationEngine(task["cfg"], model_spec, hw_params)
    mode = task["mode"]
    tp = task.get("tp", 1)

    if mode == "colocated":
        dp = task.get("dp", 1)
        metrics = engine.run(list(requests), mode=mode, tp_size=tp, dp=dp)
    else:
        d_tp = task.get("d_tp", 1)
        dp_d = task.get("dp_d", 1)
        metrics = engine.run(list(requests), mode=mode,
                             pd_ratio=task.get("pd_ratio"),
                             tp_size=tp, d_tp_size=d_tp)

    elapsed = time.perf_counter() - t0
    score = metrics.score(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])
    slo_info = metrics.slo_compliance(slo["ttft_ms"], slo["tpot_ms"], slo["p99_latency_ms"])

    total_t = metrics.total_time
    return {
        "label": task["label"],
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
        "slo_score": slo_info["score"],
    }


def _deep_copy_config(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg))
