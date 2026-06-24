"""Grid search over colocated and disaggregated strategy parameters."""

import json
import time

from tqdm import tqdm

from sim.engine import SimulationEngine
from sim.config import valid_tp_sizes, valid_pp_sizes, total_vram_gb


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
    gpu_mem_util = cfg["simulation"].get("gpu_memory_utilization", 0.85)
    max_model_len = model_spec.get("max_model_len", 8192)
    max_num_seqs = cfg["simulation"].get("max_num_seqs", 256)

    max_tokens_list = search_cfg.get("max_batched_tokens", [8192])
    thresholds_list = search_cfg.get("prefill_thresholds", [1024])
    pd_ratios = search_cfg.get("pd_ratios", [[1, 1]])
    p_tp_sizes = search_cfg.get("p_tp_sizes") or search_cfg.get("tp_sizes", [1])
    d_tp_sizes = search_cfg.get("decode_tp_sizes") or p_tp_sizes
    p_pp_sizes = search_cfg.get("p_pp_sizes") or search_cfg.get("pp_sizes", [1])
    d_pp_sizes = search_cfg.get("decode_pp_sizes") or p_pp_sizes
    hw_params = engine.hw

    # Pre-compute valid PP sizes for this model
    valid_pps = valid_pp_sizes(model_spec, total_gpus)
    p_pp_sizes = [p for p in p_pp_sizes if p in valid_pps]
    d_pp_sizes = [p for p in d_pp_sizes if p in valid_pps]
    if not p_pp_sizes:
        p_pp_sizes = [1]
    if not d_pp_sizes:
        d_pp_sizes = [1]

    # Pre-compute valid TP sizes for this model+GPU.
    # Pass pp=1 for the initial filter — the full cross-product is validated
    # in the loop below where pp is known.
    valid_tps = valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, total_gpus,
                               max_model_len, max_num_seqs,
                               gpu_memory_utilization=gpu_mem_util,
                               max_batch_tokens=max(max_tokens_list))
    p_tp_sizes = [t for t in p_tp_sizes if t in valid_tps]
    d_tp_sizes = [t for t in d_tp_sizes if t in valid_tps]
    if not p_tp_sizes:
        p_tp_sizes = [1]
    if not d_tp_sizes:
        d_tp_sizes = [1]

    tasks: list[dict] = []

    for pp in p_pp_sizes:
        for tp in p_tp_sizes:
            # Colocated: each replica = tp × pp GPUs
            replica_gpus = tp * pp
            for max_tokens in max_tokens_list:
                for threshold in thresholds_list:
                    if threshold > max_tokens:
                        continue

                    # Colocated: DP = total_gpus / (tp × pp) independent replicas
                    if mode in ("colocated", "auto") and total_gpus % replica_gpus == 0:
                        dp = total_gpus // replica_gpus
                        pp_str = f"PP{pp} " if pp > 1 else ""
                        label = f"Colo {pp_str}TP{tp} DP{dp} (batch={max_tokens}, thr={threshold})"
                        local_cfg = _deep_copy_config(cfg)
                        local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                        local_cfg["simulation"]["long_prefill_token_threshold"] = threshold
                        tasks.append({
                            "label": label, "cfg": local_cfg, "mode": "colocated",
                            "tp": tp, "dp": dp, "pp": pp,
                        })

                    # Disaggregated: P side TP=tp, PP=pp; D side TP=d_tp, PP=d_pp
                    if mode in ("disaggregated", "auto"):
                        for pd_ratio in pd_ratios:
                            p, d = pd_ratio if isinstance(pd_ratio, (list, tuple)) else (pd_ratio, 1)
                            if p % replica_gpus != 0:
                                continue
                            dp_p = p // replica_gpus
                            for d_pp in d_pp_sizes:
                                for d_tp in d_tp_sizes:
                                    d_replica = d_tp * d_pp
                                    if d % d_replica != 0:
                                        continue
                                    dp_d = d // d_replica
                                    pp_str = f"PP{pp} " if pp > 1 else ""
                                    d_pp_str = f"PP{d_pp} " if d_pp > 1 else ""
                                    label = (f"Disagg {p}P({pp_str}TP{tp}×{dp_p})"
                                             f":{d}D({d_pp_str}TP{d_tp}×{dp_d}) "
                                             f"(batch={max_tokens}, thr={threshold})")
                                    local_cfg = _deep_copy_config(cfg)
                                    local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                                    local_cfg["simulation"]["long_prefill_token_threshold"] = threshold
                                    tasks.append({
                                        "label": label, "cfg": local_cfg,
                                        "mode": "disaggregated", "pd_ratio": (p, d),
                                        "tp": tp, "dp_p": dp_p, "pp": pp,
                                        "d_tp": d_tp, "dp_d": dp_d, "d_pp": d_pp,
                                    })

    total = len(tasks)
    print(f"\n  Model={model_spec['name']} ({model_spec['total_params_b']/1e9:.1f}B params)")
    print(f"  GPU={gpu_name} x{total_gpus} ({total_vram_gb(gpu_name)}GB each)")
    print(f"  Valid P-TP sizes: {p_tp_sizes}")
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
    pp = task.get("pp", 1)

    if mode == "colocated":
        dp = task.get("dp", 1)
        metrics = engine.run(list(requests), mode=mode, tp_size=tp, dp=dp,
                             pp_size=pp)
    else:
        d_tp = task.get("d_tp", 1)
        d_pp = task.get("d_pp", 1)
        dp_d = task.get("dp_d", 1)
        metrics = engine.run(list(requests), mode=mode,
                             pd_ratio=task.get("pd_ratio"),
                             tp_size=tp, d_tp_size=d_tp,
                             pp_size=pp, d_pp_size=d_pp)

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
