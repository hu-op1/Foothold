"""Grid search over colocated and disaggregated strategy parameters.

Checkpoint/resume: each completed strategy is appended immediately to the output
CSV.  On restart completed rows are read back and matching labels are skipped so
only unfinished strategies run.
"""

import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from sim.engine import SimulationEngine
from sim.config import valid_tp_sizes, valid_pp_sizes, total_vram_gb
from sim.report import SEARCH_FIELDNAMES, flatten_result


# ── checkpoint helpers (CSV-based) ──────────────────────────────────────────

def _load_completed_labels(csv_path: str) -> set[str]:
    """Read output CSV and return set of completed strategy labels.

    Reconstructs the full label from strategy_type + batch + thr columns
    so it matches the task labels built by _search_one.
    """
    if not os.path.exists(csv_path):
        return set()
    labels = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                st = row.get("strategy_type", "")
                batch = row.get("batch", "")
                thr = row.get("thr", "")
                if st:
                    labels.add(f"{st} (batch={batch}, thr={thr})")
    except Exception:
        pass
    return labels


def _append_csv(csv_path: str, result: dict) -> None:
    """Flatten and append one search result to the output CSV."""
    from bench.utils import append_csv_row
    row = flatten_result(result)
    append_csv_row(csv_path, SEARCH_FIELDNAMES, row)


def _load_results_from_csv(csv_path: str) -> list[dict]:
    """Read output CSV and reconstruct result dicts for merge."""
    results = []
    if not os.path.exists(csv_path):
        return results
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = {
                    "throughput": float(row.get("output_throughput_tok_s", 0)),
                    "input_throughput": float(row.get("input_throughput_tok_s", 0)),
                    "output_throughput": float(row.get("output_throughput_tok_s", 0)),
                    "total_throughput": float(row.get("total_throughput_tok_s", 0)),
                    "mean_ttft_ms": float(row.get("ttft_mean_ms", 0)),
                    "p50_ttft_ms": float(row.get("ttft_p50_ms", 0)),
                    "p90_ttft_ms": float(row.get("ttft_p90_ms", 0)),
                    "p99_ttft_ms": float(row.get("ttft_p99_ms", 0)),
                    "mean_tpot_ms": float(row.get("tpot_mean_ms", 0)),
                    "p50_tpot_ms": float(row.get("tpot_p50_ms", 0)),
                    "p90_tpot_ms": float(row.get("tpot_p90_ms", 0)),
                    "p99_tpot_ms": float(row.get("tpot_p99_ms", 0)),
                    "p50_ms": float(row.get("latency_p50_ms", 0)),
                    "p90_ms": float(row.get("latency_p90_ms", 0)),
                    "p95_ms": float(row.get("latency_p95_ms", 0)),
                    "p99_ms": float(row.get("latency_p99_ms", 0)),
                    "num_requests": int(float(row.get("num_requests", 0))),
                    "total_input_tokens": int(float(row.get("total_input_tokens", 0))),
                    "total_output_tokens": int(float(row.get("total_output_tokens", 0))),
                    "total_time_s": float(row.get("total_time_s", 0)),
                    "cache_hit_rate": float(row.get("cache_hit_rate", 0)),
                    "attn_proj_pct": float(row.get("attn_proj_pct", 0)),
                    "ffn_proj_pct": float(row.get("ffn_proj_pct", 0)),
                    "attn_prefill_pct": float(row.get("attn_prefill_pct", 0)),
                    "attn_decode_pct": float(row.get("attn_decode_pct", 0)),
                    "fused_add_norm_pct": float(row.get("fused_add_norm_pct", 0)),
                    "swiglu_pct": float(row.get("swiglu_pct", 0)),
                    "rope_pct": float(row.get("rope_pct", 0)),
                    "lm_head_pct": float(row.get("lm_head_pct", 0)),
                    "all_reduce_pct": float(row.get("all_reduce_pct", 0)),
                    "inter_stage_comm_pct": float(row.get("inter_stage_comm_pct", 0)),
                    "kv_transfer_pct": float(row.get("kv_transfer_pct", 0)),
                    "swap_pct": float(row.get("swap_pct", 0)),
                }
                st = row.get("strategy_type", "")
                batch = row.get("batch", "")
                thr = row.get("thr", "")
                results.append({
                    "label": f"{st} (batch={batch}, thr={thr})",
                    "metrics_raw": m,
                    "throughput": float(row.get("output_throughput_tok_s", 0)),
                    "slo_pass": row.get("slo_pass", "False") == "True",
                    "elapsed": float(row.get("elapsed_s", 0)),
                })
    except Exception:
        pass
    return results


# ── main search entry point ─────────────────────────────────────────────────

def search(engine: SimulationEngine, requests: list, cfg: dict) -> list[dict]:
    """Run grid search over strategy configurations.

    Cartesian product: tp_sizes × max_batched_tokens × prefill_thresholds
                       × (colocated + each P:D ratio)

    On-disk checkpoint: after each strategy finishes its result is appended as
    one JSON line to ``<output>.checkpoint.jsonl``.  On restart, strategies
    whose label already appears in that file are skipped so only unfinished
    work runs.

    If cfg["strategy"]["search"]["gpu_sweep"] is set, sweeps over GPU counts
    and attaches a scalability_summary to the first result.

    Returns list of {label, metrics_raw, throughput, slo_pass, elapsed}
    sorted: SLO-passing first (by throughput), then SLO-failing.
    """
    mode = cfg["strategy"]["mode"]
    search_cfg = cfg["strategy"]["search"]
    slo = cfg["slo"]
    total_gpus = cfg["strategy"]["total_gpus"]
    model_spec = engine.model
    gpu_name = cfg.get("gpu", "3090")
    gpus_per_node = cfg["strategy"].get("gpus_per_node")
    kv_cache_gb = cfg["simulation"]["kv_cache_memory_gb"]
    gpu_mem_util = cfg["simulation"].get("gpu_memory_utilization", 0.85)
    max_model_len = model_spec.get("max_model_len", 8192)
    max_num_seqs = cfg["simulation"].get("max_num_seqs", 256)
    hw_params = engine.hw

    # ── overwrite: delete existing CSV once before any search ──
    out_path = str(cfg.get("output", "sim/output/results.csv"))
    if cfg.get("overwrite") and os.path.exists(out_path):
        os.remove(out_path)

    gpu_sweep = search_cfg.get("gpu_sweep")
    if isinstance(gpu_sweep, list) and len(gpu_sweep) > 0:
        return _search_with_sweep(
            gpu_sweep, mode, search_cfg, slo, model_spec, gpu_name,
            kv_cache_gb, gpu_mem_util, max_model_len, max_num_seqs,
            hw_params, requests, cfg, gpus_per_node=gpus_per_node)

    return _search_one(total_gpus, mode, search_cfg, slo, model_spec,
                       gpu_name, kv_cache_gb, gpu_mem_util, max_model_len,
                       max_num_seqs, hw_params, requests, cfg,
                       gpus_per_node=gpus_per_node)


def _search_with_sweep(gpu_sweep: list[int], mode, search_cfg, slo,
                       model_spec, gpu_name, kv_cache_gb, gpu_mem_util,
                       max_model_len, max_num_seqs, hw_params, requests,
                       cfg, gpus_per_node=None) -> list[dict]:
    """Sweep over GPU counts, returning all results with scalability summary."""
    all_results = []
    scalability = []  # [{gpus, best_colo, best_disagg}, ...]

    for n_gpus in gpu_sweep:
        print(f"\n{'='*60}")
        print(f"  GPU count: {n_gpus}")
        print(f"{'='*60}")
        batch = _search_one(n_gpus, mode, search_cfg, slo, model_spec,
                            gpu_name, kv_cache_gb, gpu_mem_util, max_model_len,
                            max_num_seqs, hw_params, requests, cfg,
                            gpus_per_node=gpus_per_node)

        # Tag each result with its GPU count for downstream grouping
        for r in batch:
            r["total_gpus"] = n_gpus

        all_results.extend(batch)

        # Extract best per-mode for this GPU count (prefer SLO-compliant, fall back to raw throughput)
        print("  [fallback] no SLO-compliant strategy, falling back to raw throughput")
        colo_all = sorted(
            [r for r in batch if r.get("mode_label") == "colocated"],
            key=lambda r: r["metrics_raw"]["throughput"], reverse=True)
        disagg_all = sorted(
            [r for r in batch if r.get("mode_label") == "disaggregated"],
            key=lambda r: r["metrics_raw"]["throughput"], reverse=True)
        colo_slo = [r for r in colo_all if r.get("slo_pass")]
        disagg_slo = [r for r in disagg_all if r.get("slo_pass")]

        best_colo = colo_slo[0] if colo_slo else (colo_all[0] if colo_all else None)
        best_disagg = disagg_slo[0] if disagg_slo else (disagg_all[0] if disagg_all else None)

        scalability.append({
            "total_gpus": n_gpus,
            "best_colo": best_colo,
            "best_disagg": best_disagg,
        })

        if best_colo:
            print(f"  Best colo: {best_colo['label']} "
                  f"→ throughput={best_colo['metrics_raw']['throughput']:.1f} tok/s")
        else:
            print(f"  Best colo: (none meeting SLO)")
        if best_disagg:
            print(f"  Best disaggregated: {best_disagg['label']} "
                  f"→ throughput={best_disagg['metrics_raw']['throughput']:.1f} tok/s")
        else:
            print(f"  Best disaggregated: (none meeting SLO)")

    # Print scalability summary table
    print(f"\n{'='*70}")
    print(f"  Scalability Summary")
    print(f"  {'GPUs':<6} {'Best Colo (tok/s)':<20} {'Best Disagg (tok/s)':<22} {'Winner'}")
    print(f"  {'-'*60}")
    for s in scalability:
        ct = s["best_colo"]["metrics_raw"]["throughput"] if s["best_colo"] else 0
        dt = s["best_disagg"]["metrics_raw"]["throughput"] if s["best_disagg"] else 0
        winner = "Colocated" if ct >= dt else "Disaggregated"
        print(f"  {s['total_gpus']:<6} {ct:<20.1f} {dt:<22.1f} {winner}")

    # Sort: SLO-passing first (by throughput), then SLO-failing (by throughput)
    all_results.sort(key=lambda r: (r.get("slo_pass", False), r.get("throughput", 0)), reverse=True)

    # Attach scalability summary to first result for report export
    if all_results:
        all_results[0]["_scalability"] = scalability

    return all_results


def _search_one(total_gpus: int, mode, search_cfg, slo, model_spec,
                gpu_name, kv_cache_gb, gpu_mem_util, max_model_len,
                max_num_seqs, hw_params, requests, cfg,
                gpus_per_node=None) -> list[dict]:
    """Run grid search for a single GPU count."""

    max_tokens_list = search_cfg.get("max_batched_tokens", [8192])
    thresholds_list = search_cfg.get("prefill_thresholds", [1024])
    max_workers = search_cfg.get("max_workers", 1)
    enable_tp = search_cfg.get("tp", True)
    enable_pp = search_cfg.get("pp", True)
    enable_dp = search_cfg.get("dp", True)

    # Auto-generate PD ratios: all (p, d) where p + d = total_gpus, p>=1, d>=1
    pd_ratios = [[p, total_gpus - p] for p in range(1, total_gpus)]

    # Auto-generate TP/PP sizes from model+GPU constraints
    valid_pps = valid_pp_sizes(model_spec, total_gpus)
    valid_tps = valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, total_gpus,
                               max_model_len, max_num_seqs,
                               gpu_memory_utilization=gpu_mem_util,
                               max_batch_tokens=max(max_tokens_list),
                               gpus_per_node=gpus_per_node)
    p_tp_sizes = d_tp_sizes = valid_tps if enable_tp else [1]
    p_pp_sizes = d_pp_sizes = valid_pps if enable_pp else [1]

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
                    if (mode in ("colocated", "both")
                            and total_gpus % replica_gpus == 0
                            and (enable_dp or total_gpus // replica_gpus == 1)):
                        dp = total_gpus // replica_gpus
                        label = f"Colo PP{pp} TP{tp} DP{dp} (batch={max_tokens}, thr={threshold})"
                        local_cfg = _deep_copy_config(cfg)
                        local_cfg["strategy"]["total_gpus"] = total_gpus
                        local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                        local_cfg["simulation"]["long_prefill_token_threshold"] = threshold
                        tasks.append({
                            "label": label, "cfg": local_cfg, "mode": "colocated",
                            "tp": tp, "dp": dp, "pp": pp, "total_gpus": total_gpus,
                            "mode_label": "colocated",
                        })

                    # Disaggregated: P side TP=tp, PP=pp; D side TP=d_tp, PP=d_pp
                    if mode in ("disaggregated", "both"):
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
                                    if not enable_dp and (dp_p != 1 or dp_d != 1):
                                        continue
                                    label = (f"Disagg {p}P(PP{pp} TP{tp} DP{dp_p})"
                                             f":{d}D(PP{d_pp} TP{d_tp} DP{dp_d}) "
                                             f"(batch={max_tokens}, thr={threshold})")
                                    local_cfg = _deep_copy_config(cfg)
                                    local_cfg["strategy"]["total_gpus"] = total_gpus
                                    local_cfg["simulation"]["max_num_batched_tokens"] = max_tokens
                                    local_cfg["simulation"]["long_prefill_token_threshold"] = threshold
                                    tasks.append({
                                        "label": label, "cfg": local_cfg,
                                        "mode": "disaggregated", "pd_ratio": (p, d),
                                        "tp": tp, "dp_p": dp_p, "pp": pp,
                                        "d_tp": d_tp, "dp_d": dp_d, "d_pp": d_pp,
                                        "total_gpus": total_gpus,
                                        "mode_label": "disaggregated",
                                    })

    # ── checkpoint: skip already-finished strategies (read from output CSV) ──
    out_path = str(cfg.get("output", "sim/output/results.csv"))
    completed_labels = _load_completed_labels(out_path)
    pending = [t for t in tasks if t["label"] not in completed_labels]
    skipped = len(tasks) - len(pending)

    total = len(tasks)
    print(f"  Model={model_spec['name']} ({model_spec['total_params_b']/1e9:.1f}B params)")
    print(f"  GPU={gpu_name} x{total_gpus} ({total_vram_gb(gpu_name)}GB each)")
    print(f"  Valid P-TP sizes: {p_tp_sizes}")
    if skipped:
        print(f"  Checkpoint: {skipped}/{total} already done → {len(pending)} remaining")
    else:
        print(f"  Evaluating {total} strategies...")

    results = []
    if pending:
        if max_workers > 1:
            with tqdm(total=len(pending), desc="  Searching", unit="strat",
                      bar_format="{desc}: {percentage:3.0f}% |{bar}| "
                                 "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_run_one, t, requests, model_spec, hw_params, slo): t
                              for t in pending}
                    for future in as_completed(futures):
                        r = future.result()
                        results.append(r)
                        _append_csv(out_path, r)
                        pbar.update(1)
        else:
            for t in tqdm(pending, desc="  Searching", unit="strat",
                          bar_format="{desc}: {percentage:3.0f}% |{bar}| "
                                     "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
                r = _run_one(t, requests, model_spec, hw_params, slo)
                results.append(r)
                _append_csv(out_path, r)

    # Merge with previously-checkpointed results for the final sorted list
    if skipped:
        prev_results = _load_results_from_csv(out_path)
        # Deduplicate by label (keep the freshly-computed copy when overlap exists)
        seen = {r["label"] for r in results}
        for pr in prev_results:
            if pr["label"] not in seen:
                results.append(pr)
                seen.add(pr["label"])

    results.sort(key=lambda r: (r.get("slo_pass", False), r.get("throughput", 0)), reverse=True)
    return results


def _run_one(task: dict, requests: list, model_spec: dict, hw_params: dict,
             slo: dict) -> dict:
    """Run a single strategy task. Deep-copies requests to avoid shared-state
    races when running multiple strategies concurrently via max_workers."""
    t0 = time.perf_counter()
    local_requests = copy.deepcopy(list(requests))

    engine = SimulationEngine(task["cfg"], model_spec, hw_params)
    mode = task["mode"]
    tp = task.get("tp", 1)
    pp = task.get("pp", 1)

    if mode == "colocated":
        dp = task.get("dp", 1)
        metrics = engine.run(local_requests, mode=mode, tp_size=tp, dp=dp,
                             pp_size=pp)
    else:
        d_tp = task.get("d_tp", 1)
        d_pp = task.get("d_pp", 1)
        dp_d = task.get("dp_d", 1)
        metrics = engine.run(local_requests, mode=mode,
                             pd_ratio=task.get("pd_ratio"),
                             tp_size=tp, d_tp_size=d_tp,
                             pp_size=pp, d_pp_size=d_pp)

    elapsed = time.perf_counter() - t0
    slo_info = metrics.slo_compliance(slo["p90_ttft_ms"], slo["p90_tpot_ms"])
    throughput = metrics.throughput()

    total_t = metrics.total_time
    result = {
        "label": task["label"],
        "mode_label": task.get("mode_label", task.get("mode", "")),
        "total_gpus": task.get("total_gpus", 0),
        "throughput": throughput,
        "slo_pass": slo_info["slo_pass"],
        "metrics_raw": {
            "throughput": throughput,
            "input_throughput": metrics.input_throughput(),
            "output_throughput": throughput,
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
        "elapsed": elapsed,
    }
    # Add time breakdown percentages
    breakdown = metrics.time_breakdown_pct()
    result["metrics_raw"].update(breakdown)
    return result


def _deep_copy_config(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg))
