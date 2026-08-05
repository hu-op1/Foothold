"""Fit roofline model parameters from CUDA Graph replay benchmark data.

CUDA Graph replay eliminates per-kernel CPU→GPU dispatch overhead
(~5–10 μs each).  The roofline params fitted from graph-replay data
therefore reflect the GPU's true compute and memory bandwidth without
launch-overhead contamination.  These params produce more accurate
decode-step predictions when CUDA Graph is active.

Fits follow the same split strategy as fit/matmul.py and fit/flashattn.py:
  - Matmul:   M < 256 → decode,  M ≥ 256 → prefill (shared F_peak)
  - FlashAttn: s_q = 1 → decode, s_q > 1 → prefill (shared F_peak)
  - Elementwise: per-op B_eff + overhead (no regime split needed)

Output keys are namespaced with ``_cudagraph`` suffix, e.g.:
  F_peak_prefill_cudagraph, B_peak_decode_cudagraph, p_decode_cudagraph,
  elem_b_effs_cudagraph, elem_overheads_cudagraph, …
"""

import numpy as np
from fit.utils import roofline_fit
from fit.elementwise import PROXY  # share proxy map with eager fit

M_SPLIT = 256
S_Q_SPLIT = 1


# ── Matmul CUDA Graph fit ───────────────────────────────────────────────

def _fit_cg_matmul_subset(matmul_results, label, F_fixed=None):
    flops = np.array([r["flops"] for r in matmul_results])
    bytes_moved = np.array([r["bytes"] for r in matmul_results])
    times = np.array([r["time_ms"] for r in matmul_results]) / 1000.0

    if len(matmul_results) < 5:
        print(f"  {label}: too few points ({len(matmul_results)}), skipping")
        return {}

    if F_fixed is not None:
        from scipy.optimize import curve_fit
        def model(X, B, p):
            f, b_arr = X
            return (f / F_fixed + b_arr / B) ** (1 / p)
        B0 = float(np.median(bytes_moved / times))
        popt, _ = curve_fit(model, (flops, bytes_moved), times,
                            p0=[B0 * 0.5, 2.0],
                            bounds=([1e8, 1.0], [1e13, 10.0]),
                            maxfev=20000)
        B_peak, p = popt
        F_peak = F_fixed
        predicted = model((flops, bytes_moved), B_peak, p)
        ss_res = float(np.sum((times - predicted) ** 2))
        ss_tot = float(np.sum((times - np.mean(times)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        F_obs = float(np.max(flops / times))
        F_max = F_obs * 1.05
        F_peak, B_peak, p, r2 = roofline_fit(flops, bytes_moved, times,
                                               F_max=F_max)

    # Under CUDA Graph, per-kernel launch overhead is zero.

    print(f"  {label}: F={F_peak / 1e12:.1f} TF  B={B_peak / 1e12:.2f} TB  p={p:.3f}  "
          f"R2={r2:.4f}")
    return {"F_peak": float(F_peak), "B_peak": float(B_peak),
            "p": float(p), "r2": float(r2)}


def _fit_cg_matmul_dtype(matmul_results, label_suffix=""):
    small = [r for r in matmul_results if r["M"] <= M_SPLIT]
    large = [r for r in matmul_results if r["M"] >= M_SPLIT]

    p_large = _fit_cg_matmul_subset(large, f"prefill (M>={M_SPLIT}){label_suffix}")
    F_shared = p_large.get("F_peak", 1e13)
    p_small = _fit_cg_matmul_subset(small, f"decode (M<={M_SPLIT}){label_suffix}",
                                    F_fixed=F_shared)

    params = {}
    params.update({f"{k}_decode_cudagraph{label_suffix}": v for k, v in p_small.items()})
    params.update({f"{k}_prefill_cudagraph{label_suffix}": v for k, v in p_large.items()})
    return params


def fit_cudagraph_matmul(results):
    matmul_results = [r for r in results if r["op_name"] == "cudagraph_matmul"]
    if not matmul_results:
        return {}

    dtypes = sorted(set(r.get("dtype", "float16") for r in matmul_results))

    print("=" * 60)
    print("Roofline Fit (CUDA Graph Matmul)")
    print(f"  split at M = {M_SPLIT}")
    print(f"  dtypes: {dtypes}")
    print("=" * 60)

    params = {}
    if len(dtypes) <= 1:
        params.update(_fit_cg_matmul_dtype(matmul_results))
    else:
        for dt in dtypes:
            subset = [r for r in matmul_results if r.get("dtype", "float16") == dt]
            params.update(_fit_cg_matmul_dtype(subset, label_suffix=f"_{dt}"))
    return params


# ── Elementwise CUDA Graph fit ──────────────────────────────────────────

def _fit_cg_elem_dtype(elem_results, label_suffix=""):
    """Fit per-op B_eff and overhead from CUDA Graph replay elementwise data."""
    operators = sorted(set(r["operator"] for r in elem_results))

    b_effs = {}
    overheads = {}
    for op_name in operators:
        op_results = [r for r in elem_results if r["operator"] == op_name]
        # Use large-N points for B_eff, small-N for overhead
        large_n = sorted(op_results, key=lambda r: r["N"], reverse=True)[:20]
        small_n = sorted(op_results, key=lambda r: r["N"])[:20]

        if len(large_n) >= 3:
            # Bytes / time → effective bandwidth
            bw = np.array([r["bytes"] / (r["time_ms"] / 1000.0) for r in large_n])
            b_effs[op_name] = float(np.median(bw))
        else:
            b_effs[op_name] = 0.0

        if len(small_n) >= 3:
            times = np.array([r["time_ms"] / 1000.0 for r in small_n])
            bytes_arr = np.array([r["bytes"] for r in small_n])
            if b_effs[op_name] > 0:
                residuals = times - bytes_arr / b_effs[op_name]
                overheads[op_name] = float(np.median([r for r in residuals if r > 0]))
            else:
                overheads[op_name] = 0.0
        else:
            overheads[op_name] = 0.0

        print(f"  {op_name}{label_suffix}: B_eff={b_effs[op_name]/1e9:.2f} GB/s  "
              f"overhead={overheads[op_name]*1e6:.1f} us")

    # Apply proxy mapping for unbenchmarked ops (matches fit/elementwise.py).
    # CUDA Graph benchmarks only cover residual_add, rmsnorm, rope, swiglu;
    # ops like fused_residual_norm, layernorm, causal_mask, softmax inherit
    # from their semantically closest proxy for graph-replay mode too.
    for op, proxy in PROXY.items():
        if proxy in b_effs and op not in b_effs:
            b_effs[op] = b_effs[proxy]
            overheads[op] = overheads[proxy]
            print(f"  {op}{label_suffix}: → proxied to {proxy} (CUDA Graph)")

    params = {
        f"elem_b_effs_cudagraph{label_suffix}": b_effs,
        f"elem_overheads_cudagraph{label_suffix}": overheads,
    }
    return params


def fit_cudagraph_elementwise(results):
    elem_results = [r for r in results if r["op_name"] == "cudagraph_elem"]
    if not elem_results:
        return {}

    dtypes = sorted(set(r.get("dtype", "float16") for r in elem_results))

    print("\n" + "=" * 60)
    print("Roofline Fit (CUDA Graph Elementwise)")
    print(f"  dtypes: {dtypes}")
    print("=" * 60)

    params = {}
    if len(dtypes) <= 1:
        params.update(_fit_cg_elem_dtype(elem_results))
    else:
        for dt in dtypes:
            subset = [r for r in elem_results if r.get("dtype", "float16") == dt]
            params.update(_fit_cg_elem_dtype(subset, label_suffix=f"_{dt}"))
    return params


# ── FlashAttention CUDA Graph fit ───────────────────────────────────────

def _fit_cg_fa_subset(fa_results, label, F_fixed=None):
    flops = np.array([r["flops"] for r in fa_results])
    bytes_moved = np.array([r["bytes"] for r in fa_results])
    times = np.array([r["time_ms"] for r in fa_results]) / 1000.0

    if len(fa_results) < 5:
        print(f"  {label}: too few points ({len(fa_results)}), skipping")
        return {}

    if F_fixed is not None:
        from scipy.optimize import curve_fit
        def model(X, B, p):
            f, b_arr = X
            return (f / F_fixed + b_arr / B) ** (1 / p)
        B0 = float(np.median(bytes_moved / times))
        popt, _ = curve_fit(model, (flops, bytes_moved), times,
                            p0=[B0 * 0.5, 2.0],
                            bounds=([1e8, 1.0], [1e13, 10.0]),
                            maxfev=20000)
        B_peak, p = popt
        F_peak = F_fixed
        predicted = model((flops, bytes_moved), B_peak, p)
        ss_res = float(np.sum((times - predicted) ** 2))
        ss_tot = float(np.sum((times - np.mean(times)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        F_peak, B_peak, p, r2 = roofline_fit(flops, bytes_moved, times)

    print(f"  {label}: F={F_peak / 1e12:.1f} TF  B={B_peak / 1e12:.2f} TB  "
          f"p={p:.3f}  R2={r2:.4f}")
    return {"F_peak": float(F_peak), "B_peak": float(B_peak),
            "p": float(p), "r2": float(r2)}


def _fit_cg_fa_dtype(fa_results, label_suffix=""):
    batch_sizes = sorted(set(r["b"] for r in fa_results))

    if len(batch_sizes) <= 1:
        decode = [r for r in fa_results if r["s_q"] <= S_Q_SPLIT]
        prefill = [r for r in fa_results if r["s_q"] > S_Q_SPLIT]
        p_prefill = _fit_cg_fa_subset(prefill, f"prefill (s_q > {S_Q_SPLIT}){label_suffix}")
        if not p_prefill:
            raise ValueError(
                f"Too few prefill points ({len(prefill)}) to fit the cudagraph FA "
                "roofline — cudagraph_flashattn.csv is likely contaminated by OOM "
                "rows. Restore a clean CSV and re-run the fit."
            )
        F_shared = p_prefill["F_peak"]
        p_decode = _fit_cg_fa_subset(decode, f"decode (s_q = {S_Q_SPLIT}){label_suffix}",
                                     F_fixed=F_shared)
        params = {}
        params.update({f"{k}_fa_decode_cudagraph{label_suffix}": v for k, v in p_decode.items()})
        params.update({f"{k}_fa_prefill_cudagraph{label_suffix}": v for k, v in p_prefill.items()})
        params[f"fa_bench_nh_cudagraph{label_suffix}"] = fa_results[0].get("nh", 32)
        return params

    # Multi-batch: anchor F_peak on the largest batch with ≥5 prefill points
    # AND a healthy fit (R2 ≥ 0.9; polluted batches fit with garbage R2 and
    # extrapolate F absurdly high).  Fall back down the batch list.
    anchor_b = None
    for b_val in reversed(batch_sizes):
        prefill_candidates = [r for r in fa_results
                              if r["b"] == b_val and r["s_q"] > S_Q_SPLIT]
        if len(prefill_candidates) < 5:
            continue
        p_candidate = _fit_cg_fa_subset(prefill_candidates,
                                        f"prefill b={b_val} (F_peak candidate){label_suffix}")
        if p_candidate.get("r2", 0.0) < 0.9:
            print(f"  → b={b_val} anchor fit R2={p_candidate['r2']:.3f} < 0.9 "
                  f"(likely polluted), trying smaller batch")
            continue
        anchor_b = b_val
        p_largest = p_candidate
        break
    if anchor_b is None:
        per_batch = {b_val: len([r for r in fa_results
                                 if r["b"] == b_val and r["s_q"] > S_Q_SPLIT])
                     for b_val in batch_sizes}
        raise ValueError(
            "No batch has ≥5 valid prefill points with a healthy anchor fit "
            f"(R2≥0.9; per-batch prefill point counts: {per_batch}). "
            "cudagraph_flashattn.csv is likely contaminated by OOM rows — "
            "restore a clean CSV and re-run the fit."
        )
    F_shared = p_largest["F_peak"]

    decode_B, decode_p = [], []
    prefill_B, prefill_p = [], []

    for b_val in batch_sizes:
        batch_results = [r for r in fa_results if r["b"] == b_val]
        decode = [r for r in batch_results if r["s_q"] <= S_Q_SPLIT]
        prefill_b = [r for r in batch_results if r["s_q"] > S_Q_SPLIT]
        p_p = _fit_cg_fa_subset(prefill_b, f"prefill b={b_val}{label_suffix}",
                                F_fixed=F_shared)
        p_d = _fit_cg_fa_subset(decode, f"decode  b={b_val}{label_suffix}",
                                F_fixed=F_shared)
        prefill_B.append(p_p.get("B_peak", 0.0))
        prefill_p.append(p_p.get("p", 1.0))
        decode_B.append(p_d.get("B_peak", 0.0))
        decode_p.append(p_d.get("p", 1.0))

    prefill_all = [r for r in fa_results if r["s_q"] > S_Q_SPLIT]
    decode_all = [r for r in fa_results if r["s_q"] <= S_Q_SPLIT]
    p_prefill_u = _fit_cg_fa_subset(prefill_all,
                                    f"prefill (all batches){label_suffix}",
                                    F_fixed=F_shared)
    p_decode_u = _fit_cg_fa_subset(decode_all,
                                   f"decode  (all batches){label_suffix}",
                                   F_fixed=F_shared)

    params = {}
    params.update({f"{k}_fa_decode_cudagraph{label_suffix}": v for k, v in p_decode_u.items()})
    params.update({f"{k}_fa_prefill_cudagraph{label_suffix}": v for k, v in p_prefill_u.items()})
    params[f"fa_batch_sizes_cudagraph{label_suffix}"] = batch_sizes
    params[f"fa_decode_B_cudagraph{label_suffix}"] = decode_B
    params[f"fa_decode_p_cudagraph{label_suffix}"] = decode_p
    params[f"fa_prefill_B_cudagraph{label_suffix}"] = prefill_B
    params[f"fa_prefill_p_cudagraph{label_suffix}"] = prefill_p
    params[f"fa_bench_nh_cudagraph{label_suffix}"] = fa_results[0].get("nh", 32)

    return params


def fit_cudagraph_flashattn(results):
    fa_results = [r for r in results if r["op_name"] == "cudagraph_flashattn"]
    if not fa_results:
        return {}

    dtypes = sorted(set(r.get("dtype", "float16") for r in fa_results))

    print("\n" + "=" * 60)
    print("Roofline Fit (CUDA Graph FlashAttention)")
    print(f"  split at s_q = {S_Q_SPLIT + 1} (decode: s_q=1, prefill: s_q>1)")
    print(f"  dtypes: {dtypes}")
    print("=" * 60)

    params = {}
    if len(dtypes) <= 1:
        params.update(_fit_cg_fa_dtype(fa_results))
    else:
        for dt in dtypes:
            subset = [r for r in fa_results if r.get("dtype", "float16") == dt]
            print(f"\n  ── {dt} ──")
            params.update(_fit_cg_fa_dtype(subset, label_suffix=f"_{dt}"))
    return params


# ── Top-level ───────────────────────────────────────────────────────────

def fit_cudagraph_all(results):
    """Fit all CUDA Graph roofline parameters from benchmark data."""
    params = {}
    params.update(fit_cudagraph_matmul(results))
    params.update(fit_cudagraph_elementwise(results))
    params.update(fit_cudagraph_flashattn(results))
    return params
