"""Fit FlashAttention-specific roofline parameters from benchmark data.

FlashAttention has different hardware efficiency than GEMM (different
tiling, SRAM usage, causal mask overhead).  Reusing matmul-fitted F_peak
and B_peak for attention prediction introduces systematic error.

This module fits dedicated (F_peak, B_peak, p) for FlashAttention by
splitting at s_q = 1 (decode) vs s_q > 1 (prefill), matching the two-
regime approach in fit/matmul.py.

When the benchmark includes multiple batch sizes, per-batch B_peak and p
are fitted (with a single shared F_peak from the largest-batch prefill).
This captures the GPU utilisation curve: small batches under-utilise the
GPU, achieving lower effective bandwidth.  At prediction time the sim
interpolates between batch sizes based on the number of concurrent
requests of each type (prefill / decode).
"""

import numpy as np
from fit.utils import roofline_fit

# Split: s_q = 1 is decode (memory-bound GEMV via FA), s_q > 1 is prefill.
# Both share the same F_peak (fitted on prefill, fixed for decode).
S_Q_SPLIT = 1


def _fit_subset(fa_results, label, F_fixed=None):
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


def _fit_single_batch(fa_results):
    """Original fitting path: single batch, split by s_q only."""
    decode = [r for r in fa_results if r["s_q"] <= S_Q_SPLIT]
    prefill = [r for r in fa_results if r["s_q"] > S_Q_SPLIT]

    p_prefill = _fit_subset(prefill, f"prefill (s_q > {S_Q_SPLIT})")
    F_shared = p_prefill.get("F_peak", 1e13)
    p_decode = _fit_subset(decode, f"decode (s_q = {S_Q_SPLIT})", F_fixed=F_shared)

    params = {}
    params.update({f"{k}_fa_decode": v for k, v in p_decode.items()})
    params.update({f"{k}_fa_prefill": v for k, v in p_prefill.items()})
    return params


def _fit_flashattn_dtype(fa_results, label_suffix=""):
    """Fit FA roofline for a single dtype subset.

    When multiple batch sizes are present, fits per-batch B_peak and p
    (with shared F_peak).  Returns params with optional label suffix
    (e.g. "_float16", "_bfloat16").
    """
    batch_sizes = sorted(set(r["b"] for r in fa_results))

    if len(batch_sizes) <= 1:
        p = _fit_single_batch(fa_results)
        # Add suffix and return
        if label_suffix:
            return {f"{k}{label_suffix}": v for k, v in p.items()}
        return p

    # ── Step 1: fit F_peak from largest-batch prefill ──
    largest_b = batch_sizes[-1]
    prefill_largest = [r for r in fa_results
                       if r["b"] == largest_b and r["s_q"] > S_Q_SPLIT]
    p_largest = _fit_subset(prefill_largest,
                            f"prefill b={largest_b} (F_peak anchor){label_suffix}")
    F_shared = p_largest.get("F_peak", 1e13)
    print(f"  → shared F_peak = {F_shared / 1e12:.1f} TF (from b={largest_b} prefill)")

    # ── Step 2: per-batch B_peak and p (F_peak fixed) ──
    decode_B = []
    decode_p = []
    prefill_B = []
    prefill_p = []

    for b_val in batch_sizes:
        batch_results = [r for r in fa_results if r["b"] == b_val]
        decode = [r for r in batch_results if r["s_q"] <= S_Q_SPLIT]
        prefill_b = [r for r in batch_results if r["s_q"] > S_Q_SPLIT]

        p_p = _fit_subset(prefill_b, f"prefill b={b_val}{label_suffix}", F_fixed=F_shared)
        p_d = _fit_subset(decode, f"decode  b={b_val}{label_suffix}", F_fixed=F_shared)

        prefill_B.append(p_p.get("B_peak", 0.0))
        prefill_p.append(p_p.get("p", 1.0))
        decode_B.append(p_d.get("B_peak", 0.0))
        decode_p.append(p_d.get("p", 1.0))

    # ── Step 3: unified fit (all batches) ──
    prefill_all = [r for r in fa_results if r["s_q"] > S_Q_SPLIT]
    decode_all = [r for r in fa_results if r["s_q"] <= S_Q_SPLIT]
    p_prefill_u = _fit_subset(prefill_all, f"prefill (all batches){label_suffix}", F_fixed=F_shared)
    p_decode_u = _fit_subset(decode_all, f"decode  (all batches){label_suffix}", F_fixed=F_shared)

    params = {}
    # Unified keys
    params.update({f"{k}_fa_decode{label_suffix}": v for k, v in p_decode_u.items()})
    params.update({f"{k}_fa_prefill{label_suffix}": v for k, v in p_prefill_u.items()})
    # Per-batch arrays
    params[f"fa_batch_sizes{label_suffix}"] = batch_sizes
    params[f"fa_decode_B{label_suffix}"] = decode_B
    params[f"fa_decode_p{label_suffix}"] = decode_p
    params[f"fa_prefill_B{label_suffix}"] = prefill_B
    params[f"fa_prefill_p{label_suffix}"] = prefill_p
    params[f"fa_bench_nh{label_suffix}"] = fa_results[0].get("nh", 32)

    # Print utilisation curve summary
    print(f"  decode  B(TB/s): {[f'{v/1e12:.2f}' for v in decode_B]}")
    print(f"  prefill B(TB/s): {[f'{v/1e12:.2f}' for v in prefill_B]}")

    return params


def fit_flashattn(results):
    """Fit FlashAttention roofline params from benchmark data.

    When multiple dtypes are present, fits per-dtype params with
    "_float16" / "_bfloat16" suffix, plus unsuffixed unified keys
    for backward compat.

    When multiple batch sizes are present, fits per-batch B_peak and p
    (with shared F_peak) to capture the GPU utilisation curve.

    Returns dict with keys:
      - Unified (backward compat): F_peak_fa_prefill, B_peak_fa_prefill, ...
      - Per-dtype: F_peak_fa_prefill_float16, B_peak_fa_prefill_float16, ...
      - Per-batch arrays: fa_batch_sizes, fa_decode_B, fa_decode_p, ...
    """
    fa_results = [r for r in results if r["op_name"] == "flashattn"]
    if not fa_results:
        print("\n[FlashAttn Fit] No flashattn benchmark data — skipping")
        return {}

    dtypes = sorted(set(r.get("dtype", "float16") for r in fa_results))

    print("\n" + "=" * 60)
    print("Roofline Fit (FlashAttention)")
    print(f"  split at s_q = {S_Q_SPLIT + 1} (decode: s_q=1, prefill: s_q>1)")
    print(f"  dtypes: {dtypes}")
    print("=" * 60)

    params = {}

    if len(dtypes) <= 1:
        params.update(_fit_flashattn_dtype(fa_results))
    else:
        for dt in dtypes:
            subset = [r for r in fa_results if r.get("dtype", "float16") == dt]
            print(f"\n  ── {dt} ──")
            params.update(_fit_flashattn_dtype(subset, label_suffix=f"_{dt}"))

    return params
