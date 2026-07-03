"""Fit FlashAttention-specific roofline parameters from benchmark data.

FlashAttention has different hardware efficiency than GEMM (different
tiling, SRAM usage, causal mask overhead).  Reusing matmul-fitted F_peak
and B_peak for attention prediction introduces systematic error.

This module fits dedicated (F_peak, B_peak, p) for FlashAttention by
splitting at s_q = 1 (decode) vs s_q > 1 (prefill), matching the two-
regime approach in fit/matmul.py.
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

    if len(fa_results) < 10:
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


def fit_flashattn(results):
    """Fit FlashAttention roofline params from benchmark data.

    Returns dict with keys: F_peak_fa_prefill, B_peak_fa_prefill, p_fa_prefill,
    F_peak_fa_decode, B_peak_fa_decode, p_fa_decode.
    """
    fa_results = [r for r in results if r["op_name"] == "flashattn"]
    if not fa_results:
        print("\n[FlashAttn Fit] No flashattn benchmark data — skipping")
        return {}

    print("\n" + "=" * 60)
    print("Roofline Fit (FlashAttention)")
    print(f"  split at s_q = {S_Q_SPLIT + 1} (decode: s_q=1, prefill: s_q>1)")
    print("=" * 60)

    decode = [r for r in fa_results if r["s_q"] <= S_Q_SPLIT]
    prefill = [r for r in fa_results if r["s_q"] > S_Q_SPLIT]

    # Step 1: fit prefill (large s_q) — F_peak well-constrained here
    p_prefill = _fit_subset(prefill, f"prefill (s_q > {S_Q_SPLIT})")

    # Step 2: fit decode (s_q = 1) with F_peak fixed from prefill
    F_shared = p_prefill.get("F_peak", 1e13)
    p_decode = _fit_subset(decode, f"decode (s_q = {S_Q_SPLIT})", F_fixed=F_shared)

    params = {}
    params.update({f"{k}_fa_decode": v for k, v in p_decode.items()})
    params.update({f"{k}_fa_prefill": v for k, v in p_prefill.items()})

    return params
