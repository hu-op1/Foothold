"""Fit roofline model parameters from matmul benchmark data.

Fits one shared F_peak (hardware constant), then separate B_peak and p
for memory-bound and compute-bound regimes split by M.

F_peak is estimated from the peak observed FLOP/s in the prefill regime
(M >= 256, compute-bound) and capped to the GPU's physical limit.

A per-kernel launch overhead is estimated from the smallest-M data
points where kernel launch dominates the roofline time.
"""

import numpy as np
from fit.utils import roofline_fit

M_SPLIT = 256


def _fit_subset(matmul_results, label, F_fixed=None, fit_overhead=False):
    flops = np.array([r["flops"] for r in matmul_results])
    bytes_moved = np.array([r["bytes"] for r in matmul_results])
    times = np.array([r["time_ms"] for r in matmul_results]) / 1000.0

    if len(matmul_results) < 5:
        print(f"  {label}: too few points ({len(matmul_results)}), skipping")
        return {}

    if F_fixed is not None:
        # Fit only B and p, holding F_peak fixed
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
        # Prefill fit: cap F_peak to GPU's observed peak + 5% margin
        F_obs = float(np.max(flops / times))
        F_max = F_obs * 1.05
        F_peak, B_peak, p, r2 = roofline_fit(flops, bytes_moved, times,
                                               F_max=F_max)

    # ── Fit per-kernel launch overhead ──
    # For small M (M <= 4), kernel launch overhead dominates the
    # roofline time.  We estimate overhead as the residual after
    # subtracting the roofline prediction.
    matmul_overhead = 0.0
    if fit_overhead:
        small = [r for r in matmul_results if r["M"] <= 4]
        if len(small) >= 3:
            residuals = []
            for r in small:
                t_meas = r["time_ms"] / 1000.0
                f, b = r["flops"], r["bytes"]
                t_pred = (f / F_fixed + b / B_peak) ** (1 / p) if F_fixed is not None else \
                         (f / F_peak + b / B_peak) ** (1 / p)
                residuals.append(t_meas - t_pred)
            matmul_overhead = float(np.median([r for r in residuals if r > 0]))

    print(f"  {label}: F={F_peak / 1e12:.1f} TF  B={B_peak / 1e12:.2f} TB  p={p:.3f}  "
          f"overhead={matmul_overhead * 1e6:.1f} us  R2={r2:.4f}")
    result = {"F_peak": float(F_peak), "B_peak": float(B_peak),
              "p": float(p), "r2": float(r2)}
    if fit_overhead:
        result["matmul_overhead"] = matmul_overhead
    return result


def _fit_matmul_dtype(matmul_results, dtypes, label_suffix=""):
    """Fit matmul roofline for a single dtype (or all mixed)."""
    small = [r for r in matmul_results if r["M"] <= M_SPLIT]
    large = [r for r in matmul_results if r["M"] >= M_SPLIT]

    p_large = _fit_subset(large, f"prefill (M>={M_SPLIT}){label_suffix}")
    F_shared = p_large.get("F_peak", 1e13)
    p_small = _fit_subset(small, f"decode (M<={M_SPLIT}){label_suffix}",
                          F_fixed=F_shared, fit_overhead=True)

    params = {}
    params.update({f"{k}_decode{label_suffix}": v for k, v in p_small.items()})
    params.update({f"{k}_prefill{label_suffix}": v for k, v in p_large.items()})
    # Copy overhead from decode to prefill (same hardware, same launch cost)
    if "matmul_overhead" in p_small:
        params[f"matmul_overhead_prefill{label_suffix}"] = p_small["matmul_overhead"]
    return params


def fit_matmul(results):
    matmul_results = [r for r in results if r["op_name"] == "matmul"]
    if not matmul_results:
        return {}

    # Detect dtypes present in results
    dtypes = sorted(set(r.get("dtype", "float16") for r in matmul_results))

    print("=" * 60)
    print("Roofline Fit (Matmul)")
    print(f"  split at M = {M_SPLIT}")
    print(f"  dtypes: {dtypes}")
    print("=" * 60)

    params = {}

    if len(dtypes) <= 1:
        # Single dtype — fit directly (no suffix, backward compat)
        params.update(_fit_matmul_dtype(matmul_results, dtypes))
    else:
        # Per-dtype fitting
        for dt in dtypes:
            subset = [r for r in matmul_results if r.get("dtype", "float16") == dt]
            params.update(_fit_matmul_dtype(subset, [dt], label_suffix=f"_{dt}"))
        # Unified (all dtypes) for backward compat
        params.update(_fit_matmul_dtype(matmul_results, dtypes))

    return params
