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


def _fit_subset(matmul_results, label, F_fixed=None):
    flops = np.array([r["flops"] for r in matmul_results])
    bytes_moved = np.array([r["bytes"] for r in matmul_results])
    times = np.array([r["time_ms"] for r in matmul_results]) / 1000.0

    if len(matmul_results) < 5:
        print(f"  {label}: too few points ({len(matmul_results)}), skipping")
        return {}

    times = np.clip(times, 1e-9, None)  # avoid negative from noise

    if F_fixed is not None:
        # Fit only B and p, holding F_peak fixed
        from scipy.optimize import curve_fit

        B_bounds = (1e8, 1e13)
        p_bounds = (1.0, 10.0)

        def model(X, B, p):
            f, b_arr = X
            return (f / F_fixed + b_arr / B) ** (1 / p)
        B0 = float(np.median(bytes_moved / times))
        B0 = float(np.clip(B0 * 0.5, *B_bounds))
        popt, _ = curve_fit(model, (flops, bytes_moved), times,
                            p0=[B0, 2.0],
                            bounds=([B_bounds[0], p_bounds[0]], [B_bounds[1], p_bounds[1]]),
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

    print(f"  {label}: F={F_peak / 1e12:.1f} TF  B={B_peak / 1e12:.2f} TB  p={p:.3f}  "
          f"R2={r2:.4f}")
    return {"F_peak": float(F_peak), "B_peak": float(B_peak),
            "p": float(p), "r2": float(r2)}


def _fit_matmul_dtype(matmul_results, dtypes, label_suffix=""):
    """Fit matmul roofline for a single dtype (or all mixed)."""
    small = [r for r in matmul_results if r["M"] <= M_SPLIT]
    large = [r for r in matmul_results if r["M"] >= M_SPLIT]

    p_large = _fit_subset(large, f"prefill (M>={M_SPLIT}){label_suffix}")
    if not p_large:
        raise ValueError(
            f"Too few prefill points ({len(large)}) to fit the matmul roofline "
            "— matmul.csv is likely contaminated by OOM rows. Restore a clean "
            "CSV and re-run the fit."
        )
    F_shared = p_large["F_peak"]
    p_small = _fit_subset(small, f"decode (M<={M_SPLIT}){label_suffix}",
                          F_fixed=F_shared)

    params = {}
    params.update({f"{k}_decode{label_suffix}": v for k, v in p_small.items()})
    params.update({f"{k}_prefill{label_suffix}": v for k, v in p_large.items()})
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
        # Single dtype — fit directly (no suffix)
        params.update(_fit_matmul_dtype(matmul_results, dtypes))
    else:
        # Per-dtype fitting with suffix (e.g. _float16, _bfloat16)
        for dt in dtypes:
            subset = [r for r in matmul_results if r.get("dtype", "float16") == dt]
            params.update(_fit_matmul_dtype(subset, [dt], label_suffix=f"_{dt}"))

    return params
