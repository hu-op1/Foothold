"""Fit roofline model parameters from matmul benchmark data."""

import numpy as np
from fit.utils import roofline_fit


def fit_matmul(results):
    matmul_results = [r for r in results if r["op_name"] == "matmul"]
    if not matmul_results:
        return {}

    flops = np.array([r["flops"] for r in matmul_results])
    bytes_moved = np.array([r["bytes"] for r in matmul_results])
    times = np.array([r["time_ms"] for r in matmul_results]) / 1000.0

    F_peak, B_peak, p, r2 = roofline_fit(flops, bytes_moved, times)

    print("=" * 60)
    print("Roofline Fit (Matmul)")
    print("=" * 60)
    print(f"  F_peak = {F_peak:.3e} FLOP/s  ({F_peak / 1e12:.1f} TFLOP/s)")
    print(f"  B_peak = {B_peak:.3e} bytes/s ({B_peak / 1e12:.2f} TB/s)")
    print(f"  p      = {p:.3f}")
    print(f"  R²     = {r2:.4f}")

    return {
        "F_peak": float(F_peak),
        "B_peak": float(B_peak),
        "p": float(p),
        "r2": float(r2),
    }
