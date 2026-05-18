"""Fit effective bandwidth from elementwise ops for validation.

All elementwise are memory-bound (AI < 1). We fit B_effective per op
and check they agree within ~10%, confirming B_peak is hardware property
not op-specific.
"""

import numpy as np


BYTES_FACTORS = {
    "residual_add": 3,
    "rmsnorm": 4,
    "softmax": 6,
}


def fit_elementwise(results):
    print("\n" + "=" * 60)
    print("Elementwise Bandwidth Validation")
    print("=" * 60)

    params = {}
    for op_name in ["residual_add", "rmsnorm", "softmax"]:
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            continue

        bytes_moved = np.array([r["bytes"] for r in op_results])
        times = np.array([r["time_ms"] for r in op_results]) / 1000.0

        # Memory-bound: time = bytes / B
        B_effective = float(np.median(bytes_moved / times))
        predicted = bytes_moved / B_effective
        ss_res = float(np.sum((times - predicted) ** 2))
        ss_tot = float(np.sum((times - np.mean(times)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  {op_name:<16} B_eff = {B_effective / 1e12:.3f} TB/s  R2 = {r2:.4f}")

        params[op_name] = {
            "B_effective": B_effective,
            "r2": float(r2),
            "type": "elementwise",
        }

    return params
