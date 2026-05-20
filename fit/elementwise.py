"""Fit per-op effective bandwidth + kernel launch overhead.

Model: time = bytes / B_eff + overhead

Each measured op gets independent B_eff from its large-N data.
Unmeasured ops inherit from a proxy.
"""

import numpy as np


PROXY = {
    "swiglu": "residual_add",
    "rope": "residual_add",
    "layernorm": "residual_add",
    "rmsnorm": "residual_add",
    "causal_mask": "residual_add",
}


def _fit_op(bytes_moved, times_s, op_name):
    """Fit B_eff and overhead for a single op.

    B_eff from largest 3 N points (where overhead is negligible).
    Overhead from small N points (where bandwidth term is negligible).
    """
    idx = np.argsort(bytes_moved)
    bs = bytes_moved[idx]
    ts = times_s[idx]

    # B_eff: effective bandwidth from largest 3 points
    bw = bs[-3:] / ts[-3:]
    B_eff = float(np.median(bw))

    # Overhead: from smallest 3 points, subtracting bandwidth term
    overhead = float(np.median(ts[:3] - bs[:3] / B_eff))
    overhead = max(overhead, 0.0)

    predicted = bs / B_eff + overhead
    ss_res = float(np.sum((ts - predicted) ** 2))
    ss_tot = float(np.sum((ts - np.mean(ts)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return B_eff, overhead, r2


def fit_elementwise(results):
    print("\n" + "=" * 60)
    print("Elementwise Fit (per-op B_eff + overhead)")
    print("  model: time = bytes / B_eff + overhead")
    print("=" * 60)

    b_effs = {}
    overheads = {}

    for op_name in ["residual_add", "softmax"]:
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            continue

        bytes_moved = np.array([r["bytes"] for r in op_results])
        times = np.array([r["time_ms"] for r in op_results]) / 1000.0

        B_eff, overhead, r2 = _fit_op(bytes_moved, times, op_name)
        b_effs[op_name] = B_eff
        overheads[op_name] = overhead

        print(f"  {op_name:<16} B={B_eff / 1e12:.4f} TB/s  "
              f"overhead={overhead * 1e6:>6.1f} us  R2={r2:.4f}")

    for op, proxy in PROXY.items():
        if proxy in b_effs:
            b_effs[op] = b_effs[proxy]
            overheads[op] = overheads[proxy]

    return {
        "elem_b_effs": {k: float(v) for k, v in b_effs.items()},
        "elem_overheads": {k: float(v) for k, v in overheads.items()},
        "type": "elementwise",
    }
