"""Fit elementwise streaming bandwidth + per-op kernel launch overhead.

All elementwise are memory-bound (AI < 1).
Model: time = bytes / B_stream + overhead

B_stream estimated from softmax (cleanest bandwidth signal).
Overhead fitted per op given B_stream.
"""

import numpy as np


BYTES_FACTORS = {
    "residual_add": 3,
    "rmsnorm": 4,
    "softmax": 6,
}

PROXY = {
    "swiglu": "residual_add",
    "rope": "residual_add",
    "layernorm": "rmsnorm",
    "causal_mask": "residual_add",
}


def fit_elementwise(results):
    print("\n" + "=" * 60)
    print("Elementwise Streaming Fit")
    print("  model: time = bytes / B_stream + overhead")
    print("=" * 60)

    # Step 1: B_stream from softmax largest-N bandwidth
    sm_results = [r for r in results if r["op_name"] == "softmax"]
    sm_results.sort(key=lambda r: r["bytes"])
    bw = [r["bytes"] / (r["time_ms"] / 1000.0) for r in sm_results[-3:]]
    B_stream = float(np.median(bw))
    print(f"  B_stream = {B_stream / 1e12:.3f} TB/s  (from softmax largest N)")

    # Step 2: overhead per op
    overheads = {}
    for op_name in ["residual_add", "rmsnorm", "softmax"]:
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            continue

        bytes_moved = np.array([r["bytes"] for r in op_results])
        times = np.array([r["time_ms"] for r in op_results]) / 1000.0

        # Overhead from small-N points (bytes < 10% of max)
        threshold = np.median(bytes_moved) * 0.1
        small = bytes_moved <= threshold
        if small.any():
            overhead = float(np.mean(times[small] - bytes_moved[small] / B_stream))
        else:
            residuals = times - bytes_moved / B_stream
            overhead = float(np.mean(residuals))
        overhead = max(overhead, 0.0)  # overhead can't be negative
        overheads[op_name] = overhead

        predicted = bytes_moved / B_stream + overhead
        ss_res = float(np.sum((times - predicted) ** 2))
        ss_tot = float(np.sum((times - np.mean(times)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  {op_name:<16} overhead={overhead * 1e6:>6.1f} us  R2={r2:.4f}")

    for op, proxy in PROXY.items():
        if proxy in overheads:
            overheads[op] = overheads[proxy]

    return {
        "B_stream": float(B_stream),
        "elem_overheads": overheads,
        "type": "elementwise",
    }
