"""Fit per-op effective bandwidth + kernel launch overhead.

Model: time = bytes / B_eff + overhead

Each measured op gets independent B_eff from its large-N data.
Ops without direct benchmarks (layernorm, causal_mask) inherit from a
semantically similar proxy.
"""

import numpy as np


# Proxy ops that lack direct benchmarks OR whose benchmarks are unreliable.
#
# rope is PROXIED because the PyTorch synthetic kernel uses slice indexing
# ([:, 0] / [:, 1]) which breaks kernel fusion — see §1b in accuracy doc.
#
# rmsnorm is PROXIED because the benchmark applies F.rms_norm(x, (N,), w)
# where x has shape (N,) — this is a GLOBAL reduction over all N elements.
# Real LLM rmsnorm is per-token: normalized_shape = (hidden_dim,) not (N,).
# The global reduction achieves B_eff ≈ 4.2 GB/s vs ~800 GB/s for per-token.
# See docs/accuracy-improvements.md §1c.
#
# vLLM's real RoPE and per-token rmsnorm are both memory-bandwidth-bound
# in-place kernels with similar overhead to residual_add.
PROXY = {
    "layernorm": "residual_add",          # per-token reduction, similar BW
    "causal_mask": "residual_add",        # simple add pattern
    "rope": "residual_add",               # single fused in-place kernel
    "rmsnorm": "residual_add",            # per-token norm, not global reduction
}

# Ops that have reliable direct benchmarks — fit independently.
MEASURED_OPS = ["residual_add", "softmax", "swiglu"]


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


def _fit_elementwise_dtype(results, label_suffix=""):
    """Fit elementwise B_eff + overhead for a single dtype subset."""
    b_effs = {}
    overheads = {}

    for op_name in MEASURED_OPS:
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            print(f"  {op_name:<16} (no benchmark data — skipped)")
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
            print(f"  {op:<16} → proxied to {proxy}")

    suffix = label_suffix
    return {
        f"elem_b_effs{suffix}": {k: float(v) for k, v in b_effs.items()},
        f"elem_overheads{suffix}": {k: float(v) for k, v in overheads.items()},
    }


def fit_elementwise(results):
    # Detect dtypes
    dtypes = sorted(set(r.get("dtype", "float16") for r in results if r.get("dtype")))

    print("\n" + "=" * 60)
    print("Elementwise Fit (per-op B_eff + overhead)")
    print("  model: time = bytes / B_eff + overhead")
    print(f"  dtypes: {dtypes}")
    print("=" * 60)

    params = {}

    if len(dtypes) <= 1:
        params.update(_fit_elementwise_dtype(results))
    else:
        for dt in dtypes:
            subset = [r for r in results if r.get("dtype", "float16") == dt]
            print(f"\n  ── {dt} ──")
            params.update(_fit_elementwise_dtype(subset, label_suffix=f"_{dt}"))
        # Unified (all dtypes) for backward compat
        print(f"\n  ── unified (all dtypes) ──")
        params.update(_fit_elementwise_dtype(results))

    params["type"] = "elementwise"
    return params
