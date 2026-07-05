"""Fit kernel launch overhead from CPU-wall-clock benchmark data.

The launch overhead CSV contains slope-analysis rows (n_launches=0) with
the per-launch overhead stored in cpu_time_ms (overhead in ms) and
gpu_time_ms (per-kernel GPU time in ms).  We extract the median overhead
across trivial and realistic matmul kernels for each dtype.
"""

import numpy as np


def fit_launch_overhead(results):
    """Extract kernel launch overhead from benchmark results.

    Args:
        results: list of benchmark dicts from launch_overhead.csv.
            Expected fields: op_name, dtype, n_launches, cpu_time_ms.

    Returns:
        dict with key ``kernel_launch_overhead_us`` (float) — per-kernel
        CPU→GPU dispatch overhead in microseconds.  Returns empty dict
        if no launch overhead data is present.
    """
    overhead_rows = [r for r in results
                     if r.get("op_name", "").startswith("launch_")
                     and r.get("n_launches") == 0]

    if not overhead_rows:
        print("  [launch_overhead] No slope-fit rows found — skipping")
        return {}

    # Group by dtype, take median across kernel types per dtype
    dtypes = sorted(set(r.get("dtype", "float16") for r in overhead_rows))
    all_overheads = []

    for dt in dtypes:
        dt_rows = [r for r in overhead_rows if r.get("dtype") == dt]
        overheads_us = []
        for r in dt_rows:
            overhead_ms = r.get("cpu_time_ms", 0.0)
            # n_launches=0 rows store overhead in cpu_time_ms as ms
            overheads_us.append(overhead_ms * 1000)

        if overheads_us:
            median_us = float(np.median(overheads_us))
            all_overheads.append(median_us)
            print(f"  [{dt}] kernel launch overhead: {median_us:.1f} µs "
                  f"(from {len(overheads_us)} kernel types)")

    if not all_overheads:
        return {}

    # Take median across dtypes (overhead is hardware-dependent, not dtype-dependent)
    overall_us = float(np.median(all_overheads))
    print(f"  kernel_launch_overhead = {overall_us:.1f} µs")

    return {"kernel_launch_overhead_us": overall_us}
