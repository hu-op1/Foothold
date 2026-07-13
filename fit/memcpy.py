"""Fit memory copy LUT from benchmark data.

Builds a byte-size → transfer-time lookup table for the simulator,
replacing the simple BW+latency linear model with measured data.
"""

import numpy as np


def fit_memcpy(results):
    """Build memcpy LUT arrays from benchmark data.

    Returns dict with keys:
      memcpy_d2h_bytes: sorted byte-size list
      memcpy_d2h_time_s: corresponding transfer times in seconds
      memcpy_h2d_bytes: sorted byte-size list
      memcpy_h2d_time_s: corresponding transfer times in seconds
    """
    memcpy_results = [r for r in results if r.get("op_name") == "memcpy"]
    if not memcpy_results:
        return {}

    params = {}

    for direction in ("d2h", "h2d"):
        dir_data = [r for r in memcpy_results if r.get("direction") == direction]
        if not dir_data:
            continue

        dir_data.sort(key=lambda r: r["bytes"])
        byte_arr = np.array([r["bytes"] for r in dir_data], dtype=np.float64)
        time_arr = np.array([r["time_ms"] for r in dir_data], dtype=np.float64) / 1000.0

        params[f"memcpy_{direction}_bytes"] = byte_arr.tolist()
        params[f"memcpy_{direction}_time_s"] = time_arr.tolist()

        print(f"  memcpy {direction}: {len(dir_data)} points, "
              f"{byte_arr[0]}–{byte_arr[-1]} bytes, "
              f"{time_arr[0]*1e6:.1f}–{time_arr[-1]*1e6:.1f} µs")

    return params
