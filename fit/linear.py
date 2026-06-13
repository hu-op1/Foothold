"""Linear interpolation backend for roofline prediction.

Stores raw benchmark data as lookup tables. At prediction time,
uses 2D linear interpolation in (flops, bytes) space for matmul
and 1D linear interpolation in bytes space for elementwise ops.

Unlike the roofline model, this makes no parametric assumptions —
it just pieces together measured data points.
"""

M_SPLIT = 256  # matches fit/matmul.py

# Elementwise op proxy map: ops we didn't benchmark inherit from measured ones
PROXY = {
    "swiglu": "residual_add",
    "rope": "residual_add",
    "layernorm": "residual_add",
    "rmsnorm": "residual_add",
    "causal_mask": "residual_add",
}


def fit_matmul_linear(results: list[dict]) -> dict:
    """Package matmul benchmark data split by prefill/decode regime.

    Returns {matmul_decode_data, matmul_prefill_data} each with
    {flops: [...], bytes: [...], times_ms: [...]} lists.
    """
    matmul = [r for r in results if r["op_name"] == "matmul"]
    if not matmul:
        return {}

    small = [r for r in matmul if r["M"] <= M_SPLIT]
    large = [r for r in matmul if r["M"] >= M_SPLIT]

    params = {}
    for label, subset in [("decode", small), ("prefill", large)]:
        if not subset:
            continue
        params[f"matmul_{label}_data"] = {
            "flops": [float(r["flops"]) for r in subset],
            "bytes": [float(r["bytes"]) for r in subset],
            "times_ms": [float(r["time_ms"]) for r in subset],
        }

    return params


def fit_elementwise_linear(results: list[dict]) -> dict:
    """Package elementwise benchmark data per op.

    Returns {elem_data: {op_name: {bytes: [...], times_ms: [...]}}}.
    Unmeasured ops inherit from proxy ops.
    """
    elem_data = {}
    measured_ops = {"residual_add", "rmsnorm", "softmax"}

    for op_name in measured_ops:
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            continue
        # Sort by bytes for clean 1D interpolation
        sorted_results = sorted(op_results, key=lambda r: r["bytes"])
        elem_data[op_name] = {
            "bytes": [float(r["bytes"]) for r in sorted_results],
            "times_ms": [float(r["time_ms"]) for r in sorted_results],
        }

    # Copy proxy data for unmeasured ops
    for op, proxy in PROXY.items():
        if proxy in elem_data and op not in elem_data:
            elem_data[op] = elem_data[proxy]

    return {"elem_data": elem_data}
