"""Roofline model math — shared by the simulation executor.

Extracted from the old ``perf_predict/`` module.  Only the kernel-level
roofline helpers live here; ``predict()`` / ``print_one()`` / ``print_all()``
(the static single-batch predictor) have been removed.
"""

# Bytes per element for each dtype.
DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}
# Legacy default (fp16, 2 bytes) — used when dtype is not specified.
_DTYPE_BYTES_DEFAULT = 2


def dtype_bytes(dtype_name=None):
    """Bytes per element for a named dtype.  Falls back to fp16 (2 bytes)."""
    if dtype_name is None:
        return _DTYPE_BYTES_DEFAULT
    return DTYPE_BYTES_MAP.get(dtype_name, _DTYPE_BYTES_DEFAULT)

# Tensor core tile size for fp16 (m16n8k16 on Ampere, m16n8k8 on Turing).
# The K and N dimensions are quantized to tile boundaries; non-aligned
# dimensions cause the GPU to pad internally, inflating effective FLOPs.
_TILE_K = 16
_TILE_N = 16


def _tile_waste(K, N):
    """Return time multiplier ≥ 1.0 for tensor-core tile quantization waste.

    GPU tensor cores operate on fixed-size tiles (16×16 for fp16).
    When the inner dimension K or output dimension N is not a multiple
    of the tile size, the hardware pads to the next tile boundary and
    discards unused results, wasting compute.

    For well-aligned dimensions (multiples of 16) — which includes
    all standard LLM hidden/intermediate sizes — returns 1.0.
    """
    K_pad = ((K + _TILE_K - 1) // _TILE_K) * _TILE_K
    N_pad = ((N + _TILE_N - 1) // _TILE_N) * _TILE_N
    return (K_pad / K) * (N_pad / N)


def roofline_time(flops, bytes_moved, F_peak, B_peak, p):
    c = flops / F_peak
    m = bytes_moved / B_peak
    return (c ** p + m ** p) ** (1 / p)


def matmul_time(M, K, N, F, B, p, dt_bytes=None, overhead=0.0):
    """Predicted time for [M,K] × [K,N] matmul.

    Accounts for tensor-core tile quantization: when K or N are not
    multiples of the tile size (16), the GPU pads internally and wastes
    compute.  The tile-waste factor inflates the predicted time accordingly.

    *overhead* is the per-kernel launch overhead (seconds).  For small M
    (decode, M=1) this can be comparable to the roofline time itself.
    """
    if dt_bytes is None:
        dt_bytes = _DTYPE_BYTES_DEFAULT
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * dt_bytes
    t = roofline_time(flops, bytes_moved, F, B, p)
    return t * _tile_waste(K, N) + overhead
