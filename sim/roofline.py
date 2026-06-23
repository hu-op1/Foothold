"""Roofline model math — shared by the simulation executor.

Extracted from the old ``perf_predict/`` module.  Only the kernel-level
roofline helpers live here; ``predict()`` / ``print_one()`` / ``print_all()``
(the static single-batch predictor) have been removed.
"""

DTYPE_BYTES = 2  # fp16

# Bytes-per-element factors for elementwise ops
ELEM_BYTES = {
    "residual_add": 3,
    "swiglu": 3,
    "rope": 4,
    "softmax": 6,
    "layernorm": 5,
    "rmsnorm": 4,
    "causal_mask": 3,
}


def roofline_time(flops, bytes_moved, F_peak, B_peak, p):
    c = flops / F_peak
    m = bytes_moved / B_peak
    return (c ** p + m ** p) ** (1 / p)


def matmul_time(M, K, N, F, B, p):
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * DTYPE_BYTES
    return roofline_time(flops, bytes_moved, F, B, p)


def elem_time(op_name, N, b_effs, overheads):
    factor = ELEM_BYTES.get(op_name, 3)
    B_eff = b_effs.get(op_name, 1e12)
    overhead = overheads.get(op_name, 0.0)
    return (N * factor * DTYPE_BYTES) / B_eff + overhead


# ── layer ops ───────────────────────────────────────────────────────────

def projections(M, h, inter, F, B, p, nh=None, nh_kv=None, hd=None):
    """Q/K/V/O (4×) + FFN up/gate (2×) + FFN down.

    nh, hd: if nh*hd != h, Q proj outputs nh*hd, K/V output nh_kv*hd, O inputs nh*hd.
    nh_kv defaults to nh (MHA), set for GQA.
    """
    if nh is None or (nh * hd == h and (nh_kv or nh) == nh):
        t = 4 * matmul_time(M, h, h, F, B, p)
    else:
        dim_q = nh * hd
        dim_kv = (nh_kv or nh) * hd
        t = matmul_time(M, h, dim_q, F, B, p)       # Q proj
        t += matmul_time(M, h, dim_kv, F, B, p)      # K proj
        t += matmul_time(M, h, dim_kv, F, B, p)      # V proj
        t += matmul_time(M, dim_q, h, F, B, p)       # O proj
    t += 2 * matmul_time(M, h, inter, F, B, p)
    t += matmul_time(M, inter, h, F, B, p)
    return t


def attention_fused(b, nh, s_q, s_kv, hd, F, B, p, nh_kv=None):
    """FlashAttention: QKᵀ + softmax + score×V fused in SRAM.

    FLOPs unchanged, but intermediate S×S matrix never touches HBM.
    Bytes = Q,K,V reads + O write (no S×S round-trip).

    nh_kv: number of KV heads (defaults to nh for MHA; < nh for GQA).
    """
    if nh_kv is None:
        nh_kv = nh
    M_q = b * nh * s_q
    M_kv = b * nh_kv * s_kv
    flops = 4 * M_q * s_kv * hd
    bytes_moved = b * hd * DTYPE_BYTES * (nh * s_q + nh_kv * s_kv + nh_kv * s_kv + nh * s_q)
    return roofline_time(flops, bytes_moved, F, B, p)


def elementwise_ops(b, s, h, inter, nh, hd, norm_type, b_effs, overheads, nh_kv=None):
    """All elementwise ops per layer.

    nh_kv: number of KV heads (defaults to nh for MHA; < nh for GQA).
    """
    if nh_kv is None:
        nh_kv = nh
    t = 0.0
    N = b * s * h
    t += 2 * elem_time(norm_type, N, b_effs, overheads)
    t += elem_time("swiglu", b * s * inter, b_effs, overheads)
    t += elem_time("rope", b * nh * s * hd, b_effs, overheads)
    t += elem_time("rope", b * nh_kv * s * hd, b_effs, overheads)
    t += 2 * elem_time("residual_add", N, b_effs, overheads)
    return t
