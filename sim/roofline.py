"""Roofline model math — shared by the simulation executor.

Extracted from the old ``perf_predict/`` module.  Only the kernel-level
roofline helpers live here; ``predict()`` / ``print_one()`` / ``print_all()``
(the static single-batch predictor) have been removed.
"""

DTYPE_BYTES = 2  # fp16

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

# Bytes-per-element factors for elementwise ops
ELEM_BYTES = {
    "residual_add": 3,
    "swiglu": 3,
    "rope": 4,
    "softmax": 6,
    "layernorm": 5,
    "rmsnorm": 4,
    "causal_mask": 3,
    "fused_residual_norm": 4,  # read residual + read hidden + write normed + write new residual
}


def roofline_time(flops, bytes_moved, F_peak, B_peak, p):
    c = flops / F_peak
    m = bytes_moved / B_peak
    return (c ** p + m ** p) ** (1 / p)


def matmul_time(M, K, N, F, B, p):
    """Predicted time for [M,K] × [K,N] matmul.

    Accounts for tensor-core tile quantization: when K or N are not
    multiples of the tile size (16), the GPU pads internally and wastes
    compute.  The tile-waste factor inflates the predicted time accordingly.
    """
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * DTYPE_BYTES
    t = roofline_time(flops, bytes_moved, F, B, p)
    return t * _tile_waste(K, N)


def elem_time(op_name, N, b_effs, overheads):
    factor = ELEM_BYTES.get(op_name, 3)
    B_eff = b_effs.get(op_name, 1e12)
    overhead = overheads.get(op_name, 0.0)
    return (N * factor * DTYPE_BYTES) / B_eff + overhead


# ── layer ops ───────────────────────────────────────────────────────────

def attn_projections(M, h, F, B, p, nh=None, nh_kv=None, hd=None):
    """Q/K/V/O projections for attention layers.

    Q/K/V are fused into one matmul (matching vLLM's QKVParallelLinear),
    so the input activations are read from HBM once instead of three times.
    O projection is a separate matmul.

    nh, hd: if nh*hd != h, Q proj outputs nh*hd, K/V output nh_kv*hd.
    nh_kv defaults to nh (MHA), set for GQA.
    """
    if nh is None or (nh * hd == h and (nh_kv or nh) == nh):
        # MHA: QKV fused [M,h]×[h,3h] + O [M,h]×[h,h]
        return matmul_time(M, h, 3 * h, F, B, p) + matmul_time(M, h, h, F, B, p)
    # GQA: Q dim ≠ KV dim
    dim_q = nh * hd
    dim_kv = (nh_kv or nh) * hd
    # QKV fused: [M, h] × [h, dim_q + 2·dim_kv]
    t = matmul_time(M, h, dim_q + 2 * dim_kv, F, B, p)
    # O projection: [M, dim_q] × [dim_q, h]
    t += matmul_time(M, dim_q, h, F, B, p)
    return t


def ffn_projections(M, h, inter, F, B, p):
    """FFN projections: gate/up (2×) + down."""
    t = 2 * matmul_time(M, h, inter, F, B, p)
    t += matmul_time(M, inter, h, F, B, p)
    return t


def projections(M, h, inter, F, B, p, nh=None, nh_kv=None, hd=None):
    """Q/K/V/O (4×) + FFN up/gate (2×) + FFN down — convenience wrapper."""
    return attn_projections(M, h, F, B, p, nh, nh_kv, hd) + ffn_projections(M, h, inter, F, B, p)


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
    """All elementwise ops per layer — convenience wrapper."""
    if nh_kv is None:
        nh_kv = nh
    t = norm_ops(b, s, h, norm_type, b_effs, overheads)
    t += swiglu_op(b, s, inter, b_effs, overheads)
    t += rope_op(b, s, nh, nh_kv, hd, b_effs, overheads)
    t += residual_add_ops(b, s, h, b_effs, overheads)
    return t


def norm_ops(b, s, h, norm_type, b_effs, overheads):
    """2 × norm per layer (pre-attention + pre-FFN)."""
    N = b * s * h
    return 2 * elem_time(norm_type, N, b_effs, overheads)


def swiglu_op(b, s, inter, b_effs, overheads):
    """SiLU gating in FFN (1× per layer)."""
    return elem_time("swiglu", b * s * inter, b_effs, overheads)


def rope_op(b, s, nh, nh_kv, hd, b_effs, overheads):
    """RoPE for Q and K (2× per layer)."""
    t = elem_time("rope", b * nh * s * hd, b_effs, overheads)
    t += elem_time("rope", b * nh_kv * s * hd, b_effs, overheads)
    return t


def residual_add_ops(b, s, h, b_effs, overheads):
    """2 × residual addition per layer (post-attention + post-FFN)."""
    N = b * s * h
    return 2 * elem_time("residual_add", N, b_effs, overheads)


def fused_residual_norm_ops(b, s, h, b_effs, overheads):
    """2 × fused residual+norm per layer (post-attn + post-FFN).

    Matches vLLM's fused_add_rms_norm: reads residual + hidden_states,
    computes norm in a single kernel, writes normed output + new residual.
    Eliminates the intermediate HBM traffic of separate residual_add + rmsnorm.

    When fused_residual_norm is not in b_effs, falls back to separate ops.
    """
    if "fused_residual_norm" in b_effs:
        N = b * s * h
        return 2 * elem_time("fused_residual_norm", N, b_effs, overheads)
    # Fallback: separate residual_add + rmsnorm (backward compatible)
    return residual_add_ops(b, s, h, b_effs, overheads) + norm_ops(b, s, h, "rmsnorm", b_effs, overheads)
