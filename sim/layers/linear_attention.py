"""DeltaNet linear attention builder: QK projection -> V projection -> linear scan -> output."""

from sim.layers import register_attention
from sim.graph import OpSpec


@register_attention("linear_attention")
def build_linear_attention(ctx, lt, hw) -> list[OpSpec]:
    nh_qk = lt["num_qk_heads"]
    nh_v = lt["num_v_heads"]
    hd = lt["head_dim"]
    h = ctx.hidden_dim
    M = ctx.total_tokens
    F = ctx.matmul_F
    B = ctx.matmul_B
    p = ctx.matmul_p
    ov = ctx.matmul_overhead

    dim_qk = nh_qk * hd
    dim_v = nh_v * hd

    return [
        # QK projection (Q=K, fused)
        OpSpec(
            name="qk_proj", category="matmul",
            tags=frozenset({"projection"}),
            M=M, K=h, N=2 * dim_qk,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
        # V projection
        OpSpec(
            name="v_proj", category="matmul",
            tags=frozenset({"projection"}),
            M=M, K=h, N=dim_v,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
        # Linear scan (matmul-based approximation)
        OpSpec(
            name="linear_scan", category="matmul",
            tags=frozenset({"projection"}),
            M=M, K=h, N=h,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
        # Output projection
        OpSpec(
            name="out_proj", category="matmul",
            tags=frozenset({"projection"}),
            M=M, K=dim_v, N=h,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
    ]
