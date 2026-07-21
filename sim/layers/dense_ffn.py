"""Dense FFN builder: gate+up projection -> SwiGLU -> down projection."""

from sim.layers import register_ffn
from sim.layers.common import build_swiglu
from sim.graph import OpSpec


@register_ffn("dense")
def build_dense_ffn(ctx, lt, hw) -> list[OpSpec]:
    inter = lt["intermediate_dim"]
    M = ctx.total_tokens
    h = ctx.hidden_dim
    F = ctx.matmul_F
    B = ctx.matmul_B
    p = ctx.matmul_p
    ov = ctx.matmul_overhead

    ops = [
        # Gate + Up fused projection (tag: projection -> TP cuts N)
        OpSpec(
            name="gate_up_proj", category="matmul",
            tags=frozenset({"projection"}),
            M=M, K=h, N=2 * inter,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
    ]

    # SwiGLU activation
    ops += build_swiglu(ctx, inter, hw)

    # Down projection
    ops.append(OpSpec(
        name="down_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=M, K=inter, N=h,
        F_peak=F, B_peak=B, p=p, overhead=ov,
    ))

    return ops
