"""LM head builder: final_norm -> vocab projection."""

from sim.layers.common import build_rmsnorm
from sim.graph import OpSpec


def build_lm_head(ctx, spec, hw) -> list[OpSpec]:
    """Final RMSNorm + lm_head projection."""
    h = spec["hidden_dim"]
    vs = spec["vocab_size"]
    M = ctx.total_tokens
    F = ctx.matmul_F
    B = ctx.matmul_B
    p = ctx.matmul_p
    ov = ctx.matmul_overhead

    ops = build_rmsnorm(ctx, spec, hw)
    ops.append(OpSpec(
        name="lm_head", category="matmul",
        tags=frozenset({"projection"}),
        M=M, K=h, N=vs,
        F_peak=F, B_peak=B, p=p, overhead=ov,
    ))
    return ops
