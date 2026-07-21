"""MoE FFN builder: router -> all-to-all -> expert GEMMs -> all-to-all -> shared FFN."""

from sim.layers import register_ffn
from sim.layers.dense_ffn import build_dense_ffn
from sim.layers.common import build_swiglu
from sim.graph import OpSpec


@register_ffn("moe")
def build_moe_ffn(ctx, lt, hw) -> list[OpSpec]:
    h = ctx.hidden_dim
    M = ctx.total_tokens
    F = ctx.matmul_F
    B = ctx.matmul_B
    p = ctx.matmul_p
    ov = ctx.matmul_overhead

    num_experts = lt["num_experts"]
    moe_inter = lt.get("moe_intermediate_dim", lt["intermediate_dim"] // 4)
    shared_inter = lt["intermediate_dim"]
    topk = lt.get("num_experts_per_tok", 2)
    ep = ctx.ep_size

    ops = [
        # Router projection (replicated, tag: expert_router — NOT cut by TP/EP)
        OpSpec(
            name="router", category="matmul",
            tags=frozenset({"expert_router"}),
            M=M, K=h, N=num_experts,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
    ]

    if ep > 1:
        total_expert_bytes = M * h * ctx.dt_bytes * topk
        ops.append(OpSpec(
            name="all_to_all_send", category="comm",
            tags=frozenset({"expert_comm"}),
            comm_bytes=total_expert_bytes,
            comm_type="all_to_all",
            comm_lut_bytes=ctx.comm_lut_bytes,
            comm_lut_time_s=ctx.comm_lut_time_s,
        ))

    # Per-expert GEMM (M /= ep for token distribution across EP ranks)
    expert_M = M if ep <= 1 else M * topk // ep
    ops.append(OpSpec(
        name="expert_gate_up", category="matmul",
        tags=frozenset({"expert"}),
        M=expert_M, K=h, N=2 * moe_inter,
        F_peak=F, B_peak=B, p=p, overhead=ov,
    ))
    ops += build_swiglu(ctx, moe_inter, hw)
    ops.append(OpSpec(
        name="expert_down", category="matmul",
        tags=frozenset({"expert"}),
        M=expert_M, K=moe_inter, N=h,
        F_peak=F, B_peak=B, p=p, overhead=ov,
    ))

    if ep > 1:
        ops.append(OpSpec(
            name="all_to_all_recv", category="comm",
            tags=frozenset({"expert_comm"}),
            comm_bytes=expert_M * h * ctx.dt_bytes,
            comm_type="all_to_all",
            comm_lut_bytes=ctx.comm_lut_bytes,
            comm_lut_time_s=ctx.comm_lut_time_s,
        ))

    # Shared FFN (standard dense, tag: expert_shared)
    ops += build_dense_ffn(ctx, lt, hw)

    return ops
