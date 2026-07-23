"""Fine-grained operator builders — each builds a single OpSpec.

Builders have explicit parameter signatures (h, nh, nh_kv, hd, inter, etc.)
instead of opaque dicts. Per-model files compose them into full computation
graphs.
"""

from sim.graph import OpSpec


def build_qkv_proj(ctx, h, nh, nh_kv, hd, hw) -> list[OpSpec]:
    dim_q = nh * hd
    dim_kv = nh_kv * hd
    return [OpSpec(
        name="qkv_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=h, N=dim_q + 2 * dim_kv,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_o_proj(ctx, h, nh, hd, hw) -> list[OpSpec]:
    dim_q = nh * hd
    return [OpSpec(
        name="o_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=dim_q, N=h,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_flash_attn(ctx, hw) -> list[OpSpec]:
    return [OpSpec(
        name="flash_attn", category="attention",
        tags=frozenset({"attention"}),
        prefill_flops=ctx.prefill_flops, prefill_bytes=ctx.prefill_bytes,
        decode_flops=ctx.decode_flops, decode_bytes=ctx.decode_bytes,
        fa_d=ctx.fa_decode_params, fa_p=ctx.fa_prefill_params,
    )]


def build_rope(ctx, nh, nh_kv, hd, rope_dim, hw) -> list[OpSpec]:
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    b_eff = b_effs.get("rope", 1e12)
    overhead = overheads.get("rope", 0.0)
    return [
        OpSpec(
            name="rope_q",
            category="elementwise",
            tags=frozenset({"rope"}),
            N_elems=ctx.total_tokens * nh * rope_dim,
            elem_op="rope",
            b_eff=b_eff,
            elem_overhead=overhead,
        ),
        OpSpec(
            name="rope_kv",
            category="elementwise",
            tags=frozenset({"rope"}),
            N_elems=ctx.total_tokens * nh_kv * rope_dim,
            elem_op="rope",
            b_eff=b_eff,
            elem_overhead=overhead,
        ),
    ]


def build_qk_proj(ctx, h, nh, nh_kv, hd, hw) -> list[OpSpec]:
    """Q and K fused projection (no V). Used by linear attention."""
    dim_qk = nh * hd
    return [OpSpec(
        name="qk_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=h, N=2 * dim_qk,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_v_proj(ctx, h, nh_v, hd, hw) -> list[OpSpec]:
    dim_v = nh_v * hd
    return [OpSpec(
        name="v_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=h, N=dim_v,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_linear_scan(ctx, h, hw) -> list[OpSpec]:
    """Linear scan projection (matmul approximation)."""
    return [OpSpec(
        name="linear_scan", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=h, N=h,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_linear_out_proj(ctx, h, nh_v, hd, hw) -> list[OpSpec]:
    """Output projection for linear attention (from V dim back to hidden)."""
    dim_v = nh_v * hd
    return [OpSpec(
        name="out_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=dim_v, N=h,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_gate_up_proj(ctx, h, inter, hw) -> list[OpSpec]:
    return [OpSpec(
        name="gate_up_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=h, N=2 * inter,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_down_proj(ctx, h, inter, hw) -> list[OpSpec]:
    return [OpSpec(
        name="down_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=inter, N=h,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_swiglu(ctx, inter, hw) -> list[OpSpec]:
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    return [OpSpec(
        name="swiglu",
        category="elementwise",
        tags=frozenset({"activation"}),
        N_elems=ctx.total_tokens * inter,
        elem_op="swiglu",
        b_eff=b_effs.get("swiglu", 1e12),
        elem_overhead=overheads.get("swiglu", 0.0),
    )]


def build_fused_residual_norm(ctx, h, hw) -> list[OpSpec]:
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    if "fused_residual_norm" not in b_effs:
        raise RuntimeError(
            "fused_residual_norm missing from elem_b_effs. "
            "Re-run benchmark and fit to generate this data."
        )
    return [OpSpec(
        name="fused_residual_norm",
        category="elementwise",
        tags=frozenset({"norm", "residual"}),
        N_elems=ctx.total_tokens * h,
        elem_op="fused_residual_norm",
        b_eff=b_effs["fused_residual_norm"],
        elem_overhead=overheads.get("fused_residual_norm", 0.0),
    )]


def build_rmsnorm(ctx, h, hw) -> list[OpSpec]:
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    return [OpSpec(
        name="rmsnorm",
        category="elementwise",
        tags=frozenset({"norm"}),
        N_elems=ctx.total_tokens * h,
        elem_op="rmsnorm",
        b_eff=b_effs.get("rmsnorm", b_effs.get("fused_residual_norm", 1e12)),
        elem_overhead=overheads.get("rmsnorm", 0.0),
    )]


def build_residual_add(ctx, h, hw) -> list[OpSpec]:
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    return [OpSpec(
        name="residual_add",
        category="elementwise",
        tags=frozenset({"residual"}),
        N_elems=ctx.total_tokens * h,
        elem_op="residual_add",
        b_eff=b_effs.get("residual_add", b_effs.get("fused_residual_norm", 1e12)),
        elem_overhead=overheads.get("residual_add", 0.0),
    )]


def build_lm_head_matmul(ctx, h, vocab_size, hw) -> list[OpSpec]:
    return [OpSpec(
        name="lm_head", category="matmul",
        tags=frozenset({"projection"}),
        M=ctx.total_tokens, K=h, N=vocab_size,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_router(ctx, h, num_experts, hw) -> list[OpSpec]:
    return [OpSpec(
        name="router", category="matmul",
        tags=frozenset({"expert_router"}),
        M=ctx.total_tokens, K=h, N=num_experts,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_expert_gate_up(ctx, expert_M, h, moe_inter, hw) -> list[OpSpec]:
    return [OpSpec(
        name="expert_gate_up", category="matmul",
        tags=frozenset({"expert"}),
        M=expert_M, K=h, N=2 * moe_inter,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_expert_down(ctx, expert_M, h, moe_inter, hw) -> list[OpSpec]:
    return [OpSpec(
        name="expert_down", category="matmul",
        tags=frozenset({"expert"}),
        M=expert_M, K=moe_inter, N=h,
        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
    )]


def build_all_to_all(ctx, bytes_per_token, hw) -> list[OpSpec]:
    comm_bytes = int(ctx.total_tokens * bytes_per_token)
    return [OpSpec(
        name="all_to_all", category="comm",
        tags=frozenset({"expert_comm"}),
        comm_bytes=comm_bytes,
        comm_type="all_to_all",
        comm_lut_bytes=ctx.comm_lut_bytes,
        comm_lut_time_s=ctx.comm_lut_time_s,
    )]
