"""Shared operator builders: norm, rope, swiglu — used by all layer types."""

from sim.graph import OpSpec


def build_fused_residual_norm(ctx, spec, hw) -> list[OpSpec]:
    """fused residual_add + rmsnorm (tag: residual_norm).

    2 ops per layer: post-attn + post-FFN.
    Uses hw["elem_b_effs"]["fused_residual_norm"] — must exist or errors.
    """
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
        N_elems=ctx.total_tokens * spec["hidden_dim"],
        elem_op="fused_residual_norm",
        b_eff=b_effs["fused_residual_norm"],
        elem_overhead=overheads.get("fused_residual_norm", 0.0),
    )]


def build_rmsnorm(ctx, spec, hw) -> list[OpSpec]:
    """Standalone RMSNorm (tag: norm). Used for final_norm before lm_head."""
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    return [OpSpec(
        name="rmsnorm",
        category="elementwise",
        tags=frozenset({"norm"}),
        N_elems=ctx.total_tokens * spec["hidden_dim"],
        elem_op="rmsnorm",
        b_eff=b_effs.get("rmsnorm", b_effs.get("fused_residual_norm", 1e12)),
        elem_overhead=overheads.get("rmsnorm", 0.0),
    )]


def build_rope(ctx, lt, hw) -> list[OpSpec]:
    """RoPE for Q and K (tag: rope). 2 elementwise ops per attention layer."""
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    head_dim = lt["head_dim"]
    rope_dim = lt.get("rope_dim", head_dim)
    nh = lt.get("num_q_heads", lt.get("num_qk_heads", 1))
    nh_kv = lt.get("num_kv_heads", nh)
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


def build_swiglu(ctx, inter_dim, hw) -> list[OpSpec]:
    """SwiGLU activation (tag: activation). 1 op per FFN block."""
    b_effs = ctx.b_effs
    overheads = ctx.overheads
    return [OpSpec(
        name="swiglu",
        category="elementwise",
        tags=frozenset({"activation"}),
        N_elems=ctx.total_tokens * inter_dim,
        elem_op="swiglu",
        b_eff=b_effs.get("swiglu", 1e12),
        elem_overhead=overheads.get("swiglu", 0.0),
    )]
