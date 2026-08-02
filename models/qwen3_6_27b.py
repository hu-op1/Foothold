"""Qwen3.6-27B computation graph.

Hybrid architecture: 48 linear attention layers + 16 full attention layers
(every 4th layer). Full attention layers include attn_output_gate.

Same computation graph as Qwen3.5-4B (models/qwen3_5_4b.py) — see its
docstring for the full op breakdown; dims verified against
Qwen/Qwen3.6-27B config.json.
"""

SPEC = {
    "name": "Qwen/Qwen3.6-27B",
    "hidden_dim": 5120,
    "vocab_size": 248320,
    "max_model_len": 262144,
    "norm_type": "rmsnorm",
    "tie_word_embeddings": False,
    "num_layers": 64,
    "num_attn_layers": 16,
    "num_heads": 24,
    "num_q_heads": 24,
    "num_kv_heads": 4,
    "head_dim": 256,
    "rope_dim": 64,  # partial_rotary_factor=0.25 * 256
    "intermediate_dim": 17408,
    "total_params_b": 27.0,
    "full_attention_interval": 4,
    # Linear attention sub-config
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def build_graph(spec):
    from sim.graph import ModelGraph, OpSpec
    from sim.layers import (
        build_qkv_proj, build_o_proj, build_flash_attn, build_rope,
        build_qk_proj, build_v_proj, build_linear_scan, build_linear_out_proj,
        build_gate_up_proj, build_down_proj, build_swiglu,
        build_fused_residual_norm, build_lm_head,
    )

    h = spec["hidden_dim"]
    inter = spec["intermediate_dim"]

    # Full attention params
    nh = spec["num_q_heads"]
    nh_kv = spec["num_kv_heads"]
    hd = spec["head_dim"]
    rd = spec["rope_dim"]

    # Linear attention params
    lnkh = spec["linear_num_key_heads"]
    lkhd = spec["linear_key_head_dim"]
    lnvh = spec["linear_num_value_heads"]
    lvhd = spec["linear_value_head_dim"]

    interval = spec.get("full_attention_interval", 4)
    layer_types = [
        "full" if (i + 1) % interval == 0 else "linear"
        for i in range(spec["num_layers"])
    ]

    _builders = []
    for lt in layer_types:
        if lt == "full":
            def _make_full(_h=h, _nh=nh, _nh_kv=nh_kv, _hd=hd, _rd=rd, _inter=inter):
                def _builder(ctx, hw):
                    ops = []
                    ops += build_qkv_proj(ctx, _h, _nh, _nh_kv, _hd, hw, q_gate=True)
                    b_effs = ctx.b_effs
                    overheads = ctx.overheads
                    b_eff = b_effs.get("rmsnorm", b_effs.get("fused_residual_norm", 1e12))
                    overhead = overheads.get("rmsnorm", 0.0)
                    ops.append(OpSpec(
                        name="q_norm", category="elementwise",
                        tags=frozenset({"norm"}),
                        N_elems=ctx.total_tokens * _nh * _hd,
                        elem_op="rmsnorm",
                        b_eff=b_eff,
                        elem_overhead=overhead,
                    ))
                    ops.append(OpSpec(
                        name="k_norm", category="elementwise",
                        tags=frozenset({"norm"}),
                        N_elems=ctx.total_tokens * _nh_kv * _hd,
                        elem_op="rmsnorm",
                        b_eff=b_eff,
                        elem_overhead=overhead,
                    ))
                    ops += build_rope(ctx, _nh, _nh_kv, _hd, _rd, hw)
                    ops += build_flash_attn(ctx, hw)
                    ops += build_o_proj(ctx, _h, _nh, _hd, hw)
                    ops.append(OpSpec(
                        name="attn_gate_mul",
                        category="elementwise",
                        tags=frozenset({"activation"}),
                        N_elems=ctx.total_tokens * _nh * _hd,
                        elem_op="swiglu",
                        b_eff=b_effs.get("swiglu", 1e12),
                        elem_overhead=overheads.get("swiglu", 0.0),
                    ))
                    ops += build_fused_residual_norm(ctx, _h, hw)
                    ops += build_gate_up_proj(ctx, _h, _inter, hw)
                    ops += build_swiglu(ctx, _inter, hw)
                    ops += build_down_proj(ctx, _h, _inter, hw)
                    ops += build_fused_residual_norm(ctx, _h, hw)
                    return ops
                return _builder
            _builders.append(_make_full())
        else:
            def _make_linear(_h=h, _lnkh=lnkh, _lkhd=lkhd, _lnvh=lnvh, _lvhd=lvhd, _inter=inter):
                def _builder(ctx, hw):
                    conv_dim = 2 * _lnkh * _lkhd + _lnvh * _lvhd
                    ops = []
                    ops += build_qk_proj(ctx, _h, _lnkh, _lnkh, _lkhd, hw)
                    ops += build_v_proj(ctx, _h, _lnvh, _lvhd, hw)
                    b_effs = ctx.b_effs
                    overheads = ctx.overheads
                    ops.append(OpSpec(
                        name="conv1d", category="elementwise",
                        tags=frozenset({"activation"}),
                        N_elems=ctx.total_tokens * conv_dim,
                        elem_op="conv1d",
                        b_eff=b_effs.get("conv1d", b_effs.get("swiglu", 1e12)),
                        elem_overhead=overheads.get("conv1d", 0.0),
                    ))
                    ops.append(OpSpec(
                        name="in_proj_z", category="matmul",
                        tags=frozenset({"projection", "col_parallel"}),
                        M=ctx.total_tokens, K=_h, N=_lnvh * _lvhd,
                        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
                        p=ctx.matmul_p,
                    ))
                    ops.append(OpSpec(
                        name="in_proj_b", category="matmul",
                        tags=frozenset({"projection", "col_parallel"}),
                        M=ctx.total_tokens, K=_h, N=_lnvh,
                        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
                        p=ctx.matmul_p,
                    ))
                    ops.append(OpSpec(
                        name="in_proj_a", category="matmul",
                        tags=frozenset({"projection", "col_parallel"}),
                        M=ctx.total_tokens, K=_h, N=_lnvh,
                        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
                        p=ctx.matmul_p,
                    ))
                    ops += build_linear_scan(ctx, _lnkh, _lkhd, _lnvh, _lvhd, hw)
                    ops.append(OpSpec(
                        name="linear_gate_norm",
                        category="elementwise",
                        tags=frozenset({"activation"}),
                        N_elems=ctx.total_tokens * _lnvh * _lvhd,
                        elem_op="swiglu",
                        b_eff=b_effs.get("swiglu", 1e12),
                        elem_overhead=overheads.get("swiglu", 0.0),
                    ))
                    ops += build_linear_out_proj(ctx, _h, _lnvh, _lvhd, hw)
                    ops += build_fused_residual_norm(ctx, _h, hw)
                    ops += build_gate_up_proj(ctx, _h, _inter, hw)
                    ops += build_swiglu(ctx, _inter, hw)
                    ops += build_down_proj(ctx, _h, _inter, hw)
                    ops += build_fused_residual_norm(ctx, _h, hw)
                    return ops
                return _builder
            _builders.append(_make_linear())

    return ModelGraph(
        layer_specs=[(b, 1) for b in _builders],
        head_builder=lambda ctx, hw: build_lm_head(ctx, spec, hw),
    )
