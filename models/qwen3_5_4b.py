"""Qwen3.5-4B computation graph.

Hybrid architecture: 24 linear attention layers + 8 full attention layers
(every 4th layer). Full attention layers include attn_output_gate.
"""

SPEC = {
    "name": "Qwen/Qwen3.5-4B",
    "hidden_dim": 2560,
    "vocab_size": 248320,
    "max_model_len": 262144,
    "norm_type": "rmsnorm",
    "tie_word_embeddings": True,
    "num_layers": 32,
    "num_heads": 16,
    "num_q_heads": 16,
    "num_kv_heads": 4,
    "head_dim": 256,
    "rope_dim": 64,  # partial_rotary_factor=0.25 * 256
    "intermediate_dim": 9216,
    "total_params_b": 4.0,
    # Linear attention sub-config
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    # Layer-type pattern: linear × 3, full × 1, repeating 8 times
    "layer_types": [
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
        "linear", "linear", "linear", "full",
    ],
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

    layer_types = spec["layer_types"]

    _builders = []
    for lt in layer_types:
        if lt == "full":
            def _make_full(_h=h, _nh=nh, _nh_kv=nh_kv, _hd=hd, _rd=rd, _inter=inter):
                def _builder(ctx, hw):
                    ops = []
                    ops += build_qkv_proj(ctx, _h, _nh, _nh_kv, _hd, hw)
                    ops += build_rope(ctx, _nh, _nh_kv, _hd, _rd, hw)
                    ops += build_flash_attn(ctx, hw)
                    ops += build_o_proj(ctx, _h, _nh, _hd, hw)
                    ops.append(OpSpec(
                        name="attn_gate_proj", category="matmul",
                        tags=frozenset({"projection"}),
                        M=ctx.total_tokens, K=_h, N=_h,
                        F_peak=ctx.matmul_F, B_peak=ctx.matmul_B,
                        p=ctx.matmul_p, overhead=ctx.matmul_overhead,
                    ))
                    b_effs = ctx.b_effs
                    overheads = ctx.overheads
                    ops.append(OpSpec(
                        name="attn_gate_mul",
                        category="elementwise",
                        tags=frozenset({"activation"}),
                        N_elems=ctx.total_tokens * _h,
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
                    ops = []
                    ops += build_qk_proj(ctx, _h, _lnkh, _lnkh, _lkhd, hw)
                    ops += build_v_proj(ctx, _h, _lnvh, _lvhd, hw)
                    ops += build_linear_scan(ctx, _h, hw)
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
