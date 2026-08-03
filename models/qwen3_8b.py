"""Qwen3-8B computation graph.

Standard GQA decoder: QKV fused proj → QK Norm (per-head RMSNorm) → RoPE →
FlashAttn → O proj → fused_residual_norm → gate_up proj → SwiGLU →
down proj → fused_residual_norm

Same graph as Qwen3-4B (models/qwen3_4b.py); dims per the
Qwen/Qwen3-8B config.json (hidden 4096, 32 q / 8 kv heads, head_dim 128,
intermediate 12288, 36 layers).
"""

SPEC = {
    "name": "Qwen/Qwen3-8B",
    "hidden_dim": 4096,
    "vocab_size": 151936,
    "max_model_len": 40960,
    "norm_type": "rmsnorm",
    "tie_word_embeddings": False,
    "num_layers": 36,
    "num_attn_layers": 36,
    "num_heads": 32,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "rope_dim": 128,
    "intermediate_dim": 12288,
    "activation": "silu",
    "total_params_b": 8.0,
}


def build_graph(spec):
    from sim.graph import ModelGraph, OpSpec
    from sim.layers import (
        build_qkv_proj, build_o_proj, build_flash_attn, build_rope,
        build_gate_up_proj, build_down_proj, build_swiglu,
        build_fused_residual_norm, build_lm_head,
    )

    h = spec["hidden_dim"]
    nh = spec["num_q_heads"]
    nh_kv = spec["num_kv_heads"]
    hd = spec["head_dim"]
    rd = spec.get("rope_dim", hd)
    inter = spec["intermediate_dim"]

    def _layer(ctx, hw):
        ops = []
        ops += build_qkv_proj(ctx, h, nh, nh_kv, hd, hw)
        b_effs = ctx.b_effs
        overheads = ctx.overheads
        b_eff = b_effs.get("rmsnorm", b_effs.get("fused_residual_norm", 1e12))
        overhead = overheads.get("rmsnorm", 0.0)
        ops.append(OpSpec(
            name="q_norm",
            category="elementwise",
            tags=frozenset({"norm"}),
            N_elems=ctx.total_tokens * nh * hd,
            elem_op="rmsnorm",
            b_eff=b_eff,
            elem_overhead=overhead,
        ))
        ops.append(OpSpec(
            name="k_norm",
            category="elementwise",
            tags=frozenset({"norm"}),
            N_elems=ctx.total_tokens * nh_kv * hd,
            elem_op="rmsnorm",
            b_eff=b_eff,
            elem_overhead=overhead,
        ))
        ops += build_rope(ctx, nh, nh_kv, hd, rd, hw)
        ops += build_flash_attn(ctx, hw)
        ops += build_o_proj(ctx, h, nh, hd, hw)
        ops += build_fused_residual_norm(ctx, h, hw)
        ops += build_gate_up_proj(ctx, h, inter, hw)
        ops += build_swiglu(ctx, inter, hw)
        ops += build_down_proj(ctx, h, inter, hw)
        ops += build_fused_residual_norm(ctx, h, hw)
        return ops

    return ModelGraph(
        layer_specs=[(_layer, spec["num_layers"])],
        head_builder=lambda ctx, hw: build_lm_head(ctx, spec, hw),
    )
