"""Qwen3-4B computation graph.

Standard GQA decoder: QKV fused proj → RoPE → FlashAttn → O proj →
fused_residual_norm → gate_up proj → SwiGLU → down proj → fused_residual_norm
"""

SPEC = {
    "name": "Qwen/Qwen3-4B",
    "hidden_dim": 2560,
    "vocab_size": 151936,
    "max_model_len": 40960,
    "norm_type": "rmsnorm",
    "tie_word_embeddings": True,
    "num_layers": 36,
    "num_attn_layers": 36,
    "num_heads": 32,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "rope_dim": 128,
    "intermediate_dim": 9728,
    "activation": "silu",
    "total_params_b": 4.0,
}


def build_graph(spec):
    from sim.graph import ModelGraph
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
