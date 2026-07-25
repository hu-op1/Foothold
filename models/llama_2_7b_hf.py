"""Llama-2-7B computation graph.

Standard MHA decoder: QKV fused proj → RoPE → FlashAttn → O proj →
fused_residual_norm → gate_up proj → SwiGLU → down proj → fused_residual_norm
"""

SPEC = {
    "name": "meta-llama/Llama-2-7b-hf",
    "hidden_dim": 4096,
    "vocab_size": 32000,
    "max_model_len": 4096,
    "norm_type": "rmsnorm",
    "tie_word_embeddings": False,
    "num_layers": 32,
    "num_attn_layers": 32,
    "num_heads": 32,
    "num_q_heads": 32,
    "num_kv_heads": 32,
    "head_dim": 128,
    "rope_dim": 128,
    "intermediate_dim": 11008,
    "activation": "swiglu",
    "total_params_b": 6.74,
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
