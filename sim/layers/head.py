"""LM head: final norm + vocab projection.

Kept as a convenience composite for per-model files that want both steps.
Individual builders (build_rmsnorm, build_lm_head_matmul) are in common.py.
"""
from sim.layers.common import build_rmsnorm, build_lm_head_matmul


def build_lm_head(ctx, spec, hw) -> list:
    h = spec["hidden_dim"]
    vs = spec["vocab_size"]
    ops = build_rmsnorm(ctx, h, hw)
    ops += build_lm_head_matmul(ctx, h, vs, hw)
    return ops
