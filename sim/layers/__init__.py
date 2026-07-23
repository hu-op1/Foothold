"""Fine-grained operator builders for constructing computation graphs.

Each builder takes explicit parameters (h, nh, nh_kv, hd, inter, etc.)
and returns list[OpSpec]. Per-model files in models/ compose them.
"""

from sim.layers.common import (
    build_qkv_proj,
    build_o_proj,
    build_flash_attn,
    build_rope,
    build_qk_proj,
    build_v_proj,
    build_linear_scan,
    build_linear_out_proj,
    build_gate_up_proj,
    build_down_proj,
    build_swiglu,
    build_fused_residual_norm,
    build_rmsnorm,
    build_residual_add,
    build_lm_head_matmul,
    build_router,
    build_expert_gate_up,
    build_expert_down,
    build_all_to_all,
)
from sim.layers.head import build_lm_head
