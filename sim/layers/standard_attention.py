"""Standard attention builder: QKV projection -> RoPE -> FlashAttention -> O projection."""

from sim.layers import register_attention
from sim.layers.common import build_rope
from sim.graph import OpSpec


@register_attention("standard_attention")
def build_standard_attention(ctx, lt, hw) -> list[OpSpec]:
    nh = lt["num_q_heads"]
    nh_kv = lt.get("num_kv_heads", nh)
    hd = lt["head_dim"]
    h = ctx.hidden_dim if hasattr(ctx, "hidden_dim") else lt["hidden_dim"]
    dim_q = nh * hd
    dim_kv = nh_kv * hd

    M = ctx.total_tokens
    F = ctx.matmul_F
    B = ctx.matmul_B
    p = ctx.matmul_p
    ov = ctx.matmul_overhead

    ops = [
        # QKV fused projection (tag: projection -> TP cuts N)
        OpSpec(
            name="qkv_proj", category="matmul",
            tags=frozenset({"projection"}),
            M=M, K=h, N=dim_q + 2 * dim_kv,
            F_peak=F, B_peak=B, p=p, overhead=ov,
        ),
    ]

    # RoPE
    ops += build_rope(ctx, lt, hw)

    # FlashAttention (prefill + decode batched separately in ctx)
    ops.append(OpSpec(
        name="flash_attn", category="attention",
        tags=frozenset({"attention"}),
        prefill_flops=ctx.prefill_flops, prefill_bytes=ctx.prefill_bytes,
        decode_flops=ctx.decode_flops, decode_bytes=ctx.decode_bytes,
        fa_d=ctx.fa_decode_params, fa_p=ctx.fa_prefill_params,
    ))

    # O projection
    ops.append(OpSpec(
        name="o_proj", category="matmul",
        tags=frozenset({"projection"}),
        M=M, K=dim_q, N=h,
        F_peak=F, B_peak=B, p=p, overhead=ov,
    ))

    return ops
