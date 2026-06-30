"""Roofline-based step-time prediction for mixed prefill/decode batches."""

from sim.roofline import (
    matmul_time,
    attn_projections,
    ffn_projections,
    attention_fused,
    norm_ops,
    swiglu_op,
    rope_op,
    residual_add_ops,
    DTYPE_BYTES,
)

# Threshold for selecting prefill vs decode roofline params.
# Originally 256 (matching fit/matmul.py M_SPLIT), but using 256 causes
# most simulation steps to use decode params (B_peak=786 GB/s instead of
# 4122 GB/s), making batched matmuls 5.2x slower and inflating TTFT from
# ~50ms to 5+ seconds due to queue buildup.
# 32 is chosen because for M >= 32 on RTX 3090, the batched matmul
# [M, 4096] x [4096, 4096] has arithmetic intensity >= 31.8 FLOP/byte,
# close to the GPU ridge point (38 FLOP/byte), so prefill params apply.
M_SPLIT = 32


def _select_roofline_params(M_total, hw):
    """Select prefill or decode roofline params based on total batch tokens."""
    if M_total >= M_SPLIT:
        return {
            "F": hw["F_peak_prefill"],
            "B": hw["B_peak_prefill"],
            "p": hw["p_prefill"],
        }
    else:
        return {
            "F": hw["F_peak_decode"],
            "B": hw["B_peak_decode"],
            "p": hw["p_decode"],
        }


def predict_step(scheduled_requests, model_spec, hw_params):
    """Predict GPU execution time for one scheduler step.

    Args:
        scheduled_requests: list of (request, num_new_tokens).
            Each request has current num_computed_tokens (post _update_after_schedule).
        model_spec: dict from model_specs.yaml (hidden_dim, num_heads, etc.)
        hw_params: dict from fitted_params.json

    Returns:
        step_time_s: float, seconds for this step's GPU forward pass.

    Roofline param selection:
      - Projections / LM head are batched matmuls over all tokens together.
        Their effective bandwidth depends on the total M (batch size), so we
        select F,B,p from total_new_tokens (M >= 256 → prefill params, else decode).
      - Attention is per-request (FlashAttention fused kernel).  For prefill
        requests (large s_q) the fused attention is compute-bound; for decode
        (s_q = 1) it is memory-bound.  We therefore use prefill params for
        requests still in the prefill phase and decode params for decode steps.
      - Elementwise ops use b_effs / overheads which are independent of F,B,p.
    """
    if not scheduled_requests:
        return {"total": 0.0, "attn_proj": 0.0, "ffn_proj": 0.0,
                "attn_prefill": 0.0, "attn_decode": 0.0,
                "rmsnorm": 0.0, "swiglu": 0.0, "rope": 0.0,
                "residual_add": 0.0, "lm_head": 0.0}

    h = model_spec["hidden_dim"]
    inter = model_spec.get("intermediate_dim", h * 4)
    nh = model_spec["num_heads"]
    nh_kv = model_spec.get("num_kv_heads", nh)
    hd = model_spec["head_dim"]
    vs = model_spec["vocab_size"]
    norm_type = model_spec.get("norm_type", "rmsnorm")
    nl = model_spec["num_layers"]
    na = model_spec.get("attn_layers", nl)
    nd = nl - na  # DeltaNet layers (no attention)

    b_effs = hw_params["elem_b_effs"]
    overheads = hw_params["elem_overheads"]

    # Prefill / decode params for per-request attention
    F_d = hw_params["F_peak_decode"]
    B_d = hw_params["B_peak_decode"]
    p_d = hw_params["p_decode"]
    F_p = hw_params["F_peak_prefill"]
    B_p = hw_params["B_peak_prefill"]
    p_p = hw_params["p_prefill"]

    # Total new tokens this step — used for batched projections / LM head
    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    params = _select_roofline_params(total_new_tokens, hw_params)
    F, B, p = params["F"], params["B"], params["p"]

    # ── Projections (batched over all tokens → single F,B,p per step) ──
    attn_proj_time = nl * attn_projections(total_new_tokens, h, F, B, p, nh, nh_kv, hd)
    ffn_proj_time = nl * ffn_projections(total_new_tokens, h, inter, F, B, p)

    # ── Attention (per-request, per-attention-layer) ──
    # Prefill requests (is_prefill_chunk=True) have s_q ≫ 1 and are more
    # compute-heavy; decode requests (s_q=1) are memory-bound.  Use the
    # appropriate roofline params for each.
    attn_prefill_time = 0.0
    attn_decode_time = 0.0
    for req, num_new in scheduled_requests:
        # num_computed_tokens was already incremented by _update_after_schedule,
        # so it already reflects the KV cache length after this step.
        kv_len_after = req.num_computed_tokens
        if kv_len_after > 0:
            if req.is_prefill_chunk:
                attn_prefill_time += na * attention_fused(1, nh, num_new, kv_len_after, hd, F_p, B_p, p_p, nh_kv)
            else:
                attn_decode_time += na * attention_fused(1, nh, num_new, kv_len_after, hd, F_d, B_d, p_d, nh_kv)

    # ── Elementwise ops (per-layer, multiplied by num_layers) ──
    # b_effs / overheads are independent of the prefill/decode split.
    rmsnorm_time = nl * norm_ops(1, total_new_tokens, h, norm_type, b_effs, overheads)
    swiglu_time = nl * swiglu_op(1, total_new_tokens, inter, b_effs, overheads)
    rope_time = nl * rope_op(1, total_new_tokens, nh, nh_kv, hd, b_effs, overheads)
    residual_add_time = nl * residual_add_ops(1, total_new_tokens, h, b_effs, overheads)

    # ── Output projection (single lm_head, batched) ──
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p)

    total = attn_proj_time + ffn_proj_time + attn_prefill_time + attn_decode_time + rmsnorm_time + swiglu_time + rope_time + residual_add_time + lm_head_time
    return {
        "total": total,
        "attn_proj": attn_proj_time,
        "ffn_proj": ffn_proj_time,
        "attn_prefill": attn_prefill_time,
        "attn_decode": attn_decode_time,
        "rmsnorm": rmsnorm_time,
        "swiglu": swiglu_time,
        "rope": rope_time,
        "residual_add": residual_add_time,
        "lm_head": lm_head_time,
    }


def predict_step_pp(scheduled_requests, model_spec, hw_params,
                    pp_size, intra_node_bw_gb_s):
    """Predict step time with pipeline parallelism.

    Splits model layers evenly across *pp_size* pipeline stages.
    Each stage computes ``num_layers / pp_size`` layers and sends
    hidden states to the next stage (``pp_size - 1`` transfers).

    Uses the optimistic (pipelined) model: all stages execute in parallel,
    so compute time divides by pp_size while inter-stage communication
    adds fixed overhead.

    Args:
        pp_size: number of pipeline stages (GPUs in the PP dimension).
        intra_node_bw_gb_s: intra-node bandwidth for inter-stage P2P (GB/s).

    Returns:
        step_time_s with PP overhead.
    """
    if pp_size <= 1:
        return predict_step(scheduled_requests, model_spec, hw_params)

    if not scheduled_requests:
        return {"total": 0.0, "attn_proj": 0.0, "ffn_proj": 0.0,
                "attn_prefill": 0.0, "attn_decode": 0.0,
                "rmsnorm": 0.0, "swiglu": 0.0, "rope": 0.0,
                "residual_add": 0.0, "lm_head": 0.0,
                "inter_stage_comm": 0.0}

    h = model_spec["hidden_dim"]
    inter = model_spec.get("intermediate_dim", h * 4)
    nh = model_spec["num_heads"]
    nh_kv = model_spec.get("num_kv_heads", nh)
    hd = model_spec["head_dim"]
    vs = model_spec["vocab_size"]
    norm_type = model_spec.get("norm_type", "rmsnorm")
    nl = model_spec["num_layers"]
    na = model_spec.get("attn_layers", nl)

    b_effs = hw_params["elem_b_effs"]
    overheads = hw_params["elem_overheads"]

    total_new_tokens = sum(nt for _, nt in scheduled_requests)

    # Per-stage layer counts (last stage gets remainder if uneven)
    layers_per_stage = nl // pp_size
    attn_per_stage = na // pp_size

    # ── Roofline params ──
    # Projections / LM head use total M to pick prefill vs decode params
    params = _select_roofline_params(total_new_tokens, hw_params)
    F, B, p = params["F"], params["B"], params["p"]

    F_d, B_d, p_d = hw_params["F_peak_decode"], hw_params["B_peak_decode"], hw_params["p_decode"]
    F_p, B_p, p_p = hw_params["F_peak_prefill"], hw_params["B_peak_prefill"], hw_params["p_prefill"]

    # ── Per-stage projections ──
    attn_proj_time = layers_per_stage * attn_projections(
        total_new_tokens, h, F, B, p, nh, nh_kv, hd)
    ffn_proj_time = layers_per_stage * ffn_projections(
        total_new_tokens, h, inter, F, B, p)

    # ── Per-stage attention (per-request, per-attention-layer) ──
    attn_prefill_time = 0.0
    attn_decode_time = 0.0
    for req, num_new in scheduled_requests:
        kv_len_after = req.num_computed_tokens
        if kv_len_after > 0:
            if req.is_prefill_chunk:
                attn_prefill_time += attn_per_stage * attention_fused(
                    1, nh, num_new, kv_len_after, hd, F_p, B_p, p_p, nh_kv)
            else:
                attn_decode_time += attn_per_stage * attention_fused(
                    1, nh, num_new, kv_len_after, hd, F_d, B_d, p_d, nh_kv)

    # ── Per-stage elementwise ops ──
    rmsnorm_time = layers_per_stage * norm_ops(1, total_new_tokens, h, norm_type, b_effs, overheads)
    swiglu_time = layers_per_stage * swiglu_op(1, total_new_tokens, inter, b_effs, overheads)
    rope_time = layers_per_stage * rope_op(1, total_new_tokens, nh, nh_kv, hd, b_effs, overheads)
    residual_add_time = layers_per_stage * residual_add_ops(1, total_new_tokens, h, b_effs, overheads)

    # ── LM head (only on last stage, but sequential pipeline model
    #      means it's still on the critical path) ──
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p)

    # ── Inter-stage communication: (pp_size - 1) transfers ──
    # Each transfer sends hidden states for all tokens: tokens × h × 2 bytes
    inter_stage_bytes = total_new_tokens * h * DTYPE_BYTES
    inter_stage_comm = (pp_size - 1) * inter_stage_bytes / (intra_node_bw_gb_s * 1e9)

    total = attn_proj_time + ffn_proj_time + attn_prefill_time + attn_decode_time + rmsnorm_time + swiglu_time + rope_time + residual_add_time + lm_head_time + inter_stage_comm
    return {
        "total": total,
        "attn_proj": attn_proj_time,
        "ffn_proj": ffn_proj_time,
        "attn_prefill": attn_prefill_time,
        "attn_decode": attn_decode_time,
        "rmsnorm": rmsnorm_time,
        "swiglu": swiglu_time,
        "rope": rope_time,
        "residual_add": residual_add_time,
        "lm_head": lm_head_time,
        "inter_stage_comm": inter_stage_comm,
    }


def predict_step_tp(scheduled_requests, model_spec, hw_params,
                    num_gpus, intra_node_bw_gb_s,
                    pp_size=1):
    """Predict step time with tensor parallelism (and optional pipeline parallelism).

    When pp_size > 1, pipeline parallelism is applied first (splitting layers
    across stages), then TP divides the per-stage compute and adds all-reduce.

    Args:
        num_gpus: number of GPUs in the TP group.
        intra_node_bw_gb_s: intra-node bandwidth for all-reduce (GB/s).
        pp_size: pipeline parallelism degree (1 = no PP).

    Returns:
        dict with keys: total, proj, attn_prefill, attn_decode, elem,
        lm_head, all_reduce (and inter_stage_comm when pp_size > 1).
    """
    if pp_size > 1:
        base = predict_step_pp(scheduled_requests, model_spec, hw_params,
                               pp_size, intra_node_bw_gb_s)
    else:
        base = predict_step(scheduled_requests, model_spec, hw_params)

    if num_gpus <= 1:
        base["all_reduce"] = 0.0
        return base

    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    h = model_spec["hidden_dim"]

    # All-reduce overhead: 2 * activations_size / bandwidth
    all_reduce_bytes = 2 * total_new_tokens * h * DTYPE_BYTES
    all_reduce_time = all_reduce_bytes / (intra_node_bw_gb_s * 1e9)

    # Scale compute components by 1/tp; inter_stage_comm (PP) is also divided
    result = {}
    compute_total = 0.0
    for k in ("attn_proj", "ffn_proj", "attn_prefill", "attn_decode",
              "rmsnorm", "swiglu", "rope", "residual_add",
              "lm_head", "inter_stage_comm"):
        v = base.get(k, 0.0)
        scaled = v / num_gpus
        result[k] = scaled
        compute_total += scaled
    result["all_reduce"] = all_reduce_time
    result["total"] = compute_total + all_reduce_time
    return result
