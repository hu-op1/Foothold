"""Roofline-based step-time prediction for mixed prefill/decode batches."""

from math import log, exp, log2

from sim.roofline import (
    matmul_time,
    attn_projections,
    ffn_projections,
    norm_ops,
    swiglu_op,
    rope_op,
    residual_add_ops,
    roofline_time,
    DTYPE_BYTES,
)

# ── Roofline param selection ──────────────────────────────────────────────
# The smooth-roofline fit produces two (B_peak, p) pairs:
#   - decode  (M ≤ 256 in fit/matmul.py): small-batch, memory-bound regime
#   - prefill (M ≥ 256): large-batch, compute-bound regime
#
# Rather than a hard threshold we interpolate B_peak and p in log space
# between M=32 and M=256.  This avoids the "step function" artifact where
# a single token pushes M from 31→32 and the predicted step time jumps
# discontinuously.  Log-space interpolation reflects the physical reality
# that effective bandwidth transitions smoothly as batch size increases.
#
# F_peak is shared (fitted on prefill, held fixed for decode), so only
# B and p vary between regimes.

_M_LO = 32    # below this: pure decode params
_M_HI = 256   # above this: pure prefill params
_LOG_RANGE = log2(_M_HI) - log2(_M_LO)


def _select_roofline_params(M_total, hw):
    """Smoothly interpolate between decode and prefill roofline params.

    M_total ≤ 32  → pure decode (memory-bound, low B_peak)
    M_total ≥ 256 → pure prefill (compute-bound, high B_peak)
    32 < M < 256  → log-space interpolation of B_peak and p
    """
    if M_total <= _M_LO:
        return {
            "F": hw["F_peak_decode"],
            "B": hw["B_peak_decode"],
            "p": hw["p_decode"],
        }
    if M_total >= _M_HI:
        return {
            "F": hw["F_peak_prefill"],
            "B": hw["B_peak_prefill"],
            "p": hw["p_prefill"],
        }

    w = (log2(M_total) - log2(_M_LO)) / _LOG_RANGE  # 0 → 1

    # Interpolate B in log space (physical: bandwidth ratios are multiplicative)
    log_B = log(hw["B_peak_decode"]) + w * (log(hw["B_peak_prefill"]) - log(hw["B_peak_decode"]))
    B = exp(log_B)
    p = hw["p_decode"] + w * (hw["p_prefill"] - hw["p_decode"])

    # F_peak is the same in both regimes (fit/matmul.py fixes F for decode)
    return {"F": hw["F_peak_prefill"], "B": B, "p": p}


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

    # Attention roofline params — prefer FlashAttention-specific if fitted,
    # fall back to matmul-fitted params (backward compatible with old fit data).
    F_d = hw_params.get("F_peak_fa_decode", hw_params["F_peak_decode"])
    B_d = hw_params.get("B_peak_fa_decode", hw_params["B_peak_decode"])
    p_d = hw_params.get("p_fa_decode", hw_params["p_decode"])
    F_p = hw_params.get("F_peak_fa_prefill", hw_params["F_peak_prefill"])
    B_p = hw_params.get("B_peak_fa_prefill", hw_params["B_peak_prefill"])
    p_p = hw_params.get("p_fa_prefill", hw_params["p_prefill"])

    # Total new tokens this step — used for batched projections / LM head
    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    params = _select_roofline_params(total_new_tokens, hw_params)
    F, B, p = params["F"], params["B"], params["p"]

    # ── Projections (batched over all tokens → single F,B,p per step) ──
    attn_proj_time = nl * attn_projections(total_new_tokens, h, F, B, p, nh, nh_kv, hd)
    ffn_proj_time = nl * ffn_projections(total_new_tokens, h, inter, F, B, p)

    # ── Attention: group by type, then apply roofline ONCE per type ──
    # The L^p norm roofline_time(f,b) does NOT distribute over addition:
    #   Σ roofline_time(f_i, b_i)  ≥  roofline_time(Σ f_i, Σ b_i)
    # (triangle inequality; equality only when f_i/b_i ratios are identical).
    # Real FlashAttention batches all same-type requests into one kernel call,
    # so we accumulate FLOPs + bytes for prefill and decode separately,
    # then apply roofline_time once per type.  This avoids the systematic
    # over-estimate of the old per-request loop.
    prefill_flops = 0.0
    prefill_bytes = 0.0
    decode_flops = 0.0
    decode_bytes = 0.0
    n_prefill = 0
    n_decode = 0

    for req, num_new in scheduled_requests:
        kv_len_after = req.num_computed_tokens
        if kv_len_after <= 0:
            continue
        # FLOPs = 4 * nh * s_q * s_kv * hd  (standard attention FLOP count)
        # Bytes = Q + K + V reads + O write (no S×S HBM round-trip in FA)
        f = 4 * nh * num_new * kv_len_after * hd
        b = hd * DTYPE_BYTES * (2 * nh * num_new + 2 * nh_kv * kv_len_after)
        if req.is_prefill_chunk:
            n_prefill += 1
            prefill_flops += f
            prefill_bytes += b
        else:
            n_decode += 1
            decode_flops += f
            decode_bytes += b

    attn_prefill_time = na * roofline_time(prefill_flops, prefill_bytes, F_p, B_p, p_p) if n_prefill > 0 else 0.0
    attn_decode_time = na * roofline_time(decode_flops, decode_bytes, F_d, B_d, p_d) if n_decode > 0 else 0.0

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

    F_d = hw_params.get("F_peak_fa_decode", hw_params["F_peak_decode"])
    B_d = hw_params.get("B_peak_fa_decode", hw_params["B_peak_decode"])
    p_d = hw_params.get("p_fa_decode", hw_params["p_decode"])
    F_p = hw_params.get("F_peak_fa_prefill", hw_params["F_peak_prefill"])
    B_p = hw_params.get("B_peak_fa_prefill", hw_params["B_peak_prefill"])
    p_p = hw_params.get("p_fa_prefill", hw_params["p_prefill"])

    # ── Per-stage projections ──
    attn_proj_time = layers_per_stage * attn_projections(
        total_new_tokens, h, F, B, p, nh, nh_kv, hd)
    ffn_proj_time = layers_per_stage * ffn_projections(
        total_new_tokens, h, inter, F, B, p)

    # ── Per-stage attention: group by type → roofline once per type ──
    prefill_flops = 0.0
    prefill_bytes = 0.0
    decode_flops = 0.0
    decode_bytes = 0.0
    n_prefill = 0
    n_decode = 0

    for req, num_new in scheduled_requests:
        kv_len_after = req.num_computed_tokens
        if kv_len_after <= 0:
            continue
        f = 4 * nh * num_new * kv_len_after * hd
        b = hd * DTYPE_BYTES * (2 * nh * num_new + 2 * nh_kv * kv_len_after)
        if req.is_prefill_chunk:
            n_prefill += 1
            prefill_flops += f
            prefill_bytes += b
        else:
            n_decode += 1
            decode_flops += f
            decode_bytes += b

    attn_prefill_time = attn_per_stage * roofline_time(prefill_flops, prefill_bytes, F_p, B_p, p_p) if n_prefill > 0 else 0.0
    attn_decode_time = attn_per_stage * roofline_time(decode_flops, decode_bytes, F_d, B_d, p_d) if n_decode > 0 else 0.0

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
