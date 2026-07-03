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
    fused_residual_norm_ops,
    roofline_time,
    dtype_bytes,
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


def _select_dtype_params(hw_params, dtype):
    """Return hw_params with dtype-specific values under unsuffixed keys.

    When the fit was run with multiple dtypes, params are stored as
    ``F_peak_decode_float16``, ``elem_b_effs_bfloat16``, etc.  This
    helper strips the ``_{dtype}`` suffix so downstream code can use
    the same key names regardless of precision.

    Falls back to the unsuffixed (unified) keys when no dtype-specific
    params exist (backward compat with single-dtype fit data).
    """
    suffix = f"_{dtype}"
    result = dict(hw_params)  # shallow copy
    for key, val in hw_params.items():
        if key.endswith(suffix):
            base = key[:-len(suffix)]
            # Only override if the key actually had a dtype suffix —
            # don't clobber native unsuffixed keys like "type".
            if base:
                result[base] = val
    return result


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


def _select_fa_params(n_requests, regime, hw_params, nh_model=None):
    """Select FlashAttention roofline params based on concurrent request count.

    When per-batch FA params are available (fa_batch_sizes), interpolates
    B_peak and p in log-space across batch sizes — mirroring the M-based
    interpolation in _select_roofline_params.  Falls back to unified FA
    params, then to matmul-fitted params.

    The interpolation is done on **total query heads** (n_requests × nh_model),
    normalised by the benchmark's nh (fa_bench_nh).  This makes the sweep
    valid for models with different num_heads than the benchmark config.

    Args:
        n_requests: number of concurrent requests of this type (prefill or decode).
        regime: "decode" or "prefill".
        hw_params: fitted hardware params dict.
        nh_model: model's num_heads (for query-head normalisation).  If None,
            uses n_requests directly (backward compat).

    Returns:
        dict with keys F, B, p.
    """
    batch_sizes = hw_params.get("fa_batch_sizes")
    B_key = f"fa_{regime}_B"
    p_key = f"fa_{regime}_p"
    F_key = f"F_peak_fa_{regime}"

    # Normalise to benchmark batch: effective_batch = total_query_heads / bench_nh
    if nh_model is not None:
        bench_nh = hw_params.get("fa_bench_nh", nh_model)
        effective = n_requests * nh_model / bench_nh if bench_nh > 0 else n_requests
    else:
        effective = n_requests

    if batch_sizes and len(batch_sizes) > 1 and B_key in hw_params:
        B_arr = hw_params[B_key]
        p_arr = hw_params[p_key]

        if effective <= batch_sizes[0]:
            B, p = B_arr[0], p_arr[0]
        elif effective >= batch_sizes[-1]:
            B, p = B_arr[-1], p_arr[-1]
        else:
            # Log-space interpolation between bracketing batch sizes
            for i in range(len(batch_sizes) - 1):
                if batch_sizes[i] <= effective <= batch_sizes[i + 1]:
                    lo, hi = batch_sizes[i], batch_sizes[i + 1]
                    w = (log2(effective) - log2(lo)) / (log2(hi) - log2(lo))
                    log_B = log(B_arr[i]) + w * (log(B_arr[i + 1]) - log(B_arr[i]))
                    B = exp(log_B)
                    p = p_arr[i] + w * (p_arr[i + 1] - p_arr[i])
                    break

        F = hw_params.get(F_key, hw_params.get(f"F_peak_{regime}", 1e13))
        return {"F": F, "B": B, "p": p}

    # Fallback: unified FA params → matmul params
    F_fb = hw_params.get(F_key)
    B_fb = hw_params.get(f"B_peak_fa_{regime}")
    p_fb = hw_params.get(f"p_fa_{regime}")
    if F_fb is not None and B_fb is not None and p_fb is not None:
        return {"F": F_fb, "B": B_fb, "p": p_fb}

    return {
        "F": hw_params[f"F_peak_{regime}"],
        "B": hw_params[f"B_peak_{regime}"],
        "p": hw_params[f"p_{regime}"],
    }


def predict_step(scheduled_requests, model_spec, hw_params, dtype="float16"):
    """Predict GPU execution time for one scheduler step.

    Args:
        scheduled_requests: list of (request, num_new_tokens).
            Each request has current num_computed_tokens (post _update_after_schedule).
        model_spec: dict from model_specs.yaml (hidden_dim, num_heads, etc.)
        hw_params: dict from fitted_params.json
        dtype: precision string (float16, bfloat16, float8_e4m3fn, …).

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
                "fused_add_norm": 0.0, "swiglu": 0.0, "rope": 0.0,
                "lm_head": 0.0}

    # Select dtype-specific roofline params (backward compat: no-op if single dtype)
    hw = _select_dtype_params(hw_params, dtype)
    dt_bytes = dtype_bytes(dtype)

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

    b_effs = hw["elem_b_effs"]
    overheads = hw["elem_overheads"]

    # Total new tokens this step — used for batched projections / LM head
    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    params = _select_roofline_params(total_new_tokens, hw)
    F, B, p = params["F"], params["B"], params["p"]

    # ── Projections (batched over all tokens → single F,B,p per step) ──
    attn_proj_time = nl * attn_projections(total_new_tokens, h, F, B, p, nh, nh_kv, hd, dt_bytes)
    ffn_proj_time = nl * ffn_projections(total_new_tokens, h, inter, F, B, p, dt_bytes)

    # ── Attention: group by type, then apply roofline ONCE per type ──
    # The L^p norm roofline_time(f,b) does NOT distribute over addition:
    #   Σ roofline_time(f_i, b_i)  ≥  roofline_time(Σ f_i, Σ b_i)
    # (triangle inequality; equality only when f_i/b_i ratios are identical).
    # Real FlashAttention batches all same-type requests into one kernel call,
    # so we accumulate FLOPs + bytes for prefill and decode separately,
    # then apply roofline_time once per type.  This avoids the systematic
    # over-estimate of the old per-request loop.

    # Count concurrent requests by type (maps to FA benchmark batch dimension)
    n_prefill = sum(1 for req, _ in scheduled_requests if req.is_prefill_chunk)
    n_decode = sum(1 for req, _ in scheduled_requests if not req.is_prefill_chunk)

    # Per-batch FA params selected from actual concurrency counts
    fa_d = _select_fa_params(n_decode, "decode", hw, nh)
    fa_p = _select_fa_params(n_prefill, "prefill", hw, nh)
    F_d, B_d, p_d = fa_d["F"], fa_d["B"], fa_d["p"]
    F_p, B_p, p_p = fa_p["F"], fa_p["B"], fa_p["p"]

    prefill_flops = 0.0
    prefill_bytes = 0.0
    decode_flops = 0.0
    decode_bytes = 0.0

    for req, num_new in scheduled_requests:
        kv_len_after = req.num_computed_tokens
        if kv_len_after <= 0:
            continue
        # FLOPs = 4 * nh * s_q * s_kv * hd  (standard attention FLOP count)
        # Bytes = Q + K + V reads + O write (no S×S HBM round-trip in FA)
        f = 4 * nh * num_new * kv_len_after * hd
        b = hd * dt_bytes * (2 * nh * num_new + 2 * nh_kv * kv_len_after)
        if req.is_prefill_chunk:
            prefill_flops += f
            prefill_bytes += b
        else:
            decode_flops += f
            decode_bytes += b

    attn_prefill_time = na * roofline_time(prefill_flops, prefill_bytes, F_p, B_p, p_p) if n_prefill > 0 else 0.0
    attn_decode_time = na * roofline_time(decode_flops, decode_bytes, F_d, B_d, p_d) if n_decode > 0 else 0.0

    # ── Elementwise ops (per-layer, multiplied by num_layers) ──
    # RMSNorm + residual_add are fused per vLLM's fused_add_rms_norm:
    # each layer has 2 fused residual+norm kernels (post-attn + post-FFN)
    # instead of 2 standalone norms + 2 standalone residual adds.
    # Falls back to separate ops when fused_residual_norm is not in fit data.
    fused_add_norm_time = nl * fused_residual_norm_ops(1, total_new_tokens, h, b_effs, overheads, dt_bytes)
    swiglu_time = nl * swiglu_op(1, total_new_tokens, inter, b_effs, overheads, dt_bytes)
    rope_time = nl * rope_op(1, total_new_tokens, nh, nh_kv, hd, b_effs, overheads, dt_bytes)

    # ── Output projection (single lm_head, batched) ──
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p, dt_bytes)

    total = attn_proj_time + ffn_proj_time + attn_prefill_time + attn_decode_time + fused_add_norm_time + swiglu_time + rope_time + lm_head_time
    return {
        "total": total,
        "attn_proj": attn_proj_time,
        "ffn_proj": ffn_proj_time,
        "attn_prefill": attn_prefill_time,
        "attn_decode": attn_decode_time,
        "fused_add_norm": fused_add_norm_time,
        "swiglu": swiglu_time,
        "rope": rope_time,
        "lm_head": lm_head_time,
    }


def predict_step_pp(scheduled_requests, model_spec, hw_params,
                    pp_size, intra_node_bw_gb_s, intra_latency_us=2.0,
                    dtype="float16"):
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
        intra_latency_us: intra-node latency per P2P transfer (µs).
        dtype: precision string.

    Returns:
        step_time_s with PP overhead.
    """
    if pp_size <= 1:
        return predict_step(scheduled_requests, model_spec, hw_params, dtype)

    if not scheduled_requests:
        return {"total": 0.0, "attn_proj": 0.0, "ffn_proj": 0.0,
                "attn_prefill": 0.0, "attn_decode": 0.0,
                "fused_add_norm": 0.0, "swiglu": 0.0, "rope": 0.0,
                "lm_head": 0.0,
                "inter_stage_comm": 0.0}

    hw = _select_dtype_params(hw_params, dtype)
    dt_bytes = dtype_bytes(dtype)

    h = model_spec["hidden_dim"]
    inter = model_spec.get("intermediate_dim", h * 4)
    nh = model_spec["num_heads"]
    nh_kv = model_spec.get("num_kv_heads", nh)
    hd = model_spec["head_dim"]
    vs = model_spec["vocab_size"]
    norm_type = model_spec.get("norm_type", "rmsnorm")
    nl = model_spec["num_layers"]
    na = model_spec.get("attn_layers", nl)

    b_effs = hw["elem_b_effs"]
    overheads = hw["elem_overheads"]

    total_new_tokens = sum(nt for _, nt in scheduled_requests)

    # Per-stage layer counts (last stage gets remainder if uneven)
    layers_per_stage = nl // pp_size
    attn_per_stage = na // pp_size

    # ── Roofline params ──
    # Projections / LM head use total M to pick prefill vs decode params
    params = _select_roofline_params(total_new_tokens, hw)
    F, B, p = params["F"], params["B"], params["p"]

    # ── Per-stage projections ──
    attn_proj_time = layers_per_stage * attn_projections(
        total_new_tokens, h, F, B, p, nh, nh_kv, hd, dt_bytes)
    ffn_proj_time = layers_per_stage * ffn_projections(
        total_new_tokens, h, inter, F, B, p, dt_bytes)

    # ── Per-stage attention: group by type → roofline once per type ──
    # Count concurrent requests by type first (maps to FA batch dimension)
    n_prefill = sum(1 for req, _ in scheduled_requests if req.is_prefill_chunk)
    n_decode = sum(1 for req, _ in scheduled_requests if not req.is_prefill_chunk)

    fa_d = _select_fa_params(n_decode, "decode", hw, nh)
    fa_p = _select_fa_params(n_prefill, "prefill", hw, nh)
    F_d, B_d, p_d = fa_d["F"], fa_d["B"], fa_d["p"]
    F_p, B_p, p_p = fa_p["F"], fa_p["B"], fa_p["p"]

    prefill_flops = 0.0
    prefill_bytes = 0.0
    decode_flops = 0.0
    decode_bytes = 0.0

    for req, num_new in scheduled_requests:
        kv_len_after = req.num_computed_tokens
        if kv_len_after <= 0:
            continue
        f = 4 * nh * num_new * kv_len_after * hd
        b = hd * dt_bytes * (2 * nh * num_new + 2 * nh_kv * kv_len_after)
        if req.is_prefill_chunk:
            prefill_flops += f
            prefill_bytes += b
        else:
            decode_flops += f
            decode_bytes += b

    attn_prefill_time = attn_per_stage * roofline_time(prefill_flops, prefill_bytes, F_p, B_p, p_p) if n_prefill > 0 else 0.0
    attn_decode_time = attn_per_stage * roofline_time(decode_flops, decode_bytes, F_d, B_d, p_d) if n_decode > 0 else 0.0

    # ── Per-stage elementwise ops ──
    fused_add_norm_time = layers_per_stage * fused_residual_norm_ops(1, total_new_tokens, h, b_effs, overheads, dt_bytes)
    swiglu_time = layers_per_stage * swiglu_op(1, total_new_tokens, inter, b_effs, overheads, dt_bytes)
    rope_time = layers_per_stage * rope_op(1, total_new_tokens, nh, nh_kv, hd, b_effs, overheads, dt_bytes)

    # ── LM head (only on last stage, but sequential pipeline model
    #      means it's still on the critical path) ──
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p, dt_bytes)

    # ── Inter-stage communication: (pp_size - 1) transfers ──
    # Each transfer sends hidden states for all tokens: tokens × h × dt_bytes.
    # latency + bandwidth model: real P2P has fixed per-transfer overhead
    # that dominates for small messages (e.g. decode with 1 token).
    inter_stage_bytes = total_new_tokens * h * dt_bytes
    inter_stage_comm = (pp_size - 1) * (
        inter_stage_bytes / (intra_node_bw_gb_s * 1e9) + intra_latency_us * 1e-6
    )

    total = attn_proj_time + ffn_proj_time + attn_prefill_time + attn_decode_time + fused_add_norm_time + swiglu_time + rope_time + lm_head_time + inter_stage_comm
    return {
        "total": total,
        "attn_proj": attn_proj_time,
        "ffn_proj": ffn_proj_time,
        "attn_prefill": attn_prefill_time,
        "attn_decode": attn_decode_time,
        "fused_add_norm": fused_add_norm_time,
        "swiglu": swiglu_time,
        "rope": rope_time,
        "lm_head": lm_head_time,
        "inter_stage_comm": inter_stage_comm,
    }


def predict_step_tp(scheduled_requests, model_spec, hw_params,
                    num_gpus, intra_node_bw_gb_s,
                    intra_latency_us=2.0,
                    pp_size=1, dtype="float16"):
    """Predict step time with tensor parallelism (and optional pipeline parallelism).

    When pp_size > 1, pipeline parallelism is applied first (splitting layers
    across stages), then TP divides the per-stage compute and adds all-reduce.

    Args:
        num_gpus: number of GPUs in the TP group.
        intra_node_bw_gb_s: intra-node bandwidth for all-reduce (GB/s).
        intra_latency_us: intra-node latency per all-reduce step (µs).
        pp_size: pipeline parallelism degree (1 = no PP).
        dtype: precision string.

    Returns:
        dict with keys: total, proj, attn_prefill, attn_decode, elem,
        lm_head, all_reduce (and inter_stage_comm when pp_size > 1).
    """
    if pp_size > 1:
        base = predict_step_pp(scheduled_requests, model_spec, hw_params,
                               pp_size, intra_node_bw_gb_s, intra_latency_us, dtype)
    else:
        base = predict_step(scheduled_requests, model_spec, hw_params, dtype)

    if num_gpus <= 1:
        base["all_reduce"] = 0.0
        return base

    dt_bytes = dtype_bytes(dtype)
    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    h = model_spec["hidden_dim"]

    # All-reduce overhead: bytes / bandwidth + num_gpus × latency.
    # Ring all-reduce has N communication steps; latency dominates for
    # small messages (decode with 1 token: ~16 KB → ~1.7 µs bw-only,
    # but ~20 µs with 2 µs × 8-GPU ring).
    all_reduce_bytes = 2 * total_new_tokens * h * dt_bytes
    all_reduce_time = (
        all_reduce_bytes / (intra_node_bw_gb_s * 1e9)
        + num_gpus * intra_latency_us * 1e-6
    )

    # Scale compute components by 1/tp; inter_stage_comm (PP) is also divided
    result = {}
    compute_total = 0.0
    for k in ("attn_proj", "ffn_proj", "attn_prefill", "attn_decode",
              "fused_add_norm", "swiglu", "rope",
              "lm_head", "inter_stage_comm"):
        v = base.get(k, 0.0)
        scaled = v / num_gpus
        result[k] = scaled
        compute_total += scaled
    result["all_reduce"] = all_reduce_time
    result["total"] = compute_total + all_reduce_time
    return result
