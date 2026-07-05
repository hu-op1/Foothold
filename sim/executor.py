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

    Also handles ``_cudagraph`` suffix: when present, cudagraph-specific
    params are lifted to unsuffixed keys (taking priority over eager params).

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


def _has_cudagraph_params(hw):
    """Return True if CUDA Graph-specific roofline params are available."""
    return "F_peak_decode_cudagraph" in hw or "F_peak_prefill_cudagraph" in hw


def _select_roofline_params(M_total, hw, use_cudagraph=False):
    """Smoothly interpolate between decode and prefill roofline params.

    M_total ≤ 32  → pure decode (memory-bound, low B_peak)
    M_total ≥ 256 → pure prefill (compute-bound, high B_peak)
    32 < M < 256  → log-space interpolation of B_peak and p

    When *use_cudagraph* is True and cudagraph params are available,
    uses CUDA Graph-specific fitted values (keys with ``_cudagraph`` suffix).
    CUDA Graph params reflect true hardware limits without per-kernel
    launch overhead baked in.
    """
    if use_cudagraph and _has_cudagraph_params(hw):
        key_F_d = "F_peak_decode_cudagraph"
        key_B_d = "B_peak_decode_cudagraph"
        key_p_d = "p_decode_cudagraph"
        key_F_p = "F_peak_prefill_cudagraph"
        key_B_p = "B_peak_prefill_cudagraph"
        key_p_p = "p_prefill_cudagraph"
        key_ov_d = "matmul_overhead_decode_cudagraph"
        key_ov_p = "matmul_overhead_prefill_cudagraph"
    else:
        key_F_d = "F_peak_decode"
        key_B_d = "B_peak_decode"
        key_p_d = "p_decode"
        key_F_p = "F_peak_prefill"
        key_B_p = "B_peak_prefill"
        key_p_p = "p_prefill"
        key_ov_d = "matmul_overhead_decode"
        key_ov_p = "matmul_overhead_prefill"

    if M_total <= _M_LO:
        return {
            "F": hw[key_F_d],
            "B": hw[key_B_d],
            "p": hw[key_p_d],
            "overhead": hw.get(key_ov_d, 0.0),
        }
    if M_total >= _M_HI:
        return {
            "F": hw[key_F_p],
            "B": hw[key_B_p],
            "p": hw[key_p_p],
            "overhead": hw.get(key_ov_p, hw.get(key_ov_d, 0.0)),
        }

    w = (log2(M_total) - log2(_M_LO)) / _LOG_RANGE

    log_B = log(hw[key_B_d]) + w * (log(hw[key_B_p]) - log(hw[key_B_d]))
    B = exp(log_B)
    p = hw[key_p_d] + w * (hw[key_p_p] - hw[key_p_d])

    return {"F": hw[key_F_p], "B": B, "p": p,
            "overhead": hw.get(key_ov_d, 0.0)}


def _select_fa_params(n_requests, regime, hw_params, nh_model=None,
                      use_cudagraph=False):
    """Select FlashAttention roofline params based on concurrent request count.

    When per-batch FA params are available (fa_batch_sizes), interpolates
    B_peak and p in log-space across batch sizes — mirroring the M-based
    interpolation in _select_roofline_params.  Falls back to unified FA
    params, then to matmul-fitted params.

    When *use_cudagraph* is True and cudagraph FA params are available,
    uses CUDA Graph-specific keys (``_cudagraph`` suffix).

    The interpolation is done on **total query heads** (n_requests × nh_model),
    normalised by the benchmark's nh (fa_bench_nh).  This makes the sweep
    valid for models with different num_heads than the benchmark config.

    Args:
        n_requests: number of concurrent requests of this type (prefill or decode).
        regime: "decode" or "prefill".
        hw_params: fitted hardware params dict.
        nh_model: model's num_heads (for query-head normalisation).  If None,
            uses n_requests directly (backward compat).
        use_cudagraph: if True, use CUDA Graph-specific roofline params.

    Returns:
        dict with keys F, B, p.
    """
    cg = "_cudagraph" if (use_cudagraph and "fa_decode_B_cudagraph" in hw_params) else ""
    batch_sizes = hw_params.get(f"fa_batch_sizes{cg}")
    B_key = f"fa_{regime}_B{cg}"
    p_key = f"fa_{regime}_p{cg}"
    F_key = f"F_peak_fa_{regime}{cg}"

    # Normalise to benchmark batch: effective_batch = total_query_heads / bench_nh
    if nh_model is not None:
        bench_nh = hw_params.get(f"fa_bench_nh{cg}", nh_model)
        effective = n_requests * nh_model / bench_nh if bench_nh > 0 else n_requests
    else:
        effective = n_requests

    if batch_sizes and len(batch_sizes) > 1 and B_key in hw_params:
        B_arr = hw_params[B_key]
        p_arr = hw_params[p_key]

        # Filter out entries where fit failed (B=0) — can happen when
        # a batch size has too few benchmark points for fitting.
        valid = [(bs, b, p) for bs, b, p in zip(batch_sizes, B_arr, p_arr) if b > 0]
        if not valid:
            # All entries invalid — fall through to unified/matmul fallback
            pass
        elif len(valid) == 1:
            B, p = valid[0][1], valid[0][2]
        else:
            batch_vals, B_vals, p_vals = zip(*valid)

            if effective <= batch_vals[0]:
                B, p = B_vals[0], p_vals[0]
            elif effective >= batch_vals[-1]:
                B, p = B_vals[-1], p_vals[-1]
            else:
                for i in range(len(batch_vals) - 1):
                    if batch_vals[i] <= effective <= batch_vals[i + 1]:
                        lo, hi = batch_vals[i], batch_vals[i + 1]
                        w = (log2(effective) - log2(lo)) / (log2(hi) - log2(lo))
                        log_B = log(B_vals[i]) + w * (log(B_vals[i + 1]) - log(B_vals[i]))
                        B = exp(log_B)
                        p = p_vals[i] + w * (p_vals[i + 1] - p_vals[i])
                        break

        F = hw_params.get(F_key, hw_params.get(f"F_peak_{regime}", 1e13))
        return {"F": F, "B": B, "p": p}

    # Fallback: unified FA params → matmul params
    F_fb = hw_params.get(F_key)
    B_fb = hw_params.get(f"B_peak_fa_{regime}{cg}")
    p_fb = hw_params.get(f"p_fa_{regime}{cg}")
    if F_fb is not None and B_fb is not None and p_fb is not None:
        return {"F": F_fb, "B": B_fb, "p": p_fb}

    print(f"  [fallback] FA {regime} params not found, using matmul roofline params")
    return {
        "F": hw_params[f"F_peak_{regime}{cg if cg else ''}"],
        "B": hw_params[f"B_peak_{regime}{cg if cg else ''}"],
        "p": hw_params[f"p_{regime}{cg if cg else ''}"],
    }


def predict_step(scheduled_requests, model_spec, hw_params, dtype="float16",
                 use_cudagraph=False):
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

    b_effs = hw.get("elem_b_effs_cudagraph", hw["elem_b_effs"]) if use_cudagraph else hw["elem_b_effs"]
    overheads = hw.get("elem_overheads_cudagraph", hw["elem_overheads"]) if use_cudagraph else hw["elem_overheads"]

    # Total new tokens this step — used for batched projections / LM head
    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    params = _select_roofline_params(total_new_tokens, hw, use_cudagraph=use_cudagraph)
    F, B, p = params["F"], params["B"], params["p"]
    matmul_ov = params.get("overhead", 0.0)

    # ── Projections (batched over all tokens → single F,B,p per step) ──
    attn_proj_time = nl * attn_projections(total_new_tokens, h, F, B, p, nh, nh_kv, hd, dt_bytes, matmul_ov)
    ffn_proj_time = nl * ffn_projections(total_new_tokens, h, inter, F, B, p, dt_bytes, matmul_ov)

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
    fa_d = _select_fa_params(n_decode, "decode", hw, nh, use_cudagraph=use_cudagraph)
    fa_p = _select_fa_params(n_prefill, "prefill", hw, nh, use_cudagraph=use_cudagraph)
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
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p, dt_bytes, matmul_ov)

    # ── Kernel launch overhead (CPU→GPU dispatch) ──
    # Each kernel launch has ~5-10 µs of fixed CPU→GPU scheduling cost that
    # CUDA events cannot measure.  For decode (small batch, GPU time ~100-500 µs)
    # this is 10-30% of step time; for prefill (GPU time ~10-100 ms) it's 1-3%.
    # Under CUDA Graph, graph replay uses a single host launch — zero overhead.
    # Per-layer kernel count: QKV(1) + O(1) + gate(1) + up(1) + down(1)
    #   + fused_add_norm(2) + swiglu(1) + rope(1) + attention(1) = 10
    kernel_overhead_us = hw.get("kernel_launch_overhead_us", 0.0)
    if use_cudagraph:
        launch_overhead_time = 0.0
    elif kernel_overhead_us > 0:
        num_kernels = nl * 10 + 1  # +1 for lm_head
        launch_overhead_time = num_kernels * kernel_overhead_us * 1e-6
    else:
        launch_overhead_time = 0.0

    total = attn_proj_time + ffn_proj_time + attn_prefill_time + attn_decode_time + fused_add_norm_time + swiglu_time + rope_time + lm_head_time + launch_overhead_time
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
        "launch_overhead": launch_overhead_time,
    }


def predict_step_pp(scheduled_requests, model_spec, hw_params,
                    pp_size, intra_node_bw_gb_s, intra_latency_us=2.0,
                    dtype="float16", use_cudagraph=False):
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
        return predict_step(scheduled_requests, model_spec, hw_params, dtype,
                            use_cudagraph=use_cudagraph)

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
    params = _select_roofline_params(total_new_tokens, hw, use_cudagraph=use_cudagraph)
    F, B, p = params["F"], params["B"], params["p"]
    matmul_ov = params.get("overhead", 0.0)

    # ── Per-stage projections ──
    attn_proj_time = layers_per_stage * attn_projections(
        total_new_tokens, h, F, B, p, nh, nh_kv, hd, dt_bytes, matmul_ov)
    ffn_proj_time = layers_per_stage * ffn_projections(
        total_new_tokens, h, inter, F, B, p, dt_bytes, matmul_ov)

    # ── Per-stage attention: group by type → roofline once per type ──
    # Count concurrent requests by type first (maps to FA batch dimension)
    n_prefill = sum(1 for req, _ in scheduled_requests if req.is_prefill_chunk)
    n_decode = sum(1 for req, _ in scheduled_requests if not req.is_prefill_chunk)

    fa_d = _select_fa_params(n_decode, "decode", hw, nh, use_cudagraph=use_cudagraph)
    fa_p = _select_fa_params(n_prefill, "prefill", hw, nh, use_cudagraph=use_cudagraph)
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
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p, dt_bytes, matmul_ov)

    # ── Inter-stage communication: (pp_size - 1) transfers ──
    # Each transfer sends hidden states for all tokens: tokens × h × dt_bytes.
    # latency + bandwidth model: real P2P has fixed per-transfer overhead
    # that dominates for small messages (e.g. decode with 1 token).
    inter_stage_bytes = total_new_tokens * h * dt_bytes
    inter_stage_comm = (pp_size - 1) * (
        inter_stage_bytes / (intra_node_bw_gb_s * 1e9) + intra_latency_us * 1e-6
    )

    # ── Kernel launch overhead (CPU→GPU dispatch) ──
    # Per-stage kernel count (same per-layer logic as predict_step).
    kernel_overhead_us = hw.get("kernel_launch_overhead_us", 0.0)
    if use_cudagraph:
        launch_overhead_time = 0.0
    elif kernel_overhead_us > 0:
        num_kernels = layers_per_stage * 10 + 1  # +1 for lm_head (on last stage)
        launch_overhead_time = num_kernels * kernel_overhead_us * 1e-6
    else:
        launch_overhead_time = 0.0

    total = attn_proj_time + ffn_proj_time + attn_prefill_time + attn_decode_time + fused_add_norm_time + swiglu_time + rope_time + lm_head_time + inter_stage_comm + launch_overhead_time
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
        "launch_overhead": launch_overhead_time,
    }


def predict_step_tp(scheduled_requests, model_spec, hw_params,
                    num_gpus, intra_node_bw_gb_s,
                    intra_latency_us=2.0,
                    pp_size=1, dtype="float16", use_cudagraph=False):
    """Predict step time with tensor parallelism (and optional pipeline parallelism).

    When pp_size > 1, pipeline parallelism is applied first (splitting layers
    across stages), then TP divides the per-stage compute and adds all-reduce.

    All-reduce is per-layer (matching vLLM's column-parallel linear pattern):
      - QKV projection output → 1 all-reduce per attention layer
      - FFN gate+up output    → 1 all-reduce per layer
    O-projection and FFN-down are row-parallel (no all-reduce needed).

    Args:
        num_gpus: number of GPUs in the TP group.
        intra_node_bw_gb_s: intra-node bandwidth for all-reduce (GB/s).
        intra_latency_us: intra-node latency per all-reduce step (µs).
        pp_size: pipeline parallelism degree (1 = no PP).
        dtype: precision string.
        use_cudagraph: if True, use CUDA Graph-specific roofline params.

    Returns:
        dict with keys: total, proj, attn_prefill, attn_decode, elem,
        lm_head, all_reduce (and inter_stage_comm when pp_size > 1).
    """
    if pp_size > 1:
        base = predict_step_pp(scheduled_requests, model_spec, hw_params,
                               pp_size, intra_node_bw_gb_s, intra_latency_us, dtype,
                               use_cudagraph=use_cudagraph)
    else:
        base = predict_step(scheduled_requests, model_spec, hw_params, dtype,
                            use_cudagraph=use_cudagraph)

    if num_gpus <= 1:
        base["all_reduce"] = 0.0
        return base

    dt_bytes = dtype_bytes(dtype)
    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    h = model_spec["hidden_dim"]
    inter = model_spec.get("intermediate_dim", h * 4)
    nh = model_spec.get("num_heads", h // 128)
    nh_kv = model_spec.get("num_kv_heads", nh)
    hd = model_spec.get("head_dim", h // nh)
    nl = model_spec["num_layers"]
    na = model_spec.get("attn_layers", nl)

    # ── Per-layer all-reduce data volumes ──
    # vLLM uses two TP patterns (see vllm/model_executor/layers/linear.py):
    #   ColumnParallelLinear (QKV, gate_proj, up_proj):
    #     Each GPU holds a column-wise shard of weights.  Output is LOCAL —
    #     no cross-GPU communication needed (attention uses local heads).
    #   RowParallelLinear (O_proj, down_proj):
    #     Each GPU computes a partial sum.  Needs all-reduce to combine.
    #
    # Per attention layer: O_proj + down_proj → 2 all-reduces of [tokens, h]
    # Per delta (non-attn) layer: down_proj only → 1 all-reduce of [tokens, h]
    #
    # Ring all-reduce: reduce-scatter (N−1 steps) + all-gather (N−1 steps).
    # Each step sends data/N bytes over one hop.
    # Total data: 2·(N−1)/N · data  →  ≈ 2·data for large N
    # Total latency: 2·(N−1) · hop_latency
    # vLLM executes communication serially after each layer's compute
    # (no CUDA-stream overlap).  So total = compute + communicate.
    def _ar_time(ar_bytes):
        steps = 2 * (num_gpus - 1)
        factor = steps / num_gpus          # = 2·(N−1)/N
        return (ar_bytes * factor / (intra_node_bw_gb_s * 1e9)
                + steps * intra_latency_us * 1e-6)

    ar_bytes_per_layer = total_new_tokens * h * dt_bytes
    all_reduce_time = (2 * na + (nl - na)) * _ar_time(ar_bytes_per_layer)

    # Scale compute components by 1/tp.
    # vLLM source (vllm/model_executor/models/llama.py) confirms:
    #   SHARDED by TP: QKV, O, gate, up, down, attention, SwiGLU, RoPE, lm_head
    #   NOT SHARDED:  RMSNorm, residual_add → each GPU computes on FULL hidden_size
    #   (fused_add_norm combines RMSNorm + residual_add — also NOT sharded)
    result = {}
    compute_total = 0.0
    TP_SCALED = ("attn_proj", "ffn_proj", "attn_prefill", "attn_decode",
                 "swiglu", "rope", "lm_head", "inter_stage_comm",
                 "launch_overhead")
    for k in TP_SCALED:
        v = base.get(k, 0.0)
        scaled = v / num_gpus
        result[k] = scaled
        compute_total += scaled
    # fused_add_norm is NOT sharded — each GPU does full work
    fan = base.get("fused_add_norm", 0.0)
    result["fused_add_norm"] = fan
    compute_total += fan
    result["all_reduce"] = all_reduce_time
    result["total"] = compute_total + all_reduce_time
    return result
