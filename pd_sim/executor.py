"""Roofline-based step-time prediction for mixed prefill/decode batches."""

import sys
import os

# Allow importing from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perf_predict.predict import (
    matmul_time,
    projections,
    attention_fused,
    elementwise_ops,
    DTYPE_BYTES,
)

M_SPLIT = 256  # matches fit/matmul.py M_SPLIT


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
        return 0.0

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
    proj_time = nl * projections(total_new_tokens, h, inter, F, B, p, nh, nh_kv, hd)

    # ── Attention (per-request, per-attention-layer) ──
    # Prefill requests (is_prefill_chunk=True) have s_q ≫ 1 and are more
    # compute-heavy; decode requests (s_q=1) are memory-bound.  Use the
    # appropriate roofline params for each.
    attn_time = 0.0
    for req, num_new in scheduled_requests:
        # num_computed_tokens was already incremented by _update_after_schedule,
        # so it already reflects the KV cache length after this step.
        kv_len_after = req.num_computed_tokens
        if kv_len_after > 0:
            if req.is_prefill_chunk:
                attn_time += na * attention_fused(1, nh, num_new, kv_len_after, hd, F_p, B_p, p_p, nh_kv)
            else:
                attn_time += na * attention_fused(1, nh, num_new, kv_len_after, hd, F_d, B_d, p_d, nh_kv)

    # ── Elementwise (per-layer, multiplied by num_layers) ──
    # b_effs / overheads are independent of the prefill/decode split.
    elem_time = nl * elementwise_ops(
        1, total_new_tokens, h, inter, nh, hd, norm_type, b_effs, overheads, nh_kv
    )

    # ── Output projection (single lm_head, batched) ──
    lm_head_time = matmul_time(total_new_tokens, h, vs, F, B, p)

    return proj_time + attn_time + elem_time + lm_head_time


def predict_step_tp(scheduled_requests, model_spec, hw_params,
                    num_gpus, intra_node_bw_gb_s):
    """Predict step time with tensor parallelism.

    Args:
        num_gpus: number of GPUs in the TP group.
        intra_node_bw_gb_s: intra-node bandwidth for all-reduce (GB/s).

    Returns:
        step_time_s with TP overhead.
    """
    single_time = predict_step(scheduled_requests, model_spec, hw_params)
    if num_gpus <= 1:
        return single_time

    total_new_tokens = sum(nt for _, nt in scheduled_requests)
    h = model_spec["hidden_dim"]

    # All-reduce overhead: 2 * activations_size / bandwidth
    all_reduce_bytes = 2 * total_new_tokens * h * DTYPE_BYTES
    all_reduce_time = all_reduce_bytes / (intra_node_bw_gb_s * 1e9)

    return single_time / num_gpus + all_reduce_time
