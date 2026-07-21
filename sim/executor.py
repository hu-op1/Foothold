"""Roofline-based step-time prediction for mixed prefill/decode batches."""

from math import log, exp, log2

from sim.roofline import dtype_bytes
from sim.graph import StepContext, ModelGraph, apply_tp, apply_ep, apply_pp


def predict_step(scheduled_requests, model_spec, graph, hw_params,
                 tp=1, pp=1, ep=0, pp_stage=0, cross_node_hops=0,
                 dtype="float16", use_cudagraph=False,
                 comm_lut_bytes=None, comm_lut_time_s=None) -> dict:
    """Predict GPU execution time using graph-based evaluation.

    Args:
        scheduled_requests: list of (request, num_new_tokens)
        model_spec: dict from model YAML
        graph: ModelGraph
        hw_params: dict from fitted_params.json
        dtype: precision string
        tp/pp/ep: parallelism degrees
        use_cudagraph: whether CUDA Graph is active

    Returns:
        dict with "total" + per-op breakdown
    """
    if not scheduled_requests:
        return {"total": 0.0}

    ctx = StepContext.precompute(
        scheduled_requests, model_spec, hw_params, dtype=dtype,
        use_cudagraph=use_cudagraph,
        comm_lut_bytes=comm_lut_bytes,
        comm_lut_time_s=comm_lut_time_s,
    )
    ctx.tp_size = tp
    ctx.ep_size = ep
    ctx.pp_size = pp

    g = graph
    if tp > 1:
        g = apply_tp(g, ctx, tp)
    if ep > 1:
        g = apply_ep(g, ctx, ep)
    if pp > 1:
        g = apply_pp(g, ctx, pp, pp_stage, cross_node_hops)

    return g.evaluate(ctx, hw_params)
