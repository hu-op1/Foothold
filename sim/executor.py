"""Roofline-based step-time prediction for mixed prefill/decode batches."""

from math import log, exp, log2

from sim.roofline import dtype_bytes
from sim.graph import StepContext, ModelGraph, apply_tp, apply_ep
from sim.communication import _transfer_time


def _inter_stage_time(total_tokens: int, h: int, dt_bytes: int,
                      pp: int, cross_node_hops: int,
                      comm_model: str,
                      intra_bw_gb_s: float, intra_latency_us: float,
                      inter_bw_gb_s: float, inter_latency_us: float,
                      lut_bytes=None, lut_time_s=None) -> float:
    """PP inter-stage hidden-state transfer time.

    Each inter-stage transition sends *total_tokens × h × dt_bytes* bytes.
    There are ``pp − 1`` transitions; *cross_node_hops* of them use inter-node
    parameters, the rest use intra-node.
    """
    if pp <= 1 or total_tokens <= 0:
        return 0.0
    total_bytes = total_tokens * h * dt_bytes
    intra_hops = (pp - 1) - cross_node_hops
    inter_hops = cross_node_hops

    t = 0.0
    if intra_hops > 0:
        if comm_model == "bw_latency":
            t_intra = total_bytes / (intra_bw_gb_s * 1e9) + intra_latency_us * 1e-6
        else:
            t_intra = _transfer_time(total_bytes, comm_model="lut",
                                     lut_bytes=lut_bytes, lut_time_s=lut_time_s)
        t += intra_hops * t_intra

    if inter_hops > 0:
        if comm_model == "bw_latency":
            t_inter = total_bytes / (inter_bw_gb_s * 1e9) + inter_latency_us * 1e-6
        else:
            t_inter = _transfer_time(total_bytes, comm_model="lut",
                                     lut_bytes=lut_bytes, lut_time_s=lut_time_s)
        t += inter_hops * t_inter

    return t


def predict_step(scheduled_requests, model_spec, graph, hw_params,
                 tp=1, pp=1, ep=0, cross_node_hops=0,
                 dtype="float16", use_cudagraph=False,
                 comm_model="lut", comm_intra_bw_gb_s=9.7, comm_intra_latency_us=2.0,
                 comm_inter_bw_gb_s=9.7, comm_inter_latency_us=6.9,
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
        comm_model: "lut" or "bw_latency"
        comm_intra_bw_gb_s: intra-node bandwidth for all-reduce (GB/s)
        comm_intra_latency_us: intra-node latency per hop (us)

    Returns:
        dict with "total" + per-op breakdown
    """
    if not scheduled_requests:
        return {"total": 0.0}

    ctx = StepContext.precompute(
        scheduled_requests, model_spec, hw_params, dtype=dtype,
        use_cudagraph=use_cudagraph,
        comm_model=comm_model,
        comm_intra_bw_gb_s=comm_intra_bw_gb_s,
        comm_intra_latency_us=comm_intra_latency_us,
        comm_inter_bw_gb_s=comm_inter_bw_gb_s,
        comm_inter_latency_us=comm_inter_latency_us,
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
    # PP: a scheduler step's whole batch is a single micro-batch that
    # traverses every stage serially (vLLM has no intra-step PP micro-
    # batching — see gpu_worker.py recv→compute→send per step).  The wall
    # time is therefore the FULL graph traversal (all pp stages), not a
    # single stage: PP reduces per-GPU compute but not per-step latency
    # (validated: vLLM pp=2 TPOT 18.4ms ≈ pp=1 traversal 17.4ms).
    result = g.evaluate(ctx, hw_params)

    # PP inter-stage communication (hidden-state transfer between stages)
    if pp > 1 and ctx.total_tokens > 0:
        h = model_spec["hidden_dim"]
        inter_s = _inter_stage_time(
            ctx.total_tokens, h, ctx.dt_bytes, pp, cross_node_hops,
            ctx.comm_model,
            ctx.comm_intra_bw_gb_s, ctx.comm_intra_latency_us,
            ctx.comm_inter_bw_gb_s, ctx.comm_inter_latency_us,
            lut_bytes=ctx.comm_lut_bytes, lut_time_s=ctx.comm_lut_time_s,
        )
        result["total"] += inter_s
        result["breakdown"]["inter_stage_comm"] = inter_s

    return result
