"""Model computation graph: OpSpec, StepContext, ModelGraph, transforms."""

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from math import log, exp, log2
from typing import Callable, Literal

from sim.roofline import (
    matmul_time,
    roofline_time,
    DTYPE_BYTES_MAP,
)
from sim.communication import memcpy_time, all_to_all_time

_M_LO = 32
_M_HI = 256
_LOG_RANGE = log2(_M_HI) - log2(_M_LO)


@dataclass
class OpSpec:
    """A single GPU operation with pre-computed roofline parameters."""

    name: str
    category: Literal["matmul", "attention", "elementwise", "comm"]
    tags: frozenset[str] = frozenset()

    # matmul
    M: int = 0
    K: int = 0
    N: int = 0
    F_peak: float = 0.0
    B_peak: float = 0.0
    p: float = 0.0
    overhead: float = 0.0

    # attention (FlashAttention)
    prefill_flops: float = 0.0
    prefill_bytes: float = 0.0
    decode_flops: float = 0.0
    decode_bytes: float = 0.0
    fa_d: dict = field(default_factory=dict)
    fa_p: dict = field(default_factory=dict)
    na: int = 0

    # elementwise
    N_elems: int = 0
    elem_op: str = ""
    b_eff: float = 0.0
    elem_overhead: float = 0.0

    # comm
    comm_bytes: int = 0
    comm_type: str = ""
    comm_lut_bytes: list = field(default_factory=list)
    comm_lut_time_s: list = field(default_factory=list)

    # result
    time: float = 0.0


ELEM_BYTES = {
    "fused_residual_norm": 4,
    "rmsnorm": 4,
    "layernorm": 5,
    "swiglu": 3,
    "rope": 4,
    "residual_add": 3,
}


def _compute_op_time(op: OpSpec, ctx: "StepContext") -> float:
    """Compute wall-clock time for a single OpSpec."""
    if op.category == "matmul":
        return matmul_time(
            op.M, op.K, op.N, op.F_peak, op.B_peak, op.p,
            dt_bytes=ctx.dt_bytes, overhead=op.overhead,
        )
    elif op.category == "attention":
        t = 0.0
        if op.prefill_flops > 0 or op.prefill_bytes > 0:
            t += roofline_time(
                op.prefill_flops, op.prefill_bytes,
                op.fa_p.get("F", 1e13), op.fa_p.get("B", 1e12), op.fa_p.get("p", 1.0),
            )
        if op.decode_flops > 0 or op.decode_bytes > 0:
            t += roofline_time(
                op.decode_flops, op.decode_bytes,
                op.fa_d.get("F", 1e13), op.fa_d.get("B", 1e12), op.fa_d.get("p", 1.0),
            )
        return t * op.na if op.na > 0 else t
    elif op.category == "elementwise":
        factor = ELEM_BYTES.get(op.elem_op, 4)
        return (op.N_elems * factor * ctx.dt_bytes) / op.b_eff + op.elem_overhead
    elif op.category == "comm":
        if op.comm_type == "all_reduce":
            return memcpy_time(op.comm_bytes, op.comm_lut_bytes, op.comm_lut_time_s)
        elif op.comm_type == "all_to_all":
            return all_to_all_time(
                op.comm_bytes, op.comm_lut_bytes, op.comm_lut_time_s,
                ep_size=ctx.ep_size or 1,
            )
        return 0.0
    return 0.0


@dataclass
class StepContext:
    """Pre-computed context for one scheduler step."""

    scheduled_requests: list
    total_tokens: int = 0
    dtype: str = "float16"
    dt_bytes: int = 2

    n_prefill: int = 0
    n_decode: int = 0
    prefill_flops: float = 0.0
    prefill_bytes: float = 0.0
    decode_flops: float = 0.0
    decode_bytes: float = 0.0

    matmul_F: float = 0.0
    matmul_B: float = 0.0
    matmul_p: float = 0.0

    fa_decode_params: dict = field(default_factory=dict)
    fa_prefill_params: dict = field(default_factory=dict)

    b_effs: dict = field(default_factory=dict)
    overheads: dict = field(default_factory=dict)

    kernel_overhead_us: float = 0.0

    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1

    comm_lut_bytes: list = field(default_factory=list)
    comm_lut_time_s: list = field(default_factory=list)

    @classmethod
    def precompute(cls, scheduled_requests, spec, hw_params, dtype="float16",
                   use_cudagraph=False, comm_lut_bytes=None, comm_lut_time_s=None):
        """Build StepContext from scheduled requests + model/hardware params."""
        if not scheduled_requests:
            return cls(scheduled_requests=scheduled_requests)

        hw = _select_dtype_params(hw_params, dtype)
        dt_bytes = DTYPE_BYTES_MAP.get(dtype, 2)

        nh = spec.get("num_heads", spec.get("num_q_heads", 32))
        nh_kv = spec.get("num_kv_heads", nh)
        hd = spec["head_dim"]

        total_new_tokens = sum(nt for _, nt in scheduled_requests)

        mp = _interp_roofline(total_new_tokens, hw, use_cudagraph=use_cudagraph)

        n_prefill = sum(1 for req, _ in scheduled_requests if req.is_prefill_chunk)
        n_decode = sum(1 for req, _ in scheduled_requests if not req.is_prefill_chunk)

        fa_d = _select_fa_params(n_decode, "decode", hw, nh, use_cudagraph=use_cudagraph)
        fa_p = _select_fa_params(n_prefill, "prefill", hw, nh, use_cudagraph=use_cudagraph)

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

        use_cg = use_cudagraph and _has_cudagraph_elem(hw)
        b_effs = hw.get("elem_b_effs_cudagraph", hw["elem_b_effs"]) if use_cg else hw["elem_b_effs"]
        overheads = hw.get("elem_overheads_cudagraph", hw["elem_overheads"]) if use_cg else hw["elem_overheads"]

        kernel_overhead_us = 0.0 if use_cg else hw.get("kernel_launch_overhead_us", 0.0)

        return cls(
            scheduled_requests=scheduled_requests,
            total_tokens=total_new_tokens,
            dtype=dtype,
            dt_bytes=dt_bytes,
            n_prefill=n_prefill,
            n_decode=n_decode,
            prefill_flops=prefill_flops,
            prefill_bytes=prefill_bytes,
            decode_flops=decode_flops,
            decode_bytes=decode_bytes,
            matmul_F=mp["F"],
            matmul_B=mp["B"],
            matmul_p=mp["p"],
            fa_decode_params=fa_d,
            fa_prefill_params=fa_p,
            b_effs=b_effs,
            overheads=overheads,
            kernel_overhead_us=kernel_overhead_us,
            comm_lut_bytes=comm_lut_bytes,
            comm_lut_time_s=comm_lut_time_s,
        )


@dataclass
class ModelGraph:
    """A complete model computation graph.

    layer_specs: [(builder, count), ...]
        Builders have (ctx, hw) -> list[OpSpec] signature.
    head_builder: (ctx, hw) -> list[OpSpec]
        lm_head + final_norm.
    """

    layer_specs: list[tuple[Callable, int]] = field(default_factory=list)
    head_builder: Callable | None = None

    def evaluate(self, ctx: StepContext, hw: dict) -> dict:
        """Evaluate all ops, return {total, breakdown}."""
        total = 0.0
        breakdown = defaultdict(float)
        for builder, count in self.layer_specs:
            for _ in range(count):
                for op in builder(ctx, hw):
                    t = _compute_op_time(op, ctx)
                    op.time = t
                    total += t
                    breakdown[op.name] += t
        if self.head_builder:
            for op in self.head_builder(ctx, hw):
                t = _compute_op_time(op, ctx)
                total += t
                breakdown[op.name] += t
        return {"total": total, "breakdown": dict(breakdown)}

    def transform_layers(self, fn):
        """Return new ModelGraph with each layer's builder wrapped by fn.

        fn: (ops: list[OpSpec], ctx: StepContext, hw: dict) -> list[OpSpec]
        """
        new_specs = []
        for builder, count in self.layer_specs:
            def _wrapped(b=builder, f=fn):
                def _inner(ctx, hw):
                    ops = b(ctx, hw)
                    return f(ops, ctx, hw)
                return _inner
            new_specs.append((_wrapped(), count))
        return ModelGraph(layer_specs=new_specs, head_builder=self.head_builder)


# ── Internal helpers (migrated from executor.py) ─────────────────────

def _has_cudagraph_elem(hw):
    return "elem_b_effs_cudagraph" in hw


def _has_cudagraph_params(hw):
    return "F_peak_decode_cudagraph" in hw


def _select_dtype_params(hw_params, dtype):
    """Select dtype-specific roofline params by stripping _{dtype} suffix.

    When multiple dtypes are fitted, keys are stored with a suffix
    (e.g. F_peak_decode_float16).  This strips the suffix so downstream
    code can use the same key names regardless of dtype.
    Falls back to unsuffixed keys for single-dtype fits.
    """
    suffix = f"_{dtype}"
    result = dict(hw_params)
    for key, val in hw_params.items():
        if key.endswith(suffix):
            base = key[:-len(suffix)]
            if base:
                result[base] = val
    return result


def _interp_roofline(M_total, hw, use_cudagraph=False):
    """Smooth interpolation of decode/prefill roofline params by M."""
    if use_cudagraph and _has_cudagraph_params(hw):
        keys = {
            "F_d": "F_peak_decode_cudagraph", "B_d": "B_peak_decode_cudagraph",
            "p_d": "p_decode_cudagraph", "F_p": "F_peak_prefill_cudagraph",
            "B_p": "B_peak_prefill_cudagraph", "p_p": "p_prefill_cudagraph",
        }
    else:
        keys = {
            "F_d": "F_peak_decode", "B_d": "B_peak_decode", "p_d": "p_decode",
            "F_p": "F_peak_prefill", "B_p": "B_peak_prefill", "p_p": "p_prefill",
        }

    if M_total <= _M_LO:
        return {"F": hw[keys["F_d"]], "B": hw[keys["B_d"]], "p": hw[keys["p_d"]]}
    if M_total >= _M_HI:
        return {"F": hw[keys["F_p"]], "B": hw[keys["B_p"]], "p": hw[keys["p_p"]]}

    w = (log2(M_total) - log2(_M_LO)) / _LOG_RANGE
    log_B = log(hw[keys["B_d"]]) + w * (log(hw[keys["B_p"]]) - log(hw[keys["B_d"]]))
    B = exp(log_B)
    p = hw[keys["p_d"]] + w * (hw[keys["p_p"]] - hw[keys["p_d"]])
    return {"F": hw[keys["F_p"]], "B": B, "p": p}


def _select_fa_params(n_requests, regime, hw_params, nh_model=None, use_cudagraph=False):
    """Select FlashAttention roofline params by regime + concurrent request count."""
    cg = "_cudagraph" if (use_cudagraph and "fa_decode_B_cudagraph" in hw_params) else ""
    batch_sizes = hw_params.get(f"fa_batch_sizes{cg}")
    B_key = f"fa_{regime}_B{cg}"
    p_key = f"fa_{regime}_p{cg}"
    F_key = f"F_peak_fa_{regime}{cg}"

    if nh_model is not None:
        bench_nh = hw_params.get(f"fa_bench_nh{cg}", nh_model)
        effective = n_requests * nh_model / bench_nh if bench_nh > 0 else n_requests
    else:
        effective = n_requests

    if batch_sizes and len(batch_sizes) > 1 and B_key in hw_params:
        B_arr = hw_params[B_key]
        p_arr = hw_params[p_key]
        valid = [(bs, b, p) for bs, b, p in zip(batch_sizes, B_arr, p_arr) if b > 0]
        if valid:
            if len(valid) == 1:
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


# ── Graph Transforms ─────────────────────────────────────────────────

def apply_tp(graph: ModelGraph, ctx: StepContext, tp: int) -> ModelGraph:
    """Shard projection matmuls by tp, insert all_reduce communication.

    For each OpSpec with category=="matmul" and "projection" in tags:
    - N <- N // tp
    - Insert all_reduce after
    """
    if tp <= 1:
        return graph

    def _transform(ops, _ctx, _hw):
        result = []
        for op in ops:
            if op.category == "matmul" and "projection" in op.tags:
                sharded = copy.copy(op)
                sharded.N = max(sharded.N // tp, 1)
                result.append(sharded)
                result.append(OpSpec(
                    name="all_reduce", category="comm",
                    tags=frozenset({"sync"}),
                    comm_bytes=sharded.N * ctx.dt_bytes,
                    comm_type="all_reduce",
                    comm_lut_bytes=ctx.comm_lut_bytes,
                    comm_lut_time_s=ctx.comm_lut_time_s,
                ))
            else:
                result.append(op)
        return result

    return graph.transform_layers(_transform)


def apply_ep(graph: ModelGraph, ctx: StepContext, ep: int) -> ModelGraph:
    """Insert all-to-all for MoE expert ops, scale expert token counts.

    EP does NOT shard matmul dimensions — each GPU holds different complete experts.
    Only modifies ops with "expert" tag: M <- M // ep (tokens distributed).
    Ops with "expert_comm" tag already inserted by moe_ffn builder.
    """
    if ep <= 1:
        return graph

    def _transform(ops, _ctx, _hw):
        result = []
        for op in ops:
            if op.category == "matmul" and "expert" in op.tags:
                scaled = copy.copy(op)
                scaled.M = max(scaled.M // ep, 1)
                result.append(scaled)
            else:
                result.append(op)
        return result

    return graph.transform_layers(_transform)


def apply_pp(graph: ModelGraph, ctx: StepContext, pp: int, stage: int,
             cross_node_hops: int = 0) -> ModelGraph:
    """Keep only layers belonging to the current PP stage.

    Layers distributed evenly: nl // pp per stage, remainder to first stages.
    """
    if pp <= 1:
        return graph

    all_builders = []
    for builder, count in graph.layer_specs:
        all_builders.extend([builder] * count)

    nl = len(all_builders)
    per_stage = nl // pp
    remainder = nl % pp
    start = stage * per_stage + min(stage, remainder)
    end = start + per_stage + (1 if stage < remainder else 0)

    stage_builders = all_builders[start:end]

    new_specs = [(b, 1) for b in stage_builders]
    return ModelGraph(layer_specs=new_specs, head_builder=graph.head_builder)
