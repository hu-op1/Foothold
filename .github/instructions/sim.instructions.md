---
description: "Use when working on PD disaggregation simulation in sim/. Covers event-driven engine, scheduler, roofline executor, memory pool, request lifecycle, and recording."
applyTo: "sim/**/*.py"
---

# sim/ — PD Disaggregation Simulator

## Architecture Overview

```
Requests → SimulationEngine (event loop)
             ├── ColocatedScheduler (vLLM v1 two-phase)
             ├── BlockPool (PagedAttention + prefix caching)
             ├── Executor (roofline predict_step)
             │     └── ModelGraph (sim/graph.py + sim/layers/) — compute graph
             ├── MetricsCollector
             └── SimRecorder (time-series output)
```

## Computation Graph (v1 refactor)

`sim/graph.py` + `sim/layers/` implement the v0→v1 computation graph refactor
(see `docs/plans/2026-07-21-computation-graph-refactor.md` and its design doc),
replacing the hardcoded `predict_step_*` math in `executor.py`:

- **`sim/graph.py`** — `OpSpec` (matmul / attention / elementwise / comm ops with precomputed roofline params), `StepContext.precompute()` (builds a single-step context from scheduled requests + hw params; detects `ls_*` keys to select gated-delta scan params), `ModelGraph.evaluate()` / `transform_layers()`, and graph transforms `apply_tp` / `apply_ep` / `apply_pp` (tag matching, architecture-agnostic).
- **`sim/layers/`** — layer builders (`common.py`: qkv_proj, o_proj, flash_attn, rope, linear_scan, gate_up, swiglu, fused_residual_norm, moe/expert, all_to_all; `head.py`: lm_head).
- **Wiring**: `sim/executor.py:predict_step()` now only does `StepContext.precompute` + transform + `ModelGraph.evaluate`. `sim/engine.py` loads the graph via `load_model_graph(config["model"])`.
- **Per-model graphs** live in `models/*.py` (`SPEC` dict + `build_graph(spec)`) — see `models.instructions.md`.

When adding a new op: add a builder in `sim/layers/`, wire it in the model's `build_graph()`, and add the roofline formula in `sim/roofline.py`.

## Request Lifecycle

`sim/request.py` — `Request` dataclass with `IntEnum` status:

```python
class RequestStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 4
    FINISHED_DROPPED = 5
```

Use `@dataclass` with `field(default_factory=list)` for mutable fields. All fields annotated with types. Use `X | None` for optional fields.

## Event-Driven Engine

`sim/engine.py` — `SimulationEngine` uses `heapq` for event queue:

```python
class EventType(Enum): ARRIVAL, KV_TRANSFER_DONE

@dataclass(order=True)
class SimulationEvent:
    time: float
    event_type: EventType
    request: Request | None
```

Two run modes: `_run_colocated` (DP via least-loaded routing) and `_run_disaggregated` (P pool → KV transfer → D pool with swap preemption).

## Scheduler

`sim/scheduler.py` — vLLM v1 two-phase scheduler:

- **Phase 1**: Iterate running queue. For each request: one decode token, plus chunked prefill if remaining prompt tokens exist. Handle OOM via preemption (swap victim to CPU or recompute).
- **Phase 2**: Admit from waiting queue. Enforce token budget (`max_batched_tokens`) and prefill threshold. Respect `scheduler_reserve_full_isl` gate.

When modifying scheduling logic, ensure `_rollback_if_scheduled()` properly undoes partial scheduling on OOM.

## Memory Pool

`sim/memory.py` — `BlockPool` with PagedAttention semantics:

- `alloc()` / `free()` for block management.
- Prefix caching via SHA-256 hash of token IDs.
- GPU↔CPU swap for D-side OOM recovery.
- Always validate: `if num_blocks <= 0: raise ValueError(...)`.

## Executor (Roofline Timing)

`sim/executor.py` — `predict_step()` computes single-step GPU time:

- **Attention**: Per-request param separation — prefill chunks use prefill roofline params, decode steps use decode params.
- **Projections**: Unified params based on total batch M.
- **M interpolation**: Log-space interpolation between decode params (M ≤ 32) and prefill params (M ≥ 256) for M in (32, 256).
- `predict_step_tp()`: Adds all-reduce overhead (NCCL style).
- `predict_step_pp()`: Adds inter-stage communication for pipeline parallelism.

## Core Roofline Math

`sim/roofline.py` — Pure functions, no classes:

```python
def roofline_time(flops, bytes, F_peak, B_peak, p) -> float
def matmul_time(M, N, K, dtype, F_peak, B_peak, p) -> float
def attention_fused(s_q, s_kv, nh, nh_kv, hd, ...) -> tuple[float, float]
```

When adding new ops, add the FLOPs/bytes formula here and call from executor.

## Communication

`sim/communication.py` — Pure functions for KV transfer modeling:

- `raw_transfer_time()` — raw PCIe/NVLink transfer.
- `effective_xfer_overhead()` — overlaps with prefill compute.
- `transfer_blocks()` — allocates D-side blocks, handles cache hits/misses.

## Metrics

`sim/metrics.py` — `MetricsCollector`:

- `.record(request)` at completion time.
- Percentile methods: `p50_ttft()`, `p90_ttft()`, `p99_ttft()`, `p50_tpot()`, etc.
- SLO is a **binary gate at p90** — both `p90_ttft_ms` and `p90_tpot_ms` must pass.
- Throughput methods: `throughput()` (output tok/s), `total_throughput()`.

## Pipeline Overlap

`sim/pipeline.py` — `ScheduleExecutePipeline` for overlapping CPU schedule time with GPU execute time. Enabled via `async_scheduling` config (auto-enabled for PP > 1).

## Recording

`sim/recorder.py` — `SimRecorder` outputs:
- `meta.json` — run parameters
- `requests.jsonl` — per-request lifecycle
- `timeseries.csv` — time-step snapshots (downsampled by `tick_seconds`)

## Agentic Traces

`sim/trace.py` supports two formats:
- `sharegpt`: one JSONL line per request.
- `agentic`: sessions with sub-request chains (`session_id`, `sub_request_index`, `tool_duration`).
