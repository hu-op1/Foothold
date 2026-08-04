# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM inference performance toolchain: **GPU characterization → Roofline fitting → PD disaggregation simulation**.

```
config/bench.yaml  →  bench/   →  bench/results/<gpu>/*.csv
                                │                           │
                                ▼                           │
config/bench.yaml  →  fit/  →  fit/results/<gpu>.json     │
                                    (F_peak, B_peak, p)     │
                                         │                  │
                                         └──────┬───────────┘
                                                ▼
                                    hw_params dict
                                          │
                                          ▼
                                     sim/
                                     (PD disaggregation sim)
                                     config/search.yaml
                                     config/sim.yaml
```

## Commands

```bash
uv sync --all-extras --no-build-isolation          # Install ALL deps incl. CUDA kernels (flash-attn, fla, causal-conv1d, vllm) + dev
# NOTE: bare `uv sync` installs only base deps and uninstalls extras from an existing venv — always pass --all-extras.
#       --no-build-isolation builds source dists (flash-attn etc.) against the venv's torch instead of an isolated env.

# Benchmarking (stage 1)
uv run python main.py bench                          # Run all benchmarks → bench/results/<gpu>/
uv run python main.py --bench                        # Legacy flag form

# Roofline fitting (stage 2)
uv run python main.py fit                            # Fit from bench/results/<gpu>/ → fit/results/<gpu>.json
uv run python main.py fit --dir <path>               # Override bench results dir
uv run python main.py --fit                          # Legacy flag form

# PD disaggregation simulation (stage 3)
uv run python main.py search                        # Strategy grid search from config/search.yaml
uv run python main.py search --config <path>        # Override search config
uv run python main.py sim                          # Single simulation from config/sim.yaml
uv run python main.py --sim                        # Legacy flag form
uv run python main.py --search                     # Legacy flag form

# Validate (stage 4) — visualize, send to vLLM, or compare
uv run python main.py --validate                     # Visualize sim output (CDF, throughput)
uv run python main.py --validate --vllm              # Send trace to vLLM via OpenAI API
uv run python main.py --validate --compare           # Compare sim vs vLLM output

# Trace generation tools
uv run python tools/generate_conversation_trace.py --model Qwen/Qwen3-8B --sps 1.0   # local trace dir (default)
uv run python tools/generate_conversation_trace.py --dataset <hf_ds> --model Qwen/Qwen3-8B --sps 1.0

# Tests
uv run python -m pytest test/                        # All tests
uv run python -m pytest test/test_sim.py -v       # PD sim tests verbose
```

## Architecture

### Model spec loading

Model architecture specs are defined as per-model `.py` files in `models/` — each exports a `SPEC` dict and a `build_graph(spec)` function. No HuggingFace Hub or local `config.json` files needed.

- `sim/config.py` — `load_model_spec(model_name)` resolves the name to a `models/*.py` file via `_resolve_model_path()` (full HF IDs, short names, or direct paths) and validates against `_SPEC_REQUIRED`. `load_model_graph(model_name)` additionally calls the file's `build_graph(spec)`. `total_params_b` comes from the SPEC dict.
- Key handling: GQA (`num_kv_heads < num_heads`), hybrid architectures (`layer_types` → `num_attn_layers` + `linear_*` sub-configs for Gated DeltaNet layers), MoE fields (`is_moe`, `num_experts`, `moe_intermediate_size`, `decoder_sparse_step`).
- See `.github/instructions/models.instructions.md` for the SPEC/build_graph contract and filename resolution rules.

### Stage 1: `bench/` — GPU kernel microbenchmarks

Benchmarks cover six categories over config-specified grids. Each shape iterates the Cartesian product from config, guards against OOM via `check_memory()`, warms up, then measures with `CudaTimer`.

- `bench/utils.py` — `CudaTimer` (CUDA event-based timing), `warmup()`, `benchmark()`, `save_csv()`, `check_memory()`, `estimate_memory_gb()`, `auto_warmup_iters()`, `load_completed_keys()`, `append_csv_row()`
- `bench/matmul.py` — `torch.mm` over M×K×N grid. Covers memory-bound (small M) and compute-bound (large M) regimes. Records flops + bytes per run.
- `bench/elementwise.py` — residual_add, rmsnorm, softmax, swiglu, rope over element count N
- `bench/flashattn.py` — `F.scaled_dot_product_attention` over (s_q, s_kv) grid
- `bench/memcpy.py` — GPU↔CPU D2H/H2D memory copy bandwidth over byte-size grid. Builds a lookup table for the simulator's communication model.
- `bench/cudagraph.py` — Each op under CUDA Graph replay (eliminates kernel launch overhead)
- `bench/launch_overhead.py` — CPU→GPU kernel dispatch latency via slope method

Output: `bench/results/<gpu>/matmul.csv`, `elementwise.csv`, `flashattn.csv`, `memcpy.csv`, `cudagraph_*.csv`, `launch_overhead.csv`

### Stage 2: `fit/` — Smooth roofline model fitting

Two backends, set via `fit_all(results, backend=...)`:

- **roofline** (default): `time = ((flops/F_peak)^p + (bytes/B_peak)^p)^(1/p)` via scipy `curve_fit`.
  - `fit/matmul.py` — Splits at M=256 into prefill (large M) and decode (small M). Fits shared F_peak on prefill, then fixes it for decode B_peak + p.
  - `fit/elementwise.py` — Per-op `time = bytes/B_eff + overhead`. B_eff from large-N points, overhead from small-N. Unmeasured ops (swiglu, rope, layernorm) inherit via proxy map.
  - `fit/flashattn.py` — FlashAttention-specific roofline params (split at s_q=1). Per-batch B_peak + p.
  - `fit/memcpy.py` — Builds memcpy LUT arrays (`memcpy_d2h_bytes/time_s`, `memcpy_h2d_bytes/time_s`) from benchmark data, replacing the old BW+latency linear model.
  - `fit/cudagraph.py` — CUDA Graph replay params (keys with `_cudagraph` suffix). Same split strategy as eager fits.
  - `fit/launch_overhead.py` — Extracts `kernel_launch_overhead_us` (CPU→GPU dispatch overhead per kernel in µs).

Output: `fit/results/<gpu>.json`

### Stage 3: `sim/` — PD disaggregation simulator (event-driven)

Simulates vLLM-style inference serving with colocated and disaggregated (separate prefill/decode GPU pools) configurations.

- `sim/config.py` — Loads YAML config with model-aware defaults: `load_model_spec()` loads a `SPEC` dict from `models/*.py` (resolved by model name). `activation_memory_gb()` computes peak activation from model arch × max_batched_tokens. `valid_tp_sizes()` checks head divisibility + memory constraints. `pp_cross_node_hops()` computes cross-node boundaries for pipeline stage transitions. GPU VRAM lookup table for known GPU models.
- `sim/engine.py` — `SimulationEngine` with event-driven loop. Clock advances only when all GPUs are truly idle. Two modes: `_run_colocated` (data-parallel via least-loaded routing) and `_run_disaggregated` (P pool → KV transfer → D pool with swap preemption).
- `sim/scheduler.py` — vLLM v1 two-phase scheduler: Phase 1 iterates running queue (decode tokens + chunked prefill, OOM handling via preemption or swap), Phase 2 admits from waiting queue with token budget + prefill threshold.
- `sim/memory.py` — `BlockPool` with PagedAttention block allocation, prefix caching (SHA-256 based), and GPU↔CPU swap support for D-side OOM recovery.
- `sim/executor.py` — `predict_step()` computes single-step GPU time using roofline. Attention uses **per-request params** (prefill params for prefill chunks, decode params for decode steps). Projections use unified params based on total batch M. `predict_step_tp()` adds all-reduce overhead for tensor parallelism. `predict_step_pp()` adds inter-stage communication for pipeline parallelism.
- `sim/roofline.py` — Core roofline math: `roofline_time()`, `matmul_time()`, `attention_fused()`, `attn_projections()`, `ffn_projections()`, `norm_ops()`, `swiglu_op()`, `rope_op()`, `residual_add_ops()`, `fused_residual_norm_ops()`, `elementwise_ops()`, `projections()`, `elem_time()`, `dtype_bytes()`.
- `sim/trace.py` — Loads JSONL request traces (ShareGPT format or agentic format with session chains) into `Request` objects.
- `sim/strategy.py` — Grid search over `tp_sizes × pp_sizes × max_batched_tokens × prefill_thresholds × pd_ratios`. Results scored by `throughput × SLO_compliance` with CSV checkpoint/resume.
- `sim/report.py` — Terminal tables, Matplotlib charts, CSV export, scalability CSV export.
- `sim/communication.py` — KV transfer cost modeling via measured memcpy LUT (linear interpolation on byte-size → transfer-time table), with prefill compute overlap support. Replaces the old BW+latency linear model.
- `sim/metrics.py` — TTFT/TPOT/latency distribution tracking (p50/p90/p99, SLO compliance).
- `sim/request.py` — `Request` dataclass with lifecycle state (`WAITING → RUNNING → FINISHED`). Supports agentic trace sessions.
- `sim/pipeline.py` — `ScheduleExecutePipeline` for overlapping CPU schedule time with GPU execute time.
- `sim/recorder.py` — `SimRecorder` for time-series output (`meta.json`, `requests.jsonl`, `timeseries.csv`) compatible with LLMServingSim format.
- `validate/` module — Sim visualization (`validate/plot.py`), vLLM trace send (`validate/send.py`), and sim-vs-vLLM comparison (`validate/compare.py`).
- `sim/run_single.py` — `run_single()` helper that runs one full simulation with recording.

See `docs/vllm-simulator-gaps.md` for vLLM vs simulator gap analysis and `docs/accuracy-improvements.md` for accuracy regression tracking.

### Entry point: `main.py`

CLI with subcommands: `bench`, `fit`, `search`, `sim`, `validate`. Also supports legacy `--bench`/`--fit`/`--search` flags for backward compat.

## Key design decisions

- **Roofline over linear**: smooth roofline captures both compute-bound and memory-bound behavior with 3 params (F_peak, B_peak, p) instead of per-operator coefficients.
- **Prefill/decode split**: M<256 vs M>=256 gives separate roofline params since GPUs behave differently at small vs large batch dimensions.
- **Attention per-request param separation**: prefill chunks use prefill roofline params; decode steps use decode params. Projections use unified params (batch matmul, total M decides).
- **FlashAttention model**: `attention_fused()` skips HBM round-trip for S×S score matrix — only reads Q,K,V and writes O.
- **GQA support**: when `num_kv_heads < num_heads`, projections use different output dims for K/V vs Q/O, attention HBM traffic uses nh_kv for K/V, and RoPE separately applies to Q and K.
- **Hybrid architectures**: `attn_layers` field lets models like Qwen3.5 mix full attention layers with DeltaNet (no O(s²) attention).
- **Model specs from `models/*.py`**: `sim/config.py:load_model_spec()` loads a `SPEC` dict from a per-model file in `models/` (resolved by model name). Parameter counts come from the SPEC's `total_params_b` or the architecture formula. Any name that resolves to a file works in config YAMLs.
- **Activation memory from model architecture**: computed as `batch_tokens × (2h + 3·inter) × 2 + 0.5 GB` instead of hardcoded 2 GB.
- **DP with independent schedulers**: each DP rank has its own scheduler + block pool + model weights, least-loaded routing, parallel step — matches vLLM DP architecture.
- **Memcpy LUT for communication**: GPU↔CPU transfer times are modeled via measured lookup tables (`memcpy_d2h_bytes/time_s`, `memcpy_h2d_bytes/time_s`) instead of a simple BW+latency linear model. This captures PCIe/NVLink non-linear behavior accurately. Run `--bench` + `--fit` to generate the LUT.
- **Fused residual+norm**: `fused_residual_norm_ops()` models vLLM's fused residual-add + rmsnorm kernel as a single HBM round-trip (read input + residual, write output) instead of two separate ops, reducing memory traffic by ~33%.
- **D-side swap over recompute**: disaggregated decode GPUs swap victim to CPU when OOM (preserves `num_computed_tokens`), avoiding full recompute of long contexts.

## Configuration

- `config/bench.yaml` — Hardware benchmark config (`gpu`, matmul/elementwise/memcpy grid, dtype, `max_memory_gb`). Used by `bench` and `fit`.
- `config/search.yaml` — Strategy search config (`gpu`, `model`, simulation parameters, strategy search space, SLO targets, trace path). Communication is modeled via measured memcpy LUT (generated by `--bench` + `--fit`). Used by `search`.
- `config/sim.yaml` — Single-run simulation config (same structure as search.yaml but scalar strategy values, no grid search). Used by `sim`.
- `config/validate.yaml` — Validate config (`output_dir` for sim visualization, `vllm` block for trace send, `comparison` block for sim-vs-vLLM). Used by `validate`.

## Dependencies

- Python ≥ 3.12
- PyTorch ≥ 2.11.0 with CUDA 12.8 (installed via `https://download.pytorch.org/whl/cu128`)
- numpy, scipy, pyyaml, openpyxl, tqdm, pytest, datasets, teich, matplotlib
- Package index: tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`)

## External repos (reference/ — standalone, not part of this project)

- `LLMServingSim/` — LLM serving simulation framework (Astra-Sim based)
- `SimAI/` — Alibaba AI infrastructure simulator (ns-3, Vidur, Astra-Sim forks)
- `astra-sim/` — Distributed ML system simulator
- `DistServe/` — Disaggregated prefill/decode serving simulator
- `Frontier/` — Frontier LLM inference simulator
- `vidur/` — LLM inference simulator with capacity planning
- `apex_plus/` — GPU profiling and trace analysis tool
- `vllm-0.19.0/` — Local vLLM source checkout for reference
- `PDD/` — Prefill-decode disaggregation reference text

These are standalone projects kept alongside for reference/comparison. They have their own dependencies and build systems. Not part of the main `uv sync` pipeline.
