# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM inference performance toolchain: **GPU characterization → Roofline fitting → Throughput prediction → PD disaggregation simulation**.

```
models/<vendor>/<family>/<model>/config.json    ──→  model_spec dict (auto-discovered)
                                                          │
                              ┌───────────────────────────┴──────────────┐
                              ▼                                           ▼
config/default.yaml  →  bench/   →  bench/results/<gpu>/*.xlsx    perf_predict/predict.py
                              │                           │              │
                              ▼                           │              │
config/default.yaml  →  fit/  →  fit/results/<gpu>.json   │              │
                                    (F_peak, B_peak, p)    │              │
                                         │                 │              │
                                         └──────┬──────────┴──────────────┘
                                                ▼
                                    hw_params dict
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                     perf_predict/predict.py    sim/
                     (throughput pred)          (PD disaggregation sim)
                                                config/search.yaml
```

## Commands

```bash
uv sync                                              # Install deps (PyTorch CUDA 12.8 via tsinghua mirror)

# Benchmarking (stage 1)
uv run python main.py bench                          # Run all benchmarks → bench/results/<gpu>/
uv run python main.py --bench                        # Legacy flag form

# Roofline fitting (stage 2)
uv run python main.py fit                            # Fit from bench/results/<gpu>/ → fit/results/<gpu>.json
uv run python main.py fit --dir <path>               # Override bench results dir
uv run python main.py --fit                          # Legacy flag form

# Throughput prediction (stage 3)
uv run python main.py predict                        # Predict from config/predict.yaml
uv run python main.py predict --config <path>        # Override predict config

# PD disaggregation strategy search (stage 4)
uv run python main.py search                        # Run from config/search.yaml
uv run python main.py search --config <path>        # Override search config

# Tests
uv run python -m pytest test/                        # All tests
uv run python -m pytest test/test_sim.py -v       # PD sim tests verbose
```

## Architecture

### `models/` — Model spec discovery (auto-discovered from HF configs)

Auto-discovers `config.json` files from `models/<vendor>/<family>/<model>/` using HuggingFace format.

- `models/__init__.py` — `model_spec_from_config()` maps HF fields (`hidden_size`, `num_attention_heads`, etc.) to model_spec dict. `load_model_specs()` returns `{"models": [...]}` (YAML file at `config/model_specs.yaml` is a fallback for models without a `config.json` on disk).
- Key handling: GQA (`num_key_value_heads < num_attention_heads` → `num_kv_heads`), Qwen3.5 nested `text_config`, `layer_types` → `attn_layers` for hybrid architectures, `_compute_params()` for exact parameter count from architecture dimensions.
- See [docs/architecture.md](docs/architecture.md) §1 for complete field mapping.

### Stage 1: `bench/` — GPU kernel microbenchmarks

Two benchmark categories, not per-operator. Every shape iterates the Cartesian product from config, guards against OOM via `check_memory()`, warms up, then measures with `CudaTimer`.

- `bench/utils.py` — `CudaTimer` (CUDA event-based timing), `warmup()`, `benchmark()`, `save_xlsx()`, `check_memory()`
- `bench/matmul.py` — `torch.mm` over M×K×N grid. Covers memory-bound (small M) and compute-bound (large M) regimes. Records flops + bytes per run.
- `bench/elementwise.py` — residual_add, rmsnorm, softmax over element count N. Validates bandwidth consistency across ops with different arithmetic complexity.

Output: `bench/results/<gpu>/matmul.xlsx`, `elementwise.xlsx`

### Stage 2: `fit/` — Smooth roofline model fitting

Two backends, set via `fit_all(results, backend=...)`:

- **roofline** (default): `time = ((flops/F_peak)^p + (bytes/B_peak)^p)^(1/p)` via scipy `curve_fit`.
  - `fit/matmul.py` — Splits at M=256 into prefill (large M) and decode (small M). Fits shared F_peak on prefill, then fixes it for decode B_peak + p.
  - `fit/elementwise.py` — Per-op `time = bytes/B_eff + overhead`. B_eff from large-N points, overhead from small-N. Unmeasured ops (swiglu, rope, layernorm) inherit via proxy map.
- **linear** (alternative): `fit/linear.py` packages raw benchmark data into lookup tables for 2D/1D linear interpolation — no parametric assumptions.

Output: `fit/results/<gpu>.json`

### Stage 3: `perf_predict/` — End-to-end throughput prediction

Given fitted roofline params + model architecture specs, predicts prefill/decode latency and tokens/s.

- `perf_predict/predict.py` — Computes per-layer time as sum of projections (roofline), attention (FlashAttention-fused roofline), elementwise ops (B_eff model). Multiplies by num_layers. Prefill: all tokens parallel, O(s²) attention. Decode: 1 token at a time + KV cache, O(s_kv) attention.
- `perf_predict/fitted_params.json` — Default fitted hardware params for quick testing.
- Supports GQA (Q/K/V projection dims, attention HBM traffic, RoPE elementwise) and hybrid architectures (DeltaNet + full attention mix).

### Stage 4: `sim/` — PD disaggregation simulator (event-driven)

Simulates vLLM-style inference serving with colocated and disaggregated (separate prefill/decode GPU pools) configurations.

- `sim/config.py` — Loads `config/search.yaml` with model-aware defaults: `activation_memory_gb()` computes peak activation from model arch × max_batched_tokens (no longer hardcoded). `valid_tp_sizes()` checks head divisibility + memory constraints. GPU VRAM lookup table for known GPU models.
- `sim/engine.py` — `SimulationEngine` with event-driven loop. Clock advances only when all GPUs are truly idle. Two modes: `_run_colocated` (data-parallel via least-loaded routing) and `_run_disaggregated` (P pool → KV transfer → D pool with swap preemption).
- `sim/scheduler.py` — vLLM v1 two-phase scheduler: Phase 1 iterates running queue (decode tokens + chunked prefill, OOM handling via preemption or swap), Phase 2 admits from waiting queue with token budget + prefill threshold.
- `sim/memory.py` — `BlockPool` with PagedAttention block allocation, prefix caching (SHA-256 based), and GPU↔CPU swap support for D-side OOM recovery.
- `sim/executor.py` — `predict_step()` computes single-step GPU time using roofline. Attention uses **per-request params** (prefill params for prefill chunks, decode params for decode steps). Projections use unified params based on total batch M. `predict_step_tp()` adds all-reduce overhead for tensor parallelism.
- `sim/trace.py` — Loads JSONL request traces into `Request` objects.
- `sim/strategy.py` — Grid search over `tp_sizes × max_batched_tokens × prefill_thresholds × pd_ratios × decode_tp_sizes`. Results scored by `throughput × SLO_compliance` and exported to xlsx.
- `sim/report.py` — xlsx export with per-strategy metrics.
- `sim/communication.py` — KV transfer cost modeling with overlap.
- `sim/metrics.py` — TTFT/TPOT/latency distribution tracking.
- `sim/request.py` — `Request` dataclass with lifecycle state.

See [docs/architecture.md](docs/architecture.md) §4 for detailed design rationale, DP architecture, swap mechanics, and prefix caching behavior.

### Entry point: `main.py`

CLI with subcommands: `bench`, `fit`, `search`, `sim`. Also supports legacy `--bench`/`--fit`/`--search` flags for backward compat.

## Key design decisions

- **Roofline over linear**: smooth roofline captures both compute-bound and memory-bound behavior with 3 params (F_peak, B_peak, p) instead of per-operator coefficients.
- **Prefill/decode split**: M<256 vs M>=256 gives separate roofline params since GPUs behave differently at small vs large batch dimensions.
- **Attention per-request param separation**: prefill chunks use prefill roofline params; decode steps use decode params. Projections use unified params (batch matmul, total M decides).
- **FlashAttention model**: `attention_fused()` skips HBM round-trip for S×S score matrix — only reads Q,K,V and writes O.
- **GQA support**: when `num_kv_heads < num_heads`, projections use different output dims for K/V vs Q/O, attention HBM traffic uses nh_kv for K/V, and RoPE separately applies to Q and K.
- **Hybrid architectures**: `attn_layers` field lets models like Qwen3.5 mix full attention layers with DeltaNet (no O(s²) attention).
- **config.json as single source of truth**: model specs auto-discovered from HuggingFace configs in `models/` directory. Parameter counts computed from architecture formula, not hand-maintained. YAML fallback for models without a local `config.json`.
- **Activation memory from model architecture**: computed as `batch_tokens × (2h + 3·inter) × 2 + 0.5 GB` instead of hardcoded 2 GB.
- **DP with independent schedulers**: each DP rank has its own scheduler + block pool + model weights, least-loaded routing, parallel step — matches vLLM DP architecture.
- **D-side swap over recompute**: disaggregated decode GPUs swap victim to CPU when OOM (preserves `num_computed_tokens`), avoiding full recompute of long contexts.

## Configuration

- `config/default.yaml` — Hardware benchmark config (`gpu`, matmul/elementwise grid, dtype, `max_memory_gb`). Used by `bench` and `fit`.
- `config/predict.yaml` — Predict config (`gpu`, `model`, `batch`, `input_len`, `output_len`, `params`). Used by `predict`.
- `config/search.yaml` — Strategy search config (`gpu`, `model`, communication bandwidth, simulation parameters, strategy search space, SLO targets, trace path). Used by `search`.

## Dependencies

- Python ≥ 3.14
- PyTorch ≥ 2.11.0 with CUDA 12.8 (installed via `https://download.pytorch.org/whl/cu128`)
- numpy, scipy, pyyaml, openpyxl, tqdm, pytest
- Package index: tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`)

## External repos (git submodules / cloned)

- `InferSim/` — Independent inference simulator with kernel benchmark/simulation layers
- `LLMServingSim/` — LLM serving simulation framework (Astra-Sim based)
- `SimAI/` — Alibaba AI infrastructure simulator (ns-3, Vidur, Astra-Sim forks)
- `apex_plus/` — GPU profiling and trace analysis tool
- `vllm-0.19.0/` — Local vLLM source checkout for reference

These are standalone projects kept alongside for reference/comparison. They have their own dependencies and build systems. Not part of the main `uv sync` pipeline.
