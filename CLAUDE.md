# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM inference performance toolchain: **GPU characterization → Roofline fitting → Throughput prediction**.

```
config/default.yaml  →  bench/   →  bench/results/<gpu>/*.xlsx    (raw data)
                                    ↓
                                 fit/   →  fit/results/<gpu>.json  (F_peak, B_peak, p, B_eff)
                                             ↓
config/predict.yaml + model_specs.yaml  →  perf_predict/  →  throughput prediction (tokens/s)
```

## Commands

```bash
uv sync                                              # Install deps (PyTorch CUDA 12.8 via tsinghua mirror)

# Benchmarking (stage 1)
uv run python main.py --bench                        # Run all benchmarks → bench/results/<gpu>/

# Roofline fitting (stage 2)
uv run python main.py --fit                          # Fit from bench/results/<gpu>/ → fit/results/<gpu>.json

# Throughput prediction (stage 3)
uv run python main.py --predict                      # Predict from config/predict.yaml
```

## Architecture

### Stage 1: `bench/` — GPU kernel microbenchmarks

Two benchmark categories, not per-operator. Every shape iterates the Cartesian product from config, guards against OOM via `check_memory()`, warms up, then measures with `CudaTimer`.

- `bench/utils.py` — `CudaTimer` (CUDA event-based timing), `warmup()`, `benchmark()`, `save_xlsx()`, `check_memory()`
- `bench/matmul.py` — `torch.mm` over M×K×N grid. Covers memory-bound (small M) and compute-bound (large M) regimes. Records flops + bytes per run.
- `bench/elementwise.py` — residual_add, rmsnorm, softmax over element count N. Validates bandwidth consistency across ops with different arithmetic complexity.

Output: `bench/results/<gpu>/matmul.xlsx`, `elementwise.xlsx`

### Stage 2: `fit/` — Smooth roofline model fitting

Fits a smooth roofline model to benchmark data:

```
time = ( (flops/F_peak)^p + (bytes/B_peak)^p )^(1/p)
```

- `fit/__init__.py` — exports `load_results`, `save_fitted_params`, `roofline_time`, plus `fit_all()` orchestrator
- `fit/utils.py` — `roofline_fit()` via scipy curve_fit, `lstsq_fit()`, `lstsq_log_fit()`, xlsx loader/saver
- `fit/matmul.py` — Splits matmul results at M=256 into **prefill** (large M) and **decode** (small M) regimes. Fits F_peak on prefill first, then fixes F_peak and fits B_peak + p for decode. Produces `{F_peak, B_peak, p}_{prefill,decode}`.
- `fit/elementwise.py` — Per-op effective bandwidth model: `time = bytes / B_eff + overhead`. Fits B_eff from large-N points (overhead negligible), overhead from small-N points. Unmeasured ops inherit via proxy map. Produces `{elem_b_effs, elem_overheads}`.

### Stage 3: `perf_predict/` — End-to-end throughput prediction

Given fitted roofline params + model architecture specs, predicts prefill/decode latency and tokens/s.

- `perf_predict/predict.py` — Computes per-layer time as sum of projections (roofline), attention (FlashAttention-fused roofline), elementwise ops (B_eff model). Multiplies by num_layers. Prefill: all tokens parallel, O(s²) attention. Decode: 1 token at a time + KV cache, O(s_kv) attention.
- `perf_predict/model_specs.yaml` — Model definitions. Key fields: `hidden_dim`, `intermediate_dim`, `num_heads`, `head_dim`, `num_layers`, `vocab_size`, `norm_type`. Optional: `num_kv_heads` (GQA), `attn_layers` (hybrid architectures like Qwen3.5 with DeltaNet + full attention).
- `perf_predict/fitted_params.json` — Default fitted hardware params for quick testing.

### Entry point: `main.py`

Single CLI, all paths auto-derived from config's `gpu` field. Three modes:
- **Benchmark** (`--bench`): runs matmul + elementwise benchmarks → `bench/results/<gpu>/`
- **Fit** (`--fit`): loads xlsx from `bench/results/<gpu>/`, fits roofline model → `fit/results/<gpu>.json`
- **Predict** (`--predict`): reads settings from `config/predict.yaml`, auto-loads fitted params from `fit/results/<gpu>.json`

Override auto-paths with `--fit DIR`, `--predict-config PATH`.

## Key design decisions

- **Roofline over linear**: smooth roofline captures both compute-bound and memory-bound behavior with 3 params (F_peak, B_peak, p) instead of per-operator coefficients.
- **Prefill/decode split**: M<256 vs M>=256 gives separate roofline params since GPUs behave differently at small vs large batch dimensions.
- **FlashAttention model**: `attention_fused()` skips HBM round-trip for S×S score matrix — only reads Q,K,V and writes O.
- **GQA support**: when `num_kv_heads < num_heads`, projections use different output dims for K/V vs Q/O.
- **Hybrid architectures**: `attn_layers` field lets models like Qwen3.5 mix full attention layers with DeltaNet (no O(s²) attention).

## Configuration

## Configuration

- `config/default.yaml` — Hardware benchmark config (`gpu`, matmul/elementwise grid, dtype, `max_memory_gb`). Used by `--bench` and `--fit`.
- `config/predict.yaml` — Predict config (`gpu`, `model`, `batch`, `input_len`, `output_len`, `params`). Used by `--predict`.

## Dependencies

- Python ≥ 3.14
- PyTorch ≥ 2.11.0 with CUDA 12.8 (installed via `https://download.pytorch.org/whl/cu128`)
- numpy, scipy, pyyaml, openpyxl, tqdm
- Package index: tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`)

## External repos (git submodules / cloned)

- `InferSim/` — Independent inference simulator with kernel benchmark/simulation layers
- `LLMServingSim/` — LLM serving simulation framework (Astra-Sim based)
- `SimAI/` — Alibaba AI infrastructure simulator (ns-3, Vidur, Astra-Sim forks)
- `apex_plus/` — GPU profiling and trace analysis tool

These are standalone projects kept alongside for reference/comparison. They have their own dependencies and build systems. Not part of the main `uv sync` pipeline.
