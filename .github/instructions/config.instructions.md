---
description: "Use when editing YAML config files in config/. Covers bench, search, and sim config schemas, model spec loading, and common pitfalls."
applyTo: "config/*.yaml"
---

# config/ — Configuration Files

## File Purposes

| File | Stage | Used By |
|------|-------|---------|
| `config/bench.yaml` | 1 — Benchmarking | `bench`, `fit` |
| `config/search.yaml` | 2 — Strategy Search | `search` |
| `config/sim.yaml` | 3 — Single Simulation | `sim` |
| `config/validate.yaml` | 4 — Validation | `validate` |

## Model Spec

Model architecture is **not** loaded from HuggingFace Hub. The `model:` value in a
YAML config is resolved by `sim/config.py:_resolve_model_path()` to a per-model
`.py` file in `models/` (e.g. `meta-llama/Llama-2-7b-hf` → `models/llama_2_7b_hf.py`).
Each file exports a `SPEC` dict + `build_graph(spec)`. See `models.instructions.md`
for the contract and filename resolution rules. Any string that resolves to a file
in `models/` works — full HF IDs, short names, or direct paths.

## bench.yaml Schema

```yaml
gpu: "3090"
dtype: ["float16", "bfloat16"]
matmul:
  M: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
  K: [4096, 8192]
  N: [4096, 8192]
elementwise:
  N: [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456]
  operators: [residual_add, rmsnorm, softmax, swiglu, rope]
flashattn:
  s_q: [1, 4, 16, 64, 256, 1024, 4096, 16384, 32768]
  s_kv: [128, 512, 2048, 8192, 32768, 65536, 131072, 262144]
  batch: [1, 2, 4, 6, 8]
  num_heads: 32
  num_kv_heads: 32
  head_dim: 128
cudagraph:
  matmul: { M: [...], K: [...], N: [...] }
  elementwise: { N: [...], operators: [...] }
  flashattn: { s_q: [...], s_kv: [...], batch: [...] }
launch_overhead:
  n_values: [1, 2, 4, 8, 16, 32, 64, 128]
  trials: 50
  warmup: 5
gateddelta:
  shapes: [...]        # scan shapes
  s_q: [...]
  batch: [...]
  conv_channels: [...]  # depthwise conv1d channels
overwrite: false
max_memory_gb: 24
warmup: "auto"
min_time_ms: 200
max_iters: 10000
calib_iters: 20
```

## validate.yaml Schema

See `config/validate.yaml` for the full example — key sections:

- `sim_dir` (required) — sim output dir.
- Comparison sources: `sim_csv`, `sim_log` (LLMServingSim), `vllm_dir`, `compare_dir` (null excludes that source).
- `colors` (hex per source), `title`, `prefix`.
- `vllm:` block — `embedded` (bool), `endpoint`, `model`, `api_key`, `timeout`, `max_concurrency`, `trace_path`, `trace_format` (`"sharegpt"` | `"agentic"`), `max_requests`, `tokenizer`, `output_dir`, `tick_seconds`, `engine_args` (used only when `embedded: true`).

## search.yaml / sim.yaml Schema

`sim.yaml` mirrors `search.yaml` but with scalar strategy values (no grid search):

```yaml
gpu: "3090"
model: "meta-llama/Llama-2-7b-hf"
dtype: "float16"
communication:
  intra_bw_gb_s: 9.7
  intra_latency_us: 2.0
  inter_bw_gb_s: 9.7
  inter_latency_us: 6.9
  cpu_swap_bw_gb_s: 9.7
simulation:
  block_size: 16
  max_num_seqs: 256
  max_num_batched_tokens: 2048
  kv_cache_memory_gb: null   # auto-computed
  activation_memory_gb: null # auto-computed
  gpu_memory_utilization: 0.89
  enable_prefix_caching: true
  enable_chunked_prefill: true
  use_cudagraph: true
  async_scheduling: true
  scheduler_reserve_full_isl: true
strategy:
  mode: colocated  # or disaggregated
  total_gpus: 2
  tp_size: 2
  pp_size: 1
  pd_ratio: [1, 1]    # P:D GPU count
  p_tp_size: 1
  d_tp_size: 1
slo:
  p90_ttft_ms: 500
  p90_tpot_ms: 50
```

## Pitfalls

- `kv_cache_memory_gb: null` enables auto-compute — omit or set null to use formula.
- `dtype` must match a precision in the bench config (the fit results are keyed by dtype).
- `use_cudagraph: true` requires running `cudagraph` bench + fit first, otherwise falls back to non-graph params.
- `pd_ratio` format: `[num_prefill_gpus, num_decode_gpus]` — not a ratio but absolute counts.
