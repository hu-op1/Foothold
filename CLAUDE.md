# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                              # Install deps (PyTorch CUDA 12.8 via tsinghua mirror)

# Benchmarking (stage 1)
uv run python main.py                                # Run all benchmarks with config/default.yaml
uv run python main.py --config config/custom.yaml    # Run with custom config
uv run python main.py --output results/3090          # Specify output directory

# Operator fitting (stage 2)
uv run python -m fit results/                        # Fit from results/xlsx, print summary
uv run python -m fit results/ --save fitted.json     # Fit and export JSON

# Throughput prediction (stage 3)
uv run python perf_predict/predict.py --list         # List available model specs
uv run python perf_predict/predict.py --model "Llama-2-7B" --input-len 2048 --output-len 512 --batch 4
uv run python perf_predict/predict.py --predict-all --input-len 2048 --output-len 512
```

## Architecture

Three-stage pipeline: **bench → fit → predict**.

### Stage 1: `bench/` — GPU operator microbenchmarks

`main.py` drives four benchmark modules sequentially, each writing its own Excel file plus an `all_operators.xlsx` aggregate. Each module iterates the Cartesian product of `batch_sizes × seq_lens × hidden_dims × num_heads`, guards against OOM via `check_memory()`, warms up, then measures with `CudaTimer`.

- `bench/utils.py` — `CudaTimer` (CUDA event-based GPU timing context manager), `warmup()`, `benchmark()`, `save_xlsx()`, `estimate_memory_gb()`, `check_memory()`. Every benchmark module imports from here.
- `bench/gemm.py` — Q/K/V/O projection and FFN up/gate/down as `torch.mm`. Shape functions map `(M, h) → (M, K, N)`. Also includes lm_head (M×h × h×vocab).
- `bench/attention.py` — QK^T matmul (`torch.bmm`), softmax (`F.softmax`), score×V matmul. Multi-head layout `[b, n_heads, s, head_dim]` where `head_dim = h // num_heads`.
- `bench/norm.py` — `F.layer_norm` and `F.rms_norm` over the last dimension `[h]`.
- `bench/activation.py` — SwiGLU (`F.silu` × up), RoPE rotary embedding, residual add, causal mask.

### Stage 2: `fit/` — Linear model fitting

Fits `time = a·work + b` for each operator using least squares. Exports to `fitted_params.json` (used by stage 3).

- `fit/__main__.py` — CLI: `python -m fit <results_dir> --save <out.json>`
- `fit/utils.py` — `load_results()` from xlsx, `save_fitted_params()`, `lstsq_fit()`.
- `fit/gemm.py`, `fit/attention.py`, `fit/norm.py`, `fit/activation.py` — per-category fit functions.

### Stage 3: `perf_predict/` — End-to-end throughput prediction

Given fitted operator params + model architecture specs, predicts prefill/decode latency and tokens/s.

- `perf_predict/predict.py` — Main predictor. Computes per-layer time as sum of operator times, multiplies by num_layers. Prefill: all tokens parallel, O(s²) attention. Decode: 1 token at a time + KV cache, O(s_kv) attention.
- `perf_predict/model_specs.yaml` — Model definitions (hidden_dim, num_heads, num_layers, vocab_size, etc). Supports hybrid architectures via optional `attn_layers` field (e.g., Qwen3.5 with DeltaNet + full attention).

### Entry point: `main.py`

Supports two modes:
- **Benchmark** (default): loads YAML config, runs all four bench modules, saves xlsx files.
- **Fit** (`--fit DIR`): loads existing xlsx results and runs the fit pipeline.

## Memory model

`check_memory()` checks both `torch.cuda.mem_get_info()` free memory and the config's `max_memory_gb` cap (default 6 GB, protects 8 GB GPUs). Each benchmark estimates activation footprint with multipliers tuned per operator type:

- GEMM: ~3× (input + weight + output)
- Attention: ~3× (QKV + score matrix per head)
- Norm: ~2× (input + weights)
- Activation: varies (gate/up 3×, RoPE 2×, residual 3×)

## Output format

Results saved as `.xlsx` to the output directory (default `results/`):

| File | Operators |
|------|-----------|
| `gemm.xlsx` | q_proj, k_proj, v_proj, o_proj, ffn_up, ffn_gate, ffn_down, lm_head |
| `attention.xlsx` | qk_matmul, softmax, score_v_matmul |
| `norm.xlsx` | layernorm, rmsnorm |
| `activation.xlsx` | swiglu, rope, residual_add, causal_mask |
| `all_operators.xlsx` | All of the above (aggregate, used by fit stage) |

## Configuration

Edit `config/default.yaml` to modify parameter sweep ranges. All lists are combined via Cartesian product. `head_dim = hidden_dim / num_heads` must be an integer.

## Dependencies

- Python ≥ 3.14
- PyTorch ≥ 2.11.0 with CUDA 12.8 (installed via `https://download.pytorch.org/whl/cu128`)
- numpy, pyyaml, openpyxl, tqdm
- Package index: tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`)

```bash
uv sync
```
