# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                  # Install dependencies (PyTorch CUDA 12.8 + numpy + pyyaml + openpyxl + tqdm)
uv run python run_all.py                 # Run all benchmarks with config/default.yaml
uv run python run_all.py --config config/custom.yaml  # Run with custom config
uv run python bench/gemm.py              # Run GEMM benchmarks only
uv run python bench/attention.py         # Run attention benchmarks only
uv run python bench/norm.py              # Run norm benchmarks only
uv run python bench/activation.py        # Run activation benchmarks only
```

## Architecture

Four benchmark modules in `bench/` - each follows the same pattern: iterate over the Cartesian product of `batch_sizes x seq_lens x hidden_dims x num_heads`, guard against OOM with `check_memory()`, warm up, then measure iterations with `CudaTimer`. Each module writes its own Excel file.

- `bench/utils.py` - `CudaTimer` (CUDA event-based GPU timing via context manager), `warmup()` / `benchmark()` / `save_xlsx()` / `estimate_memory_gb()` / `check_memory()`. Every benchmark module imports from here.
- `bench/gemm.py` - Q/K/V/O projection and FFN up/gate/down as `torch.mm`. Shape functions in `GEMM_OPS` dict map `(M, h) -> (M, K, N)`. Also includes lm_head benchmark.
- `bench/attention.py` - QK^T matmul (`torch.bmm`), softmax (`F.softmax`), score x V matmul. Uses multi-head layout `[b, n_heads, s, head_dim]` where `head_dim = h // num_heads`.
- `bench/norm.py` - `F.layer_norm` and `F.rms_norm`, both over the last dimension `[h]`.
- `bench/activation.py` - SwiGLU (`F.silu` x up), RoPE rotary embedding, residual add, causal mask.
- `run_all.py` - **Main entry point**. Reads config from YAML, sequentially calls the four bench modules, aggregates results (if needed).

## Memory model

`check_memory()` checks both `torch.cuda.mem_get_info()` free memory and the config's `max_memory_gb` cap. Each benchmark module estimates its own activation footprint with multipliers tuned per operator type (3x for GEMM, 3x for attention, 2x for norm).

- GEMM: ~3x (input activation + weight + output)
- Attention: ~3x QKV + score matrix per head
- Norm: ~2x (input + weights)
- Activation: varies per op (gate/up 3x, RoPE 2x, residual 3x)

The `max_memory_gb` cap exists to protect 8GB GPUs.

## Output format

Results saved with `.xlsx` extension to `results/` directory:

| File | Operators |
|------|-----------|
| `gemm.xlsx` | q_proj, k_proj, v_proj, o_proj, ffn_up, ffn_gate, ffn_down, lm_head |
| `attention.xlsx` | qk_matmul, softmax, score_v_matmul |
| `norm.xlsx` | layernorm, rmsnorm |
| `activation.xlsx` | swiglu, rope, residual_add, causal_mask |

## Configuration

Edit `config/default.yaml` to modify parameter sweep ranges:

- dtype: "float16"
- warmup_iters: 200
- bench_iters: 1000
- max_memory_gb: 6
- batch_sizes: [1, 2, 4, 8]
- seq_lens: [256, 512, 1024, 2048]
- hidden_dims: [512, 1024, 2048]
- num_heads: [12, 24, 32]
- vocab_sizes: [32000, 128256]

Parameters are combined via Cartesian product. Each [b, s, h, nh] tuple generates one benchmark shape. Note: head_dim = hidden_dim / num_heads must be an integer.

## Dependencies

- PyTorch with CUDA 12.8
- Python 3.10+
- numpy, pyyaml, openpyxl, tqdm

Install with:

```bash
uv sync
```
