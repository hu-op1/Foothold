---
description: "Use when adding or modifying per-model computation graph definitions in models/. Covers the SPEC dict contract, build_graph() contract, model name resolution, and hybrid architecture conventions."
applyTo: "models/*.py"
---

# models/ — Per-Model Computation Graph Definitions

Model specs are defined as `.py` files in `models/`, one per model — **not** loaded from HuggingFace Hub. Each file must export a `SPEC` dict and a `build_graph(spec)` function.

## Loading / Resolution

`sim/config.py` resolves the model name from a config YAML to a `.py` file via `_resolve_model_path()`:

1. Direct `.py` path → used as-is.
2. Short name (`qwen3_4b`) → `models/qwen3_4b.py`.
3. Full HF ID (`Qwen/Qwen3-4B`) → last path segment, lowercased, `.`/`-` → `_` (`qwen3_4b.py`).
4. Stripped-first-underscore fallback (`llama_3_8b` → `llama3_8b.py`).
5. Loose glob match on the first name token.
6. `models/<name>/<name>.py` subdirectory.

`load_model_spec()` validates against `_SPEC_REQUIRED` (see `sim/config.py`) and returns a copy of the dict. `load_model_graph()` additionally calls `build_graph(spec)`.

## SPEC dict contract

Required fields (validated by `_SPEC_REQUIRED` in `sim/config.py`):

```python
SPEC = {
    "name": "meta-llama/Llama-2-7b-hf",  # full HF-style ID used in YAML configs
    "hidden_dim": 4096,
    "vocab_size": 32000,
    "max_model_len": 4096,
    "norm_type": "rmsnorm",
    "tie_word_embeddings": False,
    "num_layers": 32,
    "num_attn_layers": 32,               # attention layers (≠ num_layers for hybrid)
    "num_heads": 32,
    "num_q_heads": 32,
    "num_kv_heads": 32,                  # GQA when < num_heads
    "head_dim": 128,
    "rope_dim": 128,
    "intermediate_dim": 11008,
    "activation": "swiglu",
    "total_params_b": 6.74,
}
```

MoE models add: `is_moe`, `num_experts`, `moe_intermediate_size`, `decoder_sparse_step` (consumed by `model_weight_gb()` in `sim/config.py`).

## build_graph(spec) contract

Returns a `ModelGraph` assembled with builders from `sim/layers/`:

```python
def build_graph(spec):
    from sim.graph import ModelGraph
    from sim.layers import build_qkv_proj, build_flash_attn, ...
    graph = ModelGraph(...)
    # append layer ops per layer index, in execution order
    return graph
```

- Use `sim/layers/common.py` builders (qkv_proj, o_proj, flash_attn, rope, gate_up, swiglu, fused_residual_norm, linear_scan, moe/expert, all_to_all) and `sim/layers/head.py` (lm_head).
- Per-layer execution order: QKV → RoPE → FlashAttn → O → fused_residual_norm → gate_up → SwiGLU → down → fused_residual_norm.

## Hybrid Architectures

Mixed full-attention / Gated DeltaNet models (e.g. `qwen3_5_4b.py`) use `layer_types` + `linear_*` sub-configs (Gated DeltaNet, conv1d, QK-norm, attn gate). The `ls_*`-prefixed roofline params (see `fit/gateddelta.py`) are selected automatically by `StepContext.precompute()` in `sim/graph.py`.

## Conventions

- One file per model; the filename is the resolution key (see above).
- Keep `SPEC` values from the model card / HF config — do not hand-tune.
- `total_params_b` must match the architecture parameter formula (see CLAUDE.md §0).
