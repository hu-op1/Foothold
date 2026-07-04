"""Load sim configuration from YAML with defaults."""

from pathlib import Path
import yaml
from transformers import AutoConfig

HERE = Path(__file__).parent.resolve()
DEFAULT_CONFIG = HERE.parent / "config" / "search.yaml"


def load_config(path=None, model_spec=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Derive from model spec
    if model_spec:
        cfg.setdefault("max_model_len", model_spec.get("max_model_len", 8192))

    # Defaults for null fields
    sim = cfg.setdefault("simulation", {})
    # max_num_seqs is a user-configured scheduling parameter.
    # It must be set explicitly in search.yaml; no auto-estimation.

    mem_util = sim.get("gpu_memory_utilization", 0.85)
    # Activation memory: null → auto-compute, number → override
    act_conf = sim.get("activation_memory_gb")
    if act_conf is None and model_spec:
        max_tokens = sim.get("max_num_batched_tokens", 8192)
        act_conf = activation_memory_gb(model_spec, max_tokens)
    elif act_conf is None:
        act_conf = 2.0

    if sim.get("kv_cache_memory_gb") is None:
        if model_spec:
            weight_gb = model_weight_gb(model_spec)
            eff_vram = total_vram_gb(cfg.get("gpu", "3090")) * mem_util
            sim["kv_cache_memory_gb"] = max(1, int(eff_vram - weight_gb - act_conf))
        else:
            sim["kv_cache_memory_gb"] = _default_vram(cfg.get("gpu", "3090")) * mem_util

    # Store resolved activation for downstream consumers
    sim["activation_memory_gb"] = act_conf

    strat = cfg.setdefault("strategy", {})
    strat.setdefault("mode", "both")

    # Use search list first value as simulation default
    search = strat.setdefault("search", {})
    if "max_num_batched_tokens" not in sim:
        tokens = search.get("max_batched_tokens", [8192])
        sim["max_num_batched_tokens"] = tokens[0] if tokens else 8192
    if "long_prefill_token_threshold" not in sim:
        thrs = search.get("prefill_thresholds", [1024])
        sim["long_prefill_token_threshold"] = thrs[0] if thrs else 1024

    slo = cfg.setdefault("slo", {})
    for k, v in [("p90_ttft_ms", 500), ("p90_tpot_ms", 50)]:
        if slo.get(k) is None:
            slo[k] = v

    return cfg


GPU_VRAM = {
    "3090": 24, "4090": 24, "A100": 80, "A100-80GB": 80,
    "H100": 80, "H200": 141, "A6000": 48, "L40S": 48,
}


def _default_vram(gpu_name):
    """Return 80% of known GPU VRAM in GB."""
    gb = GPU_VRAM.get(gpu_name, 24)
    return int(gb * 0.8)


def total_vram_gb(gpu_name):
    """Return total GPU VRAM in GB."""
    return GPU_VRAM.get(gpu_name, 24)


def model_weight_gb(model_spec):
    """Model weight size in GB (fp16)."""
    params_b = model_spec.get("total_params_b", 0)
    if params_b == 0:
        # Rough estimate: hidden² × layers × 12
        h = model_spec["hidden_dim"]
        nl = model_spec["num_layers"]
        params_b = 12 * h * h * nl
    return params_b * 2 / 1e9  # fp16 = 2 bytes


def kv_cache_per_token_bytes(model_spec):
    """KV cache bytes per token (all layers, fp16)."""
    nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])
    hd = model_spec["head_dim"]
    nl = model_spec["num_layers"]
    return 2 * nl * nh_kv * hd * 2  # 2 (K+V) × layers × heads × dim × 2 bytes


def activation_memory_gb(model_spec, max_batch_tokens=8192, tp=1):
    """Estimate peak activation memory during inference (GB).

    Peak HBM activation = the moment during a layer's FFN block when the
    largest set of intermediate tensors coexist:

        residual → RMSNorm → gate  ─┐
          [h]       [h]     [inter] ├→ SiLU(gate)×up → down
                             up   ──┘    [inter]
                           [inter]

    Simultaneously alive: 2×h + 3×inter elements (fp16).

    Plus a fixed ~0.5 GB for CUDA context / allocator overhead.

    Key assumptions (matching FlashAttention / vLLM):
      - Intermediates are freed per-layer; only one layer peaks at a time.
      - FlashAttention keeps S×S in SRAM — does NOT consume HBM.
      - The attention-block peak (input+Q+K+V+attn_out) is smaller than FFN
        peak for all practical models (intermediate > hidden).

    Set ``activation_memory_gb`` in search.yaml to override with a fixed value.
    """
    h = model_spec["hidden_dim"]
    inter = model_spec.get("intermediate_dim", h * 4)

    # Peak FFN-block simultaneously-alive elements per token
    per_token_elems = 2 * h + 3 * inter  # residual, norm, gate, up, silu_result
    peak_bytes = max_batch_tokens * per_token_elems * 2  # fp16

    # CUDA context, allocator overhead, workspace buffers (~0.5 GB)
    cuda_overhead = 0.5 * 1024 ** 3

    return (peak_bytes + cuda_overhead) / 1e9 / tp


def valid_pp_sizes(model_spec, num_gpus):
    """Return list of PP sizes that are valid for this model.

    Constraints:
    1. num_layers % pp == 0 (layer divisibility)
    2. pp <= num_gpus

    Returns sorted list of all valid PP sizes ≤ num_gpus.
    """
    nl = model_spec["num_layers"]
    valid = [pp for pp in range(1, num_gpus + 1) if nl % pp == 0]
    return valid if valid else [1]


def valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, num_gpus,
                   max_model_len=8192, max_num_seqs=256,
                   gpu_memory_utilization=0.85,
                   max_batch_tokens=8192,
                   pp=1):
    """Return list of TP sizes that fit in GPU memory.

    Constraints:
    1. num_heads % tp == 0 (attention head divisibility)
    2. num_kv_heads % tp == 0 (KV head divisibility for GQA)
    3. model_weight/(tp×pp) + activation < usable VRAM (weights must fit)
    4. KV cache at expected context must fit in remaining usable VRAM.
       Uses estimated average seq length (not max_model_len) since the
       block pool (PagedAttention) handles dynamic allocation at runtime.

    When pp > 1, each GPU only holds weights for nl/pp layers and KV cache
    for those layers only, reducing per-GPU memory pressure.

    Activation memory is computed from model architecture × max_batch_tokens,
    not a fixed constant.

    Returns sorted list of valid TP sizes.
    """
    usable_vram = total_vram_gb(gpu_name) * gpu_memory_utilization
    weight_gb = model_weight_gb(model_spec)
    kv_per_tok = kv_cache_per_token_bytes(model_spec)
    nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])

    valid = []
    for tp in range(1, num_gpus + 1):
        if tp * pp > num_gpus:
            continue
        if model_spec["num_heads"] % tp != 0:
            continue
        if nh_kv % tp != 0:
            continue

        # Activation per GPU is the same regardless of PP (peak per-layer).
        # TP divides the activation tensor.
        act_gb = activation_memory_gb(model_spec, max_batch_tokens, tp)
        # PP splits layers → each GPU holds 1/(tp×pp) of weights
        weight_per_gpu = weight_gb / (tp * pp)
        if weight_per_gpu + act_gb >= usable_vram:
            continue

        # KV cache: PP splits layers, TP splits heads → 1/(pp×tp) per GPU
        kv_budget_per_gpu = usable_vram - weight_per_gpu - act_gb
        kv_per_seq_per_gpu_gb = (kv_per_tok * max_model_len) / 1e9 / (pp * tp)
        if kv_per_seq_per_gpu_gb <= kv_budget_per_gpu:
            valid.append(tp)

    return valid if valid else [1]


def memory_report(model_spec, gpu_name, tp, max_model_len=8192, max_num_seqs=256,
                  gpu_memory_utilization=0.85, max_batch_tokens=8192,
                  pp=1):
    """Print a memory breakdown for a given config.

    When pp > 1, weights and KV cache are split across pipeline stages.
    """
    usable_vram = total_vram_gb(gpu_name) * gpu_memory_utilization
    weight_gb = model_weight_gb(model_spec)
    kv_per_tok = kv_cache_per_token_bytes(model_spec)
    act_gb = activation_memory_gb(model_spec, max_batch_tokens, tp)

    w_gpu = weight_gb / (tp * pp)
    kv_seq_gb = (kv_per_tok * max_model_len) / 1e9 / (pp * tp)  # per-GPU for one seq
    kv_total_gb = kv_seq_gb * max_num_seqs
    used = w_gpu + kv_total_gb + act_gb
    free = usable_vram - used

    return {
        "vram": usable_vram,
        "weight_per_gpu": w_gpu,
        "kv_per_seq": kv_seq_gb,
        "kv_total": kv_total_gb,
        "activation": act_gb,
        "used": used,
        "free": free,
        "fits": used < usable_vram,
    }


# ── Model spec loading via transformers ───────────────────────────────

def _first_of(obj, names: list[str]):
    """Return the first attribute of *obj* that exists and is not None."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return None


def load_model_spec(model_name: str) -> dict | None:
    """Load a model spec dict from HuggingFace Hub via AutoConfig.

    Handles common naming variations across model families:

    ==================== =================================================
    Field                Fallback chain
    ==================== =================================================
    hidden_size          cfg.hidden_size (standard across all HF models)
    num_attention_heads  cfg.num_attention_heads
    num_key_value_heads  cfg.num_key_value_heads → num_attention_heads
    head_dim             cfg.head_dim → hidden_size // num_attention_heads
    num_hidden_layers    cfg.num_hidden_layers (also checks n_layer)
    intermediate_size    cfg.intermediate_size → hidden_size × 4
    vocab_size           cfg.vocab_size
    max_position_emb     cfg.max_position_embeddings → max_sequence_length
                         → n_positions → 4096
    norm_type            rms_norm_eps present → "rmsnorm", else "layernorm"
    layer_types          cfg.layer_types (Qwen3.5 hybrid only)
    text_config          Unwrapped for multimodal models (Qwen3.5, etc.)
    ==================== =================================================

    Args:
        model_name: Full HF model ID (e.g. 'Qwen/Qwen3-8B',
                    'meta-llama/Llama-2-7b-hf').

    Returns:
        model_spec dict, or None if config can't be loaded.
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR loading config for '{model_name}': {e}")
        return None

    # Unwrap nested text_config (Qwen3.5 multimodal, etc.)
    text_cfg = config
    if hasattr(config, "text_config") and config.text_config is not None:
        text_cfg = config.text_config

    # ── Resolve each field with explicit fallback chains ──
    # Missing fields are collected; if any remain unresolved after exhausting
    # all known aliases, the function fails rather than using a guess.
    missing = []  # list of (field_label, tried_chain)

    # hidden_size — standard across all HF decoder models
    h = _first_of(text_cfg, ["hidden_size", "d_model", "n_embd"])
    if h is None:
        missing.append(("hidden_size", "hidden_size / d_model / n_embd"))

    # num_attention_heads
    nh = _first_of(text_cfg, ["num_attention_heads", "n_head"])
    if nh is None:
        missing.append(("num_attention_heads", "num_attention_heads / n_head"))
        nh = 1  # dummy to allow head_dim derivation below

    # num_key_value_heads — defaults to num_attention_heads (no GQA, always correct)
    nh_kv = _first_of(text_cfg, ["num_key_value_heads", "n_kv_head"])
    if nh_kv is None:
        nh_kv = nh

    # num_hidden_layers
    nl = _first_of(text_cfg, ["num_hidden_layers", "n_layer"])
    if nl is None:
        missing.append(("num_hidden_layers", "num_hidden_layers / n_layer"))

    # head_dim — derived when not explicit (always possible if h and nh are known)
    hd = _first_of(text_cfg, ["head_dim"])
    if hd is None and h is not None and nh is not None:
        hd = h // nh

    # intermediate_size
    inter = _first_of(text_cfg, ["intermediate_size", "ffn_dim"])
    if inter is None:
        missing.append(("intermediate_size", "intermediate_size / ffn_dim"))

    # vocab_size
    vocab = _first_of(text_cfg, ["vocab_size", "padded_vocab_size"])
    if vocab is None:
        missing.append(("vocab_size", "vocab_size / padded_vocab_size"))

    # max_position_embeddings
    max_len = _first_of(text_cfg, [
        "max_position_embeddings", "max_sequence_length",
        "n_positions", "max_seq_len",
    ])
    if max_len is None:
        missing.append(("max_position_embeddings",
                        "max_position_embeddings / max_sequence_length / n_positions / max_seq_len"))

    # norm_type — safe to default to layernorm (doesn't affect FLOPs, only which
    # elementwise op name is selected; most modern models use rmsnorm anyway)
    norm_type = "layernorm"
    if getattr(text_cfg, "rms_norm_eps", None) is not None:
        norm_type = "rmsnorm"
    elif getattr(text_cfg, "norm_type", "").lower() == "rms_norm":
        norm_type = "rmsnorm"

    # tie_word_embeddings — safe to default to False
    tied = getattr(text_cfg, "tie_word_embeddings", False)

    # ── Fail if any required field could not be resolved ──
    if missing:
        print(f"ERROR: cannot determine architecture for '{model_name}':")
        for field_label, tried in missing:
            print(f"  - {field_label}: tried {tried} — none found on config object")
        print(f"  Config class: {type(config).__name__}")
        print(f"  Available attributes: {sorted(k for k in dir(text_cfg) if not k.startswith('_'))}")
        return None

    spec: dict = {
        "name": model_name,
        "hidden_dim": h,
        "num_heads": nh,
        "head_dim": hd,
        "num_layers": nl,
        "vocab_size": vocab,
        "intermediate_dim": inter,
        "max_model_len": max_len,
        "total_params_b": _compute_params_from_attrs(
            h=h, inter=inter, nl=nl, nh=nh, nkv=nh_kv, hd=hd, vocab=vocab, tied=tied
        ),
        "norm_type": norm_type,
    }

    if nh_kv < nh:
        spec["num_kv_heads"] = nh_kv

    # Hybrid architectures (Qwen3.5 DeltaNet)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        attn = sum(1 for t in layer_types if t == "full_attention")
        if attn < nl:
            spec["attn_layers"] = attn

    return spec


def _compute_params_from_attrs(*, h, inter, nl, nh, nkv, hd, vocab, tied):
    """Compute total parameter count (fp16) from architecture dimensions.

    Uses the standard Llama/Qwen decoder-only formula.
    For hybrid architectures (Qwen3.5 DeltaNet) this is approximate.
    """
    # Per-layer weights (no biases):
    #   Q proj: h × (nh × hd)
    #   K proj: h × (nkv × hd)
    #   V proj: h × (nkv × hd)
    #   O proj: (nh × hd) × h
    #   Gate:   h × inter
    #   Up:     h × inter
    #   Down:   inter × h
    #   2× RMSNorm: 2 × h
    per_layer = (
        2 * h * nh * hd
        + 2 * h * nkv * hd
        + 3 * h * inter
        + 2 * h
    )

    embed = vocab * h
    lm_head = 0 if tied else vocab * h
    final_norm = h

    return embed + nl * per_layer + lm_head + final_norm
