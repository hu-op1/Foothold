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
            sim["kv_cache_memory_gb"] = max(1.0, eff_vram - weight_gb - act_conf)
        else:
            sim["kv_cache_memory_gb"] = _default_vram(cfg.get("gpu", "3090")) * mem_util

    # Store resolved activation for downstream consumers
    sim["activation_memory_gb"] = act_conf

    strat = cfg.setdefault("strategy", {})
    strat.setdefault("mode", "both")

    # ── Node topology: derive total_gpus from nodes × gpus_per_node ──
    nodes = strat.get("nodes")
    gpus_per_node = strat.get("gpus_per_node")
    if nodes is not None and gpus_per_node is not None:
        strat["total_gpus"] = int(nodes) * int(gpus_per_node)
    else:
        strat.setdefault("total_gpus", 1)

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


def model_weight_gb(model_spec, tp=1, pp=1, ep=1):
    """Model weight size in GB (fp16).

    With EP: expert weights are sharded by EP, dense weights by TP×PP.
    When EP is 1, falls back to the standard formula.
    """
    params_b = model_spec.get("total_params_b", 0)
    if params_b == 0:
        h = model_spec["hidden_dim"]
        nl = model_spec["num_layers"]
        params_b = 12 * h * h * nl

    if ep > 1 and model_spec.get("is_moe"):
        num_experts = model_spec.get("num_experts", 0)
        moe_inter = model_spec.get("moe_intermediate_size", 0)
        decoder_sparse_step = model_spec.get("decoder_sparse_step", 1)
        moe_layers = model_spec["num_layers"] // decoder_sparse_step
        h = model_spec["hidden_dim"]

        expert_params = moe_layers * num_experts * (3 * h * moe_inter)
        router_params = moe_layers * (h * num_experts)
        non_expert_params = params_b - expert_params - router_params

        expert_per_gpu = expert_params / ep
        non_expert_per_gpu = non_expert_params / (tp * pp)
        router_per_gpu = router_params

        weight_gb = (expert_per_gpu + non_expert_per_gpu + router_per_gpu) * 2 / 1e9
    else:
        weight_gb = params_b * 2 / 1e9 / (tp * pp)

    return weight_gb


def kv_cache_per_token_bytes(model_spec):
    """KV cache bytes per token (all layers, fp16)."""
    nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])
    hd = model_spec["head_dim"]
    nl = model_spec["num_layers"]
    return 2 * nl * nh_kv * hd * 2  # 2 (K+V) × layers × heads × dim × 2 bytes


def activation_memory_gb(model_spec, max_batch_tokens=8192, tp=1, pp=1):
    """Estimate peak activation memory during inference (GB).

    Peak HBM activation = the moment during a layer's FFN block when the
    largest set of intermediate tensors coexist:

        residual → RMSNorm → gate  ─┐
          [h]       [h]     [inter] ├→ SiLU(gate)×up → down
                             up   ──┘    [inter]
                           [inter]

    Simultaneously alive: 2×h + 3×inter elements (fp16).

    With PP > 2, middle pipeline stages (ranks 1..pp-2) hold an extra
    inter-stage hidden state buffer for simultaneous send/receive:
    + max_batch_tokens × h elements (fp16).

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

    # Middle pipeline stages (pp > 2): extra inter-stage hidden state buffer
    # for simultaneous async send to next stage + receive from previous.
    if pp > 2:
        peak_bytes += max_batch_tokens * h * 2  # fp16

    # CUDA context, allocator overhead, workspace buffers (~0.5 GB)
    cuda_overhead = 0.5 * 1024 ** 3

    return (peak_bytes + cuda_overhead) / 1e9 / tp


def pp_cross_node_hops(pp_size, tp_size, gpus_per_node):
    """Number of PP inter-stage transfers that cross node boundaries.

    Each PP stage occupies ``tp_size`` consecutive GPUs.  When the number of
    stages per node is not an integer, some stage transitions must cross node
    boundaries and therefore use the slower inter-node bandwidth.

    Returns 0 when ``gpus_per_node`` is None (flat topology, all intra-node).
    """
    if gpus_per_node is None or pp_size <= 1:
        return 0
    stages_per_node = gpus_per_node // tp_size
    if stages_per_node == 0:
        return pp_size - 1
    cross = 0
    for k in range(pp_size - 1):
        if (k + 1) % stages_per_node == 0:
            cross += 1
    return cross


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
                   pp=1, gpus_per_node=None):
    """Return list of TP sizes that fit in GPU memory.

    Constraints:
    1. num_heads % tp == 0 (attention head divisibility)
    2. num_kv_heads % tp == 0 (KV head divisibility for GQA)
    3. tp <= gpus_per_node (TP groups stay within a single node)
    4. model_weight/(tp×pp) + activation < usable VRAM (weights must fit)
    5. KV cache at expected context must fit in remaining usable VRAM.
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

    max_tp = num_gpus
    if gpus_per_node is not None:
        max_tp = min(max_tp, gpus_per_node)

    valid = []
    for tp in range(1, max_tp + 1):
        if tp * pp > num_gpus:
            continue
        if model_spec["num_heads"] % tp != 0:
            continue
        if nh_kv % tp != 0:
            continue

        # TP divides the activation tensor. PP > 2 adds an extra
        # inter-stage buffer for middle pipeline ranks.
        act_gb = activation_memory_gb(model_spec, max_batch_tokens, tp, pp)
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
    act_gb = activation_memory_gb(model_spec, max_batch_tokens, tp, pp)

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
        print(f"WARNING: failed to load config from HuggingFace for '{model_name}': {e}")
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

    # ── MoE detection ──
    is_moe = False
    num_experts = _first_of(text_cfg, ["num_experts", "num_local_experts"])
    moe_inter = None
    decoder_sparse_step = 1
    num_experts_per_tok = 1
    if num_experts is not None and num_experts > 1:
        is_moe = True
        moe_inter = _first_of(text_cfg, ["moe_intermediate_size"]) or inter // 4
        decoder_sparse_step = getattr(text_cfg, "decoder_sparse_step", 1)
        num_experts_per_tok = _first_of(text_cfg, ["num_experts_per_tok", "num_experts_per_token"]) or 1

    # ── Fail if any required field could not be resolved ──
    if missing:
        print(f"WARNING: cannot determine architecture for '{model_name}' (config.json missing keywords):")
        for field_label, tried in missing:
            print(f"  - {field_label}: tried {tried} — not found")
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
            h=h, inter=inter, nl=nl, nh=nh, nkv=nh_kv, hd=hd, vocab=vocab, tied=tied,
            is_moe=is_moe,
            num_experts=num_experts or 0,
            moe_inter=moe_inter or 0,
            decoder_sparse_step=decoder_sparse_step,
        ),
        "norm_type": norm_type,
    }
    if is_moe:
        spec["is_moe"] = True
        spec["num_experts"] = num_experts
        spec["num_experts_per_tok"] = num_experts_per_tok
        spec["moe_intermediate_size"] = moe_inter
        spec["decoder_sparse_step"] = decoder_sparse_step
        spec["shared_expert_intermediate_size"] = inter
        norm_topk = getattr(text_cfg, "norm_topk_prob", False)
        if norm_topk:
            spec["norm_topk_prob"] = True
        mlp_only = getattr(text_cfg, "mlp_only_layers", None)
        if mlp_only:
            spec["mlp_only_layers"] = mlp_only

    if nh_kv < nh:
        spec["num_kv_heads"] = nh_kv

    # Hybrid architectures (Qwen3.5 DeltaNet)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        attn = sum(1 for t in layer_types if t == "full_attention")
        if attn < nl:
            spec["attn_layers"] = attn

    return spec


def _compute_params_from_attrs(*, h, inter, nl, nh, nkv, hd, vocab, tied,
                                is_moe=False, num_experts=0, moe_inter=0,
                                decoder_sparse_step=1):
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

    total = embed + nl * per_layer + lm_head + final_norm

    # MoE expert weights (per layer with MoE)
    if is_moe and num_experts > 0 and moe_inter > 0:
        moe_layers = nl // decoder_sparse_step
        # Each expert: gate(h×moe_inter) + up(h×moe_inter) + down(moe_inter×h)
        expert_params = num_experts * (3 * h * moe_inter)
        total += moe_layers * expert_params
        # Router: h × num_experts per MoE layer
        total += moe_layers * (h * num_experts)

    return total


# ── YAML-based model loading (v1) ────────────────────────────────────

_MODELS_DIR = Path(__file__).parent.parent / "models"


def _resolve_model_path(model_name: str) -> Path:
    p = Path(model_name)
    if p.exists() and p.suffix in (".yaml", ".yml"):
        return p
    q = _MODELS_DIR / f"{model_name}.yaml"
    if q.exists():
        return q
    r = _MODELS_DIR / model_name / f"{model_name}.yaml"
    if r.exists():
        return r
    raise FileNotFoundError(
        f"Model YAML not found for '{model_name}'. "
        f"Checked: {p}, {q}, {r}"
    )


def load_model_spec_yaml(model_name: str) -> dict:
    """Load model spec from YAML file (v1)."""
    path = _resolve_model_path(model_name)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate_yaml(raw, str(path))
    first_lt = next(iter(raw["layer_types"].values()))

    # Normalize defaults for layer_type fields
    for lt_name, lt in raw["layer_types"].items():
        attn = lt.get("attention", "none")
        if attn in ("standard_attention",):
            if "num_kv_heads" not in lt:
                lt["num_kv_heads"] = lt["num_q_heads"]
            if "rope_dim" not in lt:
                lt["rope_dim"] = lt["head_dim"]

    spec = {
        "name": raw["name"],
        "hidden_dim": raw["hidden_dim"],
        "num_layers": raw["num_layers"],
        "vocab_size": raw["vocab_size"],
        "max_model_len": raw.get("max_model_len", 8192),
        "norm_type": raw.get("norm_type", "rmsnorm"),
        "tie_word_embeddings": raw.get("tie_word_embeddings", False),
        "total_params_b": raw["total_params_b"],
        "layer_types": raw["layer_types"],
        "layers": raw["layers"],
        # Backward-compat fields
        "num_heads": first_lt.get("num_q_heads", first_lt.get("num_qk_heads", 1)),
        "num_kv_heads": first_lt.get("num_kv_heads", first_lt.get("num_q_heads", 1)),
        "head_dim": first_lt["head_dim"],
        "intermediate_dim": first_lt.get("intermediate_dim", raw["hidden_dim"] * 4),
    }
    return spec


def load_model_graph(model_name: str):
    """Parse model YAML, construct ModelGraph with layer builders."""
    spec = load_model_spec_yaml(model_name)

    from sim.graph import ModelGraph
    from sim.layers import ATTENTION_REGISTRY, FFN_REGISTRY
    from sim.layers.common import build_fused_residual_norm
    from sim.layers.head import build_lm_head

    layer_specs = []
    repeat = 1
    for entry in spec["layers"]:
        if "repeat" in entry:
            repeat = entry["repeat"]
            continue
        lt_name = entry["type"]
        count = entry["count"]
        lt = spec["layer_types"][lt_name]

        # Merge hidden_dim into layer_type for builder access
        lt_dict = dict(lt)
        lt_dict["hidden_dim"] = spec["hidden_dim"]

        attn_fn = ATTENTION_REGISTRY.get(lt["attention"])
        ffn_fn = FFN_REGISTRY.get(lt["ffn"])
        if attn_fn is None:
            available = list(ATTENTION_REGISTRY.keys())
            raise RuntimeError(
                f"Unknown attention '{lt['attention']}' for layer type '{lt_name}'. "
                f"Available: {available}"
            )
        if ffn_fn is None:
            available = list(FFN_REGISTRY.keys())
            raise RuntimeError(
                f"Unknown ffn '{lt['ffn']}' for layer type '{lt_name}'. "
                f"Available: {available}"
            )

        def _make(ltype=lt_dict, attn=attn_fn, ffn=ffn_fn, fspec=spec):
            def builder(ctx, hw):
                ops = []
                ops += attn(ctx, ltype, hw)
                ops += build_fused_residual_norm(ctx, fspec, hw)
                ops += ffn(ctx, ltype, hw)
                ops += build_fused_residual_norm(ctx, fspec, hw)
                return ops
            return builder

        layer_specs.append((_make(), count))

    return ModelGraph(
        layer_specs=layer_specs,
        head_builder=lambda ctx, hw: build_lm_head(ctx, spec, hw),
    )


def _validate_yaml(raw: dict, path: str):
    required = ["name", "hidden_dim", "vocab_size", "num_layers",
                "total_params_b", "layer_types", "layers"]
    for key in required:
        if key not in raw or raw[key] is None:
            raise RuntimeError(f"Missing required field '{key}' in {path}")

    for lt_name, lt in raw["layer_types"].items():
        if "attention" not in lt or "ffn" not in lt:
            raise RuntimeError(
                f"layer_type '{lt_name}' in {path} must have 'attention' and 'ffn' fields"
            )
