"""Load sim configuration from YAML with defaults."""

from pathlib import Path
import yaml

HERE = Path(__file__).parent.resolve()
DEFAULT_CONFIG = HERE.parent / "config" / "search.yaml"


def load_config(path=None, model_spec=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Derive from model spec
    if model_spec:
        cfg.setdefault("max_model_len", model_spec.get("max_model_len", 8192))

    # Strip whitespace from model name (common copy-paste error)
    if "model" in cfg and isinstance(cfg["model"], str):
        cfg["model"] = cfg["model"].strip()

    # Defaults for null fields
    sim = cfg.setdefault("simulation", {})
    # max_num_seqs is a user-configured scheduling parameter.
    # It must be set explicitly in search.yaml; no auto-estimation.

    mem_util = sim.get("gpu_memory_utilization", 0.85)
    gpu_name = cfg.get("gpu", "3090")

    # ── Strategy: parse early so tp/pp are available for memory checks ──
    strat = cfg.setdefault("strategy", {})
    strat.setdefault("mode", "both")

    # Node topology: derive total_gpus from nodes × gpus_per_node
    nodes = strat.get("nodes")
    gpus_per_node = strat.get("gpus_per_node")
    if nodes is not None and gpus_per_node is not None:
        strat["total_gpus"] = int(nodes) * int(gpus_per_node)
    else:
        strat.setdefault("total_gpus", 1)

    tp = strat.get("tp_size", 1)
    pp = strat.get("pp_size", 1)

    # Activation memory: null → auto-compute, number → override
    act_conf = sim.get("activation_memory_gb")
    if act_conf is None and model_spec:
        max_tokens = sim.get("max_num_batched_tokens", 8192)
        act_conf = activation_memory_gb(model_spec, max_tokens, tp, pp)
    elif act_conf is None:
        act_conf = 2.0

    usable_vram = total_vram_gb(gpu_name) * mem_util

    # ── Memory validation: fail fast if model doesn't fit ──
    if model_spec:
        weight_per_gpu = model_weight_gb(model_spec, tp=tp, pp=pp)
        required = weight_per_gpu + act_conf
        if required >= usable_vram:
            total_params_b = model_spec.get("total_params_b", 0)
            raise RuntimeError(
                f"Model '{cfg.get('model', 'unknown')}' "
                f"({total_params_b:.1f}B params, {cfg.get('dtype', 'bf16')}) "
                f"does not fit on {gpu_name} ({total_vram_gb(gpu_name)} GB) "
                f"with TP={tp}, PP={pp}.\n"
                f"  Weight per GPU:  {weight_per_gpu:.1f} GB\n"
                f"  Activation:      {act_conf:.1f} GB\n"
                f"  Required:        {required:.1f} GB\n"
                f"  VRAM usable:     {usable_vram:.1f} GB "
                f"({(1 - mem_util) * 100:.0f}% reserved)\n"
                f"  Shortfall:       {required - usable_vram:.1f} GB\n"
                f"\n"
                f"Options: use more GPUs (increase gpus_per_node or nodes),\n"
                f"enable pipeline parallelism (PP), or choose a smaller model."
            )

        if sim.get("kv_cache_memory_gb") is None:
            sim["kv_cache_memory_gb"] = max(1.0, usable_vram - weight_per_gpu - act_conf)
    else:
        if sim.get("kv_cache_memory_gb") is None:
            sim["kv_cache_memory_gb"] = max(1.0, usable_vram - act_conf)

    # Store resolved activation for downstream consumers
    sim["activation_memory_gb"] = act_conf

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
    "3080": 10,
}


def _default_vram(gpu_name):
    """Return 80% of known GPU VRAM in GB."""
    gb = GPU_VRAM.get(gpu_name)
    if gb is None:
        raise KeyError(f"Unknown GPU '{gpu_name}'. Add it to GPU_VRAM in sim/config.py.")
    return int(gb * 0.8)


def total_vram_gb(gpu_name):
    """Return total GPU VRAM in GB."""
    gb = GPU_VRAM.get(gpu_name)
    if gb is None:
        raise KeyError(f"Unknown GPU '{gpu_name}'. Add it to GPU_VRAM in sim/config.py.")
    return gb


def model_weight_gb(model_spec, tp=1, pp=1, ep=1):
    """Model weight size in GB (fp16).

    With EP: expert weights are sharded by EP, dense weights by TP×PP.
    When EP is 1, falls back to the standard formula.

    total_params_b is in billions (e.g. 27.0 = 27B params); the fallback
    formula produces raw counts.
    """
    params_b = model_spec.get("total_params_b", 0)
    if params_b == 0:
        h = model_spec["hidden_dim"]
        nl = model_spec["num_layers"]
        params_raw = 12 * h * h * nl
    else:
        params_raw = params_b * 1e9

    if ep > 1 and model_spec.get("is_moe"):
        num_experts = model_spec.get("num_experts", 0)
        moe_inter = model_spec.get("moe_intermediate_size", 0)
        decoder_sparse_step = model_spec.get("decoder_sparse_step", 1)
        moe_layers = model_spec["num_layers"] // decoder_sparse_step
        h = model_spec["hidden_dim"]

        expert_params = moe_layers * num_experts * (3 * h * moe_inter)
        router_params = moe_layers * (h * num_experts)
        non_expert_params = params_raw - expert_params - router_params

        expert_per_gpu = expert_params / ep
        non_expert_per_gpu = non_expert_params / (tp * pp)
        router_per_gpu = router_params

        weight_gb = (expert_per_gpu + non_expert_per_gpu + router_per_gpu) * 2 / 1e9
    else:
        weight_gb = params_raw * 2 / 1e9 / (tp * pp)

    return weight_gb


def kv_cache_per_token_bytes(model_spec):
    """KV cache bytes per token (all layers, fp16)."""
    nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])
    hd = model_spec["head_dim"]
    nl = model_spec.get("num_attn_layers", model_spec["num_layers"])
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




# ── Per-model .py loading ─────────────────────────────────────────────

from importlib.util import spec_from_file_location, module_from_spec

_MODELS_DIR = Path(__file__).parent.parent / "models"

_SPEC_REQUIRED = [
    "name", "hidden_dim", "vocab_size", "max_model_len",
    "num_layers", "num_heads", "num_kv_heads", "head_dim",
    "intermediate_dim", "total_params_b",
]


def _resolve_model_path(model_name: str) -> Path:
    """Resolve a model name to a .py file in models/."""
    p = Path(model_name)
    if p.exists() and p.suffix == ".py":
        return p
    # Short name (e.g. "qwen3_4b")
    q = _MODELS_DIR / f"{model_name}.py"
    if q.exists():
        return q
    # Full HF ID (e.g. "Qwen/Qwen3-4B") → short name
    short = model_name.split("/")[-1].lower().replace(".", "-").replace("_", "-").replace("-", "_")
    r = _MODELS_DIR / f"{short}.py"
    if r.exists():
        return r
    # Also try stripping first underscore (e.g. "llama_3_8b" → "llama3_8b")
    r2 = _MODELS_DIR / f"{short.replace('_', '', 1)}.py"
    if r2.exists():
        return r2
    # Loose glob match — handle naming variations like llama3_8b vs llama_3_8b
    from glob import glob
    pattern = _MODELS_DIR / f"*{short.split('_')[0]}*{'_'.join(short.split('_')[1:])}.py"
    for cand in sorted(glob(str(pattern))):
        return Path(cand)
    d = _MODELS_DIR / model_name / f"{model_name}.py"
    if d.exists():
        return d
    raise FileNotFoundError(
        f"Model .py not found for '{model_name}'. "
        f"Checked: {p}, {q}, {r}, {r2}, {d}"
    )


def _import_module(model_name: str):
    """Import a per-model .py file and return the module object."""
    path = _resolve_model_path(model_name)
    spec = spec_from_file_location(path.stem, str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_spec(model_name: str) -> dict:
    """Load model SPEC dict from per-model .py file."""
    mod = _import_module(model_name)
    spec = getattr(mod, "SPEC", None)
    if spec is None:
        raise RuntimeError(f"Model file for '{model_name}' must define SPEC dict")
    for key in _SPEC_REQUIRED:
        if key not in spec:
            raise RuntimeError(
                f"SPEC for '{model_name}' missing required field '{key}'"
            )
    return dict(spec)


def load_model_graph(model_name: str):
    """Load model graph by calling the per-model file's build_graph()."""
    spec = load_model_spec(model_name)
    mod = _import_module(model_name)
    build = getattr(mod, "build_graph", None)
    if build is None:
        raise RuntimeError(
            f"Model file for '{model_name}' must define build_graph(spec) function"
        )
    return build(spec)
