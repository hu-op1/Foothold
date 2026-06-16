"""Load pd_sim configuration from YAML with defaults."""

from pathlib import Path
import yaml

HERE = Path(__file__).parent.resolve()
DEFAULT_CONFIG = HERE.parent / "config" / "pd_sim.yaml"


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
    # It must be set explicitly in pd_sim.yaml; no auto-estimation.

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
    strat.setdefault("mode", "auto")

    # Use search list first value as simulation default
    search = strat.setdefault("search", {})
    if "max_num_batched_tokens" not in sim:
        tokens = search.get("max_batched_tokens", [8192])
        sim["max_num_batched_tokens"] = tokens[0] if tokens else 8192
    if "long_prefill_token_threshold" not in sim:
        thrs = search.get("prefill_thresholds", [1024])
        sim["long_prefill_token_threshold"] = thrs[0] if thrs else 1024

    slo = cfg.setdefault("slo", {})
    for k, v in [("ttft_ms", 500), ("tpot_ms", 50), ("p99_latency_ms", 2000)]:
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

    Set ``activation_memory_gb`` in pd_sim.yaml to override with a fixed value.
    """
    h = model_spec["hidden_dim"]
    inter = model_spec.get("intermediate_dim", h * 4)

    # Peak FFN-block simultaneously-alive elements per token
    per_token_elems = 2 * h + 3 * inter  # residual, norm, gate, up, silu_result
    peak_bytes = max_batch_tokens * per_token_elems * 2  # fp16

    # CUDA context, allocator overhead, workspace buffers (~0.5 GB)
    cuda_overhead = 0.5 * 1024 ** 3

    return (peak_bytes + cuda_overhead) / 1e9 / tp


def valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, num_gpus,
                   max_model_len=8192, max_num_seqs=256,
                   gpu_memory_utilization=0.85,
                   max_batch_tokens=8192):
    """Return list of TP sizes that fit in GPU memory.

    Constraints:
    1. num_heads % tp == 0 (attention head divisibility)
    2. num_kv_heads % tp == 0 (KV head divisibility for GQA)
    3. model_weight/tp + activation < usable VRAM (weights must fit)
    4. KV cache at expected context must fit in remaining usable VRAM.
       Uses estimated average seq length (not max_model_len) since the
       block pool (PagedAttention) handles dynamic allocation at runtime.

    Activation memory is computed from model architecture × max_batch_tokens,
    not a fixed constant.

    Returns sorted list of valid TP sizes.
    """
    usable_vram = total_vram_gb(gpu_name) * gpu_memory_utilization
    weight_gb = model_weight_gb(model_spec)
    kv_per_tok = kv_cache_per_token_bytes(model_spec)
    nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])

    valid = []
    for tp in [1, 2, 4, 8]:
        if tp > num_gpus:
            continue
        if model_spec["num_heads"] % tp != 0:
            continue
        if nh_kv % tp != 0:
            continue

        act_gb = activation_memory_gb(model_spec, max_batch_tokens, tp)
        weight_per_gpu = weight_gb / tp
        if weight_per_gpu + act_gb >= usable_vram:
            continue

        # Must fit at least 1 request at full context
        kv_budget_per_gpu = usable_vram - weight_per_gpu - act_gb
        kv_per_seq_per_gpu_gb = (kv_per_tok * max_model_len) / 1e9 / tp
        if kv_per_seq_per_gpu_gb <= kv_budget_per_gpu:
            valid.append(tp)

    return valid if valid else [1]


def memory_report(model_spec, gpu_name, tp, max_model_len=8192, max_num_seqs=256,
                  gpu_memory_utilization=0.85, max_batch_tokens=8192):
    """Print a memory breakdown for a given config."""
    usable_vram = total_vram_gb(gpu_name) * gpu_memory_utilization
    weight_gb = model_weight_gb(model_spec)
    kv_per_tok = kv_cache_per_token_bytes(model_spec)
    act_gb = activation_memory_gb(model_spec, max_batch_tokens, tp)

    w_gpu = weight_gb / tp
    kv_seq_gb = (kv_per_tok * max_model_len) / 1e9 / tp  # per-GPU for one seq
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
