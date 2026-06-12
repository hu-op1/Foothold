"""Load pd_sim configuration from YAML with defaults."""

from pathlib import Path
import yaml

HERE = Path(__file__).parent.resolve()
DEFAULT_CONFIG = HERE.parent / "config" / "pd_sim.yaml"


def load_config(path=None, model_spec=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Defaults for null fields
    sim = cfg.setdefault("simulation", {})
    if sim.get("max_num_seqs") is None and model_spec:
        # Auto-cap max_num_seqs based on available KV memory
        vram = total_vram_gb(cfg.get("gpu", "3090"))
        weight_gb = model_weight_gb(model_spec)
        kv_per_tok = kv_cache_per_token_bytes(model_spec)
        kv_per_seq_gb = (kv_per_tok * cfg.get("max_model_len", 8192)) / 1e9
        kv_budget = max(1, vram - weight_gb - 2)
        sim["max_num_seqs"] = max(1, int(kv_budget / kv_per_seq_gb))

    if sim.get("kv_cache_memory_gb") is None:
        if model_spec:
            weight_gb = model_weight_gb(model_spec)
            sim["kv_cache_memory_gb"] = max(1, int(total_vram_gb(cfg.get("gpu", "3090"))
                                                    - weight_gb - 2))
        else:
            sim["kv_cache_memory_gb"] = _default_vram(cfg.get("gpu", "3090"))

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


def valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, num_gpus,
                   max_model_len=8192, max_num_seqs=256):
    """Return list of TP sizes that fit in GPU memory.

    Constraints:
    1. num_heads % tp == 0 (attention head divisibility)
    2. num_kv_heads % tp == 0 (KV head divisibility for GQA)
    3. model_weight/tp + activation_overhead < VRAM (weights must fit)
    4. KV cache at max context must fit in remaining VRAM:
       kv_per_token × max_model_len × max_num_seqs / tp < VRAM - weight/tp - activation

    Returns sorted list of valid TP sizes.
    """
    vram = total_vram_gb(gpu_name)
    weight_gb = model_weight_gb(model_spec)
    kv_per_tok = kv_cache_per_token_bytes(model_spec)
    nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])
    activation_gb = 2.0

    valid = []
    for tp in [1, 2, 4, 8]:
        if tp > num_gpus:
            continue
        if model_spec["num_heads"] % tp != 0:
            continue
        if nh_kv % tp != 0:
            continue

        weight_per_gpu = weight_gb / tp
        if weight_per_gpu + activation_gb >= vram:
            continue

        # Per-GPU available for KV
        kv_budget_per_gpu = vram - weight_per_gpu - activation_gb
        # KV per seq is split across tp GPUs (head parallelism)
        kv_per_seq_per_gpu_gb = (kv_per_tok * max_model_len) / 1e9 / tp

        # Must fit at least 1 request at full context
        if kv_per_seq_per_gpu_gb > kv_budget_per_gpu:
            continue

        if kv_per_seq_per_gpu_gb * max_num_seqs <= kv_budget_per_gpu:
            valid.append(tp)

    return valid if valid else [1]


def memory_report(model_spec, gpu_name, tp, max_model_len=8192, max_num_seqs=256):
    """Print a memory breakdown for a given config."""
    vram = total_vram_gb(gpu_name)
    weight_gb = model_weight_gb(model_spec)
    kv_per_tok = kv_cache_per_token_bytes(model_spec)
    activation_gb = 2.0

    w_gpu = weight_gb / tp
    kv_seq_gb = (kv_per_tok * max_model_len) / 1e9 / tp  # per-GPU for one seq
    kv_total_gb = kv_seq_gb * max_num_seqs
    used = w_gpu + kv_total_gb + activation_gb
    free = vram - used

    return {
        "vram": vram,
        "weight_per_gpu": w_gpu,
        "kv_per_seq": kv_seq_gb,
        "kv_total": kv_total_gb,
        "activation": activation_gb,
        "used": used,
        "free": free,
        "fits": used < vram,
    }
