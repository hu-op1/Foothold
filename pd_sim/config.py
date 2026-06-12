"""Load pd_sim configuration from YAML with defaults."""

from pathlib import Path
import yaml

HERE = Path(__file__).parent.resolve()
DEFAULT_CONFIG = HERE.parent / "config" / "pd_sim.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Defaults for null fields
    sim = cfg.setdefault("simulation", {})
    if sim.get("kv_cache_memory_gb") is None:
        sim["kv_cache_memory_gb"] = _default_vram(cfg.get("gpu", "3090"))

    strat = cfg.setdefault("strategy", {})
    strat.setdefault("mode", "auto")

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


def valid_tp_sizes(model_spec, gpu_name, kv_cache_gb, num_gpus):
    """Return list of TP sizes that fit in GPU memory.

    Constraints:
    1. num_heads % tp == 0 (attention head divisibility)
    2. (model_weight/tp + kv_cache/tp + activation_overhead) < VRAM
    3. tp <= num_gpus

    For decode-only, TP doesn't help → only TP=1 is useful for D side.
    Returns sorted list of valid TP sizes.
    """
    vram = total_vram_gb(gpu_name)
    weight_gb = model_weight_gb(model_spec)
    activation_overhead = max(1.0, kv_cache_gb * 0.1)  # ~10% overhead

    valid = []
    for tp in [1, 2, 4, 8]:
        if tp > num_gpus:
            continue
        if model_spec["num_heads"] % tp != 0:
            continue
        per_gpu = weight_gb / tp + kv_cache_gb / tp + activation_overhead
        if per_gpu < vram:
            valid.append(tp)
    return valid if valid else [1]
