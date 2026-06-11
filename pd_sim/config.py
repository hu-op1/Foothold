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


def _default_vram(gpu_name):
    """Return 80% of known GPU VRAM in GB."""
    vram = {
        "3090": 24, "4090": 24, "A100": 80, "A100-80GB": 80,
        "H100": 80, "H200": 141, "A6000": 48, "L40S": 48,
    }
    gb = vram.get(gpu_name, 24)
    return int(gb * 0.8)
