"""Model spec discovery from models/ directory.

Scans models/<vendor>/<family>/<model>/config.json and builds
model_spec dicts compatible with the existing model_specs.yaml format.

Architecture parameters are read directly from HF config.json; total_params_b
is computed from the architecture formula rather than relying on manual input.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ── config.json → model_spec mapping ──────────────────────────────────────

def _unwrap_config(cfg: dict) -> dict:
    """Handle nested Qwen3.5 text_config; return flat config dict."""
    if "text_config" in cfg:
        return cfg["text_config"]
    return cfg


def _head_dim(cfg: dict, nh: int) -> int:
    """Extract head_dim — explicit or computed."""
    if "head_dim" in cfg:
        return cfg["head_dim"]
    return cfg.get("hidden_size", 4096) // nh


def _norm_type(cfg: dict) -> str:
    """Infer norm type from config."""
    if "rms_norm_eps" in cfg:
        return "rmsnorm"
    return "layernorm"


def _count_attn_layers(cfg: dict, nl: int) -> int | None:
    """Count full-attention layers from Qwen3.5 layer_types; return None for dense models."""
    layer_types = cfg.get("layer_types")
    if not layer_types:
        return None  # dense model — all layers have full attention
    return sum(1 for t in layer_types if t == "full_attention")


def _compute_params(cfg: dict) -> int:
    """Compute total parameter count (fp16) from architecture dimensions.

    Uses the standard Llama/Qwen decoder-only formula.
    For hybrid architectures (Qwen3.5 DeltaNet) this is approximate.
    """
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    nl = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = _head_dim(cfg, nh)
    vocab = cfg["vocab_size"]
    tied = cfg.get("tie_word_embeddings", False)

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


# ── discovery ─────────────────────────────────────────────────────────────

def discover_models(root: Path | None = None) -> list[dict]:
    """Walk models/ directory and return model_spec list.

    Directory convention:  models/<vendor>/<family>/<model>/config.json
    e.g.                    models/Qwen/Qwen3/Qwen3-8B/config.json
                            models/meta-llama/Llama-2-7b-hf/config.json

    The model name is taken from the leaf directory name.
    """
    root = Path(root) if root else HERE
    specs: list[dict] = []

    for config_path in sorted(root.rglob("config.json")):
        model_dir = config_path.parent
        model_name = model_dir.name  # e.g., "Qwen3-8B"

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        spec = model_spec_from_config(raw, name=model_name)
        if spec:
            specs.append(spec)

    return specs


def model_spec_from_config(cfg: dict, name: str = "") -> dict:
    """Convert a HF config.json dict to our model_spec format.

    Handles:
    - Flat configs (Llama-2, Qwen3)
    - Nested text_config (Qwen3.5 multimodal)
    - GQA (num_key_value_heads < num_attention_heads)
    - Hybrid architectures (Qwen3.5 layer_types → attn_layers)
    """
    c = _unwrap_config(cfg)

    h = c.get("hidden_size")
    if not h:
        return {}

    inter = c.get("intermediate_size", h * 4)
    nh = c.get("num_attention_heads", 32)
    nkv = c.get("num_key_value_heads", nh)
    hd = _head_dim(c, nh)
    nl = c.get("num_hidden_layers", 32)
    vocab = c.get("vocab_size", 32000)
    max_len = c.get("max_position_embeddings", 4096)
    norm = _norm_type(c)

    spec: dict = {
        "name": name,
        "total_params_b": _compute_params(c),
        "max_model_len": max_len,
        "hidden_dim": h,
        "intermediate_dim": inter,
        "num_heads": nh,
        "head_dim": hd,
        "num_layers": nl,
        "vocab_size": vocab,
        "norm_type": norm,
    }

    if nkv < nh:
        spec["num_kv_heads"] = nkv

    attn_layers = _count_attn_layers(c, nl)
    if attn_layers is not None and attn_layers < nl:
        spec["attn_layers"] = attn_layers

    return spec


# ── integration helpers ───────────────────────────────────────────────────

def load_model_specs(path=None, yaml_fallback=True):
    """Load model specs: auto-discover from models/ with optional YAML fallback.

    Args:
        path: Path to model_specs.yaml for fallback / override.
        yaml_fallback: If True, merge in any models from YAML not found on disk.

    Returns:
        {"models": [spec_dict, ...]}  — same format as model_specs.yaml.
    """
    specs = discover_models()

    if yaml_fallback:
        # Merge models from YAML that don't have a config.json on disk
        import yaml
        yaml_path = Path(path) if path else Path(__file__).parent.parent / "config" / "model_specs.yaml"
        disk_names = {s["name"] for s in specs}
        try:
            with open(yaml_path, encoding="utf-8") as f:
                yaml_specs = yaml.safe_load(f)
            for m in yaml_specs.get("models", []):
                if m["name"] not in disk_names:
                    specs.append(m)
        except (FileNotFoundError, yaml.YAMLError):
            pass

    return {"models": specs}


def lookup_model(model_sel: str):
    """Find a single model spec by name (exact or case-insensitive match)."""
    all_specs = load_model_specs()["models"]
    for m in all_specs:
        if m["name"] == model_sel:
            return m
    lower = model_sel.lower()
    for m in all_specs:
        if m["name"].lower() == lower:
            return m
    return None
