"""
Predict model inference throughput from architecture specs + hardware roofline params.

Usage:
    uv run python perf_predict/predict.py --list
    uv run python perf_predict/predict.py --model "Llama-3.1-8B" --input-len 2048 --output-len 512 --batch 4
    uv run python perf_predict/predict.py --predict-all --input-len 2048 --output-len 512
"""

import yaml
import json
import argparse
from pathlib import Path

HERE = Path(__file__).parent.resolve()
DEFAULT_PARAMS = HERE / "fitted_params.json"
DEFAULT_SPECS = HERE / "model_specs.yaml"
DTYPE_BYTES = 2  # fp16


def load_model_specs(path=None):
    path = path or str(DEFAULT_SPECS)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hw_params(path=None):
    path = path or str(DEFAULT_PARAMS)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── roofline ────────────────────────────────────────────────────────────

def roofline_time(flops, bytes_moved, F_peak, B_peak, p):
    c = flops / F_peak
    m = bytes_moved / B_peak
    return (c ** p + m ** p) ** (1 / p)


def matmul_time(M, K, N, F, B, p):
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * DTYPE_BYTES
    return roofline_time(flops, bytes_moved, F, B, p)


# Bytes-per-element factors for elementwise ops
ELEM_BYTES = {
    "residual_add": 3,
    "swiglu": 3,
    "rope": 4,
    "softmax": 6,
    "layernorm": 5,
    "rmsnorm": 4,
    "causal_mask": 3,
}


def elem_time(op_name, N, B_peak):
    factor = ELEM_BYTES.get(op_name, 3)
    return (N * factor * DTYPE_BYTES) / B_peak


# ── layer ops ───────────────────────────────────────────────────────────

def projections(M, h, inter, F, B, p):
    """Q/K/V/O (4×) + FFN up/gate (2×) + FFN down."""
    t = 4 * matmul_time(M, h, h, F, B, p)
    t += 2 * matmul_time(M, h, inter, F, B, p)
    t += matmul_time(M, inter, h, F, B, p)
    return t


def attention_matmuls(b, nh, s_q, s_kv, hd, F, B, p):
    """QK^T + score×V matmuls."""
    t = matmul_time(b * nh * s_q, hd, s_kv, F, B, p)
    t += matmul_time(b * nh * s_q, s_kv, hd, F, B, p)
    return t


def elementwise_ops(b, s, h, inter, nh, hd, norm_type, B_peak):
    """All elementwise ops per layer."""
    t = 0.0
    N = b * s * h
    # 2 norm ops (attention + ffn)
    t += 2 * elem_time(norm_type, N, B_peak)
    # SwiGLU
    t += elem_time("swiglu", b * s * inter, B_peak)
    # RoPE (on Q)
    t += elem_time("rope", b * nh * s * hd, B_peak)
    # 2 residual adds
    t += 2 * elem_time("residual_add", N, B_peak)
    return t


# ── prediction ──────────────────────────────────────────────────────────

def predict(model, batch, input_len, output_len, hw_params):
    F = hw_params["F_peak"]
    B_peak = hw_params["B_peak"]
    p = hw_params["p"]

    nl = model["num_layers"]
    na = model.get("attn_layers", nl)
    nd = nl - na

    h = model["hidden_dim"]
    inter = model.get("intermediate_dim", h * 4)
    nh = model["num_heads"]
    hd = model["head_dim"]
    vs = model["vocab_size"]
    norm_type = model.get("norm_type", "rmsnorm")

    # ---- prefill ----
    M = batch * input_len
    s = input_len

    p_proj = projections(M, h, inter, F, B_peak, p)
    p_elem = elementwise_ops(batch, s, h, inter, nh, hd, norm_type, B_peak)

    # Full attention: matmuls + softmax
    p_attn_matmul = attention_matmuls(batch, nh, s, s, hd, F, B_peak, p)
    p_attn_softmax = elem_time("softmax", batch * nh * s * s, B_peak)

    p_layer_full = p_proj + p_elem + p_attn_matmul + p_attn_softmax
    p_layer_delta = p_proj + p_elem

    p_total = na * p_layer_full + nd * p_layer_delta

    # lm_head (once)
    p_total += matmul_time(M, h, vs, F, B_peak, p)
    # causal_mask (once, if any full-attn layers)
    if na > 0:
        p_total += elem_time("causal_mask", batch * nh * s * s, B_peak)

    prefill_time_s = p_total

    # ---- decode (one step) ----
    M = batch * 1
    s_kv = input_len

    d_proj = projections(M, h, inter, F, B_peak, p)
    d_elem = elementwise_ops(batch, 1, h, inter, nh, hd, norm_type, B_peak)

    d_attn_matmul = attention_matmuls(batch, nh, 1, s_kv, hd, F, B_peak, p)
    d_attn_softmax = elem_time("softmax", batch * nh * 1 * s_kv, B_peak)

    d_layer_full = d_proj + d_elem + d_attn_matmul + d_attn_softmax
    d_layer_delta = d_proj + d_elem

    d_step = na * d_layer_full + nd * d_layer_delta
    d_step += matmul_time(M, h, vs, F, B_peak, p)

    decode_time_s = output_len * d_step

    # ---- totals ----
    total_s = prefill_time_s + decode_time_s
    total_tokens = batch * (input_len + output_len)
    overall_tps = total_tokens / total_s if total_s > 0 else float("inf")

    return {
        "prefill_ms": round(prefill_time_s * 1000, 2),
        "decode_step_ms": round(d_step * 1000, 4),
        "decode_total_ms": round(decode_time_s * 1000, 2),
        "total_latency_ms": round(total_s * 1000, 2),
        "total_tokens": total_tokens,
        "overall_tps": round(overall_tps, 1),
    }


# ── display ─────────────────────────────────────────────────────────────

def print_one(model_name, model, r, batch, input_len, output_len):
    h = model["hidden_dim"]
    inter = model.get("intermediate_dim", h * 4)
    nh = model["num_heads"]
    nl = model["num_layers"]
    na = model.get("attn_layers", nl)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  hidden_dim={h}, intermediate_dim={inter}, num_heads={nh}, num_layers={nl}")
    if na < nl:
        print(f"  attn_layers={na}/{nl}  (DeltaNet: {nl - na} layers)")
    print(f"  batch={batch}, input_len={input_len}, output_len={output_len}")
    print()
    p_tokens = batch * input_len
    print(f"  prefill ({p_tokens} tokens, b={batch}x{input_len}):")
    print(f"    time:        {r['prefill_ms']:.2f} ms")
    print(f"    throughput:  {p_tokens / (r['prefill_ms'] / 1000):.1f} tokens/s")
    print(f"  decode (b={batch}, {output_len} steps, 1 token/step):")
    print(f"    time/step:   {r['decode_step_ms']:.4f} ms")
    print(f"    total:       {r['decode_total_ms']:.2f} ms")
    print(f"  ----")
    print(f"  total latency: {r['total_latency_ms']:.2f} ms  ({r['total_latency_ms'] / 1000:.3f} s)")
    print(f"  total tokens:  {r['total_tokens']}")
    print(f"  throughput:    {r['overall_tps']:.1f} tokens/s")
    print()


def print_all(models, batch, input_len, output_len, hw_params):
    print(f"\n  batch={batch}, input_len={input_len}, output_len={output_len}")
    print(f"  {'Model':<20} {'Params':>8}  {'Prefill':>10} {'DecStep':>10} {'Total':>10} {'tokens/s':>10}")
    print(f"  {'':20} {'':8}  {'ms':>10} {'ms':>10} {'ms':>10} {'':>10}")
    print(f"  {'-' * 70}")

    for m in models:
        r = predict(m, batch, input_len, output_len, hw_params)
        pb = m.get("total_params_b", 0) / 1e9
        print(f"  {m['name']:<20} {pb:>7.1f}B  {r['prefill_ms']:>10.1f} {r['decode_step_ms']:>10.4f} "
              f"{r['total_latency_ms']:>10.1f} {r['overall_tps']:>10.1f}")


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict model inference throughput")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--input-len", type=int, default=2048, help="Prompt length (tokens)")
    parser.add_argument("--output-len", type=int, default=512, help="Generation length (tokens)")
    parser.add_argument("--params", type=str, help="Path to fitted_params.json")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--predict-all", action="store_true", help="Predict all models")

    args = parser.parse_args()

    specs = load_model_specs()
    models = specs.get("models", [])

    if args.list:
        print("Available models:")
        for m in models:
            na = m.get("attn_layers", m["num_layers"])
            print(f"  {m['name']:<20}  h={m['hidden_dim']}, nh={m['num_heads']}, "
                  f"nl={m['num_layers']}, attn_layers={na}")
        return

    params_path = args.params or str(DEFAULT_PARAMS)
    try:
        hw_params = load_hw_params(params_path)
    except FileNotFoundError:
        print(f"Fitted params not found: {params_path}")
        print("Run `uv run python -m fit results/ --save perf_predict/fitted_params.json` first.")
        return

    if "F_peak" not in hw_params:
        print(f"Error: fitted params missing 'F_peak'. Expected roofline model params.")
        return

    if args.model:
        model = next((m for m in models if m["name"] == args.model), None)
        if not model:
            print(f"Model not found: {args.model}")
            return
        r = predict(model, args.batch, args.input_len, args.output_len, hw_params)
        print_one(args.model, model, r, args.batch, args.input_len, args.output_len)

    elif args.predict_all:
        print_all(models, args.batch, args.input_len, args.output_len, hw_params)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
