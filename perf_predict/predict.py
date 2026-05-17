"""
Predict model inference throughput from architecture specs + fitted operator params.

Args are input_len (prompt) and output_len (generation):
  prefill_time  = one forward over input_len tokens
  decode_time   = output_len * one decode step (1 token + KV cache)
  total_latency = prefill_time + decode_time
  throughput    = (input_len + output_len) / total_latency

Usage:
    uv run python perf_predict/predict.py --list
    uv run python perf_predict/predict.py --model "Llama-3.1-8B" --input-len 2048 --output-len 512
    uv run python perf_predict/predict.py --predict-all --input-len 2048 --output-len 512
"""

import yaml
import json
import argparse
from pathlib import Path

HERE = Path(__file__).parent.resolve()
DEFAULT_PARAMS = HERE / "fitted_params.json"
DEFAULT_SPECS = HERE / "model_specs.yaml"


def load_model_specs(path=None):
    path = path or str(DEFAULT_SPECS)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fitted_params(path=None):
    path = path or str(DEFAULT_PARAMS)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── workloads ───────────────────────────────────────────────────────────

def prefill_workloads(model, b, s):
    h = model["hidden_dim"]
    inter = model.get("intermediate_dim", h * 4)
    nh = model["num_heads"]
    hd = model["head_dim"]
    vs = model["vocab_size"]
    M = b * s
    return {
        "q_proj":       M * h * h,
        "k_proj":       M * h * h,
        "v_proj":       M * h * h,
        "o_proj":       M * h * h,
        "ffn_up":       M * h * inter,
        "ffn_gate":     M * h * inter,
        "ffn_down":     M * inter * h,
        "lm_head":      M * h * vs,
        "qk_matmul":    b * nh * s * s * hd,
        "softmax":      b * nh * s * s,
        "score_v_matmul": b * nh * s * s * hd,
        "layernorm":    b * s * h,
        "rmsnorm":      b * s * h,
        "swiglu":       b * s * inter,
        "rope":         b * nh * s * hd,
        "residual_add": b * s * h,
        "causal_mask":  b * nh * s * s,
    }


def decode_workloads(model, b, s_kv):
    h = model["hidden_dim"]
    inter = model.get("intermediate_dim", h * 4)
    nh = model["num_heads"]
    hd = model["head_dim"]
    vs = model["vocab_size"]
    M = b * 1
    return {
        "q_proj":       M * h * h,
        "k_proj":       M * h * h,
        "v_proj":       M * h * h,
        "o_proj":       M * h * h,
        "ffn_up":       M * h * inter,
        "ffn_gate":     M * h * inter,
        "ffn_down":     M * inter * h,
        "lm_head":      M * h * vs,
        "qk_matmul":    b * nh * 1 * s_kv * hd,
        "softmax":      b * nh * 1 * s_kv,
        "score_v_matmul": b * nh * 1 * s_kv * hd,
        "layernorm":    b * 1 * h,
        "rmsnorm":      b * 1 * h,
        "swiglu":       b * 1 * inter,
        "rope":         b * nh * 1 * hd,
        "residual_add": b * 1 * h,
    }


# ── per-layer op counts ─────────────────────────────────────────────────

BASE_OPS = {
    "q_proj": 1, "k_proj": 1, "v_proj": 1, "o_proj": 1,
    "ffn_up": 1, "ffn_gate": 1, "ffn_down": 1,
    "swiglu": 1, "rope": 1, "residual_add": 2,
}

ATTN_OPS = {"qk_matmul": 1, "softmax": 1, "score_v_matmul": 1}


def _sum_ops(ops, w, fitted):
    t = 0.0
    for op, count in ops.items():
        work = w.get(op, 0)
        if op in fitted and work > 0:
            t += count * (fitted[op]["a"] * work + fitted[op]["b"])
    return t


def base_layer_time(w, fitted, norm_type):
    t = _sum_ops(BASE_OPS, w, fitted)
    if norm_type in fitted and w.get(norm_type, 0) > 0:
        a, b = fitted[norm_type]["a"], fitted[norm_type]["b"]
        t += 2 * (a * w[norm_type] + b)
    return t


def attn_layer_time(w, fitted):
    return _sum_ops(ATTN_OPS, w, fitted)


# ── prediction ──────────────────────────────────────────────────────────

def predict(model, batch, input_len, output_len, fitted):
    nl = model["num_layers"]
    na = model.get("attn_layers", nl)  # layers with full attention (default: all)
    nd = nl - na                        # layers without attention (DeltaNet, etc.)
    norm = model.get("norm_type", "rmsnorm")

    # ---- prefill ----
    pw = prefill_workloads(model, batch, input_len)
    p_base = base_layer_time(pw, fitted, norm)
    p_attn = attn_layer_time(pw, fitted)

    p_total = nl * p_base + na * p_attn

    if "causal_mask" in fitted and pw.get("causal_mask", 0) > 0:
        a, b = fitted["causal_mask"]["a"], fitted["causal_mask"]["b"]
        p_total += a * pw["causal_mask"] + b
    if "lm_head" in fitted and pw.get("lm_head", 0) > 0:
        a, b = fitted["lm_head"]["a"], fitted["lm_head"]["b"]
        p_total += a * pw["lm_head"] + b

    prefill_time_s = p_total / 1000.0

    # ---- decode (one step) ----
    dw = decode_workloads(model, batch, input_len)
    d_base = base_layer_time(dw, fitted, norm)
    d_attn = attn_layer_time(dw, fitted)

    d_step = nl * d_base + na * d_attn
    if "lm_head" in fitted and dw.get("lm_head", 0) > 0:
        a, b = fitted["lm_head"]["a"], fitted["lm_head"]["b"]
        d_step += a * dw["lm_head"] + b

    decode_time_s = output_len * d_step / 1000.0

    # ---- totals ----
    total_s = prefill_time_s + decode_time_s
    total_tokens = batch * (input_len + output_len)
    overall_tps = total_tokens / total_s if total_s > 0 else float("inf")

    return {
        "prefill_ms":        round(p_total, 2),
        "decode_step_ms":    round(d_step, 4),
        "decode_total_ms":   round(output_len * d_step, 2),
        "total_latency_ms":  round(total_s * 1000, 2),
        "total_tokens":      total_tokens,
        "overall_tps":       round(overall_tps, 1),
    }


# ── display ─────────────────────────────────────────────────────────────

def print_one(model_name, model, r, batch, input_len, output_len):
    h = model["hidden_dim"]
    inter = model["intermediate_dim"]
    nh = model["num_heads"]
    nl = model["num_layers"]

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  hidden_dim={h}, intermediate_dim={inter}, num_heads={nh}, num_layers={nl}")
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


def print_all(models, batch, input_len, output_len, fitted):
    print(f"\n  batch={batch}, input_len={input_len}, output_len={output_len}")
    print(f"  {'Model':<20} {'Params':>8}  {'Prefill':>10} {'DecStep':>10} {'Total':>10} {'tokens/s':>10}")
    print(f"  {'':20} {'':8}  {'ms':>10} {'ms':>10} {'ms':>10} {'':>10}")
    print(f"  {'-' * 70}")

    for m in models:
        r = predict(m, batch, input_len, output_len, fitted)
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

    params_path = args.params or str(DEFAULT_PARAMS)
    try:
        fitted = load_fitted_params(params_path)
    except FileNotFoundError:
        print(f"Fitted params not found: {params_path}")
        print("Run `uv run python -m fit results/<gpu> --save perf_predict/fitted_params.json` first.")
        return

    if args.list:
        print("Available models:")
        for m in models:
            print(f"  {m['name']:<20}  h={m['hidden_dim']}, nh={m['num_heads']}, nl={m['num_layers']}")
        return

    if args.model:
        model = next((m for m in models if m["name"] == args.model), None)
        if not model:
            print(f"Model not found: {args.model}")
            return
        r = predict(model, args.batch, args.input_len, args.output_len, fitted)
        print_one(args.model, model, r, args.batch, args.input_len, args.output_len)

    elif args.predict_all:
        print_all(models, args.batch, args.input_len, args.output_len, fitted)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
