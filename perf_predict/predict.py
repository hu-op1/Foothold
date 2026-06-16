"""Predict model inference throughput from architecture specs + hardware roofline params.

Library module — CLI entry point is main.py.
"""

import yaml
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DEFAULT_PARAMS = HERE / "fitted_params.json"
DEFAULT_SPECS = ROOT / "config" / "model_specs.yaml"
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


def elem_time(op_name, N, b_effs, overheads):
    factor = ELEM_BYTES.get(op_name, 3)
    B_eff = b_effs.get(op_name, 1e12)  # fallback: huge B = ~zero time
    overhead = overheads.get(op_name, 0.0)
    return (N * factor * DTYPE_BYTES) / B_eff + overhead


# ── layer ops ───────────────────────────────────────────────────────────

def projections(M, h, inter, F, B, p, nh=None, nh_kv=None, hd=None):
    """Q/K/V/O (4x) + FFN up/gate (2x) + FFN down.

    nh, hd: if nh*hd != h, Q proj outputs nh*hd, K/V output nh_kv*hd, O inputs nh*hd.
    nh_kv defaults to nh (MHA), set for GQA.
    """
    if nh is None or (nh * hd == h and (nh_kv or nh) == nh):
        t = 4 * matmul_time(M, h, h, F, B, p)
    else:
        dim_q = nh * hd
        dim_kv = (nh_kv or nh) * hd
        t = matmul_time(M, h, dim_q, F, B, p)       # Q proj
        t += matmul_time(M, h, dim_kv, F, B, p)      # K proj
        t += matmul_time(M, h, dim_kv, F, B, p)      # V proj
        t += matmul_time(M, dim_q, h, F, B, p)       # O proj
    t += 2 * matmul_time(M, h, inter, F, B, p)
    t += matmul_time(M, inter, h, F, B, p)
    return t


def attention_fused(b, nh, s_q, s_kv, hd, F, B, p, nh_kv=None):
    """FlashAttention: QK^T + softmax + score x V fused in SRAM.

    FLOPs unchanged, but intermediate SxS matrix never touches HBM.
    Bytes = Q,K,V reads + O write (no SxS round-trip).

    nh_kv: number of KV heads (defaults to nh for MHA; < nh for GQA).
    """
    if nh_kv is None:
        nh_kv = nh
    M_q = b * nh * s_q
    M_kv = b * nh_kv * s_kv
    flops = 4 * M_q * s_kv * hd
    # HBM traffic: Q(read) + K(read) + V(read) + O(write)
    # Q,O use nh heads; K,V use nh_kv heads
    bytes_moved = b * hd * DTYPE_BYTES * (nh * s_q + nh_kv * s_kv + nh_kv * s_kv + nh * s_q)
    return roofline_time(flops, bytes_moved, F, B, p)


def elementwise_ops(b, s, h, inter, nh, hd, norm_type, b_effs, overheads, nh_kv=None):
    """All elementwise ops per layer.

    nh_kv: number of KV heads (defaults to nh for MHA; < nh for GQA).
    """
    if nh_kv is None:
        nh_kv = nh
    t = 0.0
    N = b * s * h
    t += 2 * elem_time(norm_type, N, b_effs, overheads)
    t += elem_time("swiglu", b * s * inter, b_effs, overheads)
    # RoPE applied to Q (nh heads) and K (nh_kv heads)
    t += elem_time("rope", b * nh * s * hd, b_effs, overheads)
    t += elem_time("rope", b * nh_kv * s * hd, b_effs, overheads)
    t += 2 * elem_time("residual_add", N, b_effs, overheads)
    return t


# ── prediction ──────────────────────────────────────────────────────────

def predict(model, batch, input_len, output_len, hw_params):
    F_p = hw_params["F_peak_prefill"]
    B_p = hw_params["B_peak_prefill"]
    p_p = hw_params["p_prefill"]
    F_d = hw_params["F_peak_decode"]
    B_d = hw_params["B_peak_decode"]
    p_d = hw_params["p_decode"]
    b_effs = hw_params["elem_b_effs"]
    overheads = hw_params["elem_overheads"]

    nl = model["num_layers"]
    na = model.get("attn_layers", nl)
    nd = nl - na

    h = model["hidden_dim"]
    inter = model.get("intermediate_dim", h * 4)
    nh = model["num_heads"]
    nh_kv = model.get("num_kv_heads", nh)
    hd = model["head_dim"]
    vs = model["vocab_size"]
    norm_type = model.get("norm_type", "rmsnorm")

    # ---- prefill ----
    M = batch * input_len
    s = input_len

    p_proj = projections(M, h, inter, F_p, B_p, p_p, nh, nh_kv, hd)
    p_elem = elementwise_ops(batch, s, h, inter, nh, hd, norm_type, b_effs, overheads, nh_kv)

    p_attn = attention_fused(batch, nh, s, s, hd, F_p, B_p, p_p, nh_kv)

    p_layer_full = p_proj + p_elem + p_attn
    p_layer_delta = p_proj + p_elem

    p_total = na * p_layer_full + nd * p_layer_delta

    p_total += matmul_time(M, h, vs, F_p, B_p, p_p)
    if na > 0:
        p_total += elem_time("causal_mask", batch * nh * s * s, b_effs, overheads)

    prefill_time_s = p_total

    # ---- decode (one step) ----
    M = batch * 1
    s_kv = input_len

    d_proj = projections(M, h, inter, F_d, B_d, p_d, nh, nh_kv, hd)
    d_elem = elementwise_ops(batch, 1, h, inter, nh, hd, norm_type, b_effs, overheads, nh_kv)

    d_attn = attention_fused(batch, nh, 1, s_kv, hd, F_d, B_d, p_d, nh_kv)

    d_layer_full = d_proj + d_elem + d_attn
    d_layer_delta = d_proj + d_elem

    d_step = na * d_layer_full + nd * d_layer_delta
    d_step += matmul_time(M, h, vs, F_d, B_d, p_d)

    decode_time_s = output_len * d_step

    # ---- totals ----
    total_s = prefill_time_s + decode_time_s
    r = {
        "prefill_ms": round(prefill_time_s * 1000),
        "prefill_tps_1x": round(input_len / prefill_time_s) if prefill_time_s > 0 else float("inf"),
        "prefill_tps_batch": round(batch * input_len / prefill_time_s) if prefill_time_s > 0 else float("inf"),
        "decode_ms": round(decode_time_s * 1000),
        "decode_tps_1x": round(output_len / decode_time_s) if decode_time_s > 0 else float("inf"),
        "decode_tps_batch": round(batch * output_len / decode_time_s) if decode_time_s > 0 else float("inf"),
        "total_ms": round(total_s * 1000),
        "total_s": round(total_s, 2),
        "e2e_tps_1x": round((input_len + output_len) / total_s) if total_s > 0 else float("inf"),
        "e2e_tps_batch": round(batch * (input_len + output_len) / total_s) if total_s > 0 else float("inf"),
    }
    return r


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
    s_req = f"{input_len}->{output_len}"
    if batch > 1:
        print(f"  prefill  {r['prefill_ms']}ms  --  1x {r['prefill_tps_1x']} tok/s,  {batch}x {r['prefill_tps_batch']} tok/s")
        print(f"  decode   {r['decode_ms']}ms  --  1x {r['decode_tps_1x']} tok/s,   {batch}x {r['decode_tps_batch']} tok/s")
        print(f"  e2e      {r['total_ms']}ms ({r['total_s']}s)  --  1x {r['e2e_tps_1x']} tok/s,  {batch}x {r['e2e_tps_batch']} tok/s  [{s_req}]")
    else:
        print(f"  prefill  {r['prefill_ms']}ms  --  {r['prefill_tps_1x']} tok/s")
        print(f"  decode   {r['decode_ms']}ms  --  {r['decode_tps_1x']} tok/s")
        print(f"  e2e      {r['total_ms']}ms ({r['total_s']}s)  --  {r['e2e_tps_1x']} tok/s  [{s_req}]")
    print()


def print_all(models, batch, input_len, output_len, hw_params):
    print(f"\n  batch={batch}, input_len={input_len}, output_len={output_len}\n")

    for m in models:
        r = predict(m, batch, input_len, output_len, hw_params)
        pb = m.get("total_params_b", 0) / 1e9
        na = m.get("attn_layers", m["num_layers"])
        s_req = f"{input_len}->{output_len}"
        if batch > 1:
            print(f"  {m['name']:<18} {pb:>4.1f}B"
                  f"  |  prefill {r['prefill_ms']:>6}ms  1x {r['prefill_tps_1x']:>6}  {batch}x {r['prefill_tps_batch']:>6} t/s"
                  f"  |  decode {r['decode_ms']:>7}ms  1x {r['decode_tps_1x']:>5}  {batch}x {r['decode_tps_batch']:>5} t/s"
                  f"  |  e2e {r['total_ms']:>7}ms  1x {r['e2e_tps_1x']:>5}  {batch}x {r['e2e_tps_batch']:>5} t/s"
                  f"  |  [{s_req}]")
        else:
            print(f"  {m['name']:<18} {pb:>4.1f}B"
                  f"  |  prefill {r['prefill_ms']:>6}ms {r['prefill_tps_1x']:>6} t/s"
                  f"  |  decode {r['decode_ms']:>7}ms {r['decode_tps_1x']:>5} t/s"
                  f"  |  e2e {r['total_ms']:>7}ms {r['e2e_tps_1x']:>5} t/s"
                  f"  |  [{s_req}]")
