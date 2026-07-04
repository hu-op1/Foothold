"""Benchmark FlashAttention (via torch SDPA) over a (s_q, s_kv) grid.

FlashAttention is the dominant kernel in LLM inference attention layers.
Its hardware efficiency differs from matmul due to tiling, causal masking,
and SRAM-aware scheduling.  This benchmark provides measured data so the
roofline fit can produce FA-specific (F_peak, B_peak, p) parameters instead
of reusing matmul-fitted values.

Prefers Dao-AILab's flash_attn package when available (Linux + CUDA).
Falls back to torch.nn.functional.scaled_dot_product_attention on other
platforms (Windows, CPU).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import (warmup, benchmark, check_memory,
                         load_completed_keys, append_csv_row)

# Try native flash_attn first (matches vLLM backend), fall back to PyTorch SDPA.
try:
    from flash_attn import flash_attn_func as _fa_native
    _HAS_NATIVE_FA = True
except ImportError:
    _HAS_NATIVE_FA = False


# Bytes per element for each dtype.
DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}

# Fixed CSV column order (for incremental append).
FA_FIELDS = ["op_name", "dtype", "b", "nh", "nh_kv", "hd",
             "s_q", "s_kv", "time_ms", "flops", "bytes"]
FA_KEY_FIELDS = ["op_name", "dtype", "b", "s_q", "s_kv"]


def _dtype_list(config):
    """Resolve dtype field (scalar for backward compat, list for multi-precision)."""
    raw = config["dtype"]
    return raw if isinstance(raw, list) else [raw]


def _fa_bytes(b, nh, s_q, nh_kv, s_kv, hd, dt_bytes):
    """Analytical bytes moved by FlashAttention (no S×S HBM round-trip).

    Q read + K read + V read + O write.
    """
    return b * hd * dt_bytes * (nh * s_q + nh_kv * s_kv + nh_kv * s_kv + nh * s_q)


def _fa_flops(b, nh, s_q, s_kv, hd):
    """Analytical FLOPs for attention: 2·s_q·s_kv·hd per head, times 2 for mul+add."""
    return 4 * b * nh * s_q * s_kv * hd


def bench_flashattn(config, output_path="results/flashattn.csv"):
    dtypes = _dtype_list(config)
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    fa_cfg = config["flashattn"]
    batch_raw = fa_cfg.get("batch", 1)
    batch_list = batch_raw if isinstance(batch_raw, list) else [batch_raw]
    nh = fa_cfg.get("num_heads", 32)
    nh_kv = fa_cfg.get("num_kv_heads", 8)
    hd = fa_cfg.get("head_dim", 128)

    backend = "flash_attn (native)" if _HAS_NATIVE_FA else "torch SDPA"
    print(f"FlashAttn backend: {backend}")
    print(f"FlashAttn dtypes: {dtypes}")
    print(f"FlashAttn batch sizes: {batch_list}")

    # ── resume: load already-completed combos ──
    done_keys = load_completed_keys(output_path, FA_KEY_FIELDS)

    results = []
    new_count = 0
    skip_count = 0

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        # float8 FlashAttention: neither flash_attn_func nor torch SDPA
        # accept float8 inputs.  Skip for now; float8 attention via
        # dedicated Hopper kernels (cuDNN/cuBLAS) is a different codepath.
        is_float8 = dt_name in ("float8_e4m3fn", "float8_e5m2")
        if is_float8:
            print(f"\n  [skip] float8 FlashAttn: flash_attn / SDPA do not "
                  f"accept float8 inputs")
            continue

        combos = [(b_val, sq, skv) for b_val in batch_list
                  for sq in fa_cfg["s_q"] for skv in fa_cfg["s_kv"]]
        for b_val, s_q, s_kv in tqdm(combos, desc=f"FlashAttn {dt_name}"):
            key = ("flashattn", dt_name, b_val, s_q, s_kv)
            if key in done_keys:
                skip_count += 1
                continue

            # Memory check
            act_bytes = b_val * (nh * s_q + 2 * nh_kv * s_kv + nh * s_q) * hd * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                row = {
                    "op_name": "flashattn", "dtype": dt_name,
                    "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                    "s_q": s_q, "s_kv": s_kv,
                    "time_ms": "OOM", "flops": 0, "bytes": 0,
                }
                results.append(row)
                append_csv_row(output_path, FA_FIELDS, row)
                done_keys.add(key)
                continue

            # Create tensors — flash_attn / SDPA expect (batch, seqlen, nheads, hd)
            q = torch.randn(b_val, s_q, nh, hd, dtype=dtype, device=device)
            k = torch.randn(b_val, s_kv, nh_kv, hd, dtype=dtype, device=device)
            v = torch.randn(b_val, s_kv, nh_kv, hd, dtype=dtype, device=device)

            if _HAS_NATIVE_FA:
                def fa_fn(q=q, k=k, v=v):
                    _fa_native(q, k, v, causal=True)
            else:
                def fa_fn(q=q, k=k, v=v):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)

            warmup(fa_fn, warmup_iters)
            ms = benchmark(fa_fn, bench_iters)

            row = {
                "op_name": "flashattn", "dtype": dt_name,
                "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                "s_q": s_q, "s_kv": s_kv,
                "time_ms": f"{ms:.6f}",
                "flops": _fa_flops(b_val, nh, s_q, s_kv, hd),
                "bytes": _fa_bytes(b_val, nh, s_q, nh_kv, s_kv, hd, dt_bytes),
            }
            results.append(row)
            append_csv_row(output_path, FA_FIELDS, row)
            done_keys.add(key)
            new_count += 1

            del q, k, v

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif output_path and new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")
    return results
