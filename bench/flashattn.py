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
from bench.utils import warmup, benchmark, save_xlsx, check_memory

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


def bench_flashattn(config, output_path="results/flashattn.xlsx"):
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

    results = []

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)
        combos = [(b_val, sq, skv) for b_val in batch_list for sq in fa_cfg["s_q"] for skv in fa_cfg["s_kv"]]
        for b_val, s_q, s_kv in tqdm(combos, desc=f"FlashAttn {dt_name}"):
            # Memory check
            act_bytes = b_val * (nh * s_q + 2 * nh_kv * s_kv + nh * s_q) * hd * dt_bytes
            act_gb = act_bytes / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                results.append({
                    "op_name": "flashattn",
                    "dtype": dt_name,
                    "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                    "s_q": s_q, "s_kv": s_kv,
                    "time_ms": "OOM",
                    "flops": 0,
                    "bytes": 0,
                })
                continue

            # Create tensors and benchmark
            q = torch.randn(b_val, nh, s_q, hd, dtype=dtype, device=device)
            k = torch.randn(b_val, nh_kv, s_kv, hd, dtype=dtype, device=device)
            v = torch.randn(b_val, nh_kv, s_kv, hd, dtype=dtype, device=device)

            if _HAS_NATIVE_FA:
                def fa_fn(q=q, k=k, v=v):
                    _fa_native(q, k, v, causal=True)
            else:
                def fa_fn(q=q, k=k, v=v):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)

            warmup(fa_fn, warmup_iters)
            ms = benchmark(fa_fn, bench_iters)

            results.append({
                "op_name": "flashattn",
                "dtype": dt_name,
                "b": b_val, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                "s_q": s_q, "s_kv": s_kv,
                "time_ms": f"{ms:.6f}",
                "flops": _fa_flops(b_val, nh, s_q, s_kv, hd),
                "bytes": _fa_bytes(b_val, nh, s_q, nh_kv, s_kv, hd, dt_bytes),
            })

            del q, k, v

    if output_path:
        save_xlsx(results, output_path)
    return results
