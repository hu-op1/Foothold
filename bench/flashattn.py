"""Benchmark FlashAttention (via torch SDPA) over a (s_q, s_kv) grid.

FlashAttention is the dominant kernel in LLM inference attention layers.
Its hardware efficiency differs from matmul due to tiling, causal masking,
and SRAM-aware scheduling.  This benchmark provides measured data so the
roofline fit can produce FA-specific (F_peak, B_peak, p) parameters instead
of reusing matmul-fitted values.

Uses torch.nn.functional.scaled_dot_product_attention which dispatches to
the best available backend (FlashAttention-2, Memory-Efficient, or Math).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory


DTYPE_BYTES = 2  # fp16


def _fa_bytes(b, nh, s_q, nh_kv, s_kv, hd):
    """Analytical bytes moved by FlashAttention (no S×S HBM round-trip).

    Q read + K read + V read + O write.
    """
    return b * hd * DTYPE_BYTES * (nh * s_q + nh_kv * s_kv + nh_kv * s_kv + nh * s_q)


def _fa_flops(b, nh, s_q, s_kv, hd):
    """Analytical FLOPs for attention: 2·s_q·s_kv·hd per head, times 2 for mul+add."""
    return 4 * b * nh * s_q * s_kv * hd


def bench_flashattn(config, output_path="results/flashattn.xlsx"):
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    fa_cfg = config["flashattn"]
    b = fa_cfg.get("batch", 1)
    nh = fa_cfg.get("num_heads", 32)
    nh_kv = fa_cfg.get("num_kv_heads", 8)
    hd = fa_cfg.get("head_dim", 128)

    results = []

    combos = [(sq, skv) for sq in fa_cfg["s_q"] for skv in fa_cfg["s_kv"]]
    for s_q, s_kv in tqdm(combos, desc="FlashAttn"):
        # Memory check: Q[b,nh,s_q,hd] + K[b,nh_kv,s_kv,hd] + V[b,nh_kv,s_kv,hd] + O[b,nh,s_q,hd]
        act_bytes = b * (nh * s_q + 2 * nh_kv * s_kv + nh * s_q) * hd * DTYPE_BYTES
        act_gb = act_bytes / (1024 ** 3)
        oom = not check_memory(act_gb, max_mem)
        if oom:
            results.append({
                "op_name": "flashattn",
                "b": b, "nh": nh, "nh_kv": nh_kv, "hd": hd,
                "s_q": s_q, "s_kv": s_kv,
                "time_ms": "OOM",
                "flops": 0,
                "bytes": 0,
            })
            continue

        # Create tensors
        q = torch.randn(b, nh, s_q, hd, dtype=dtype, device=device)
        k = torch.randn(b, nh_kv, s_kv, hd, dtype=dtype, device=device)
        v = torch.randn(b, nh_kv, s_kv, hd, dtype=dtype, device=device)

        def fa_fn(q=q, k=k, v=v):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)

        warmup(fa_fn, warmup_iters)
        ms = benchmark(fa_fn, bench_iters)

        results.append({
            "op_name": "flashattn",
            "b": b, "nh": nh, "nh_kv": nh_kv, "hd": hd,
            "s_q": s_q, "s_kv": s_kv,
            "time_ms": f"{ms:.6f}",
            "flops": _fa_flops(b, nh, s_q, s_kv, hd),
            "bytes": _fa_bytes(b, nh, s_q, nh_kv, s_kv, hd),
        })

        del q, k, v

    if output_path:
        save_xlsx(results, output_path)
    return results
