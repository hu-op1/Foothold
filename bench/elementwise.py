"""Benchmark elementwise ops to characterize per-op effective bandwidth.

Each op has a distinct memory-access pattern and arithmetic intensity.  We
benchmark every op individually so the roofline simulator can use measured
per-op bandwidths instead of falling back to a single proxy.

Ops covered:
  - residual_add: pure add  (2 reads + 1 write)
  - rmsnorm:      reduction + rsqrt + normalize  (1 read + 1 write + reduction)
  - softmax:      multi-pass with exp  (heaviest elemwise)
  - swiglu:       SiLU gate × up  (2 reads + 1 write + sigmoid compute)
  - rope:         pairwise rotation  (1 read + 1 write + trig compute)
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import (warmup, benchmark, auto_warmup_iters, check_memory,
                         supports_float8_matmul, make_float8_tensor,
                         load_completed_keys, append_csv_row)


# Bytes per element for each dtype.
DTYPE_BYTES_MAP = {
    "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}


def _dtype_list(config):
    """Resolve dtype field (scalar for backward compat, list for multi-precision)."""
    raw = config["dtype"]
    return raw if isinstance(raw, list) else [raw]

# Bytes-per-element factors: must match sim/roofline.py ELEM_BYTES so the
# fitted B_eff is interpreted with the same multiplier at prediction time.
BYTES_FACTORS = {
    "residual_add": 3,   # read A + read B + write C
    "rmsnorm": 4,        # read + write, with rsqrt reduction
    "softmax": 6,        # multi-pass: max/exp + sum + normalize
    "swiglu": 3,         # read gate + read up + write result
    "rope": 4,           # read Q/K + read cos/sin tables + write
}

# Fixed CSV column order (for incremental append).
ELEM_FIELDS = ["op_name", "dtype", "N", "time_ms", "flops", "bytes"]
ELEM_KEY_FIELDS = ["op_name", "dtype", "N"]


def bench_elementwise(config, output_path="results/elementwise.csv"):
    dtypes = _dtype_list(config)
    warmup_cfg = config.get("warmup", config.get("warmup_iters", 10))
    warmup_ratio = config.get("warmup_ratio", 0.1)
    bench_min_time = config.get("min_time_ms", 200)
    bench_max_iters = config.get("max_iters", 10000)
    bench_calib = config.get("calib_iters", 20)
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    grid = config["elementwise"]
    ops_to_run = grid["operators"]

    # ── overwrite: delete existing CSV so all combos re-run ──
    if config.get("overwrite") and output_path and os.path.exists(output_path):
        os.remove(output_path)

    # ── resume: load already-completed combos ──
    done_keys = load_completed_keys(output_path, ELEM_KEY_FIELDS)

    results = []
    new_count = 0
    skip_count = 0

    def _wu(fn):
        """Warmup — auto-scaled or fixed iterations."""
        if warmup_cfg == "auto":
            w = auto_warmup_iters(fn, bench_min_time, bench_max_iters,
                                  bench_calib, warmup_ratio)
        else:
            w = int(warmup_cfg)
        warmup(fn, w)

    def _measure(fn, op_name, flops, byt):
        """Warmup + benchmark one elementwise op; record OOM rows instead of crashing."""
        nonlocal new_count
        try:
            _wu(fn)
            ms = benchmark(fn, min_time_ms=bench_min_time,
                           max_iters=bench_max_iters, calib_iters=bench_calib)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            key = (op_name, dt_name, N)
            if key in done_keys:
                return
            row = {
                "op_name": op_name, "dtype": dt_name, "N": N,
                "time_ms": "OOM", "flops": flops, "bytes": byt,
            }
            results.append(row)
            append_csv_row(output_path, ELEM_FIELDS, row)
            done_keys.add(key)
            return
        _append(op_name, flops, byt, ms)

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)

        # float8 elementwise: most elementwise ops (rmsnorm, softmax, silu,
        # residual_add) do not have native float8 kernels — they upcast
        # internally.  Skip float8 for elementwise; the matmul + flashattn
        # benches cover the relevant float8 data paths.
        is_float8 = dt_name in ("float8_e4m3fn", "float8_e5m2")
        if is_float8:
            print(f"\n  [skip] float8 elementwise: no native float8 kernels "
                  f"for rmsnorm/softmax/silu/residual_add/rope")
            continue

        for N in tqdm(grid["N"], desc=f"Elementwise {dt_name}"):
            # Check which ops are already done for this (dtype, N)
            pending_ops = [
                op for op in ops_to_run
                if (op, dt_name, N) not in done_keys
            ]
            if not pending_ops:
                skip_count += len(ops_to_run)
                continue

            act_gb = (3 * N * dt_bytes) / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                for op in ops_to_run:
                    key = (op, dt_name, N)
                    if key in done_keys:
                        skip_count += 1
                        continue
                    row = {
                        "op_name": op, "dtype": dt_name, "N": N,
                        "time_ms": "OOM", "flops": 0,
                        "bytes": BYTES_FACTORS[op] * N * dt_bytes,
                    }
                    results.append(row)
                    append_csv_row(output_path, ELEM_FIELDS, row)
                    done_keys.add(key)
                continue

            # Create tensors
            x = torch.randn(N, dtype=dtype, device=device)
            y = torch.randn(N, dtype=dtype, device=device)

            def _append(op_name, flops, byt, ms):
                key = (op_name, dt_name, N)
                if key in done_keys:
                    return
                row = {
                    "op_name": op_name, "dtype": dt_name, "N": N,
                    "time_ms": f"{ms:.6f}", "flops": flops, "bytes": byt,
                }
                results.append(row)
                append_csv_row(output_path, ELEM_FIELDS, row)
                done_keys.add(key)
                nonlocal new_count
                new_count += 1

            # --- residual_add ---
            if "residual_add" in pending_ops:
                def add_fn(x=x, y=y):
                    x + y
                _measure(add_fn, "residual_add", N, 3 * N * dt_bytes)

            # --- rmsnorm ---
            if "rmsnorm" in pending_ops:
                w = torch.ones(N, dtype=dtype, device=device)
                def rms_fn(x=x, w=w):
                    F.rms_norm(x, (N,), w, 1e-5)
                _measure(rms_fn, "rmsnorm", 4 * N, 4 * N * dt_bytes)

            # --- softmax ---
            if "softmax" in pending_ops:
                def soft_fn(x=x):
                    F.softmax(x, dim=0)
                _measure(soft_fn, "softmax", 5 * N, 6 * N * dt_bytes)

            # --- swiglu ---
            if "swiglu" in pending_ops:
                gate = torch.randn(N, dtype=dtype, device=device)
                up = torch.randn(N, dtype=dtype, device=device)
                def swiglu_fn(gate=gate, up=up):
                    F.silu(gate) * up
                _measure(swiglu_fn, "swiglu", 5 * N, 3 * N * dt_bytes)

            # --- rope ---
            # WARNING: This synthetic kernel uses PyTorch slice indexing
            # ([:, 0] / [:, 1]) which breaks kernel fusion — a single call
            # triggers ~8 separate GPU kernel launches instead of 1.
            # vLLM's real RoPE is a single fused in-place CUDA kernel
            # (csrc/pos_encoding_kernels.cu).  The fitted overhead (261 µs)
            # is ~7× the real value (~37 µs), and B_eff (274 GB/s) is
            # ~3× below peak.  Do NOT use this data directly — fit/elementwise.py
            # proxies rope → residual_add instead.
            # See docs/accuracy-improvements.md §1b for details.
            if "rope" in pending_ops:
                rope_q = torch.randn(N, dtype=dtype, device=device)
                def rope_fn(q=rope_q):
                    q2 = q.view(-1, 2)
                    out = torch.empty_like(q2)
                    out[:, 0] = q2[:, 0] * 0.5 - q2[:, 1] * 0.866
                    out[:, 1] = q2[:, 1] * 0.5 + q2[:, 0] * 0.866
                    return out.view(-1)
                _measure(rope_fn, "rope", 6 * N, 4 * N * dt_bytes)
            del x, y

    if skip_count:
        print(f"  [resume] skipped {skip_count} completed, {new_count} new → {output_path}")
    elif output_path and new_count > 0:
        print(f"  Saved {new_count} rows → {output_path}")
    return results
