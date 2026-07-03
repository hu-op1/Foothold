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

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory


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


def bench_elementwise(config, output_path="results/elementwise.xlsx"):
    dtypes = _dtype_list(config)
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    grid = config["elementwise"]
    results = []

    for dt_name in dtypes:
        dtype = getattr(torch, dt_name)
        dt_bytes = DTYPE_BYTES_MAP.get(dt_name, 2)
        for N in tqdm(grid["N"], desc=f"Elementwise {dt_name}"):
            act_gb = (3 * N * dt_bytes) / (1024 ** 3)
            oom = not check_memory(act_gb, max_mem)
            if oom:
                for op in grid["operators"]:
                    results.append({
                        "op_name": op,
                        "dtype": dt_name,
                        "N": N,
                        "time_ms": "OOM",
                        "flops": 0,
                        "bytes": BYTES_FACTORS[op] * N * dt_bytes,
                    })
                continue

            x = torch.randn(N, dtype=dtype, device=device)
            y = torch.randn(N, dtype=dtype, device=device)

            # --- residual_add ---
            def add_fn(x=x, y=y):
                x + y

            warmup(add_fn, warmup_iters)
            ms = benchmark(add_fn, bench_iters)
            results.append({
                "op_name": "residual_add",
                "dtype": dt_name,
                "N": N,
                "time_ms": f"{ms:.6f}",
                "flops": N,
                "bytes": 3 * N * dt_bytes,
            })

            # --- rmsnorm ---
            w = torch.ones(N, dtype=dtype, device=device)

            def rms_fn(x=x, w=w):
                F.rms_norm(x, (N,), w, 1e-5)

            warmup(rms_fn, warmup_iters)
            ms = benchmark(rms_fn, bench_iters)
            results.append({
                "op_name": "rmsnorm",
                "dtype": dt_name,
                "N": N,
                "time_ms": f"{ms:.6f}",
                "flops": 4 * N,
                "bytes": 4 * N * dt_bytes,
            })

            # --- softmax ---
            def soft_fn(x=x):
                F.softmax(x, dim=0)

            warmup(soft_fn, warmup_iters)
            ms = benchmark(soft_fn, bench_iters)
            results.append({
                "op_name": "softmax",
                "dtype": dt_name,
                "N": N,
                "time_ms": f"{ms:.6f}",
                "flops": 5 * N,
                "bytes": 6 * N * dt_bytes,
            })

            # --- swiglu ---
            gate = torch.randn(N, dtype=dtype, device=device)
            up = torch.randn(N, dtype=dtype, device=device)

            def swiglu_fn(gate=gate, up=up):
                F.silu(gate) * up

            warmup(swiglu_fn, warmup_iters)
            ms = benchmark(swiglu_fn, bench_iters)
            results.append({
                "op_name": "swiglu",
                "dtype": dt_name,
                "N": N,
                "time_ms": f"{ms:.6f}",
                "flops": 5 * N,   # silu(~4) + multiply(1)
                "bytes": 3 * N * dt_bytes,
            })

            # --- rope ---
            # RoPE applies a 2D rotation to each pair of elements.
            # q is [N] → view as [N/2, 2] → apply 2×2 rotation → flatten back.
            rope_q = torch.randn(N, dtype=dtype, device=device)

            def rope_fn(q=rope_q):
                q2 = q.view(-1, 2)
                out = torch.empty_like(q2)
                out[:, 0] = q2[:, 0] * 0.5 - q2[:, 1] * 0.866
                out[:, 1] = q2[:, 1] * 0.5 + q2[:, 0] * 0.866
                return out.view(-1)

            warmup(rope_fn, warmup_iters)
            ms = benchmark(rope_fn, bench_iters)
            results.append({
                "op_name": "rope",
                "dtype": dt_name,
                "N": N,
                "time_ms": f"{ms:.6f}",
                "flops": 6 * N,   # 2 mul + 2 add per pair → 4 per pair → 2 per element
                "bytes": 4 * N * dt_bytes,
            })

            del x, y, w, gate, up, rope_q

    if output_path:
        save_xlsx(results, output_path)
    return results
