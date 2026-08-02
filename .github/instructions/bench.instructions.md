---
description: "Use when adding or modifying GPU kernel microbenchmarks in bench/. Covers CUDA timing, checkpoint/resume, OOM handling, and benchmark conventions."
applyTo: "bench/**/*.py"
---

# bench/ — GPU Kernel Microbenchmarks

## Function Signature Convention

Every bench module exports a function with the signature:

```python
def bench_<op>(config: dict, output_dir: str | Path) -> list[dict]
```

Flat keyword arguments extracted from `config` dict (never pass config sections directly):

```python
dtypes = config.get("dtype", ["float16"])
warmup_cfg = config.get("warmup", "auto")
bench_min_time = config.get("min_time_ms", 1000)
bench_max_iters = config.get("max_iters", 100000)
overwrite = config.get("overwrite", False)
max_mem = config.get("max_memory_gb", 24)
```

## GPU Timing

Use `CudaTimer` context manager — wraps `torch.cuda.Event(enable_timing=True)`:

```python
from bench.utils import CudaTimer
with CudaTimer() as timer:
    fn()
ms = timer.elapsed_ms
```

For adaptive iteration count, use `benchmark()`:

```python
from bench.utils import warmup, benchmark
warmup(fn, iters=5)
avg_time_ms = benchmark(fn, min_time_ms=200, max_iters=10000)
```

## OOM Handling

Always guard large tensor allocations:

```python
from bench.utils import check_memory
if not check_memory(required_gb, max_gb=max_mem):
    row["time_ms"] = "OOM"
    row["flops"] = 0
    row["bytes"] = 0
    append_csv_row(output_path, FIELDS, row)
    continue
```

## Checkpoint / Resume

Use the checkpoint helpers from `bench/utils.py`:

```python
from bench.utils import load_completed_keys, append_csv_row

KEY_FIELDS = ["op_name", "dtype", "M", "K", "N"]  # per-module
done_keys = load_completed_keys(output_path, KEY_FIELDS)

for shape in grid:
    key = (op_name, dtype, M, K, N)
    if key in done_keys:
        continue  # already measured in a previous run
    # ... measure ...
    append_csv_row(output_path, FIELDS, row)
```

- OOM rows are skipped by `load_completed_keys` — they will be re-measured on next run.
- Always append (never rewrite the whole file) for crash resilience.

## Output Format

Results go to `bench/results/<gpu>/<op>.csv`. Define field lists as module-level constants:

```python
FIELDS = ["op_name", "dtype", "M", "K", "N", "time_ms", "flops", "bytes"]
```

Use `product()` loops with `tqdm` for progress:

```python
from itertools import product
from tqdm import tqdm

for M, K, N in tqdm(list(product(grid["M"], grid["K"], grid["N"]))):
    ...
```

## Benchmark Categories

| Module | Coverage |
|--------|----------|
| `matmul.py` | `torch.mm` over M×K×N grid. Memory-bound (small M) and compute-bound (large M). Supports float16/bfloat16/fp8. |
| `elementwise.py` | residual_add, rmsnorm, softmax, swiglu, rope — each with separate `BYTES_FACTORS` entry. |
| `flashattn.py` | `F.scaled_dot_product_attention` over (s_q, s_kv) grid. Prefer `flash_attn` on Linux, fallback to torch SDPA. |
| `cudagraph.py` | Each op under CUDA Graph replay — namespaced with `cudagraph_` prefix. |
| `launch_overhead.py` | CPU wall-clock vs GPU event-time slope analysis via numpy linear regression. |
| `gateddelta.py` | Hybrid-architecture kernels: gated delta rule scan (chunked prefill / recurrent decode) + causal depthwise conv1d. |

## GatedDelta (`bench/gateddelta.py`)

Measures two kernels for hybrid (e.g. Qwen3.5) architectures:

- **Gated delta rule scan** — chunked prefill / recurrent decode. Cost scales with `nvh` (hidden-dim based), not `nhk` (KV-dim based).
- **Causal depthwise conv1d** — elementwise-style kernel.

Prefer fused kernels from `fla` (flash-linear-attention) / `causal_conv1d`; fall back to the vendored torch reference implementation (same path as `modeling_qwen3_5.py`). Config grid comes from the `gateddelta:` section of `config/bench.yaml`; output is `bench/results/<gpu>/gateddelta.csv` (scan + conv1d rows).
