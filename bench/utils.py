"""Benchmark utilities: CUDA timer, warmup, csv output."""

import os
import csv
import torch


class CudaTimer:
    """Precise GPU timer using CUDA events."""

    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *args):
        self._end.record()
        torch.cuda.synchronize()

    @property
    def elapsed_ms(self):
        return self._start.elapsed_time(self._end)


def warmup(fn, iters=5):
    """Warm up GPU by running fn several times."""
    for _ in range(iters):
        fn()


def benchmark(fn, iters=100):
    """Return average execution time of fn in milliseconds."""
    timer = CudaTimer()
    total = 0.0
    for _ in range(iters):
        with timer:
            fn()
        total += timer.elapsed_ms
    return total / iters


def save_csv(results, path):
    """Save list of dicts to csv. Creates parent directories if needed."""
    if not results:
        print(f"  [skip] No results to save for {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in results for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved {len(results)} rows → {path}")


def estimate_memory_gb(b, s, h, dtype_size=2):
    """Estimate GPU memory (GiB) for a [b, s, h] fp16 tensor."""
    return (b * s * h * dtype_size) / (1024**3)


def check_memory(required_gb, max_gb=7.5):
    """Check if required_gb fits in available and allowed memory."""
    free_gb, total_gb = torch.cuda.mem_get_info()
    free_gb = free_gb / (1024**3)
    return required_gb < free_gb and required_gb <= max_gb


def get_compute_capability():
    """Return (major, minor) CUDA compute capability for the current GPU."""
    major, minor = torch.cuda.get_device_capability()
    return major, minor


def supports_float8_matmul():
    """Check if GPU supports float8 matmul via torch._scaled_mm.

    Requires sm ≥ 8.9 (Ada Lovelace RTX 4090 / Hopper H100+).
    RTX 3090 (sm 8.6) and older do NOT support float8 matmul in hardware.
    """
    major, minor = get_compute_capability()
    return major * 10 + minor >= 89  # sm_89 = Ada, sm_90 = Hopper


def make_float8_tensor(*size, device="cuda"):
    """Create a float8_e4m3fn tensor by converting from float16.

    torch.randn does not support float8 dtypes directly.
    """
    return torch.randn(*size, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
