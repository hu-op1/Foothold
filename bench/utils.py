"""Benchmark utilities: CUDA timer, warmup, csv output, checkpoint/resume."""

import csv
import os
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


def benchmark(fn, min_time_ms=200, max_iters=10000, calib_iters=20):
    """Return average execution time of fn in milliseconds.

    Runs enough iterations to accumulate at least *min_time_ms* of GPU time
    (capped at *max_iters*).  Adapts iteration count to kernel size so that
    both a 5 µs rmsnorm and a 50 ms matmul get statistically stable averages
    without wasting wall-clock.

    *calib_iters* controls the initial calibration batch size — higher values
    give better per-iteration time estimates at the cost of more warm-up work.
    """
    timer = CudaTimer()
    total = 0.0
    iters = 0

    # ── calibration batch ──
    calib = min(calib_iters, max_iters)
    for _ in range(calib):
        with timer:
            fn()
        total += timer.elapsed_ms
    iters = calib

    if total >= min_time_ms or iters >= max_iters:
        return total / iters

    # ── scale up to fill the time budget ──
    per_iter = total / iters if iters > 0 else 0.001
    remaining = min_time_ms - total
    extra = min(int(remaining / per_iter) + 1, max_iters - iters)

    for _ in range(extra):
        with timer:
            fn()
        total += timer.elapsed_ms
    iters += extra

    return total / iters


def auto_warmup_iters(fn, min_time_ms, max_iters, calib_iters, ratio=0.1):
    """Return warmup iters as a fraction of estimated benchmark iters.

    Runs a quick calibration to estimate how many iterations *benchmark()*
    would execute, then returns ``max(1, int(total_est * ratio))``.
    """
    timer = CudaTimer()
    total = 0.0
    calib = min(calib_iters, max_iters)
    for _ in range(calib):
        with timer:
            fn()
        total += timer.elapsed_ms

    if total >= min_time_ms or calib >= max_iters:
        total_est = calib
    else:
        per_iter = total / calib if calib > 0 else 0.001
        total_est = min(max_iters, int(min_time_ms / per_iter) + 1)

    return max(1, int(total_est * ratio))


def check_memory(required_gb, max_gb=7.5):
    """Check if required_gb fits in available and allowed memory."""
    free_gb, _ = torch.cuda.mem_get_info()
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


# ── checkpoint / resume ────────────────────────────────────────────────────

def _csv_key(result, key_fields):
    """Extract a hashable key tuple from a result dict.

    All values are coerced through str then attempted as int for type-safe
    comparison between CSV-loaded strings and in-memory Python values.
    """
    vals = []
    for f in key_fields:
        v = result.get(f)
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                pass
        vals.append(v)
    return tuple(vals)


def load_completed_keys(csv_path, key_fields):
    """Read existing CSV and return {key_tuple, ...} for resume."""
    keys = set()
    if not os.path.exists(csv_path):
        return keys
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip OOM rows — they need to be re-measured
                if row.get("time_ms") == "OOM":
                    continue
                keys.add(_csv_key(row, key_fields))
    except Exception:
        pass
    return keys


def append_csv_row(path, fieldnames, row_dict):
    """Append a single row to CSV. Writes header if file is new or empty."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
