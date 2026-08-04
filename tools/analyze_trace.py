"""Analyze trace JSONL workload distribution.

Streams JSONL traces line by line (no full-file load), auto-detects the format
per file — ShareGPT flat (one request per line) or Agentic session (one session
per line with chained sub_requests) — and prints workload distribution stats:

  - request/session counts, arrival time span, effective arrival rate
  - input/output token length percentiles (min/p50/p90/p99/max/mean) + total
  - decile table + ASCII histogram for input and output lengths
  - agentic-only: sub-requests per session, tool_duration stats

Usage:
  uv run python tools/analyze_trace.py traces/qwen3-8b-conversation.jsonl
  uv run python tools/analyze_trace.py traces/*.jsonl --bins 20
"""

import argparse
import json
import statistics
from pathlib import Path


def load_entry(line):
    """Parse a JSONL line into a (kind, entry) tuple.

    kind is "sharegpt" or "agentic" depending on top-level keys.
    """
    obj = json.loads(line)
    if "sub_requests" in obj:
        return "agentic", obj
    return "sharegpt", obj


def percentiles(values, pcts=(50, 90, 99)):
    """Compute percentile values from a sorted list."""
    n = len(values)
    if n == 0:
        return [0.0] * len(pcts)
    return [values[min(n - 1, int(p / 100 * n))] for p in pcts]


def decile_table(label, values):
    """Print a compact decile row (0/10/20/.../90/100)."""
    n = len(values)
    if n == 0:
        print(f"  {label:<6} (empty)")
        return
    print(f"  {label:<6} deciles: " + " ".join(
        f"{values[min(n - 1, int(p / 100 * n))]}" for p in range(0, 101, 10)))


def histogram(label, values, bins):
    """Print an ASCII histogram with `bins` equal-width buckets."""
    if not values:
        return
    lo, hi = values[0], values[-1]
    if hi == lo:
        print(f"  {label} histogram: all values = {lo}")
        return
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        counts[min(bins - 1, idx)] += 1
    peak = max(counts)
    print(f"  {label} histogram ({bins} bins, min={lo}, max={hi}):")
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / peak * 40)) if c else ""
        lo_b, hi_b = lo + i * width, lo + (i + 1) * width
        print(f"    [{lo_b:>8.0f} - {hi_b:>8.0f}) {c:>6d}  {bar}")


def analyze_sharegpt(path):
    """Analyze a flat trace: one request per line."""
    arrivals, inputs, outputs = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, entry = load_entry(line)
            arrivals.append(entry["arrival_time_ns"] / 1e9)
            inputs.append(entry["input_toks"])
            outputs.append(entry["output_toks"])
    print(f"  format: sharegpt")
    _print_common_stats(len(inputs), arrivals, inputs, outputs)


def analyze_agentic(path):
    """Analyze a session trace: one session per line with sub_requests."""
    arrivals, inputs, outputs, n_subs, tool_durs = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, session = load_entry(line)
            arrivals.append(session.get("arrival_time_ns", 0) / 1e9)
            subs = session.get("sub_requests", [])
            n_subs.append(len(subs))
            for sub in subs:
                inputs.append(sub["input_toks"])
                outputs.append(sub["output_toks"])
                tool_durs.append(sub.get("tool_duration_ns", 0) / 1e9)
    n_tools = sum(1 for t in tool_durs if t > 0)
    print(f"  format: agentic  sessions={len(n_subs)}  sub_requests={len(inputs)}"
          f"  tool_calls={n_tools} ({n_tools / max(1, len(tool_durs)) * 100:.0f}%)")
    if n_subs:
        print(f"  sub_requests/session: min={min(n_subs)} mean={statistics.mean(n_subs):.1f} max={max(n_subs)}")
    if tool_durs:
        td = sorted(tool_durs)
        print(f"  tool_duration (s): p50={td[len(td)//2]:.2f} mean={statistics.mean(td):.2f} "
              f"p99={percentiles(td, (99,))[0]:.2f} max={td[-1]:.2f}")
    _print_common_stats(len(n_subs), arrivals, inputs, outputs)


def _print_common_stats(n_reqs, arrivals, inputs, outputs):
    in_s, out_s = sorted(inputs), sorted(outputs)
    p50, p90, p99 = percentiles(in_s)
    o50, o90, o99 = percentiles(out_s)
    span = (arrivals[-1] - arrivals[0]) if arrivals else 0.0
    rate = n_reqs / span if span > 0 else 0.0

    print(f"  requests={n_reqs}  span={span:.1f}s  rate={rate:.2f} req/s")
    print(f"  input_toks : min={min(in_s)} p50={p50} p90={p90} p99={p99} "
          f"max={max(in_s)} mean={statistics.mean(in_s):.1f} total={sum(inputs)}")
    print(f"  output_toks: min={min(out_s)} p50={o50} p90={o90} p99={o99} "
          f"max={max(out_s)} mean={statistics.mean(out_s):.1f} total={sum(outputs)}")
    decile_table("input", in_s)
    decile_table("output", out_s)
    histogram("input", in_s, args.bins)
    histogram("output", out_s, args.bins)


def main():
    global args
    parser = argparse.ArgumentParser(description="Print workload distribution of JSONL traces")
    parser.add_argument("paths", nargs="+", help="Trace file(s) to analyze")
    parser.add_argument("--bins", type=int, default=10, help="Histogram bins (default 10)")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]

    for path in paths:
        if not path.exists():
            print(f"SKIP: not found: {path}")
            continue
        print(f"\n=== {path.name} ({path.stat().st_size / 1e6:.1f} MB) ===")
        try:
            with open(path, encoding="utf-8") as f:
                kind, _ = load_entry(f.readline())
        except json.JSONDecodeError as e:
            print(f"  ERROR: invalid JSON: {e}")
            continue
        (analyze_agentic if kind == "agentic" else analyze_sharegpt)(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
