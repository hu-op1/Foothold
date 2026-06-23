"""Compare a foothold sim run against (optional) LLMServingSim output.

Reads the three sim artifacts and generates comparison plots + summary::

    <output_dir>/throughput.png     prompt + gen throughput over time
    <output_dir>/requests.png       running / waiting over time
    <output_dir>/latency.png        TTFT / TPOT / latency CDFs
    <output_dir>/summary.txt        mean / median / P90 / P95 / P99

Usage::

    # Standalone — just visualize a foothold sim run
    uv run python -m sim.validate -o out/my_run

    # Side-by-side with LLMServingSim
    uv run python -m sim.validate -o out/my_run \\
        --sim-csv path/to/sim.csv --sim-log path/to/sim.log
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Sequence


# ── public API ──────────────────────────────────────────────────────────

def run(output_dir: str, *,
        sim_csv: str | None = None,
        sim_log: str | None = None,
        title: str = "foothold sim",
        prefix: str = ""):
    """Generate validation plots + summary for a finished sim run.

    Args:
        output_dir: path to a finished sim run (contains timeseries.csv, requests.jsonl).
        sim_csv: optional LLMServingSim sim.csv for side-by-side comparison.
        sim_log: optional LLMServingSim sim.log for side-by-side comparison.
        title: plot title suffix.
        prefix: output filename prefix.
    """
    out = Path(output_dir)
    if not out.is_dir():
        print(f"Not a directory: {out}")
        return

    print(f"Loading foothold data from {out} ...")
    bench_reqs = _load_requests(out / "requests.jsonl")
    bench_ts = _load_timeseries(out / "timeseries.csv")
    bench_ttft, bench_tpot, bench_lat = _compute_latencies(bench_reqs)
    print(f"  {len(bench_reqs)} requests, {len(bench_ts)} timeseries rows")

    sim_ts, sim_ttft, sim_tpot, sim_lat = [], [], [], []
    has_sim = bool(sim_csv and sim_log)
    if has_sim:
        print("Loading LLMServingSim data ...")
        sim_reqs = _load_sim_csv(Path(sim_csv))
        sim_ts = _load_sim_log(Path(sim_log))
        sim_ttft, sim_tpot, sim_lat = _sim_latencies(sim_reqs)
        print(f"  {len(sim_reqs)} requests, {len(sim_ts)} timeseries rows")

    # ── Plot ───────────────────────────────────────────────────────
    print("Generating plots ...")

    if bench_ts:
        if has_sim and sim_ts:
            _plot_throughput(out, prefix,
                             [r["t"] for r in bench_ts],
                             [r["prompt_throughput"] for r in bench_ts],
                             [r["gen_throughput"] for r in bench_ts],
                             [r["t"] for r in sim_ts],
                             [r["prompt_throughput"] for r in sim_ts],
                             [r["gen_throughput"] for r in sim_ts],
                             title)
            _plot_requests(out, prefix,
                           [r["t"] for r in bench_ts],
                           [r["running"] for r in bench_ts],
                           [r["waiting"] for r in bench_ts],
                           [r["t"] for r in sim_ts],
                           [r["running"] for r in sim_ts],
                           [r["waiting"] for r in sim_ts],
                           title)
        else:
            _plot_throughput_single(out, prefix,
                                    [r["t"] for r in bench_ts],
                                    [r["prompt_throughput"] for r in bench_ts],
                                    [r["gen_throughput"] for r in bench_ts],
                                    title)
            _plot_requests_single(out, prefix,
                                  [r["t"] for r in bench_ts],
                                  [r["running"] for r in bench_ts],
                                  [r["waiting"] for r in bench_ts],
                                  title)

    if has_sim:
        _plot_latency_cdfs(out, prefix,
                           bench_ttft, sim_ttft,
                           bench_tpot, sim_tpot,
                           bench_lat, sim_lat,
                           title)
        _write_summary(out, prefix,
                       bench_ttft, sim_ttft,
                       bench_tpot, sim_tpot,
                       bench_lat, sim_lat,
                       "foothold", "LLMServingSim")
    else:
        _plot_latency_cdfs_single(out, prefix,
                                  bench_ttft, bench_tpot, bench_lat,
                                  title)
        _write_summary_single(out, prefix,
                              bench_ttft, bench_tpot, bench_lat,
                              "foothold")

    print(f"Done → {out}")


# ── CLI (python -m sim.validate) ───────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Validate foothold sim output (optionally vs LLMServingSim)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  uv run python -m sim.validate -o out/my_run\n"
               "  uv run python -m sim.validate -o out/my_run --sim-csv s.csv --sim-log s.log",
    )
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--sim-csv", default=None)
    p.add_argument("--sim-log", default=None)
    p.add_argument("--title", default="foothold sim")
    p.add_argument("--prefix", default="")
    args = p.parse_args()
    run(args.output_dir, sim_csv=args.sim_csv, sim_log=args.sim_log,
        title=args.title, prefix=args.prefix)


# ── foothold loaders ────────────────────────────────────────────────────

def _load_requests(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_timeseries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "t": float(row["t"]),
                "prompt_throughput": float(row["prompt_throughput"]),
                "gen_throughput": float(row["gen_throughput"]),
                "running": int(float(row["running"])),
                "waiting": int(float(row["waiting"])),
                "kv_cache_pct": float(row.get("kv_cache_pct", 0.0)),
            })
    return out


def _compute_latencies(reqs: list[dict]) -> tuple[list[float], list[float], list[float]]:
    ttft, tpot, lat = [], [], []
    for r in reqs:
        arr = r.get("arrival_time")
        first = r.get("first_token_ts")
        last = r.get("last_token_ts")
        out_toks = max(1, int(r.get("output_toks", 1)))

        if arr is not None and first is not None:
            ttft.append((first - arr) * 1000.0)
        if arr is not None and last is not None:
            lat.append((last - arr) * 1000.0)
        if first is not None and last is not None and out_toks > 1:
            tpot.append((last - first) / (out_toks - 1) * 1000.0)
    return ttft, tpot, lat


# ── LLMServingSim loaders ───────────────────────────────────────────────

def _load_sim_csv(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "ttft_ns": float(row["TTFT"]),
                "tpot_ns": float(row["TPOT"]),
                "latency_ns": float(row["latency"]),
            })
    return out


def _sim_latencies(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    ttft = [r["ttft_ns"] / 1e6 for r in rows]
    tpot = [r["tpot_ns"] / 1e6 for r in rows]
    lat = [r["latency_ns"] / 1e6 for r in rows]
    return ttft, tpot, lat


_TS_RE = re.compile(r"^\[(\d+\.?\d*)s\]")
_TPUT_RE = re.compile(
    r"Avg prompt throughput:\s*(\d+\.?\d*).*generation throughput:\s*(\d+\.?\d*)"
)
_INST_RE = re.compile(
    r"Running Instance\[(\d+)\]:\s*(\d+) reqs, Waiting:\s*(\d+) reqs"
)


def _load_sim_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    cur: dict | None = None

    def _flush(c):
        if c is not None and "t" in c:
            rows.append({
                "t": c["t"],
                "prompt_throughput": c.get("prompt_throughput", 0.0),
                "gen_throughput": c.get("gen_throughput", 0.0),
                "running": c.get("running", 0),
                "waiting": c.get("waiting", 0),
            })

    with path.open(encoding="utf-8") as f:
        for line in f:
            m_ts = _TS_RE.match(line)
            if m_ts:
                _flush(cur)
                cur = {"t": float(m_ts.group(1)), "running": 0, "waiting": 0}
                m_t = _TPUT_RE.search(line)
                if m_t:
                    cur["prompt_throughput"] = float(m_t.group(1))
                    cur["gen_throughput"] = float(m_t.group(2))
                continue
            m_i = _INST_RE.search(line)
            if m_i and cur is not None:
                cur["running"] += int(m_i.group(2))
                cur["waiting"] += int(m_i.group(3))
    _flush(cur)
    return rows


# ── plot helpers ────────────────────────────────────────────────────────

def _cdf(series):
    if not series:
        return [], []
    xs = sorted(series)
    n = len(xs)
    return xs, [(i + 1) / n for i in range(n)]


def _percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


_STATS = [
    ("Mean", _mean),
    ("Median", lambda xs: _percentile(xs, 50)),
    ("P90", lambda xs: _percentile(xs, 90)),
    ("P95", lambda xs: _percentile(xs, 95)),
    ("P99", lambda xs: _percentile(xs, 99)),
]


# ── side-by-side plots ──────────────────────────────────────────────────

def _plot_throughput(out, prefix, ft_t, ft_p, ft_g, st_t, st_p, st_g, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(ft_t, ft_p, label="foothold", color="C0")
    axes[0].plot(st_t, st_p, label="LLMServingSim", color="C1", linestyle="--")
    axes[0].set_ylabel("Prompt tokens/s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ft_t, ft_g, label="foothold", color="C0")
    axes[1].plot(st_t, st_g, label="LLMServingSim", color="C1", linestyle="--")
    axes[1].set_ylabel("Generation tokens/s")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Throughput — {title}")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}throughput.png" if prefix else out / "throughput.png", dpi=150)
    plt.close(fig)


def _plot_requests(out, prefix, ft_t, ft_r, ft_w, st_t, st_r, st_w, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(ft_t, ft_r, label="foothold", color="C0")
    axes[0].plot(st_t, st_r, label="LLMServingSim", color="C1", linestyle="--")
    axes[0].set_ylabel("Running")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ft_t, ft_w, label="foothold", color="C0")
    axes[1].plot(st_t, st_w, label="LLMServingSim", color="C1", linestyle="--")
    axes[1].set_ylabel("Waiting")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Running / Waiting — {title}")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}requests.png" if prefix else out / "requests.png", dpi=150)
    plt.close(fig)


def _plot_latency_cdfs(out, prefix, ft_tt, st_tt, ft_tp, st_tp, ft_lat, st_lat, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (label, fa, sa) in zip(
        axes,
        [("TTFT (ms)", ft_tt, st_tt),
         ("TPOT (ms)", ft_tp, st_tp),
         ("Latency (ms)", ft_lat, st_lat)],
    ):
        for series, name, color, style in [
            (fa, "foothold", "C0", "-"),
            (sa, "LLMServingSim", "C1", "--"),
        ]:
            xs, ys = _cdf(series)
            ax.plot(xs, ys, label=name, color=color, linestyle=style)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Latency CDFs — {title}")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}latency.png" if prefix else out / "latency.png", dpi=150)
    plt.close(fig)


# ── standalone plots (foothold only) ────────────────────────────────────

def _plot_throughput_single(out, prefix, t, prompt, gen, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, prompt, color="C0")
    axes[0].set_ylabel("Prompt tokens/s")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, gen, color="C0")
    axes[1].set_ylabel("Generation tokens/s")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Throughput — {title}")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}throughput.png" if prefix else out / "throughput.png", dpi=150)
    plt.close(fig)


def _plot_requests_single(out, prefix, t, running, waiting, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, running, color="C0")
    axes[0].set_ylabel("Running")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, waiting, color="C0")
    axes[1].set_ylabel("Waiting")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Running / Waiting — {title}")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}requests.png" if prefix else out / "requests.png", dpi=150)
    plt.close(fig)


def _plot_latency_cdfs_single(out, prefix, ttft, tpot, lat, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (label, series) in zip(
        axes,
        [("TTFT (ms)", ttft), ("TPOT (ms)", tpot), ("Latency (ms)", lat)],
    ):
        xs, ys = _cdf(series)
        ax.plot(xs, ys, color="C0")
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Latency CDFs — {title}")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}latency.png" if prefix else out / "latency.png", dpi=150)
    plt.close(fig)


# ── summary ─────────────────────────────────────────────────────────────

def _write_summary(out, prefix, ft_tt, st_tt, ft_tp, st_tp, ft_lat, st_lat,
                   name_a, name_b):
    lines = []
    header = f"{'Metric':<25}{name_a:>12}{name_b:>12}{'Diff%':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for metric, va, sa in [("TTFT", ft_tt, st_tt),
                            ("TPOT", ft_tp, st_tp),
                            ("Latency", ft_lat, st_lat)]:
        for stat_label, fn in _STATS:
            v = fn(va) if va else float("nan")
            s = fn(sa) if sa else float("nan")
            diff = (s - v) / v * 100.0 if v else float("nan")
            lines.append(
                f"{metric + ' ' + stat_label:<25}"
                f"{v:>12.1f}{s:>12.1f}{diff:>+9.1f}%"
            )
        lines.append("")

    fname = f"{prefix}summary.txt" if prefix else "summary.txt"
    (out / fname).write_text("\n".join(lines))


def _write_summary_single(out, prefix, ttft, tpot, lat, name):
    lines = [f"{'Metric':<25}{name:>12}"]
    lines.append("-" * 37)

    for metric, vals in [("TTFT", ttft), ("TPOT", tpot), ("Latency", lat)]:
        for stat_label, fn in _STATS:
            v = fn(vals) if vals else float("nan")
            lines.append(f"{metric + ' ' + stat_label:<25}{v:>12.1f}")
        lines.append("")

    fname = f"{prefix}summary.txt" if prefix else "summary.txt"
    (out / fname).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
