"""Plot helpers and statistics for sim/vLLM output visualization."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def load_requests(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_timeseries(path: Path) -> list[dict]:
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


def compute_latencies(reqs: list[dict]) -> tuple[list[float], list[float], list[float]]:
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


def plot_throughput(out_dir: Path, prefix: str, datasets: list[dict], title: str) -> None:
    """Multi-line throughput chart.

    datasets: list of {"t": [...], "prompt": [...], "gen": [...],
                        "label": str, "color": str, "linestyle": str}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ds in datasets:
        axes[0].plot(ds["t"], ds["prompt"], label=ds["label"],
                     color=ds["color"], linestyle=ds["linestyle"])
        axes[1].plot(ds["t"], ds["gen"], label=ds["label"],
                     color=ds["color"], linestyle=ds["linestyle"])

    axes[0].set_ylabel("Prompt tokens/s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("Generation tokens/s")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Throughput — {title}")
    fig.tight_layout()
    fname = f"{prefix}throughput.png" if prefix else "throughput.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def plot_requests(out_dir: Path, prefix: str, datasets: list[dict], title: str) -> None:
    """Multi-line running/waiting chart.

    datasets: list of {"t": [...], "running": [...], "waiting": [...],
                        "label": str, "color": str, "linestyle": str}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ds in datasets:
        axes[0].plot(ds["t"], ds["running"], label=ds["label"],
                     color=ds["color"], linestyle=ds["linestyle"])
        axes[1].plot(ds["t"], ds["waiting"], label=ds["label"],
                     color=ds["color"], linestyle=ds["linestyle"])

    axes[0].set_ylabel("Running")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("Waiting")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Running / Waiting — {title}")
    fig.tight_layout()
    fname = f"{prefix}requests.png" if prefix else "requests.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def plot_latency_cdfs(out_dir: Path, prefix: str, datasets: list[dict], title: str) -> None:
    """Multi-line latency CDF chart.

    datasets: list of {"ttft": [...], "tpot": [...], "lat": [...],
                        "label": str, "color": str, "linestyle": str}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (xlabel, key) in zip(
        axes,
        [("TTFT (ms)", "ttft"), ("TPOT (ms)", "tpot"), ("Latency (ms)", "lat")],
    ):
        for ds in datasets:
            xs, ys = _cdf(ds[key])
            ax.plot(xs, ys, label=ds["label"],
                    color=ds["color"], linestyle=ds["linestyle"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Latency CDFs — {title}")
    fig.tight_layout()
    fname = f"{prefix}latency.png" if prefix else "latency.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def write_summary(out_dir: Path, prefix: str, latency_datasets: list[dict]) -> None:
    """Multi-column latency summary.

    latency_datasets: list of {"ttft": [...], "tpot": [...], "lat": [...],
                                "label": str}
    First dataset is the reference for Diff% columns.
    """
    labels = [ds["label"] for ds in latency_datasets]
    ref_label = labels[0]

    col_w = 12
    lines = []
    header = f"{'Metric':<25}"
    for lb in labels:
        header += f"{lb:>{col_w}}"
    for lb in labels[1:]:
        header += f"{'Δ vs ' + ref_label:>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    metrics = [
        ("TTFT", "ttft"),
        ("TPOT", "tpot"),
        ("Latency", "lat"),
    ]

    for metric_name, key in metrics:
        for stat_label, fn in _STATS:
            vals = [fn(ds[key]) if ds[key] else float("nan") for ds in latency_datasets]
            ref = vals[0]
            line = f"{metric_name + ' ' + stat_label:<25}"
            for v in vals:
                line += f"{v:>{col_w}.1f}"
            for v in vals[1:]:
                diff = (v - ref) / ref * 100.0 if ref and not (ref != ref) else float("nan")
                line += f"{diff:>{col_w}.1f}%"
            lines.append(line)
        lines.append("")

    fname = f"{prefix}summary.txt" if prefix else "summary.txt"
    (out_dir / fname).write_text("\n".join(lines))


# ── LLMServingSim log / CSV loaders ─────────────────────────────────────

_TS_RE = re.compile(r"^\[(\d+\.?\d*)s\]")
_TPUT_RE = re.compile(
    r"Avg prompt throughput:\s*(\d+\.?\d*).*generation throughput:\s*(\d+\.?\d*)"
)
_INST_RE = re.compile(
    r"Running Instance\[(\d+)\]:\s*(\d+) reqs, Waiting:\s*(\d+) reqs"
)


def load_sim_log(path: Path) -> list[dict]:
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


def load_sim_csv(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "ttft_ns": float(row["TTFT"]),
                "tpot_ns": float(row["TPOT"]),
                "latency_ns": float(row["latency"]),
            })
    return out


def sim_latencies(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    ttft = [r["ttft_ns"] / 1e6 for r in rows]
    tpot = [r["tpot_ns"] / 1e6 for r in rows]
    lat = [r["latency_ns"] / 1e6 for r in rows]
    return ttft, tpot, lat
