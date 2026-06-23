"""Simulation time-series recorder.

Captures per-tick scheduler state and per-request lifecycle timestamps
during a simulation run, then writes the three standard artifacts::

    <output_dir>/meta.json
    <output_dir>/requests.jsonl
    <output_dir>/timeseries.csv

Schema matches LLMServingSim's bench output so the same validate.py
can compare the two without translation.
"""

from __future__ import annotations

import csv
import json
import time as _time
from pathlib import Path
from typing import Any

META_SCHEMA_VERSION = 1


class SimRecorder:
    """Records per-tick + per-request data for one simulation run."""

    def __init__(self, *, tick_seconds: float = 0.5):
        self.tick_seconds = tick_seconds
        self._tick_rows: list[dict] = []
        self._request_records: list[dict] = []
        self._started_at: str | None = None
        self._finished_at: str | None = None

        # Per-tick accumulator state
        self._cur_bucket: dict | None = None
        self._bucket_prompt_tokens: int = 0
        self._bucket_gen_tokens: int = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._started_at = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())

    def finish(self) -> None:
        """Close the final tick bucket so the last interval is captured."""
        self._flush_bucket()
        self._finished_at = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())

    # ── per-tick recording ───────────────────────────────────────────────

    def record_tick(
        self,
        clock: float,
        running: int,
        waiting: int,
        prompt_tokens: int,
        gen_tokens: int,
        kv_cache_pct: float,
    ) -> None:
        """Accumulate tokens into the current time bucket.

        Called once per simulation step.  The recorder downsamples to
        *tick_seconds*-wide buckets so the CSV stays small while
        preserving the shape of the throughput curve.
        """
        bucket_idx = int(clock // self.tick_seconds)
        bucket_end = (bucket_idx + 1) * self.tick_seconds

        if self._cur_bucket is None:
            self._cur_bucket = {
                "t": round(bucket_end, 3),
                "running": running,
                "waiting": waiting,
                "kv_cache_pct": kv_cache_pct,
            }
            self._bucket_prompt_tokens = prompt_tokens
            self._bucket_gen_tokens = gen_tokens
            return

        if bucket_end == self._cur_bucket["t"]:
            # Still in same bucket — accumulate
            self._bucket_prompt_tokens += prompt_tokens
            self._bucket_gen_tokens += gen_tokens
            self._cur_bucket["running"] = running
            self._cur_bucket["waiting"] = waiting
            self._cur_bucket["kv_cache_pct"] = kv_cache_pct
        else:
            # New bucket — flush old one first
            self._flush_bucket()
            self._cur_bucket = {
                "t": round(bucket_end, 3),
                "running": running,
                "waiting": waiting,
                "kv_cache_pct": kv_cache_pct,
            }
            self._bucket_prompt_tokens = prompt_tokens
            self._bucket_gen_tokens = gen_tokens

    def _flush_bucket(self) -> None:
        if self._cur_bucket is None:
            return
        self._cur_bucket["prompt_throughput"] = round(
            self._bucket_prompt_tokens / self.tick_seconds, 1
        )
        self._cur_bucket["gen_throughput"] = round(
            self._bucket_gen_tokens / self.tick_seconds, 1
        )
        self._tick_rows.append(self._cur_bucket)
        self._cur_bucket = None
        self._bucket_prompt_tokens = 0
        self._bucket_gen_tokens = 0

    # ── per-request recording ────────────────────────────────────────────

    def record_request(self, request) -> None:
        """Capture lifecycle timestamps for one completed request.

        Call this when the request finishes (from MetricsCollector.record
        or directly after drain_finished).
        """
        if request.finish_time is None or request.arrival_time is None:
            return

        first = request.arrival_time + (request.ttft or 0.0) if request.ttft is not None else None

        self._request_records.append({
            "request_id": request.request_id,
            "input_toks": request.prompt_len,
            "output_toks": request.num_output_tokens,
            "arrival_time": request.arrival_time,
            "queued_ts": request.arrival_time,  # same clock domain in our sim
            "scheduled_ts": getattr(request, "scheduled_ts", None) or request.arrival_time,
            "first_token_ts": first,
            "last_token_ts": request.finish_time,
        })

    # ── write artifacts ──────────────────────────────────────────────────

    def write(self, output_dir: str | Path,
              model: str = "",
              trace_path: str = "",
              extra_meta: dict | None = None) -> None:
        """Write meta.json, requests.jsonl, timeseries.csv."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._write_meta(out, model=model, trace_path=trace_path,
                         extra_meta=extra_meta)
        self._write_requests(out)
        self._write_timeseries(out)

    def _write_meta(self, out: Path, *, model: str, trace_path: str,
                    extra_meta: dict | None) -> None:
        payload: dict[str, Any] = {
            "schema_version": META_SCHEMA_VERSION,
            "model": model,
            "simulator": "foothold-pd-sim",
            "dataset_path": trace_path,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "num_requests": len(self._request_records),
        }
        if extra_meta:
            payload.update(extra_meta)
        (out / "meta.json").write_text(json.dumps(payload, indent=2))

    def _write_requests(self, out: Path) -> None:
        with (out / "requests.jsonl").open("w") as f:
            for r in self._request_records:
                f.write(json.dumps(r) + "\n")

    def _write_timeseries(self, out: Path) -> None:
        header = [
            "t", "prompt_throughput", "gen_throughput",
            "running", "waiting", "kv_cache_pct",
        ]
        with (out / "timeseries.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in self._tick_rows:
                w.writerow([
                    row["t"],
                    row.get("prompt_throughput", 0.0),
                    row.get("gen_throughput", 0.0),
                    row.get("running", 0),
                    row.get("waiting", 0),
                    row.get("kv_cache_pct", 0.0),
                ])
