"""Simulation time-series recorder.

Captures per-tick scheduler state and per-request lifecycle timestamps
during a simulation run, then writes the three standard artifacts::

    <output_dir>/meta.json
    <output_dir>/requests.jsonl
    <output_dir>/timeseries.csv

Schema matches LLMServingSim's bench output so the validate module
can compare the two without translation.
"""

from __future__ import annotations

import csv
import json
import time as _time
from pathlib import Path
from typing import Any

META_SCHEMA_VERSION = 1


def _json_default(o):
    """Fallback serializer for numpy types that Python's json can't handle."""
    name = type(o).__name__
    if name in ("bool", "bool_"):
        return bool(o)
    if name in ("float64", "float32", "float16", "int64", "int32", "int16", "int8",
                "uint64", "uint32", "uint16", "uint8", "ndarray"):
        return o.item()
    raise TypeError(f"Object of type {name} is not JSON serializable")


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
            # New bucket — flush the old one first, then fill any tick
            # buckets the sim clock skipped over.  A single long step or an
            # idle catch-up can advance the clock across several
            # tick_seconds windows; without zero-throughput rows for those
            # buckets the timeseries has holes, and downstream trapezoidal
            # integration (validate/plot.py _integral_over) bridges each
            # hole as if tokens flowed continuously, over-counting the
            # prompt/generation area.
            prev_t = self._cur_bucket["t"]
            prev_state = {k: self._cur_bucket[k]
                          for k in ("running", "waiting", "kv_cache_pct")}
            self._flush_bucket()
            self._fill_skipped_buckets(prev_t, bucket_idx, prev_state)
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

    def _fill_skipped_buckets(self, prev_t: float, new_bucket_idx: int,
                              state: dict) -> None:
        """Emit zero-throughput rows for tick buckets the clock skipped.

        ``prev_t`` is the just-flushed bucket's ``t`` (=(old_idx + 1) * tick).
        Every bucket index in ``(old_idx, new_bucket_idx)`` gets a row with
        zero throughput and the last observed running/waiting/kv state, so
        the timeseries stays contiguous (one row per tick_seconds).
        """
        old_idx = int(prev_t / self.tick_seconds) - 1
        for idx in range(old_idx + 1, new_bucket_idx):
            self._tick_rows.append({
                "t": round((idx + 1) * self.tick_seconds, 3),
                "running": state.get("running", 0),
                "waiting": state.get("waiting", 0),
                "kv_cache_pct": state.get("kv_cache_pct", 0.0),
                "prompt_throughput": 0.0,
                "gen_throughput": 0.0,
            })

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
            "finish_reason": request.finish_reason.value if request.finish_reason else None,
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
        (out / "meta.json").write_text(json.dumps(payload, indent=2, default=_json_default))

    def _write_requests(self, out: Path) -> None:
        with (out / "requests.jsonl").open("w") as f:
            for r in self._request_records:
                f.write(json.dumps(r, default=_json_default) + "\n")

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
