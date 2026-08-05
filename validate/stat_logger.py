"""Custom vLLM stat logger that captures per-iteration scheduler + iteration stats.

Subclasses ``StatLoggerBase`` (from ``vllm.v1.metrics.loggers``) and stores
every ``record()`` call as a row in memory.  The embedded runner downsamples
these rows to ``tick_seconds`` buckets when writing ``timeseries.csv``.

vLLM calls ``record()`` once per scheduling iteration, giving us exact
per-iteration token counts — no cumulative counter desync, no polling gaps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.metrics.loggers import StatLoggerBase


@dataclass
class _Sample:
    t: float
    num_running: int
    num_waiting: int
    num_prompt_tokens: int
    num_generation_tokens: int
    kv_cache_pct: float
    engine_idx: int


class BenchStatLogger(StatLoggerBase):
    """Records per-iteration vLLM scheduler stats into an in-memory list.

    One instance is created per DP engine (vLLM passes ``engine_index``).
    Samples from all engines share a single ``samples`` list at the class
    level so the runner can write them out as one timeseries.
    """

    samples: list[_Sample] = []
    _t0: float | None = None

    def __init__(self, vllm_config: Any, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        if BenchStatLogger._t0 is None:
            BenchStatLogger._t0 = time.monotonic()

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None:
            return
        running = getattr(scheduler_stats, "num_running_reqs", 0)
        waiting = getattr(scheduler_stats, "num_waiting_reqs", 0)
        cache_pct = getattr(scheduler_stats, "gpu_cache_usage", 0.0) * 100.0

        prompt_toks = 0
        gen_toks = 0
        if iteration_stats is not None:
            prompt_toks = getattr(iteration_stats, "num_prompt_tokens", 0)
            gen_toks = getattr(iteration_stats, "num_generation_tokens", 0)

        BenchStatLogger.samples.append(_Sample(
            t=time.monotonic() - (BenchStatLogger._t0 or time.monotonic()),
            num_running=running,
            num_waiting=waiting,
            num_prompt_tokens=prompt_toks,
            num_generation_tokens=gen_toks,
            kv_cache_pct=cache_pct,
            engine_idx=engine_idx if engine_idx else self.engine_index,
        ))

    def log_engine_initialized(self) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.samples = []
        cls._t0 = None

    @classmethod
    def downsample_to_csv_rows(
        cls, tick_seconds: float
    ) -> tuple[list[str], list[list]]:
        """Bucket raw samples into ``tick_seconds`` windows.

        Within each tick, sum prompt/gen tokens across engines (matches how
        bench-side throughput reports aggregate over DP), and take the last
        running/waiting/cache_pct snapshot in the bucket.  Buckets with no
        samples (idle engine, e.g. tool-call gaps) get a zero-throughput row
        carrying the last observed state, keeping the timeseries contiguous.
        """
        header = [
            "t",
            "prompt_throughput",
            "gen_throughput",
            "running",
            "waiting",
            "kv_cache_pct",
        ]
        if not cls.samples:
            return header, []

        samples = sorted(cls.samples, key=lambda s: s.t)
        end_t = samples[-1].t
        rows: list[list] = []
        last_state = (0, 0, 0.0)  # running, waiting, kv_cache_pct

        bucket_idx = 0
        while bucket_idx * tick_seconds <= end_t:
            t_lo = bucket_idx * tick_seconds
            t_hi = t_lo + tick_seconds
            in_bucket = [s for s in samples if t_lo <= s.t < t_hi]
            if not in_bucket:
                # No scheduler iteration in this window (engine idle, e.g.
                # tool-call gaps): emit a zero-throughput row with the last
                # observed state so the timeseries stays contiguous.  Holes
                # would make trapezoidal area integration
                # (validate/plot.py _integral_over) bridge each gap with a
                # linear ramp, over-counting prompt/gen area — mirrors
                # sim/recorder.py _fill_skipped_buckets.
                rows.append([
                    round(t_hi, 3),
                    0.0,
                    0.0,
                    last_state[0],
                    last_state[1],
                    last_state[2],
                ])
                bucket_idx += 1
                continue

            prompt_sum = sum(s.num_prompt_tokens for s in in_bucket)
            gen_sum = sum(s.num_generation_tokens for s in in_bucket)

            latest_per_engine: dict[int, _Sample] = {}
            for s in in_bucket:
                if (
                    s.engine_idx not in latest_per_engine
                    or s.t > latest_per_engine[s.engine_idx].t
                ):
                    latest_per_engine[s.engine_idx] = s
            running = sum(s.num_running for s in latest_per_engine.values())
            waiting = sum(s.num_waiting for s in latest_per_engine.values())
            cache_pct = (
                sum(s.kv_cache_pct for s in latest_per_engine.values())
                / max(1, len(latest_per_engine))
            )

            rows.append([
                round(t_hi, 3),
                round(prompt_sum / tick_seconds, 1),
                round(gen_sum / tick_seconds, 1),
                running,
                waiting,
                round(cache_pct, 2),
            ])
            last_state = (running, waiting, round(cache_pct, 2))
            bucket_idx += 1

        return header, rows
