"""Schedule/execute pipeline model — overlaps CPU scheduling with GPU execution.

In vLLM, the CPU scheduler can prepare the next batch while the current batch
executes on the GPU.  This is mandatory for pipeline parallelism (PP > 1,
microbatch pipeline) and optional for colocated inference (async scheduling,
where the CPU assumes decode will produce 1 token and pre-allocates).

This module provides a minimal two-stage pipeline model: the CPU *schedule*
stage and the GPU *execute* stage.  Each stage tracks its own ``busy_until``
timestamp.  A step advances both stages, overlapping schedule of step N+1
with execution of step N.

Usage::

    pipeline = ScheduleExecutePipeline()
    for step in simulation:
        step_time = predict_step(...)
        sched_time = estimate_schedule_time(running, waiting)
        clock = pipeline.step(clock, sched_time, step_time)

When PP=1 and async_scheduling=False, the pipeline degenerates to serial
behaviour: schedule_time + step_time added linearly (no overlap).
"""

from dataclasses import dataclass


@dataclass
class ScheduleExecutePipeline:
    """Two-stage pipeline: CPU schedule → GPU execute.

    Tracks busy-until timestamps for the CPU scheduler stage and the GPU
    execution stage.  When both stages are active (PP > 1 or async scheduling
    enabled), the schedule of step *N+1* can overlap with the execution of
    step *N*, matching vLLM's batch queue / async scheduler behaviour.

    When disabled (``enabled=False``), falls back to fully serial execution —
    schedule and execute are summed without overlap.

    Attributes:
        enabled: Whether to allow schedule/execute overlap.
        schedule_busy_until: Timestamp when the CPU scheduler becomes free.
        execute_busy_until: Timestamp when the GPU becomes free.
    """

    enabled: bool = False
    schedule_busy_until: float = 0.0
    execute_busy_until: float = 0.0

    def step(self, clock: float, schedule_time_s: float,
             gpu_time_s: float) -> float:
        """Advance the pipeline by one scheduler step.

        Args:
            clock: Current simulation clock (seconds).  Typically the
                timestamp when all prerequisite events (arrivals, transfers)
                are complete.
            schedule_time_s: CPU time to run ``schedule()`` +
                ``_update_after_schedule()`` for this batch.  Estimated from
                running/waiting queue lengths (see P2-7).
            gpu_time_s: GPU time predicted by ``predict_step()``.

        Returns:
            New simulation clock value — the moment the GPU step completes.

        When *enabled* is False (default, backward-compatible):
            Returns ``clock + schedule_time_s + gpu_time_s``.
        When *enabled* is True:
            Overlaps schedule and execute as much as possible:

            ::

                       |── sched N ──|
                clock ─┘             |── exec N ──|
                       |── sched N+1 ──|          |── exec N+1 ──|
        """
        if not self.enabled:
            # Backward-compatible: schedule time is hidden (P2-7 adds it
            # separately).  Only GPU time advances the clock.
            return clock + gpu_time_s

        # ── CPU schedule stage ──
        sched_start = max(clock, self.schedule_busy_until)
        sched_end = sched_start + schedule_time_s
        self.schedule_busy_until = sched_end

        # ── GPU execute stage ──
        exec_start = max(sched_end, self.execute_busy_until)
        exec_end = exec_start + gpu_time_s
        self.execute_busy_until = exec_end

        return exec_end

    def reset(self):
        """Reset pipeline state for a new simulation run."""
        self.schedule_busy_until = 0.0
        self.execute_busy_until = 0.0


def estimate_schedule_time(num_running: int, num_waiting: int,
                           base_us: float = 100.0,
                           per_running_us: float = 5.0,
                           per_waiting_us: float = 10.0) -> float:
    """Estimate CPU time for one ``schedule()`` call.

    Based on vLLM profiling: ~100 µs base overhead + per-request traversal
    costs.  Running requests are cheaper (simple block allocation check);
    waiting requests are more expensive (hash computation for prefix cache).

    Args:
        num_running: Number of requests in the running queue.
        num_waiting: Number of requests in the waiting queue.
        base_us: Fixed overhead per schedule call (µs).
        per_running_us: Per-running-request traversal cost (µs).
        per_waiting_us: Per-waiting-request hash/check cost (µs).

    Returns:
        Estimated schedule time in seconds.
    """
    return (base_us + num_running * per_running_us
            + num_waiting * per_waiting_us) * 1e-6
