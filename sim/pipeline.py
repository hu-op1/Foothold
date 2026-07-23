"""Schedule/execute pipeline model — overlaps CPU scheduling with GPU execution.

In vLLM, the CPU scheduler can prepare the next batch while the current batch
executes on the GPU.  This is mandatory for pipeline parallelism (PP > 1,
microbatch pipeline) and optional for colocated inference (async scheduling).

When enabled, schedule time is hidden behind GPU time — only GPU time extends
the clock.  When disabled, schedule and GPU time are serial (P2-7).
"""

from dataclasses import dataclass


@dataclass
class ScheduleExecutePipeline:
    """Two-stage pipeline: CPU schedule → GPU execute.

    When enabled (PP > 1 or async scheduling), the CPU schedules step N
    while the GPU executes step N−1.  Schedule time is therefore hidden
    behind GPU time and does NOT extend the clock.

    When disabled, schedule and execute are serial — both contribute to
    the clock (P2-7).
    """

    enabled: bool = False

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
            # Schedule CPU overhead is always modeled (P2-7).
            # When pipelining is off, schedule and execute are serial.
            return clock + schedule_time_s + gpu_time_s

        # ── Enabled: schedule overlaps with previous GPU execution ──
        # The CPU schedules step N while the GPU executes step N−1.
        # By the time we reach this point, the schedule for THIS step
        # has already completed (it ran during the previous step's GPU
        # time).  Only the GPU time extends the clock.
        #
        # For PP > 1, the pipeline model is still correct because each
        # stage's schedule overlaps with the previous stage's execution.
        return clock + gpu_time_s

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
