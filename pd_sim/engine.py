"""Event-driven simulation engine for PD disaggregation."""

import heapq
from enum import Enum, auto
from dataclasses import dataclass, field

from pd_sim.request import Request, RequestStatus, FinishReason
from pd_sim.memory import BlockPool, compute_block_hashes
from pd_sim.scheduler import ColocatedScheduler
from pd_sim.executor import predict_step, predict_step_tp
from pd_sim.communication import (
    effective_xfer_overhead,
    transfer_blocks,
)


class EventType(Enum):
    ARRIVAL = auto()
    KV_TRANSFER_DONE = auto()


@dataclass(order=True)
class SimulationEvent:
    time: float
    event_type: EventType = field(compare=False)
    request: Request | None = field(compare=False, default=None)


class SimulationEngine:
    """Event-driven PD disaggregation simulator."""

    def __init__(self, config, model_spec, hw_params):
        self.cfg = config
        self.model = model_spec
        self.hw = hw_params
        self.clock: float = 0.0

        # Model dimensions
        self.nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])
        self.hd = model_spec["head_dim"]
        self.nl = model_spec["num_layers"]
        self.block_size = config["simulation"]["block_size"]

        # Bytes per KV cache block
        self.bytes_per_block = (
            2 * self.nl * self.nh_kv * self.hd
            * self.block_size * 2  # 2 bytes dtype
        )
        self.num_blocks = self._compute_num_blocks()

        # Network params
        self.bw_gb_s = config["communication"]["inter_bw_gb_s"]
        self.latency_us = config["communication"]["inter_latency_us"]
        self.intra_bw_gb_s = config["communication"]["intra_bw_gb_s"]

    def _compute_num_blocks(self):
        kv_mem_gb = self.cfg["simulation"]["kv_cache_memory_gb"]
        return max(1, int(kv_mem_gb * 1024**3) // self.bytes_per_block)

    def run(self, requests: list[Request], mode="colocated",
            chunk_size=None, pd_ratio=None, tp_size=1):
        """Run simulation over a list of requests.

        Args:
            requests: list of Request objects, sorted by arrival_time.
            mode: "colocated" or "disaggregated"
            chunk_size: for colocated, override chunk size (None = use config)
            pd_ratio: for disaggregated, (num_prefill, num_decode) GPU counts
            tp_size: tensor parallelism degree (1 = no TP)

        Returns:
            metrics_collector with recorded per-request metrics.
        """
        self.tp_size = tp_size
        # Reset request state for fresh simulation run
        for r in requests:
            r.num_computed_tokens = 0
            r.num_output_tokens = 0
            r.status = RequestStatus.WAITING
            r.finish_reason = None
            r.finish_time = None
            r.ttft = None
            r.is_prefill_chunk = True
            r.block_hashes = compute_block_hashes(r.prompt_token_ids, self.block_size)
            r.block_table.clear()
            r.kv_transfer_start = None
            r.kv_transfer_end = None

        # Safety iteration limit scales with workload (worst case: 1 token/step)
        total_decode_tokens = sum(r.max_output_len for r in requests)
        self._max_iterations = max(100_000, total_decode_tokens * 2 + len(requests) * 5)

        if mode == "colocated":
            return self._run_colocated(requests)
        else:
            return self._run_disaggregated(requests, pd_ratio or (1, 1))

    def _run_colocated(self, requests: list[Request]):
        """Run colocated simulation."""
        from pd_sim.metrics import MetricsCollector

        pool = BlockPool(self.num_blocks)
        sched = ColocatedScheduler(pool, self.cfg)
        metrics = MetricsCollector()

        event_queue: list[SimulationEvent] = []
        for r in requests:
            heapq.heappush(event_queue, SimulationEvent(r.arrival_time, EventType.ARRIVAL, r))

        loop_count = 0
        while event_queue or sched.has_requests():
            # Pop all events at current time
            if event_queue:
                self.clock = event_queue[0].time
            while event_queue and event_queue[0].time <= self.clock + 1e-9:
                ev = heapq.heappop(event_queue)
                if ev.event_type == EventType.ARRIVAL and ev.request:
                    sched.add_request(ev.request)

            # Schedule and execute one step
            output = sched.schedule()
            if output.total_num_scheduled_tokens == 0:
                if not sched.has_requests():
                    break
                # If there are future arrivals, jump to next arrival; else idle tick
                if event_queue:
                    self.clock = event_queue[0].time
                else:
                    self.clock += 0.001
                continue

            sched._update_after_schedule(output)
            step_time = self._predict_step(
                [(r, nt) for r, nt, _ in output.scheduled_requests])
            self.clock += step_time
            sched.update_from_output(output, self.clock)

            # Record finished requests
            for r in sched.drain_finished():
                metrics.record(r)

            loop_count += 1
            if loop_count > self._max_iterations:
                raise RuntimeError(
                    f"_run_colocated: exceeded {self._max_iterations} iterations. "
                    f"running={len(sched.running)}, waiting={len(sched.waiting)}, "
                    f"skipped={len(sched.skipped_waiting)}, clock={self.clock:.6f}"
                )

        return metrics

    def _run_disaggregated(self, requests: list[Request], pd_ratio: tuple[int, int]):
        """Run disaggregated simulation with multiple independent D instances.

        Requests are routed to the least-loaded D instance and admitted directly
        to the D running queue (with KV transfer). Waiting requests never hold
        D-side blocks — only running requests do.
        """
        from pd_sim.metrics import MetricsCollector

        num_p, num_d = pd_ratio

        # P side: pooled GPUs with larger batch budget
        pool_p = BlockPool(self.num_blocks * num_p)
        p_cfg = _deep_copy_config(self.cfg)
        p_cfg["simulation"]["max_num_batched_tokens"] = (
            self.cfg["simulation"]["max_num_batched_tokens"] * num_p)
        p_cfg["simulation"]["max_num_seqs"] = self.cfg["simulation"]["max_num_seqs"]
        p_sched = ColocatedScheduler(pool_p, p_cfg)

        # D side: num_d independent instances, each with own pool + scheduler.
        # D-side never resets num_computed_tokens on preempt — KV cache was
        # already computed on the P side and transferred.
        d_cfg = _deep_copy_config(self.cfg)
        d_pools = [BlockPool(self.num_blocks) for _ in range(num_d)]
        d_scheds = [ColocatedScheduler(d_pools[i], d_cfg, reset_on_preempt=False)
                    for i in range(num_d)]

        metrics = MetricsCollector()

        # Stalled prefill-done: (request, p_side_blocks, target_d_idx)
        stalled_xfers: list[tuple[Request, list[int], int]] = []

        event_queue: list[SimulationEvent] = []
        for r in requests:
            heapq.heappush(event_queue, SimulationEvent(r.arrival_time, EventType.ARRIVAL, r))

        def _pick_d_instance() -> int:
            loads = [len(s.running) + len(s.waiting) for s in d_scheds]
            return loads.index(min(loads))

        def _try_admit_to_d(req, p_side_blocks, d_idx, xfer_start_time):
            sched = d_scheds[d_idx]
            if len(sched.running) >= sched.max_num_seqs:
                return False
            xfer_time = transfer_blocks(req, d_pools[d_idx], self.bytes_per_block,
                                        self.bw_gb_s, self.latency_us,
                                        self.block_size)
            if xfer_time == float("inf"):
                return False
            pool_p.free_blocks(p_side_blocks)
            req.kv_transfer_start = xfer_start_time
            req.kv_transfer_end = xfer_start_time + xfer_time
            req.status = RequestStatus.RUNNING
            sched.running.append(req)
            return True

        p_time_val: float = 0.0
        loop_count = 0

        while (event_queue or stalled_xfers
               or p_sched.has_requests()
               or any(s.has_requests() for s in d_scheds)):

            if (not p_sched.has_requests()
                    and not any(s.has_requests() for s in d_scheds)
                    and not stalled_xfers and event_queue):
                self.clock = max(self.clock, event_queue[0].time)

            while event_queue and event_queue[0].time <= self.clock + 1e-9:
                ev = heapq.heappop(event_queue)
                if ev.event_type == EventType.ARRIVAL and ev.request:
                    p_sched.add_request(ev.request)

            p_out = p_sched.schedule()
            d_outs = [s.schedule() for s in d_scheds]

            p_total = p_out.total_num_scheduled_tokens
            d_totals = [o.total_num_scheduled_tokens for o in d_outs]

            if p_total == 0 and all(t == 0 for t in d_totals):
                if event_queue:
                    self.clock = max(self.clock, event_queue[0].time)
                elif (not p_sched.has_requests()
                        and not any(s.has_requests() for s in d_scheds)
                        and not stalled_xfers):
                    break
                else:
                    self.clock += 0.001
                continue

            p_sched._update_after_schedule(p_out)
            for i, s in enumerate(d_scheds):
                s._update_after_schedule(d_outs[i])

            p_time_val = self._predict_step(
                [(r, nt) for r, nt, _ in p_out.scheduled_requests]
            ) if p_total > 0 else 0.0

            d_times: list[float] = []
            for i, o in enumerate(d_outs):
                if o.total_num_scheduled_tokens > 0:
                    dt = self._predict_step(
                        [(r, nt) for r, nt, _ in o.scheduled_requests])
                    d_scheds[i].update_from_output(o, self.clock + dt)
                    d_times.append(dt)
                else:
                    d_times.append(0.0)

            # Retry stalled transfers (D-side may have freed blocks via finished requests)
            still_stalled: list[tuple[Request, list[int], int]] = []
            for req, p_blocks, d_idx in stalled_xfers:
                if not _try_admit_to_d(req, p_blocks, d_idx, self.clock + p_time_val):
                    still_stalled.append((req, p_blocks, d_idx))
            stalled_xfers = still_stalled

            # Prefill-done → route to least-loaded D and admit with transfer
            p_done = p_sched.drain_prefill_done()
            for req in p_done:
                if req in p_sched.running:
                    p_sched.running.remove(req)
                p_side_blocks = [b for b in req.block_table if b >= 0]
                d_idx = _pick_d_instance()
                if not _try_admit_to_d(req, p_side_blocks, d_idx, self.clock + p_time_val):
                    stalled_xfers.append((req, p_side_blocks, d_idx))

            max_d_time = max(d_times) if d_times else 0.0
            step_time = max(p_time_val, max_d_time)
            if step_time == 0:
                step_time = 0.001
            self.clock += step_time

            for s in d_scheds:
                for r in s.drain_finished():
                    metrics.record(r)

            loop_count += 1
            if loop_count > self._max_iterations:
                d_loads = [len(s.running) + len(s.waiting) for s in d_scheds]
                raise RuntimeError(
                    f"_run_disaggregated: exceeded {self._max_iterations} iterations. "
                    f"p_running={len(p_sched.running)}, p_waiting={len(p_sched.waiting)}, "
                    f"d_loads={d_loads}, stalled_xfers={len(stalled_xfers)}, "
                    f"events={len(event_queue)}, clock={self.clock:.6f}"
                )

        return metrics

    def _predict_step(self, scheduled_requests):
        """Predict step time, using TP if configured."""
        tp = getattr(self, "tp_size", 1)
        if tp > 1:
            return predict_step_tp(scheduled_requests, self.model, self.hw,
                                   tp, self.intra_bw_gb_s)
        return predict_step(scheduled_requests, self.model, self.hw)

    def _compute_xfer(self, request: Request, prefill_time: float) -> float:
        """Compute KV transfer time for a completed prefill, accounting for overlap."""
        return effective_xfer_overhead(
            request.prompt_len, self.nl, self.nh_kv, self.hd,
            self.bw_gb_s, self.latency_us, prefill_time,
        )


def _deep_copy_config(cfg: dict) -> dict:
    """Simple deep copy via JSON round-trip."""
    import json
    return json.loads(json.dumps(cfg))
