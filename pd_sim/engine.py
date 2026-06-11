"""Event-driven simulation engine for PD disaggregation."""

import heapq
from enum import Enum, auto
from dataclasses import dataclass, field

from pd_sim.request import Request, RequestStatus, FinishReason
from pd_sim.memory import BlockPool, compute_block_hashes
from pd_sim.scheduler import ColocatedScheduler
from pd_sim.executor import predict_step
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
        self.bw_gb_s = config["communication"]["bandwidth_gb_s"]
        self.latency_us = config["communication"]["latency_us"]

    def _compute_num_blocks(self):
        kv_mem_gb = self.cfg["simulation"]["kv_cache_memory_gb"]
        return max(1, int(kv_mem_gb * 1024**3) // self.bytes_per_block)

    def run(self, requests: list[Request], mode="colocated",
            chunk_size=None, pd_ratio=None):
        """Run simulation over a list of requests.

        Args:
            requests: list of Request objects, sorted by arrival_time.
            mode: "colocated" or "disaggregated"
            chunk_size: for colocated, override chunk size (None = use config)
            pd_ratio: for disaggregated, (num_prefill, num_decode) GPU counts

        Returns:
            metrics_collector with recorded per-request metrics.
        """
        for r in requests:
            r.block_hashes = compute_block_hashes(r.prompt_token_ids, self.block_size)

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
            step_time = predict_step(
                [(r, nt) for r, nt, _ in output.scheduled_requests],
                self.model, self.hw,
            )
            self.clock += step_time
            sched.update_from_output(output, self.clock)

            # Record finished requests
            for r in sched.drain_finished():
                metrics.record(r)

        return metrics

    def _run_disaggregated(self, requests: list[Request], pd_ratio: tuple[int, int]):
        """Run disaggregated simulation with separate P and D pools/schedulers."""
        from pd_sim.metrics import MetricsCollector

        num_p, num_d = pd_ratio

        # Separate memory pools
        pool_p = BlockPool(self.num_blocks * num_p)
        pool_d = BlockPool(self.num_blocks * num_d)

        # Separate schedulers
        p_cfg = _deep_copy_config(self.cfg)
        p_cfg["simulation"]["max_num_batched_tokens"] = self.cfg["simulation"]["max_num_batched_tokens"] * num_p
        p_cfg["simulation"]["max_num_seqs"] = self.cfg["simulation"]["max_num_seqs"]

        d_cfg = _deep_copy_config(self.cfg)
        d_cfg["simulation"]["max_num_seqs"] = self.cfg["simulation"]["max_num_seqs"] * num_d

        p_sched = ColocatedScheduler(pool_p, p_cfg)
        d_sched = ColocatedScheduler(pool_d, d_cfg)

        metrics = MetricsCollector()

        # Pending transfers: list of (transfer_done_time, request), sorted
        pending_xfers: list[tuple[float, Request]] = []

        event_queue: list[SimulationEvent] = []
        for r in requests:
            heapq.heappush(event_queue, SimulationEvent(r.arrival_time, EventType.ARRIVAL, r))

        while event_queue or pending_xfers or p_sched.has_requests() or d_sched.has_requests():
            # If waiting for events and nothing to process, jump clock
            if event_queue and not p_sched.has_requests() and not d_sched.has_requests() and not pending_xfers:
                self.clock = max(self.clock, event_queue[0].time)

            # Pop arrival events at current clock
            while event_queue and event_queue[0].time <= self.clock + 1e-9:
                ev = heapq.heappop(event_queue)
                if ev.event_type == EventType.ARRIVAL and ev.request:
                    p_sched.add_request(ev.request)

            # Process completed KV transfers
            while pending_xfers and pending_xfers[0][0] <= self.clock + 1e-9:
                _, req = pending_xfers.pop(0)
                req.status = RequestStatus.WAITING
                d_sched.add_request(req)

            # Schedule both sides
            p_out = p_sched.schedule()
            d_out = d_sched.schedule()

            p_total = p_out.total_num_scheduled_tokens
            d_total = d_out.total_num_scheduled_tokens

            if p_total == 0 and d_total == 0 and not pending_xfers:
                if not p_sched.has_requests() and not d_sched.has_requests():
                    break
                # Jump to next arrival
                if event_queue:
                    self.clock = event_queue[0].time
                else:
                    self.clock += 0.001
                continue

            p_sched._update_after_schedule(p_out)
            d_sched._update_after_schedule(d_out)

            # Execute both sides
            p_time = predict_step(
                [(r, nt) for r, nt, _ in p_out.scheduled_requests],
                self.model, self.hw,
            ) if p_total > 0 else 0.0

            d_time = predict_step(
                [(r, nt) for r, nt, _ in d_out.scheduled_requests],
                self.model, self.hw,
            ) if d_total > 0 else 0.0

            # Process prefill completions and compute KV transfer
            p_sched.update_from_output(p_out, self.clock + p_time)
            d_sched.update_from_output(d_out, self.clock + d_time)

            # Find requests that completed prefill on P side
            p_done = p_sched.drain_prefill_done()
            for req in p_done:
                xfer_time = self._compute_xfer(req, p_time)
                req.kv_transfer_start = self.clock + p_time
                req.kv_transfer_end = req.kv_transfer_start + xfer_time
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS

                # Transfer blocks to D pool (touch cached, allocate for new)
                transfer_blocks(req, pool_d, self.bytes_per_block,
                                self.bw_gb_s, self.latency_us)

                pending_xfers.append((req.kv_transfer_end, req))

            # Sort pending transfers by completion time
            pending_xfers.sort(key=lambda x: x[0])

            # Compute effective KV transfer overhead on critical path
            effective_xfer = 0.0
            if pending_xfers and p_time > 0:
                first_xfer = pending_xfers[0][1]
                kv_len = first_xfer.prompt_len
                effective_xfer = effective_xfer_overhead(
                    kv_len, self.nl, self.nh_kv, self.hd,
                    self.bw_gb_s, self.latency_us, p_time,
                )

            step_time = max(p_time, d_time + effective_xfer)
            if step_time == 0:
                step_time = 0.001
            self.clock += step_time

            # Record finished from D side
            for r in d_sched.drain_finished():
                metrics.record(r)

        return metrics

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
