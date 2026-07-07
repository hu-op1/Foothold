"""Event-driven simulation engine for PD disaggregation."""

import heapq
from enum import Enum, auto
from dataclasses import dataclass, field

from sim.request import Request, RequestStatus, FinishReason
from sim.memory import BlockPool, compute_block_hashes
from sim.scheduler import ColocatedScheduler
from sim.executor import predict_step, predict_step_tp
from sim.roofline import dtype_bytes
from sim.communication import (
    effective_xfer_overhead,
    transfer_blocks,
)
from sim.pipeline import ScheduleExecutePipeline, estimate_schedule_time


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
        self.pp_size = 1

        # Precision (for per-dtype roofline params + bytes-per-element)
        self.dtype = config.get("dtype", "float16")

        # Model dimensions
        self.nh_kv = model_spec.get("num_kv_heads", model_spec["num_heads"])
        self.hd = model_spec["head_dim"]
        self.nl = model_spec["num_layers"]
        self.block_size = config["simulation"]["block_size"]

        # Bytes per KV cache block (all layers — used for KV transfer).
        dt_b = dtype_bytes(self.dtype)
        self.bytes_per_block = (
            2 * self.nl * self.nh_kv * self.hd
            * self.block_size * dt_b
        )
        # Base num_blocks (pp=1).  Callers should scale by pp_size when
        # creating pools because each GPU stores KV for only nl/pp layers.
        self.num_blocks = self._compute_num_blocks()

        # Network params
        self.bw_gb_s = config["communication"]["inter_bw_gb_s"]
        self.latency_us = config["communication"]["inter_latency_us"]
        self.intra_bw_gb_s = config["communication"]["intra_bw_gb_s"]
        self.intra_latency_us = config["communication"].get("intra_latency_us", 2.0)
        # GPU↔CPU swap bandwidth for D-side preemption (bytes/s)
        self.cpu_swap_bw = config["communication"].get("cpu_swap_bw_gb_s", 32) * 1e9
        # Prefix caching
        self.enable_cache = config["simulation"].get("enable_prefix_caching", True)

        # Per-step time breakdown accumulator (reset in run())
        self.time_acc: dict[str, float] = {}

    @staticmethod
    def _zero_step_dict() -> dict[str, float]:
        return {"total": 0.0, "attn_proj": 0.0, "ffn_proj": 0.0,
                "attn_prefill": 0.0, "attn_decode": 0.0,
                "fused_add_norm": 0.0, "swiglu": 0.0, "rope": 0.0,
                "lm_head": 0.0,
                "all_reduce": 0.0, "inter_stage_comm": 0.0}

    def _compute_num_blocks(self):
        kv_mem_gb = self.cfg["simulation"]["kv_cache_memory_gb"]
        return max(1, int(kv_mem_gb * 1024**3) // self.bytes_per_block)

    def run(self, requests: list[Request], mode="colocated",
            chunk_size=None, pd_ratio=None, tp_size=1, d_tp_size=1, dp=1,
            pp_size=1, d_pp_size=1,
            recorder=None):
        """Run simulation over a list of requests.

        Args:
            requests: list of Request objects, sorted by arrival_time.
            mode: "colocated" or "disaggregated"
            chunk_size: for colocated, override chunk size (None = use config)
            pd_ratio: for disaggregated, (num_prefill, num_decode) GPU counts
            tp_size: tensor parallelism degree on P side (1 = no TP)
            d_tp_size: tensor parallelism degree on D side (1 = no TP on D)
            pp_size: pipeline parallelism degree on P side (1 = no PP)
            d_pp_size: pipeline parallelism degree on D side (1 = no PP on D)
            dp: data-parallel degree — number of independent ranks
            recorder: optional SimRecorder for per-tick + per-request output

        Returns:
            metrics_collector with recorded per-request metrics.
        """
        self.tp_size = tp_size
        self.d_tp_size = d_tp_size
        self.pp_size = pp_size
        self.d_pp_size = d_pp_size

        # ── TP/PP-aware KV cache correction ──
        # load_config() computes kv_cache_memory_gb assuming the FULL model
        # on one GPU.  With TP>1 or PP>1, each GPU only holds a fraction of
        # the weights (1/tp for TP, nl/pp for PP), freeing VRAM for KV cache.
        # The existing tp_size × pp multiplier in blocks_per_pool already
        # accounts for smaller block sizes (fewer heads / fewer layers).
        # Here we add the VRAM freed by weight reduction.
        eff_parallel = max(tp_size * pp_size, d_tp_size * d_pp_size)
        if eff_parallel > 1:
            total_params = self.model.get("total_params_b", 0)
            if total_params > 0:
                weight_gb = total_params * 2 / 1e9  # float16
                extra_kv_gb = weight_gb * (eff_parallel - 1) / eff_parallel
                extra_blocks = int(extra_kv_gb * 1024**3) // self.bytes_per_block
                self.num_blocks += max(0, extra_blocks)
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
            # Store output_tok_ids for incremental hash extension during decode.
            r.block_table.clear()
            r.kv_transfer_start = None
            r.kv_transfer_end = None
            r.scheduled_ts = None

        # Reset time breakdown accumulator
        self.time_acc = {"attn_proj": 0.0, "ffn_proj": 0.0,
                         "attn_prefill": 0.0, "attn_decode": 0.0,
                         "fused_add_norm": 0.0, "swiglu": 0.0, "rope": 0.0,
                         "lm_head": 0.0, "launch_overhead": 0.0,
                         "all_reduce": 0.0, "inter_stage_comm": 0.0,
                         "kv_transfer": 0.0, "swap": 0.0}

        # Safety iteration limit scales with workload (worst case: 1 token/step)
        total_decode_tokens = sum(r.max_output_len for r in requests)
        self._max_iterations = max(100_000, total_decode_tokens * 2 + len(requests) * 5)

        # ── Schedule/execute pipeline ──
        # Enabled when PP > 1 (microbatch pipeline requires batch queue) or
        # when async_scheduling is explicitly turned on (PP=1 optional overlap).
        sim_cfg = self.cfg.get("simulation", {})
        async_sched = sim_cfg.get("async_scheduling", False)
        pipeline_enabled = (pp_size > 1 or d_pp_size > 1 or async_sched)
        self._pipeline = ScheduleExecutePipeline(enabled=pipeline_enabled)

        if mode == "colocated":
            return self._run_colocated(requests, dp=dp, pp=pp_size, recorder=recorder)
        else:
            return self._run_disaggregated(requests, pd_ratio or (1, 1),
                                           pp=pp_size, d_pp=d_pp_size,
                                           recorder=recorder)

    def _run_colocated(self, requests: list[Request], dp: int = 1,
                        pp: int = 1, recorder=None):
        """Run colocated simulation.

        When dp > 1, creates dp independent schedulers (each with its own
        KV cache pool and model weights), routes arrivals to the least-loaded
        rank, and steps all ranks in parallel.  This matches vLLM's data-
        parallel architecture where each DP rank is a full EngineCore.

        Each DP rank represents a PP×TP group (pp_size × tp_size GPUs).
        Total GPU count = dp × tp_size × pp_size.
        """
        from sim.metrics import MetricsCollector

        # num_blocks is for a single GPU with pp=1.
        # PP: nl/pp layers → pp× more blocks per GPU.
        # TP: nh_kv/tp heads → tp× more blocks per GPU.
        # Each DP rank pool = num_blocks × tp_size × pp_size.
        blocks_per_pool = self.num_blocks * self.tp_size * pp
        pools = [BlockPool(blocks_per_pool,
                           enable_caching=self.enable_cache) for _ in range(dp)]
        for p in pools:
            p.bytes_per_block = self.bytes_per_block
        scheds = [ColocatedScheduler(pools[i], self.cfg) for i in range(dp)]

        metrics = MetricsCollector()

        event_queue: list[SimulationEvent] = []
        for r in requests:
            # Only enqueue first sub_request of each session (or standalone ShareGPT).
            # Chained sub_requests arrive when their predecessor finishes.
            if r.sub_request_index == 0:
                heapq.heappush(event_queue, SimulationEvent(r.arrival_time, EventType.ARRIVAL, r))

        def _pick_rank() -> int:
            """Least-loaded routing — matches vLLM's DP load balancing."""
            loads = [len(s.running) + len(s.waiting) for s in scheds]
            return loads.index(min(loads))

        def _all_idle() -> bool:
            return all(not s.has_requests() for s in scheds)

        loop_count = 0
        idle_ticks = 0

        # ── Progress logging ──
        _tick_seconds = self.cfg.get("tick_seconds", 0.5)
        _last_print = 0.0
        _cum_prompt = 0
        _cum_gen = 0
        _last_cum_prompt = 0
        _last_cum_gen = 0
        _total_reqs = len(requests)

        while event_queue or not _all_idle():
            # Only advance clock to next event if ALL ranks are truly idle
            if event_queue and _all_idle():
                self.clock = max(self.clock, event_queue[0].time)
            while event_queue and event_queue[0].time <= self.clock + 1e-9:
                ev = heapq.heappop(event_queue)
                if ev.event_type == EventType.ARRIVAL and ev.request:
                    target = _pick_rank()
                    scheds[target].add_request(ev.request)

            # All ranks schedule independently
            outputs = [s.schedule() for s in scheds]

            if all(o.total_num_scheduled_tokens == 0 for o in outputs):
                if _all_idle():
                    break
                if event_queue:
                    idle_ticks = 0
                    self.clock = max(self.clock, event_queue[0].time)
                else:
                    idle_ticks += 1
                    if idle_ticks > 1000:
                        raise RuntimeError(
                            "Colocated DP deadlock: all ranks stalled "
                            f"(1000 idle ticks). clock={self.clock:.4f}")
                    self.clock += 0.001
                continue
            idle_ticks = 0

            # Execute all ranks in parallel; wall-clock = max step time
            max_step = 0.0
            step_prompt: int = 0
            step_gen: int = 0
            for i, output in enumerate(outputs):
                if output.total_num_scheduled_tokens > 0:
                    scheds[i]._update_after_schedule(output)
                    step = self._predict_step(
                        [(r, nt) for r, nt, _ in output.scheduled_requests],
                        pp=pp)
                    scheds[i].update_from_output(output, self.clock + step["total"])
                    max_step = max(max_step, step["total"])
                    # Accumulate time breakdown components
                    for k, v in step.items():
                        if k != "total":
                            self.time_acc[k] += v
                    # Count prefill vs decode tokens for this step
                    for req, num_new, _ in output.scheduled_requests:
                        pre_step = req.num_computed_tokens - num_new
                        prompt_rem = max(0, req.num_prompt_tokens - pre_step)
                        dec = num_new - min(num_new, prompt_rem)
                        step_prompt += (num_new - dec)
                        step_gen += dec

            # ── Advance clock with schedule/execute pipeline ──
            # Pipeline allows CPU scheduling of step N+1 to overlap with
            # GPU execution of step N.  Enabled when PP > 1 or async_scheduling.
            total_running = sum(len(s.running) for s in scheds)
            total_waiting = sum(len(s.waiting) for s in scheds)
            sched_time = estimate_schedule_time(total_running, total_waiting)
            self.clock = self._pipeline.step(self.clock, sched_time, max_step)

            # Record per-tick timeseries
            if recorder:
                running = sum(len(s.running) for s in scheds)
                waiting = sum(len(s.waiting) for s in scheds)
                usage = (sum(p.get_usage() for p in pools) / max(1, len(pools))) * 100.0
                recorder.record_tick(self.clock, running, waiting,
                                     step_prompt, step_gen, usage)

            # Collect finished requests from all ranks
            for s in scheds:
                for r in s.drain_finished():
                    metrics.record(r)
                    if recorder:
                        recorder.record_request(r)
                    # Agentic trace chaining: enqueue next sub_request after tool pause
                    if r.next_sub_request is not None:
                        next_req = r.next_sub_request
                        next_req.arrival_time = self.clock + r.tool_duration
                        heapq.heappush(event_queue, SimulationEvent(
                            next_req.arrival_time, EventType.ARRIVAL, next_req))

            # ── Progress logging ──
            _cum_prompt += step_prompt
            _cum_gen += step_gen
            if self.clock - _last_print >= _tick_seconds:
                interval = self.clock - _last_print or 1e-9
                p_tput = (_cum_prompt - _last_cum_prompt) / interval
                g_tput = (_cum_gen - _last_cum_gen) / interval
                _total_q = sum(p._cache_queries for p in pools)
                _total_h = sum(p._cache_hits for p in pools)
                ch = _total_h / _total_q * 100 if _total_q > 0 else 0.0
                kv_usage = sum(p.get_usage() for p in pools) / len(pools) if pools else 0.0
                kv_total_gb = (self.num_blocks * self.bytes_per_block) / 1e9
                kv_used_gb = kv_usage * kv_total_gb
                print(f"[{self.clock:.1f}s] prompt={p_tput:.1f} gen={g_tput:.1f} tok/s "
                      f"| run={total_running} wait={total_waiting} "
                      f"| cache={ch:.1f}% ({_total_h}/{_total_q}) "
                      f"| mem={kv_used_gb:.1f}/{kv_total_gb:.1f} GiB "
                      f"| done={metrics.num_requests}/{_total_reqs}")
                _last_print = self.clock
                _last_cum_prompt = _cum_prompt
                _last_cum_gen = _cum_gen

            loop_count += 1
            if loop_count > self._max_iterations:
                loads = [len(s.running) + len(s.waiting) for s in scheds]
                raise RuntimeError(
                    f"_run_colocated: exceeded {self._max_iterations} iterations. "
                    f"loads={loads}, events={len(event_queue)}, clock={self.clock:.6f}"
                )

        # Aggregate cache hit rate across all pools
        total_q = sum(p._cache_queries for p in pools)
        total_h = sum(p._cache_hits for p in pools)
        metrics.cache_hit_rate = total_h / total_q if total_q > 0 else 0.0
        metrics.time_breakdown = dict(self.time_acc)
        return metrics

    def _run_disaggregated(self, requests: list[Request], pd_ratio: tuple[int, int],
                            pp: int = 1, d_pp: int = 1,
                            recorder=None):
        """Run disaggregated simulation with multiple independent D instances.

        Requests are routed to the least-loaded D instance and admitted directly
        to the D running queue (with KV transfer). Waiting requests never hold
        D-side blocks — only running requests do.

        P side: pp × tp GPUs per replica, dp_p = p / (pp × tp) replicas.
        D side: d_pp × d_tp GPUs per replica, dp_d = d / (d_pp × d_tp) replicas.
        """
        from sim.metrics import MetricsCollector

        num_p, num_d = pd_ratio

        # P side: pooled GPUs with larger batch budget.
        # Each P GPU gets num_blocks × pp (nl/pp layers → pp× more blocks).
        # num_p GPUs total → pool sized for all of them.
        pool_p = BlockPool(self.num_blocks * num_p * pp, enable_caching=self.enable_cache)
        pool_p.bytes_per_block = self.bytes_per_block
        p_cfg = _deep_copy_config(self.cfg)
        p_cfg["simulation"]["max_num_batched_tokens"] = (
            self.cfg["simulation"]["max_num_batched_tokens"] * num_p)
        p_cfg["simulation"]["max_num_seqs"] = self.cfg["simulation"]["max_num_seqs"]
        p_sched = ColocatedScheduler(pool_p, p_cfg)

        # D side: num_d GPUs grouped into (d_tp × d_pp) groups per replica.
        # Each TP×PP group is one ColocatedScheduler with a pool of
        # d_tp × num_blocks blocks. num_d_groups = DP degree on D side.
        d_tp = getattr(self, "d_tp_size", 1)
        d_pp_size = d_pp
        per_replica_gpus = d_tp * d_pp_size
        if num_d % per_replica_gpus != 0:
            raise ValueError(
                f"num_d ({num_d}) must be divisible by d_tp×d_pp "
                f"({d_tp}×{d_pp_size}={per_replica_gpus})")
        num_d_groups = num_d // per_replica_gpus

        d_cfg = _deep_copy_config(self.cfg)
        d_pools = [BlockPool(self.num_blocks * d_tp * d_pp_size,
                             enable_caching=self.enable_cache,
                             cpu_swap_bw_bytes_per_s=self.cpu_swap_bw)
                   for _ in range(num_d_groups)]
        for dp in d_pools:
            dp.bytes_per_block = self.bytes_per_block
        d_scheds = [ColocatedScheduler(d_pools[i], d_cfg, reset_on_preempt=False)
                    for i in range(num_d_groups)]

        metrics = MetricsCollector()

        # Stalled prefill-done: (request, p_side_blocks, target_d_idx)
        stalled_xfers: list[tuple[Request, list[int], int]] = []

        event_queue: list[SimulationEvent] = []
        for r in requests:
            if r.sub_request_index == 0:
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
        idle_ticks = 0

        # ── Progress logging ──
        _tick_seconds = self.cfg.get("tick_seconds", 0.5)
        _last_print = 0.0
        _cum_prompt = 0
        _cum_gen = 0
        _last_cum_prompt = 0
        _last_cum_gen = 0
        _total_reqs = len(requests)

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
                # No forward progress this step.
                if event_queue:
                    idle_ticks = 0
                    self.clock = max(self.clock, event_queue[0].time)
                elif (not p_sched.has_requests()
                        and not any(s.has_requests() for s in d_scheds)
                        and not stalled_xfers):
                    break
                else:
                    idle_ticks += 1
                    if idle_ticks > 1000:
                        d_loads = [len(s.running) + len(s.waiting) for s in d_scheds]
                        raise RuntimeError(
                            "Disaggregated deadlock: all D instances stalled "
                            f"(1000 consecutive idle ticks). "
                            f"d_loads={d_loads}, stalled_xfers={len(stalled_xfers)}, "
                            f"clock={self.clock:.4f}. "
                            "Likely cause: max_num_seqs too high for the KV cache "
                            "block pool — reduce max_num_seqs or increase kv_cache_memory_gb."
                        )
                    self.clock += 0.001
                continue

            p_sched._update_after_schedule(p_out)
            for i, s in enumerate(d_scheds):
                s._update_after_schedule(d_outs[i])

            # Pull prefill-done requests out of P running immediately.
            # If left in, Phase 1 of the next schedule() would allocate P-side
            # decode blocks that are never freed (decode belongs to D side).
            p_done = p_sched.drain_prefill_done()
            for req in p_done:
                if req in p_sched.running:
                    p_sched.running.remove(req)

            p_step = self._predict_step(
                [(r, nt) for r, nt, _ in p_out.scheduled_requests],
                pp=pp
            ) if p_total > 0 else self._zero_step_dict()
            p_time_val = p_step["total"]
            # Accumulate P-side time breakdown
            for k, v in p_step.items():
                if k != "total":
                    self.time_acc[k] += v
            # P-side update: sets scheduled_ts for newly admitted requests
            if p_total > 0:
                p_sched.update_from_output(p_out, self.clock + p_time_val)

            # Count P-side tokens (all prefill)
            step_prompt = sum(nt for _, nt, _ in p_out.scheduled_requests)
            step_gen = 0

            d_times: list[float] = []
            for i, o in enumerate(d_outs):
                if o.total_num_scheduled_tokens > 0:
                    d_step = self._predict_step(
                        [(r, nt) for r, nt, _ in o.scheduled_requests],
                        tp=d_tp, pp=d_pp)
                    d_scheds[i].update_from_output(o, self.clock + d_step["total"])
                    d_times.append(d_step["total"])
                    # Accumulate D-side time breakdown
                    for k, v in d_step.items():
                        if k != "total":
                            self.time_acc[k] += v
                    # D-side tokens are decode (gen)
                    step_gen += sum(nt for _, nt, _ in o.scheduled_requests)
                else:
                    d_times.append(0.0)

            # Drain finished from D first (frees blocks), then retry stalled transfers
            d_had_finished = False
            for s in d_scheds:
                for r in s.drain_finished():
                    metrics.record(r)
                    if recorder:
                        recorder.record_request(r)
                    d_had_finished = True
                    # Agentic trace chaining: enqueue next sub_request after tool pause
                    if r.next_sub_request is not None:
                        next_req = r.next_sub_request
                        next_req.arrival_time = self.clock + r.tool_duration
                        heapq.heappush(event_queue, SimulationEvent(
                            next_req.arrival_time, EventType.ARRIVAL, next_req))

            # Only retry stalled transfers if D-side freed blocks.
            # Each D completion frees blocks for at most 1-2 transfers.
            if d_had_finished and stalled_xfers:
                still_stalled: list[tuple[Request, list[int], int]] = []
                admitted = 0
                for req, p_blocks, d_idx in stalled_xfers:
                    if admitted < 4 and _try_admit_to_d(req, p_blocks, d_idx, self.clock + p_time_val):
                        admitted += 1
                    else:
                        still_stalled.append((req, p_blocks, d_idx))
                stalled_xfers = still_stalled

            # Transfer prefill-done requests to D side
            for req in p_done:
                p_side_blocks = [b for b in req.block_table if b >= 0]
                d_idx = _pick_d_instance()
                if not _try_admit_to_d(req, p_side_blocks, d_idx, self.clock + p_time_val):
                    stalled_xfers.append((req, p_side_blocks, d_idx))

            # Accumulate KV transfer time for successfully admitted requests
            for req in p_done:
                if req.kv_transfer_end is not None and req.kv_transfer_start is not None:
                    self.time_acc["kv_transfer"] += req.kv_transfer_end - req.kv_transfer_start

            max_d_time = max(d_times) if d_times else 0.0
            step_time = max(p_time_val, max_d_time)
            total_swap = p_out.swap_time + sum(o.swap_time for o in d_outs)
            self.time_acc["swap"] += total_swap
            if step_time == 0 and total_swap == 0:
                step_time = 0.001
            # ── Advance clock with schedule/execute pipeline ──
            # P-side + D-side schedule overheads can overlap with GPU time.
            # Swap (GPU↔CPU) is asynchronous in vLLM — issued before the GPU
            # step and completed in the background.  Model this by overlapping
            # swap time with GPU time: only the longer of the two matters.
            p_running = len(p_sched.running) if p_total > 0 else 0
            p_waiting = len(p_sched.waiting)
            d_running = sum(len(s.running) for s in d_scheds)
            d_waiting = sum(len(s.waiting) for s in d_scheds)
            sched_time = estimate_schedule_time(p_running + d_running,
                                                 p_waiting + d_waiting)
            gpu_time = max(step_time, total_swap)  # swap overlaps with GPU
            self.clock = self._pipeline.step(self.clock, sched_time, gpu_time)

            # Record per-tick timeseries
            if recorder:
                all_pools = [pool_p] + d_pools
                running = (len(p_sched.running)
                           + sum(len(s.running) for s in d_scheds))
                waiting = (len(p_sched.waiting)
                           + sum(len(s.waiting) for s in d_scheds))
                usage = (sum(p.get_usage() for p in all_pools)
                         / max(1, len(all_pools))) * 100.0
                recorder.record_tick(self.clock, running, waiting,
                                     step_prompt, step_gen, usage)

            idle_ticks = 0

            # ── Progress logging ──
            _cum_prompt += step_prompt
            _cum_gen += step_gen
            if self.clock - _last_print >= _tick_seconds:
                interval = self.clock - _last_print or 1e-9
                p_tput = (_cum_prompt - _last_cum_prompt) / interval
                g_tput = (_cum_gen - _last_cum_gen) / interval
                all_pools = [pool_p] + d_pools
                _total_q = sum(p._cache_queries for p in all_pools)
                _total_h = sum(p._cache_hits for p in all_pools)
                ch = _total_h / _total_q * 100 if _total_q > 0 else 0.0
                kv_usage = sum(p.get_usage() for p in all_pools) / len(all_pools) if all_pools else 0.0
                kv_total_gb = (self.num_blocks * self.bytes_per_block) / 1e9
                kv_used_gb = kv_usage * kv_total_gb
                print(f"[{self.clock:.1f}s] prompt={p_tput:.1f} gen={g_tput:.1f} tok/s "
                      f"| run={p_running + d_running} wait={p_waiting + d_waiting} "
                      f"| cache={ch:.1f}% ({_total_h}/{_total_q}) "
                      f"| mem={kv_used_gb:.1f}/{kv_total_gb:.1f} GiB "
                      f"| done={metrics.num_requests}/{_total_reqs}")
                _last_print = self.clock
                _last_cum_prompt = _cum_prompt
                _last_cum_gen = _cum_gen

            loop_count += 1
            if loop_count > self._max_iterations:
                d_loads = [len(s.running) + len(s.waiting) for s in d_scheds]
                raise RuntimeError(
                    f"_run_disaggregated: exceeded {self._max_iterations} iterations. "
                    f"p_running={len(p_sched.running)}, p_waiting={len(p_sched.waiting)}, "
                    f"d_loads={d_loads}, stalled_xfers={len(stalled_xfers)}, "
                    f"events={len(event_queue)}, clock={self.clock:.6f}"
                )

        # Aggregate cache hit rate across P + all D pools
        all_pools = [pool_p] + d_pools
        total_queries = sum(p._cache_queries for p in all_pools)
        total_hits = sum(p._cache_hits for p in all_pools)
        metrics.cache_hit_rate = total_hits / total_queries if total_queries > 0 else 0.0
        metrics.time_breakdown = dict(self.time_acc)
        return metrics

    def _predict_step(self, scheduled_requests, tp=None, pp=None):
        """Predict step time, using TP and/or PP if configured.

        Args:
            scheduled_requests: list of (request, num_new_tokens)
            tp: tensor parallelism degree (defaults to self.tp_size)
            pp: pipeline parallelism degree (defaults to self.pp_size)
        """
        if tp is None:
            tp = getattr(self, "tp_size", 1)
        if pp is None:
            pp = getattr(self, "pp_size", 1)
        use_cg = self.cfg.get("simulation", {}).get("use_cudagraph", False)
        if tp > 1 or pp > 1:
            return predict_step_tp(scheduled_requests, self.model, self.hw,
                                   tp, self.intra_bw_gb_s,
                                   self.intra_latency_us,
                                   pp_size=pp, dtype=self.dtype,
                                   use_cudagraph=use_cg)
        return predict_step(scheduled_requests, self.model, self.hw, self.dtype,
                            use_cudagraph=use_cg)

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
