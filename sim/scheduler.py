"""Scheduling logic — based on vllm v1/core/sched/scheduler.py."""

from collections import deque

from sim.request import Request, RequestStatus, FinishReason


class SchedulingPolicy:
    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue:
    """Simple FIFO request queue."""

    def __init__(self):
        self._queue: deque[Request] = deque()

    def add(self, request: Request) -> None:
        self._queue.append(request)

    def prepend(self, request: Request) -> None:
        self._queue.appendleft(request)

    def peek(self) -> Request | None:
        return self._queue[0] if self._queue else None

    def pop(self) -> Request | None:
        return self._queue.popleft() if self._queue else None

    def remove(self, request: Request) -> None:
        try:
            self._queue.remove(request)
        except ValueError:
            pass

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(list(self._queue))

    def pop_all(self) -> list[Request]:
        items = list(self._queue)
        self._queue.clear()
        return items


class SchedulerOutput:
    """Result of one schedule() call."""

    def __init__(self):
        self.scheduled_new_reqs: list[tuple[Request, int, list[int]]] = []
        self.scheduled_running_reqs: list[tuple[Request, int, list[int]]] = []
        self.preempted_reqs: list[Request] = []
        self.total_num_scheduled_tokens: int = 0
        self.swap_time: float = 0.0  # GPU↔CPU swap time for this step

    @property
    def scheduled_requests(self) -> list[tuple[Request, int, list[int]]]:
        return self.scheduled_running_reqs + self.scheduled_new_reqs


class ColocatedScheduler:
    """Scheduler for colocated (P+D on same GPU) or D-only (disaggregated) deployment.

    When *reset_on_preempt* is False and the pool has CPU swap enabled,
    OOM on running requests triggers swap-out (GPU→CPU) rather than
    recompute-on-admit.  Swapped-out requests keep their computed tokens
    and are swapped back in when re-admitted.
    """

    def __init__(self, memory_pool, config, *, reset_on_preempt: bool = True):
        self.pool = memory_pool
        self.max_num_batched_tokens = config["simulation"]["max_num_batched_tokens"]
        self.max_num_seqs = config["simulation"]["max_num_seqs"]
        self.block_size = config["simulation"]["block_size"]
        self.enable_chunked_prefill = config["simulation"]["enable_chunked_prefill"]
        self.long_prefill_token_threshold = config["simulation"]["long_prefill_token_threshold"]
        self.policy = config["simulation"].get("scheduling_policy", SchedulingPolicy.FCFS)
        self.max_model_len = config.get("max_model_len", 131072)
        self.reset_on_preempt = reset_on_preempt
        # Use swap instead of skip when OOM on D-side
        self.use_swap = (not reset_on_preempt
                         and getattr(memory_pool, "cpu_swap_bw", None) is not None)

        self.running: list[Request] = []
        self.waiting = RequestQueue()
        self.skipped_waiting = RequestQueue()
        self._finished_requests: list[Request] = []
        self._prefill_done_requests: list[Request] = []

    def has_requests(self) -> bool:
        return bool(self.running) or bool(self.waiting) or bool(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add(request)

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling step. Returns SchedulerOutput."""
        output = SchedulerOutput()
        token_budget = self.max_num_batched_tokens
        preempted = False

        # ── Phase 1: Service RUNNING requests ──
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if request.is_finished():
                req_index += 1
                continue

            num_new = request.num_tokens_with_spec - request.num_computed_tokens
            if num_new <= 0:
                # Decode: always generate 1 token per step after prefill completes
                if not request.is_prefill_chunk and request.num_computed_tokens >= request.prompt_len:
                    num_new = 1
                else:
                    req_index += 1
                    continue

            if self.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)
            num_new = min(num_new, self.max_model_len - 1 - request.num_computed_tokens)
            if num_new <= 0:
                req_index += 1
                continue

            new_blocks = self.pool.allocate_slots(request, num_new, self.block_size)
            if new_blocks is None:
                if self.use_swap:
                    # D-side with swap: swap out the last running request.
                    # Its blocks go to CPU memory; num_computed_tokens is preserved
                    # so it can resume later via swap-in without recomputation.
                    victim = self.running[-1]
                    swap_time = self.pool.swap_out(victim, self.block_size)
                    output.swap_time += swap_time
                    victim.status = RequestStatus.PREEMPTED
                    self.running.remove(victim)
                    self.waiting.prepend(victim)
                    preempted = True
                    output.preempted_reqs.append(victim)
                    if victim == request:
                        break
                    new_blocks = self.pool.allocate_slots(request, num_new, self.block_size)
                    if new_blocks is None:
                        break
                elif not self.reset_on_preempt:
                    # D-side without swap: skip this request for this step
                    req_index += 1
                    continue
                else:
                    # Colocated: preempt with recompute
                    if self.policy == SchedulingPolicy.PRIORITY:
                        victim = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                    else:
                        victim = self.running[-1]

                    self._preempt(victim)
                    preempted = True
                    output.preempted_reqs.append(victim)
                    if victim == request:
                        break
                    # Retry after freeing victim's blocks
                    new_blocks = self.pool.allocate_slots(request, num_new, self.block_size)
                    if new_blocks is None:
                        break

            output.scheduled_running_reqs.append((request, num_new, new_blocks))
            token_budget -= num_new
            req_index += 1

        # ── Phase 2: Admit WAITING requests ──
        if not preempted:
            for queue in (self.waiting, self.skipped_waiting):
                while queue and token_budget > 0:
                    if len(self.running) >= self.max_num_seqs:
                        break

                    request = queue.peek()
                    if request is None:
                        break

                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        queue.pop()
                        # Only move to skipped_waiting if not already there
                        if queue is self.waiting:
                            self.skipped_waiting.add(request)
                        continue

                    # Swapped-out request: restore blocks from CPU
                    if self.pool.is_swapped(request.request_id):
                        t_swap = self.pool.swap_in(request, self.block_size)
                        if t_swap == float("inf"):
                            # Not enough free blocks — wait for next step
                            break
                        output.swap_time += t_swap

                    # Prefix cache hit
                    cached_blocks, num_cached_blocks = self.pool.get_computed_blocks(
                        request.block_hashes
                    )
                    cache_touched = False
                    if num_cached_blocks > 0:
                        num_cached_tokens = num_cached_blocks * self.block_size
                        request.num_computed_tokens = num_cached_tokens
                        request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens_with_spec
                        self.pool.touch(cached_blocks)
                        cache_touched = True
                        for bid in cached_blocks:
                            if bid not in request.block_table:
                                request.block_table.append(bid)

                    num_new = request.num_tokens - request.num_computed_tokens
                    if self.long_prefill_token_threshold > 0:
                        num_new = min(num_new, self.long_prefill_token_threshold)

                    if not self.enable_chunked_prefill and num_new > token_budget:
                        if cache_touched:
                            self.pool.free_blocks(cached_blocks)
                        break

                    num_new = min(num_new, token_budget)
                    if num_new <= 0:
                        # Decode request from disaggregated P→D transfer:
                        # prefill already done, need to generate 1 token
                        if not request.is_prefill_chunk:
                            num_new = 1
                        else:
                            queue.pop()
                            if queue is self.waiting:
                                self.skipped_waiting.add(request)
                            continue

                    new_blocks = self.pool.allocate_slots(request, num_new, self.block_size)
                    if new_blocks is None:
                        if cache_touched:
                            self.pool.free_blocks(cached_blocks)
                            # Undo block_table append
                            for bid in cached_blocks:
                                if bid in request.block_table:
                                    request.block_table.remove(bid)
                        break

                    queue.pop()
                    output.scheduled_new_reqs.append((request, num_new, new_blocks))
                    token_budget -= num_new
                    request.status = RequestStatus.RUNNING
                    self.running.append(request)

        output.total_num_scheduled_tokens = sum(
            nt for _, nt, _ in output.scheduled_requests
        )
        return output

    def _preempt(self, request: Request) -> None:
        """Evict request from running, free KV cache, move to waiting."""
        if request in self.running:
            self.running.remove(request)
        self.pool.free_request(request)
        request.status = RequestStatus.PREEMPTED
        if self.reset_on_preempt:
            request.num_computed_tokens = 0  # must recompute from scratch
        self.waiting.prepend(request)

    def _update_after_schedule(self, output: SchedulerOutput) -> None:
        """Advance num_computed_tokens after scheduling, before execution."""
        for req, num_new, _ in output.scheduled_requests:
            was_prefill = req.is_prefill_chunk
            req.num_computed_tokens += num_new
            req.is_prefill_chunk = req.num_computed_tokens < req.num_tokens_with_spec
            # Detect prefill completion (transition from prefill to decode)
            if was_prefill and not req.is_prefill_chunk:
                self._prefill_done_requests.append(req)

    def update_from_output(self, output: SchedulerOutput, clock: float) -> None:
        """Simulate token generation after step execution."""
        self.pool.clock = clock

        for req, num_new, _ in output.scheduled_requests:
            # Record first schedule timestamp (for timeseries output)
            if req.scheduled_ts is None:
                req.scheduled_ts = clock

            # Determine how many of num_new are actual decode (output) tokens.
            # num_computed_tokens was already incremented in _update_after_schedule,
            # so pre-step value is num_computed_tokens - num_new.
            pre_step_computed = req.num_computed_tokens - num_new
            prompt_remaining = max(0, req.num_prompt_tokens - pre_step_computed)
            decode_tokens = num_new - min(num_new, prompt_remaining)

            if decode_tokens > 0:
                if req.num_output_tokens == 0 and req.ttft is None:
                    req.ttft = clock - req.arrival_time
                req.num_output_tokens += decode_tokens

            # Check stop
            if req.num_output_tokens >= req.max_output_len:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                req.finish_reason = FinishReason.LENGTH
                req.finish_time = clock
                self._finish_request(req)

    def _finish_request(self, request: Request) -> None:
        """Mark request finished and release resources."""
        request.finish_time = request.finish_time or self.pool.clock
        self.pool.free_request(request)
        if request in self.running:
            self.running.remove(request)
        self._finished_requests.append(request)

    def drain_finished(self) -> list[Request]:
        """Return and clear finished requests since last call."""
        finished = list(self._finished_requests)
        self._finished_requests.clear()
        return finished

    def drain_prefill_done(self) -> list[Request]:
        """Return and clear requests that just completed prefill."""
        done = list(self._prefill_done_requests)
        self._prefill_done_requests.clear()
        return done
