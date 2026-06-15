"""Demonstrate how max_batched_tokens and prefill_threshold shape scheduling."""
import sys
sys.path.insert(0, ".")

from pd_sim.scheduler import ColocatedScheduler, SchedulerOutput
from pd_sim.memory import BlockPool, compute_block_hashes
from pd_sim.request import Request, RequestStatus

def build_sched(max_batch, threshold, max_seqs=16):
    pool = BlockPool(4096)
    cfg = {"simulation": {
        "max_num_batched_tokens": max_batch,
        "max_num_seqs": max_seqs,
        "block_size": 16,
        "enable_chunked_prefill": True,
        "long_prefill_token_threshold": threshold,
        "scheduling_policy": "fcfs",
    }, "max_model_len": 8192}
    return ColocatedScheduler(pool, cfg), pool

def add_running(sched, count, prompt_len=50, output_so_far=30):
    """Add running decode requests."""
    for i in range(count):
        r = Request(f"r{i}", 0.0, list(range(prompt_len)), 200)
        r.num_computed_tokens = prompt_len + output_so_far
        r.num_output_tokens = output_so_far
        r.is_prefill_chunk = False
        r.status = RequestStatus.RUNNING
        r.block_table = list(range(i * 20, i * 20 + 20))
        sched.running.append(r)

def add_waiting(sched, count, prompt_len):
    """Add waiting requests with diverse prompt lengths."""
    for i in range(count):
        r = Request(f"w{i}", 0.0, list(range(prompt_len)), 200)
        r.block_hashes = compute_block_hashes(list(range(prompt_len)), 16)
        sched.waiting.add(r)

# ================================================================
# Experiment 1: Fix threshold=256, vary max_batch
# ================================================================
print("=" * 72)
print("EXPERIMENT 1: prefill_threshold=256, vary max_batched_tokens")
print("  12 running (decode) + 3 waiting (prompt=500, 200, 30)")
print("=" * 72)

for max_batch in [128, 256, 512, 2048]:
    sched, _ = build_sched(max_batch, 256)
    add_running(sched, 12)
    add_waiting(sched, 3, 500)   # all 3 have 500-token prompts
    sched.waiting._queue[0].request_id = "w0-long"
    sched.waiting._queue[1].request_id = "w1-long"
    sched.waiting._queue[2].request_id = "w2-long"

    output = sched.schedule()
    d_tok = sum(nt for _, nt, _ in output.scheduled_running_reqs)
    admitted = [(r.request_id, nt) for r, nt, _ in output.scheduled_new_reqs]

    w0_tok = next((nt for rid, nt in admitted if rid == "w0-long"), 0)
    n_admitted = len(admitted)
    print(f"  max_batch={max_batch:4d}: decode={d_tok:3d} tok, "
          f"admitted {n_admitted} new (w0 gets {w0_tok} tok), "
          f"budget used={d_tok+sum(nt for _,nt in admitted)}/{max_batch}")
print()
print("  → max_batch=128: decode 占用全部 budget，prefill 完全饿死")
print("  → max_batch=512: w0 拿到 256 tok, 但 budget 只够 1 个新请求")
print("  → max_batch=2048: 3 个新请求全部准入, w0 仍被切到 256/chunk")

# ================================================================
# Experiment 2: Fix max_batch=1024, vary threshold
# ================================================================
print()
print("=" * 72)
print("EXPERIMENT 2: max_batched_tokens=1024, vary prefill_threshold")
print("  12 running + 1 waiting (prompt=3000)")
print("=" * 72)

for threshold in [256, 512, 1024, 2048, 4096]:
    sched, _ = build_sched(1024, threshold)
    add_running(sched, 12)
    add_waiting(sched, 1, 3000)
    sched.waiting._queue[0].request_id = "w0-huge"

    output = sched.schedule()
    d_tok = sum(nt for _, nt, _ in output.scheduled_running_reqs)
    w0_tok = 0
    for r, nt, _ in output.scheduled_new_reqs:
        w0_tok = nt

    chunks_needed = (3000 + w0_tok - 1) // w0_tok if w0_tok > 0 else float("inf")
    budget_remaining = 1024 - d_tok
    actual_tok = min(3000, threshold, budget_remaining)
    print(f"  threshold={threshold:4d}: decode={d_tok} tok, w0 gets {w0_tok} tok, "
          f"need {chunks_needed} prefill chunks to finish")

print()
print("  → threshold=256: 需要 ceil(3000/256)=12 个 prefill chunk，TTFT 延迟很大")
print("  → threshold=4096: 1 个 chunk 搞定，但单步要处理 1012 tok（重 prefill step）")

# ================================================================
# Experiment 3: The preemption path — OOM triggers preemption
# ================================================================
print()
print("=" * 72)
print("EXPERIMENT 3: Block pool OOM → preemption path")
print("  Small pool (12 blocks), 8 running + 1 waiting")
print("=" * 72)

sched, pool = build_sched(2048, 512, max_seqs=16)
# Artificially shrink pool
pool.free_block_ids = list(range(1, 12))  # only 11 free blocks
pool.num_blocks = 12

# Running requests each hold some blocks
for i in range(8):
    r = Request(f"r{i}", 0.0, list(range(200)), 200)
    r.num_computed_tokens = 200
    r.num_output_tokens = 50
    r.is_prefill_chunk = False
    r.status = RequestStatus.RUNNING
    # Each holds 13 blocks (200/16 ≈ 13)
    r.block_table = list(range(i * 15 + 20, i * 15 + 33))
    sched.running.append(r)

# Waiting request
wr = Request("w0", 0.0, list(range(100)), 200)
wr.block_hashes = compute_block_hashes(list(range(100)), 16)
sched.waiting.add(wr)

# Pre-consume most free blocks
pool.free_block_ids = pool.free_block_ids[:2]  # only 2 free
print(f"  Free blocks before schedule: {len(pool.free_block_ids)}")
print(f"  Running requests: {len(sched.running)}, each holds ~13 blocks")

output = sched.schedule()
print(f"  Preempted: {[(r.request_id, r.num_computed_tokens) for r in output.preempted_reqs]}")
print(f"  Phase 1 (decode): {[(r.request_id, nt) for r, nt, _ in output.scheduled_running_reqs]}")
print(f"  Phase 2 (new):    {[(r.request_id, nt) for r, nt, _ in output.scheduled_new_reqs]}")
print(f"  Waiting queue:    {[r.request_id for r in sched.waiting]}")
print()
print("  → OOM 时抢占 running 末尾请求，释放 blocks 后重试")
print("  → 被抢占的请求回到 waiting 队列头部，num_computed_tokens 归零")

# ================================================================
# Experiment 4: Step time impact of batch composition
# ================================================================
print()
print("=" * 72)
print("EXPERIMENT 4: How batch composition drives step time")
print("=" * 72)

from pd_sim.executor import predict_step
from perf_predict.predict import load_model_specs, load_hw_params

specs = load_model_specs()
model = next(m for m in specs["models"] if m["name"] == "Llama-2-7B")
hw = load_hw_params("fit/results/3090.json")

# Build fake scheduled_requests for different scenarios
M_SPLIT = 256

def fake_req(req_id, num_new, kv_len, is_prefill):
    r = Request(req_id, 0.0, list(range(kv_len if is_prefill else 100)), 200)
    r.num_computed_tokens = kv_len  # already incremented
    r.num_output_tokens = max(0, kv_len - 100)
    r.is_prefill_chunk = is_prefill
    return r

scenarios = [
    ("pure decode (16 reqs)", [fake_req(f"d{i}", 1, 100+i*10, False) for i in range(16)]),
    ("pure decode (32 reqs)", [fake_req(f"d{i}", 1, 100+i*5, False) for i in range(32)]),
    ("mix: 16 dec + 1 prefill(256)",
     [fake_req(f"d{i}", 1, 100+i*10, False) for i in range(16)] +
     [fake_req("p0", 256, 256, True)]),
    ("mix: 16 dec + 1 prefill(512)",
     [fake_req(f"d{i}", 1, 100+i*10, False) for i in range(16)] +
     [fake_req("p0", 512, 512, True)]),
    ("pure prefill (1024)", [fake_req("p0", 1024, 1024, True)]),
]

for label, reqs in scenarios:
    total = sum(r.num_computed_tokens - (r.num_computed_tokens - nt)
                for r, nt in [(r, r.num_computed_tokens - (r.num_computed_tokens - 1)) for r in reqs])
    # Simplify: just sum the num_new
    total_new = sum(1 if not r.is_prefill_chunk else
                    r.num_computed_tokens - (r.num_computed_tokens -
                     (r.num_computed_tokens - (r.num_computed_tokens - r.num_output_tokens - 100)))
                    for r in reqs)
    # Actually let me just compute directly
    scheduled = []
    for r in reqs:
        if r.is_prefill_chunk:
            nt = r.num_computed_tokens  # full prefill tokens
        else:
            nt = 1
        scheduled.append((r, nt))

    t = predict_step(scheduled, model, hw)
    total_tok = sum(nt for _, nt in scheduled)
    print(f"  {label:35s}: {total_tok:4d} tok/step, step_time={t*1000:5.1f}ms, "
          f"thr={total_tok/t:.0f} tok/s")
