"""Trace one complete simulation step with full detail."""
import sys
sys.path.insert(0, ".")

from pd_sim.config import load_config
from pd_sim.trace import load_trace
from pd_sim.engine import SimulationEngine
from pd_sim.scheduler import ColocatedScheduler
from pd_sim.memory import BlockPool, compute_block_hashes
from pd_sim.executor import predict_step
from pd_sim.engine import SimulationEvent, EventType
from perf_predict.predict import (
    load_model_specs, load_hw_params, matmul_time,
    projections, attention_fused, elementwise_ops,
)
import heapq

specs = load_model_specs()
model = next(m for m in specs["models"] if m["name"] == "Llama-2-7B")
cfg = load_config("config/pd_sim.yaml", model_spec=model)
hw = load_hw_params("fit/results/3090.json")

requests = load_trace("traces/myllama-2-7b-light.jsonl", max_requests=3)
engine = SimulationEngine(cfg, model, hw)

# Reset state same as engine.run()
for r in requests:
    r.num_computed_tokens = 0
    r.num_output_tokens = 0
    r.status = 0
    r.finish_reason = None
    r.finish_time = None
    r.ttft = None
    r.is_prefill_chunk = True
    r.block_hashes = compute_block_hashes(r.prompt_token_ids, engine.block_size)
    r.block_table.clear()

pool = BlockPool(engine.num_blocks)
sched = ColocatedScheduler(pool, cfg)
metrics_coll = __import__("pd_sim.metrics", fromlist=["MetricsCollector"]).MetricsCollector()

event_queue = []
for r in requests:
    heapq.heappush(event_queue, SimulationEvent(r.arrival_time, EventType.ARRIVAL, r))

# Shortcut names
h = model["hidden_dim"]
inter = model.get("intermediate_dim", h * 4)
nh = model["num_heads"]
nh_kv = model.get("num_kv_heads", nh)
hd = model["head_dim"]
vs_ = model["vocab_size"]
nl = model["num_layers"]
na = model.get("attn_layers", nl)
b_effs = hw["elem_b_effs"]
overheads = hw["elem_overheads"]
norm = model.get("norm_type", "rmsnorm")

M_SPLIT = 256

block_size = cfg["simulation"]["block_size"]
max_tok = cfg["simulation"]["max_num_batched_tokens"]
max_seqs = cfg["simulation"]["max_num_seqs"]
threshold = cfg["simulation"]["long_prefill_token_threshold"]

print("=" * 72)
print("SYSTEM CONFIGURATION")
print("=" * 72)
print(f"  GPU: {cfg['gpu']}  |  Model: {model['name']}")
print(f"  hidden_dim={h}, inter_dim={inter}, heads={nh}, kv_heads={nh_kv}, head_dim={hd}")
print(f"  layers={nl}, attn_layers={na}, vocab={vs_}")
print(f"  block_size={block_size}, max_num_seqs={max_seqs}")
print(f"  max_num_batched_tokens={max_tok}, long_prefill_threshold={threshold}")
print(f"  num_blocks={engine.num_blocks}, bytes_per_block={engine.bytes_per_block}")
print(f"  F_prefill={hw['F_peak_prefill']/1e12:.1f}TF  B_prefill={hw['B_peak_prefill']/1e9:.0f}GB/s")
print(f"  F_decode ={hw['F_peak_decode']/1e12:.1f}TF  B_decode ={hw['B_peak_decode']/1e9:.0f}GB/s")
print()

# ================================================================
# STEP 1: req-0 arrives, gets prefilled (16 tokens)
# ================================================================
print("=" * 72)
print("STEP 1: req-0 (prompt=16, output=139) arrives at t=0.2346s")
print("=" * 72)

engine.clock = event_queue[0].time
ev = heapq.heappop(event_queue)
sched.add_request(ev.request)
print(f"[Clock] {engine.clock:.4f}s")
print(f"[Event] {ev.request.request_id} added to waiting queue")

output = sched.schedule()
print(f"[Scheduler]")
print(f"  Phase 1 (running, {len(sched.running)} reqs): nothing")
for r, nt, blks in output.scheduled_new_reqs:
    print(f"  Phase 2 (new): {r.request_id}  num_new={nt}  blocks_used={len(blks)}")
print(f"  Total scheduled tokens: {output.total_num_scheduled_tokens}")
print(f"  Token budget used: {output.total_num_scheduled_tokens}/{max_tok}")
print(f"  Remaining budget: {max_tok - output.total_num_scheduled_tokens}")

# _update_after_schedule
sched._update_after_schedule(output)
scheduled = [(r, nt) for r, nt, _ in output.scheduled_requests]

# Decompose predict_step
total_new = sum(nt for _, nt in scheduled)
params = "decode" if total_new < M_SPLIT else "prefill"
F = hw["F_peak_decode"] if total_new < M_SPLIT else hw["F_peak_prefill"]
B = hw["B_peak_decode"] if total_new < M_SPLIT else hw["B_peak_prefill"]
p = hw["p_decode"] if total_new < M_SPLIT else hw["p_prefill"]
print(f"\n[Executor.predict_step] total_new_tokens={total_new} -> {params} params")
print(f"  F={F/1e12:.1f} TFLOPS, B={B/1e9:.0f} GB/s, p={p:.3f}")

# --- projections ---
proj_per_layer = projections(total_new, h, inter, F, B, p, nh, nh_kv, hd)
proj_total = nl * proj_per_layer
print(f"\n[Projections]  ({nl} layers)")
print(f"  Per layer (7 matmuls batched over {total_new} tokens):")
# Q/K/V/O
t_q = matmul_time(total_new, h, h, F, B, p)
t_k = t_q
t_v = t_q
t_o = matmul_time(total_new, h, h, F, B, p) if nh * hd == h else matmul_time(total_new, nh * hd, h, F, B, p)
# Gate/Up/Down
t_gate = matmul_time(total_new, h, inter, F, B, p)
t_up = matmul_time(total_new, h, inter, F, B, p)
t_down = matmul_time(total_new, inter, h, F, B, p)
print(f"    Q proj: matmul({total_new}x{h}x{h}) = {t_q*1e6:.1f}us")
print(f"    K proj: matmul({total_new}x{h}x{h}) = {t_k*1e6:.1f}us")
print(f"    V proj: matmul({total_new}x{h}x{h}) = {t_v*1e6:.1f}us")
print(f"    O proj: matmul({total_new}x{h}x{h}) = {t_o*1e6:.1f}us")
print(f"    gate:   matmul({total_new}x{h}x{inter}) = {t_gate*1e6:.1f}us")
print(f"    up:     matmul({total_new}x{h}x{inter}) = {t_up*1e6:.1f}us")
print(f"    down:   matmul({total_new}x{inter}x{h}) = {t_down*1e6:.1f}us")
print(f"    per-layer sum: {proj_per_layer*1e6:.1f}us")
print(f"  Total ({nl} layers): {proj_total*1000:.1f}ms")

# --- attention per request ---
attn_total = 0.0
print(f"\n[Attention]  ({na} attention layers, per-request, summed)")
for req, num_new in scheduled:
    kv_len = req.num_computed_tokens  # already incremented by _update_after_schedule
    attn_per_layer = attention_fused(1, nh, num_new, kv_len, hd, F, B, p)
    attn_req = na * attn_per_layer
    attn_total += attn_req
    print(f"  {req.request_id}: batch=1, s_q={num_new}, s_kv={kv_len}")
    print(f"    flops=4*1*{nh}*{num_new}*{kv_len}*{hd}={4*1*nh*num_new*kv_len*hd/1e6:.1f}M")
    print(f"    bytes=4*1*{nh}*max({num_new},{kv_len})*{hd}*2={4*1*nh*max(num_new,kv_len)*hd*2/1e6:.1f}MB")
    print(f"    per_layer={attn_per_layer*1e6:.2f}us, total={attn_req*1e6:.1f}us")
print(f"  Attention sum: {attn_total*1000:.1f}ms")

# --- elementwise ---
elem_per_layer = elementwise_ops(1, total_new, h, inter, nh, hd, norm, b_effs, overheads)
elem_total = nl * elem_per_layer
N_elem = 1 * total_new * h
print(f"\n[Elementwise]  ({nl} layers, N=1*{total_new}*{h}={N_elem})")
print(f"  Per layer: rmsnorm*2 + swiglu + rope + residual_add*2 = {elem_per_layer*1e6:.1f}us")
print(f"  Total ({nl} layers): {elem_total*1000:.1f}ms")

# --- lm_head ---
lm_time = matmul_time(total_new, h, vs_, F, B, p)
lm_flops = 2 * total_new * h * vs_
lm_bytes = (total_new * h + h * vs_ + total_new * vs_) * 2
print(f"\n[LM head]  matmul({total_new}x{h}x{vs_})")
print(f"  flops={lm_flops/1e9:.2f}G, bytes={lm_bytes/1e6:.1f}MB")
print(f"  time={lm_time*1e6:.0f}us = {lm_time*1000:.1f}ms")

# --- TOTAL ---
step_time = proj_total + attn_total + elem_total + lm_time
print(f"\n[STEP TIME]")
print(f"  projections:  {proj_total*1000:7.1f}ms  ({proj_total/step_time*100:.0f}%)")
print(f"  attention:    {attn_total*1000:7.1f}ms  ({attn_total/step_time*100:.0f}%)")
print(f"  elementwise:  {elem_total*1000:7.1f}ms  ({elem_total/step_time*100:.0f}%)")
print(f"  lm_head:      {lm_time*1000:7.1f}ms  ({lm_time/step_time*100:.0f}%)")
print(f"  ─────────────────────────")
print(f"  TOTAL:        {step_time*1000:7.1f}ms")

engine.clock += step_time
sched.update_from_output(output, engine.clock)
print(f"\n[After step] clock={engine.clock:.4f}s, running={len(sched.running)}, waiting={len(sched.waiting)}")
for r in sched.drain_finished():
    print(f"[Finish] {r.request_id}: ttft={r.ttft*1000 if r.ttft else 0:.0f}ms")

# ================================================================
# STEP 2: Pure decode (single request, no new arrivals)
# ================================================================
print()
print("=" * 72)
print("STEP 2: req-0 decode (1 token), no new arrivals")
print("=" * 72)
print(f"[Clock] {engine.clock:.4f}s (no jump: GPU not idle)")

output2 = sched.schedule()
print(f"[Scheduler]")
print(f"  Phase 1 (running, {len(sched.running)} reqs):")
for r, nt, blks in output2.scheduled_running_reqs:
    print(f"    {r.request_id}: decode {nt} token")
print(f"  Total scheduled tokens: {output2.total_num_scheduled_tokens}")
print(f"  Token budget used: {output2.total_num_scheduled_tokens}/{max_tok}")

sched._update_after_schedule(output2)
sched2 = [(r, nt) for r, nt, _ in output2.scheduled_requests]
total_new2 = sum(nt for _, nt in sched2)

params2 = "decode" if total_new2 < M_SPLIT else "prefill"
F2 = hw["F_peak_decode"] if total_new2 < M_SPLIT else hw["F_peak_prefill"]
B2 = hw["B_peak_decode"] if total_new2 < M_SPLIT else hw["B_peak_prefill"]
p2 = hw["p_decode"] if total_new2 < M_SPLIT else hw["p_prefill"]

proj2 = nl * projections(total_new2, h, inter, F2, B2, p2, nh, nh_kv, hd)
attn2 = 0.0
for req, num_new in sched2:
    kv_len = req.num_computed_tokens
    attn2 += na * attention_fused(1, nh, num_new, kv_len, hd, F2, B2, p2)
elem2 = nl * elementwise_ops(1, total_new2, h, inter, nh, hd, norm, b_effs, overheads)
lm2 = matmul_time(total_new2, h, vs_, F2, B2, p2)
step2_time = proj2 + attn2 + elem2 + lm2

print(f"\n[Executor.predict_step] total_new_tokens={total_new2} -> {params2} params")
print(f"[STEP TIME]")
print(f"  projections:  {proj2*1000:7.1f}ms  ({proj2/step2_time*100:.0f}%)")
print(f"  attention:    {attn2*1000:7.1f}ms  ({attn2/step2_time*100:.0f}%)")
print(f"  elementwise:  {elem2*1000:7.1f}ms  ({elem2/step2_time*100:.0f}%)")
print(f"  lm_head:      {lm2*1000:7.1f}ms  ({lm2/step2_time*100:.0f}%)")
print(f"  TOTAL:        {step2_time*1000:7.1f}ms")

engine.clock += step2_time
sched.update_from_output(output2, engine.clock)
print(f"\n[After step] clock={engine.clock:.4f}s")
for r in sched.drain_finished():
    print(f"[Finish] {r.request_id}: ttft={r.ttft*1000 if r.ttft else 0:.0f}ms")

# Show TTFT for req-0
for r in sched.running:
    if r.ttft is not None:
        print(f"[TTFT] {r.request_id} = {r.ttft*1000:.0f}ms (= {engine.clock:.4f} - {r.arrival_time:.4f})")

# ================================================================
# STEP 3+: Show steady state with multiple running requests
# ================================================================
print()
print("=" * 72)
print("STEADY STATE: 3 requests running, decode-only step")
print("=" * 72)

# Manually add the other 2 requests to running (simulate after their prefill done)
req1 = requests[1]
req2 = requests[2]
req1.num_computed_tokens = req1.prompt_len + 1  # prefill done + 1 decode
req1.num_output_tokens = 1
req1.is_prefill_chunk = False
req1.status = 2  # RUNNING
req2.num_computed_tokens = req2.prompt_len + 1
req2.num_output_tokens = 1
req2.is_prefill_chunk = False
req2.status = 2
sched.running.extend([req1, req2])

output3 = sched.schedule()
sched._update_after_schedule(output3)
sched3 = [(r, nt) for r, nt, _ in output3.scheduled_requests]
total_new3 = sum(nt for _, nt in sched3)

params3 = "decode" if total_new3 < M_SPLIT else "prefill"
F3 = hw["F_peak_decode"] if total_new3 < M_SPLIT else hw["F_peak_prefill"]
B3 = hw["B_peak_decode"] if total_new3 < M_SPLIT else hw["B_peak_prefill"]
p3 = hw["p_decode"] if total_new3 < M_SPLIT else hw["p_prefill"]

proj3 = nl * projections(total_new3, h, inter, F3, B3, p3, nh, nh_kv, hd)
attn3 = 0.0
for req, num_new in sched3:
    kv_len = req.num_computed_tokens
    a = attention_fused(1, nh, num_new, kv_len, hd, F3, B3, p3)
    attn3 += na * a
    print(f"[Attention] {req.request_id}: q={num_new}, kv={kv_len}, per_layer={a*1e6:.2f}us, total={na*a*1e6:.1f}us")
elem3 = nl * elementwise_ops(1, total_new3, h, inter, nh, hd, norm, b_effs, overheads)
lm3 = matmul_time(total_new3, h, vs_, F3, B3, p3)
step3_time = proj3 + attn3 + elem3 + lm3

print(f"\n[Executor.predict_step] total_new_tokens={total_new3} -> {params3} params")
print(f"[STEP TIME]")
print(f"  projections:  {proj3*1000:7.1f}ms  ({proj3/step3_time*100:.0f}%)")
print(f"  attention:    {attn3*1000:7.1f}ms  ({attn3/step3_time*100:.0f}%)")
print(f"  elementwise:  {elem3*1000:7.1f}ms  ({elem3/step3_time*100:.0f}%)")
print(f"  lm_head:      {lm3*1000:7.1f}ms  ({lm3/step3_time*100:.0f}%)")
print(f"  TOTAL:        {step3_time*1000:7.1f}ms")
print(f"  Throughput: {total_new3}/{step3_time*1000:.1f}ms = {total_new3/step3_time:.0f} tok/s (decode only)")
