# vLLM v0.19.0 源码 vs 模拟器差异分析

基于对 vLLM v0.19.0 核心源码的逐行对比分析，按影响程度从高到低排列。
已完成的标记 ✅，未完成的标记 ⬜。

---

## 🔴 P0-1: CUDA Graph Decode 加速未建模 ✅

**日期**: 2026-07-05　**分支**: main

**解决方案**: 新增 `bench/cudagraph.py` + `fit/cudagraph.py`，对每个 op 在 CUDA Graph replay 下单独 benchmark，独立拟合 roofline 参数（key 含 `_cudagraph` 后缀）。模拟器通过 `use_cudagraph` 配置开关选择使用 graph 专用参数。详见下方修正建议。

**源码**: `vllm-0.19.0/vllm/v1/engine/core.py` — `_initialize_kv_caches()` (L~230-270)  
**相关**: `vllm-0.19.0/vllm/v1/worker/gpu_model_runner.py`

**问题**: vLLM 在 warmup 阶段捕获 CUDA Graphs，decode 时通过 **graph replay** 执行 forward pass。CUDA Graph replay 消除了所有 kernel launch overhead（每个 kernel ~5-10 μs），decode 延迟因此降低 2-5×。

你的模拟器通过 roofline 模型（`sim/executor.py:predict_step()`）计算时间，该模型基于 micro-benchmark 的 FLOPs/bytes 外推，**完全不知道 CUDA Graph 的存在**。如果 benchmark 数据来自非 graph 模式的测量，decode step time 会被系统性地高估。

**影响**:
- Decode step 延迟高估 **2-5×**
- TTFT（首 token 延迟）基本准确（prefill 不用 CUDA Graph）
- TPOT（每 token 输出延迟）被严重高估
- 整体 throughput 被低估

**修正建议**:

1. **在 benchmark 阶段**：确保 `bench/` 中的 micro-benchmark 测量了 CUDA Graph replay 的性能（或单独测量 graph vs non-graph 的加速比）

2. **在 fit 阶段**：为 decode 拟合一组独立的 CUDA Graph 加速因子 `cuda_graph_speedup_decode`：
   ```python
   # fit 阶段: 对比 decode 场景下 graph replay vs eager 的延迟比
   # 典型值: 3-5x for M=1, batch_size=1
   ```

3. **在 sim 阶段**：在 `predict_step()` 中将 decode 部分乘以加速因子：
   ```python
   # sim/executor.py
   decode_speedup = hw_params.get("cuda_graph_speedup_decode", 1.0)
   attn_decode_time /= decode_speedup
   ffn_proj_time /= decode_speedup  # 仅 decode 贡献部分
   # 注意: 不能直接除整个 step time，prefill 不受影响
   ```

4. **更精确的方案**：将 decode 的 roofline 参数（B_peak, p）直接拟合到 CUDA Graph replay 的实测数据上，而非从非 graph benchmark 转换。

**难度**: 中　**影响**: ⭐⭐⭐⭐⭐

---

## 🔴 P0-2: Kernel Launch Overhead 完全缺失 ✅

**日期**: 2026-07-05　**分支**: main

**解决方案**: 新增 `bench/launch_overhead.py` + `fit/launch_overhead.py`，使用 CPU wall-clock vs GPU event 斜率法测量纯 CPU→GPU dispatch 开销。结果存入 `kernel_launch_overhead_us` 参数，`predict_step()` 中按每步 kernel 数量累加（`nl × 10 + 1` 个 kernel × overhead），CUDA Graph 模式下自动归零。

**源码**: `vllm-0.19.0/vllm/v1/worker/gpu_model_runner.py`

**问题**: 除了 CUDA Graph 场景外，vLLM 每 step 执行数十到上百个 CUDA kernel。每个 kernel launch 有 ~5-10 μs 的固定 CPU→GPU 调度开销。对于 decode（小 batch，总计算量仅 ~100-500 μs），kernel launch overhead 占比可达 **10-30%**。

你的 roofline 模型假设 kernel 执行时间是"瞬间开始、完美流水线"的，未加入任何固定开销项。即使 prefill 场景（计算量大），kernel launch overhead 也可能占 1-3%。

**影响**:
- Decode step 延迟被**低估** 10-30%（如果同时修复了 P0-1，这一项部分抵消 CUDA Graph 加速）
- Prefill step 延迟略微低估 1-3%

**修正建议**:

在 `predict_step()` 中增加固定开销项：

```python
# sim/executor.py - predict_step() 末尾
KERNEL_LAUNCH_OVERHEAD = 8e-6  # 8 μs per kernel launch (保守估计)

# 估算本 step 的 kernel 数量
# 每层: QKV_proj + O_proj + gate_proj + up_proj + down_proj
#      + fused_add_norm×2 + swiglu + rope + attention(1-2 dependent on backend)
# ≈ 10 kernel / layer
num_kernels = nl * 10 + 1  # +1 for lm_head
total += num_kernels * KERNEL_LAUNCH_OVERHEAD
```

注意：这个开销在 CUDA Graph 模式下为零（graph replay 只 launch 1 次）。因此需要根据是否启用 graph 来决定是否加这项。

**难度**: 低　**影响**: ⭐⭐⭐⭐

---

## 🔴 P0-3: 异步调度 / Batch Queue "零气泡"优化未建模 ✅

**日期**: 2026-07-05　**分支**: main

**解决方案**: 新增 `sim/pipeline.py` — `ScheduleExecutePipeline` 两阶段流水线模型（CPU schedule + GPU execute），通过 `busy_until` 时间戳追踪实现 schedule/execute 重叠。`pp_size > 1` 时自动启用（PP 场景必须），`pp_size == 1` 时可通过 `async_scheduling` 配置手动开启。`estimate_schedule_time()` 提供 schedule CPU 耗时估值（与 P2-7 共享）。

**源码**:
- `vllm-0.19.0/vllm/v1/engine/core.py` — `step_with_batch_queue()` (L~440-530)
- `vllm-0.19.0/vllm/v1/core/sched/async_scheduler.py` — `AsyncScheduler`

**问题**: vLLM v0.19.0 有两种机制来重叠 CPU 调度和 GPU 执行：

### (A) Batch Queue（Pipeline Parallelism 场景）
- `batch_queue` 是一个 deque，容量 = `max_concurrent_batches`
- 当前一个 batch 在 GPU 执行时，CPU 已经开始调度下一个 batch
- 调度和执行的流水线化消除了 "bubble"

### (B) AsyncScheduler（异步调度）
- `AsyncScheduler._update_after_schedule()` 在调度后**立即假设**下一个 decode token 会生成（预分配 `num_output_placeholders`）
- 下一个 step 的调度不需要等待当前 step 的 GPU 结果
- 对于 decode-only batch，这完全消除了 CPU-GPU 同步延迟

你的模拟器始终是 **串行** 模型：
```
schedule → predict_step → clock += step_time → schedule → ...
```

而在 vLLM 中（batch_queue_size > 1 或 async_scheduling 启用时），实际流水线是：
```
        |--- GPU step N ---|
|--- schedule N+1 ---|     |--- GPU step N+1 ---|
                        |--- schedule N+2 ---|
```

**影响**:
- 对于 PP > 1：pipeline bubble 约占 10-25% 的 step time，你的模拟器高估了延迟
- 对于 async scheduling 的 decode：你的模拟器多加了 ~100-500 μs/step 的虚假设想等待时间
- 对于普通 colocated 推理（无 PP、无 async scheduling）：影响较小

**修正建议**:

1. **最简单的修正**：在 `engine.py:_run_colocated()` 和 `_run_disaggregated()` 中，对 step time 乘以一个 overlap 因子：
   ```python
   # 当 batch_queue_size > 1 或 async_scheduling 时
   schedule_overlap_factor = 0.85  # 15% 重叠节省
   self.clock += max_step * schedule_overlap_factor
   ```

2. **更精确的修正**：实现真正的流水线模型
   ```python
   # 伪代码
   class PipelineStage:
       def __init__(self):
           self.busy_until = 0.0
   
   schedule_stage = PipelineStage()
   execute_stage = PipelineStage()
   
   schedule_end = max(self.clock, schedule_stage.busy_until) + schedule_time
   schedule_stage.busy_until = schedule_end
   
   exec_start = max(schedule_end, execute_stage.busy_until)
   exec_end = exec_start + gpu_time
   execute_stage.busy_until = exec_end
   
   self.clock = exec_end
   ```

**难度**: 中　**影响**: ⭐⭐⭐⭐（PP 场景）/ ⭐⭐（colocated 场景）

---

## 🟡 P1-4: 抢占时"撤销已调度请求"的回退逻辑缺失 ✅

**日期**: 2026-07-05　**分支**: main

**解决方案**: 新增 `_rollback_if_scheduled()` 辅助函数，在 Phase 1 抢占 victim 后检查其是否已在 `scheduled_running_reqs` 中。若已调度则回退：从列表中移除、恢复 `token_budget`、回退 `req_index`。同时适用于 swap 和 recompute 两种抢占路径。

**源码**: `vllm-0.19.0/vllm/v1/core/sched/scheduler.py` L~470-483

**问题**: 在 Phase 1（running 队列调度）中，当因 KV cache 不足需要抢占一个请求时，如果被抢占的请求**恰好在本轮已被调度**（即已在 `scheduled_running_reqs` 中），vLLM 会执行回退操作：

1. 从 `scheduled_running_reqs` 中移除
2. 恢复 `token_budget`（加上它占用的 token 数）
3. 从 `req_to_new_blocks` 和 `scheduled_spec_decode_tokens` 中移除
4. 如果被抢占的请求有 encoder inputs，恢复 encoder compute budget
5. `req_index -= 1`（重新处理当前位置）

你的模拟器（`sim/scheduler.py:schedule()` Phase 1）在抢占时没有这个回退逻辑。抢占即 `_preempt(victim)`，但不会检查 victim 是否已在 `scheduled_running_reqs` 中。

**影响**:
- 当被抢占请求已在本步调度时，token_budget 被错误地多扣除了（该请求的 num_new 没有恢复）
- 可能导致本步实际调度的 token 数小于应有的值
- 在高负载（KV cache 压力大）时误差更明显

**修正建议**:

在 `sim/scheduler.py:schedule()` 的抢占路径中增加检查：

```python
# 在 ColocatedScheduler.schedule() Phase 1 中，抢占后:
if victim in [r for r, _, _ in output.scheduled_running_reqs]:
    # 回退已调度状态
    output.scheduled_running_reqs = [
        (r, nt, blk) for r, nt, blk in output.scheduled_running_reqs
        if r != victim
    ]
    token_budget += num_new_for_victim
    req_index -= 1  # 重新处理当前位置
```

**难度**: 低　**影响**: ⭐⭐⭐（高负载时）

---

## 🟡 P1-5: `scheduler_reserve_full_isl` 准入门控缺失 ✅

**日期**: 2026-07-05　**分支**: main

**解决方案**: 在 `ColocatedScheduler.__init__` 中新增 `scheduler_reserve_full_isl` 配置参数（默认 `true`）。Phase 2 中，对首次调度的 chunked prefill 请求，在分配 block 之前检查整个 prompt 的 block 数是否 ≤ 空闲 block 数，不足则 `break` 等待——而非先调度第一个 chunk 再反复抢占。

**源码**: `vllm-0.19.0/vllm/v1/core/sched/scheduler.py` L~680-690  
**相关**: `vllm-0.19.0/vllm/v1/core/kv_cache_manager.py` — `can_fit_full_sequence()` (L~230-260)

**问题**: 当启用 chunked prefill 时，一个新请求的 prefill 可能跨越多个 step（每个 step 只处理 `long_prefill_token_threshold` 个 token）。vLLM 有一个重要的保护机制：在**第一次调度该请求时**检查其**整个输入序列**的 KV cache 是否放得下（通过 `can_fit_full_sequence()`）。如果放不下，直接拒绝调度该请求（`break`），而不是先调度第一个 chunk 然后在后续 chunk 中反复触发抢占。

你的模拟器（`sim/scheduler.py:schedule()` Phase 2）没有这个检查。它会乐观地先将第一个 chunk 调度进 running，后续 chunk 如果 KV cache 不够则触发抢占。这在以下场景导致偏差：

- 长 prompt + 小 KV cache：你的模拟器频繁抢占，实际 vLLM 会等待有足够空间才接纳
- 延迟估计偏差：频繁抢占 → 更多 step 数 → 更高延迟

**影响**:
- 对长 prompt 请求，模拟器可能**低估**延迟（因为频繁抢占相比"等待"更快出结果？不一定——取决于抢占策略）
- 实际上可能导致模拟结果**忽高忽低**的不稳定表现

**修正建议**:

在 `ColocatedScheduler.schedule()` Phase 2 开始前增加准入检查：

```python
# 在调度 waiting 请求之前
if self.cfg["simulation"].get("scheduler_reserve_full_isl", False):
    num_blocks_needed = request.num_prompt_tokens // self.block_size + 1
    if num_blocks_needed > self.pool.get_num_free_blocks():
        break  # 不放行，等待更多空间
```

或在配置中新增参数 `scheduler_reserve_full_isl` 控制此行为。

**难度**: 低　**影响**: ⭐⭐⭐

---

## 🟡 P1-6: `skipped_waiting` 双队列优先级选择逻辑不完整

**源码**: `vllm-0.19.0/vllm/v1/core/sched/scheduler.py` — `_select_waiting_queue_for_scheduling()` (L~1570-1578)

**问题**: vLLM 在 PRIORITY 模式下，当 `waiting` 和 `skipped_waiting` 两个队列都非空时，会比较两个队首请求的优先级，选择优先级更高的先调度。你的模拟器在 Phase 2 中总是先遍历 `waiting` 再遍历 `skipped_waiting`，相当于始终 FCFS。

vLLM 的相关状态（`skipped_waiting` 中的请求）：
- `WAITING_FOR_REMOTE_KVS` — P/D 场景下等待 KV 传输
- `WAITING_FOR_STREAMING_REQ` — streaming 输入等待下一段
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` — 等待 structured output grammar 编译

**影响**:
- 仅在 `scheduling_policy = "priority"` 时有影响
- 对于 FCFS 模式（默认），行为一致

**修正建议**:

实现类似 vLLM 的 `_select_waiting_queue_for_scheduling()` 逻辑：

```python
def _select_waiting_queue(self):
    """Select which queue to pop from based on scheduling policy."""
    if self.policy == SchedulingPolicy.FCFS:
        if self.skipped_waiting:
            return self.skipped_waiting  # 优先处理跳过的
        return self.waiting if self.waiting else None
    
    # PRIORITY: 比较两个队首
    if self.waiting and self.skipped_waiting:
        w = self.waiting.peek()
        s = self.skipped_waiting.peek()
        if w and s:
            return self.waiting if w.priority >= s.priority else self.skipped_waiting
    return self.waiting if self.waiting else (self.skipped_waiting if self.skipped_waiting else None)
```

**难度**: 低　**影响**: ⭐⭐（仅 PRIORITY + 高负载时）

---

## 🟢 P2-7: Schedule CPU 开销未建模

**源码**: `vllm-0.19.0/vllm/v1/core/sched/scheduler.py` — `schedule()` (~200 行逻辑)

**问题**: vLLM 的 `schedule()` 不是瞬时操作。它执行：
- 遍历 running 队列（prefix cache lookup、block 分配）
- 遍历 waiting 队列（hash 计算、cache 查询）
- KV connector 元数据构建
- Encoder cache 管理

这些 CPU 操作在实际运行中耗时 ~100 μs — 2 ms，取决于 running/waiting 队列长度和 prefix cache 命中率。你的模拟器将 schedule 视为零开销。

**影响**:
- 每 step 低估 ~0.1-2 ms
- 对于 decode step（GPU 时间 ~1-5 ms），CPU 开销占 ~5-20%
- 对于 prefill step（GPU 时间 ~10-100 ms），CPU 开销可忽略

**修正建议**:

在 `engine.py` 中每个 step 增加固定 CPU 开销：

```python
# sim/engine.py - 每个 schedule() 之后
schedule_overhead = (
    100e-6  # 基础开销
    + len(running_requests) * 5e-6  # 每个 running 请求的遍历开销
    + len(waiting_requests) * 10e-6  # 每个 waiting 请求的 hash 计算开销
)
self.clock += schedule_overhead
```

**难度**: 低　**影响**: ⭐⭐

---

## 🟢 P2-8: CPU Offloading / Swap 的异步性缺失

**源码**:
- `vllm-0.19.0/vllm/v1/simple_kv_offload/manager.py` — `SimpleCPUOffloadScheduler`
- `vllm-0.19.0/vllm/v1/kv_offload/abstract.py` — `OffloadingManager`

**问题**: vLLM v0.19.0 的 CPU offloading 是**异步**的——`prepare_load()` 返回 metadata，load 操作在 worker 进程中异步执行，完成后通过 event 通知 scheduler。你的模拟器的 `swap_out()` / `swap_in()` 是同步的——swap time 直接加到 `self.clock` 上。

vLLM 的实际 swap 流水线：
```
CPU: schedule → issue load → continue scheduling other requests
GPU: execute step N (no swap blocks yet)
CPU: load complete → request ready for next schedule
GPU: execute step N+1 (with restored blocks)
```

你的模拟器：
```
swap_in(request) → clock += swap_time → schedule → execute
```

**影响**:
- Swap 时间被**全额计入**延迟，而实际可以部分或完全隐藏
- 对于 decode 为主的 workload（swap 不频繁），影响极小
- 对于 prefill-heavy workload（频繁 swap），可能高估延迟 10-30%

**修正建议**:

将 swap 建模为可以与 GPU 计算重叠的后台操作：

```python
# 记录 swap 操作但不立即加到 clock
swap_finish_time = self.clock + swap_time
# 后续步骤中，swap 完成的请求在 swap_finish_time 之后才能被调度
request.kv_ready_time = swap_finish_time
```

**难度**: 中　**影响**: ⭐⭐

---

## 🟢 P2-9: 多 KV Cache Group 不支持

**源码**:
- `vllm-0.19.0/vllm/v1/core/kv_cache_coordinator.py` — `KVCacheCoordinator`
- `vllm-0.19.0/vllm/v1/core/single_type_kv_cache_manager.py` — `SingleTypeKVCacheManager`

**问题**: vLLM 支持多个 KV cache group。例如：
- Group 0: FullAttention（标准 KV cache）
- Group 1: SlidingWindowAttention（滑动窗口 KV cache，block_size 可能不同）
- Group 2: Mamba（SSM 状态缓存，与 attention block 完全不同）

每个 group 有独立的 block pool 和 block table。你的模拟器假设单一 block pool，不支持混合 attention 类型的模型（如 Qwen3.5 混合 FullAttention + DeltaNet）。

**影响**:
- 对于标准全注意力模型（Llama、Qwen）：**无影响**
- 对于混合架构模型：block 分配数量计算错误

**修正建议**:

暂不修改——当前支持的模型均为全注意力架构。当需要支持混合架构时，将 `BlockPool` 扩展为支持多 group：

```python
# memory.py
class MultiGroupBlockPool:
    def __init__(self, group_configs: list[GroupConfig]):
        self.pools = [BlockPool(cfg.num_blocks) for cfg in group_configs]
```

**难度**: 高　**影响**: ⭐（仅混合架构模型）

---

## 🟢 P3-10: KV Cache Block Caching 时机差异

**源码**:
- `vllm-0.19.0/vllm/v1/core/sched/scheduler.py` — `update_from_output()` 中调用 `kv_cache_manager.cache_blocks()`
- `vllm-0.19.0/vllm/v1/core/single_type_kv_cache_manager.py` — `cache_blocks()` (L~230-260)

**问题**: vLLM 的 block 缓存（prefix cache 写入）是在 **GPU 执行完成后**才进行的（`update_from_output` → `cache_blocks`），因为只有此时 block 内容才真正被写入。你的模拟器在 `allocate_slots` 中**分配时就立即缓存**了 block（`memory.py:allocate_slots()` 末尾的 `self._cache_block(block)`）。

这不会影响**功能正确性**（因为 block 内容在你的模拟器中是虚构的），但会影响 **cache 驱逐的时序**：
- vLLM：一个 block 在 step N 分配 → step N 执行 → step N+1 才进入 cache（可被其他请求命中）
- 你的模拟器：一个 block 在 step N 分配时就进入 cache（可被同一 step 的后续请求命中）

这可能导致你的模拟器略微**高估** cache hit rate。

**影响**:
- Cache hit rate 可能略微偏高（同一步内新分配的 block 被后续请求命中）
- 对整体性能影响很小（~1-2% 的 token 层面差异）

**修正建议**:

将 `_cache_block()` 的调用从 `allocate_slots()` 移到 `update_from_output()` 中（模拟 GPU 执行完成后）：

```python
# scheduler.py: update_from_output()
def update_from_output(self, output, clock):
    # ... 现有逻辑 ...
    
    # 在 GPU 执行完成后才缓存新满的 blocks
    for req, num_new, new_blocks in output.scheduled_requests:
        self.pool.cache_new_full_blocks(req, num_new, self.block_size)
```

**难度**: 低　**影响**: ⭐

---

## 优先级矩阵

```
影响 ↑
 5 │  P0-1 CUDA Graph (🔴)
   │  P0-2 Kernel Launch (🔴)
 4 │  P0-3 Async/Batch Queue (🔴)
   │
 3 │  P1-4 抢占回退 (🟡)
   │  P1-5 准入门控 (🟡)
   │
 2 │  P1-6 双队列优先级 (🟡)
   │  P2-7 CPU 开销 (🟢)
   │  P2-8 Swap 异步 (🟢)
   │
 1 │  P2-9 多 KV Group (🟢)
   │  P3-10 Cache 时机 (🟢)
   └──────────────────────────→ 难度
      低          中          高
```

## 修正路线图建议

| 阶段 | 项目 | 状态 | 预计消除误差 |
|------|------|------|-------------|
| **Phase 1** (立刻) | P0-2 Kernel Launch Overhead + P2-7 Schedule CPU 开销 | ✅ P0-2 已完成，⬜ P2-7 | ~15-30% |
| **Phase 2** (本周) | P0-1 CUDA Graph 加速因子 | ✅ 已完成 | ~40-60% |
| **Phase 3** (本周) | P1-4 抢占回退逻辑 + P1-5 准入门控 | ✅ 已完成 | ~5-10% |
| **Phase 4** (后续) | P2-8 Swap 异步 | ⬜ | ~5-15% |
| **Phase 5** (按需) | P1-6, P2-7, P2-9, P3-10 | ⬜ | <5% |

**P0-1 + P0-2 + P0-3 完成后预计总体误差可从当前的 ±50-200% 降至 ±20-40%。**
