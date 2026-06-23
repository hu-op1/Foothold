# Foothold 技术架构

## 系统总览

LLM 推理性能工具链：**GPU 微基准 → Roofline 拟合 → 吞吐量预测 → PD 分离仿真**

```
models/<vendor>/<series>/<model>/config.json    ← HF 模型配置（唯一数据源）
              │
              ▼
        model_spec dict   (hidden_dim, num_heads, num_kv_heads, ...)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
bench/              perf_predict/predict.py
  │                      │
  ▼                      │
bench/results/<gpu>/     │
  *.xlsx                 │
  │                      │
  ▼                      │
fit/  →  fit/results/<gpu>.json   (F_peak, B_peak, p, B_eff, overhead)
              │                    │
              └────────┬───────────┘
                       │
                 hw_params dict
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     perf_predict/predict.py  sim/
       (throughput pred)      (PD disaggregation sim)
                                       │
                                  config/search.yaml
                                  (sim params, SLO, strategy search)
```

## 1. 模型建模 (`models/`)

### 1.1 目录约定

```
models/<vendor>/<family>/<model>/config.json
```

例：

```
models/
├── Qwen/
│   ├── Qwen3/
│   │   ├── Qwen3-4B/config.json
│   │   └── Qwen3-8B/config.json
│   └── Qwen3.5/
│       ├── Qwen3.5-2B/config.json
│       ├── Qwen3.5-4B/config.json
│       └── Qwen3.5-9B/config.json
└── meta-llama/
    ├── Llama-2-7b-hf/config.json
    └── Llama-2-13b-hf/config.json
```

### 1.2 config.json → model_spec 映射

`models/__init__.py::model_spec_from_config()` 负责字段映射：

| HF config.json 字段 | model_spec 字段 | 说明 |
|---|---|---|
| `hidden_size` | `hidden_dim` | |
| `intermediate_size` | `intermediate_dim` | |
| `num_attention_heads` | `num_heads` | |
| `num_key_value_heads` | `num_kv_heads` | 仅当 `< num_heads` 时输出（GQA） |
| `head_dim` | `head_dim` | 显式提供，或 `hidden_size / num_heads` |
| `num_hidden_layers` | `num_layers` | |
| `vocab_size` | `vocab_size` | |
| `max_position_embeddings` | `max_model_len` | |
| `rms_norm_eps` 存在 | `norm_type = "rmsnorm"` | |
| `layer_types` | `attn_layers` | Qwen3.5 专用：统计 `"full_attention"` 个数 |
| — | `total_params_b` | 由架构公式**精确计算**，非手动填写 |

### 1.3 嵌套 text_config 处理

Qwen3.5 的 config.json 为多模态结构，文本参数嵌套在 `text_config` 下：

```json
{
  "model_type": "qwen3_5",
  "text_config": {
    "hidden_size": 4096,
    "layer_types": ["linear_attention", ..., "full_attention", ...],
    ...
  },
  "vision_config": { ... }
}
```

`_unwrap_config()` 自动检测并提取 `text_config`。

### 1.4 参数总量计算

`_compute_params()` 从架构维度精确计算参数总量，公式：

```
per_layer = 2·h·nh·hd + 2·h·nkv·hd + 3·h·inter + 2·h
            \________/   \_________/   \_______/   \_/
             Q, O 投影    K, V 投影    Gate,Up,Down  2×RMSNorm

total = vocab·h  +  nl·per_layer  +  lm_head  +  final_norm
        \____/                      \______/
        token embedding            0 if tie_word_embeddings else vocab·h
```

- GQA 下 K/V 投影使用 `nkv·hd`，比 MHA 的 `nh·hd` 小
- `tie_word_embeddings: true` 时 lm_head 与 embedding 共享权重，不重复计算

## 2. GQA（Grouped Query Attention）支持

### 2.1 问题

GQA 模型（如 Qwen3-8B: `nh=32, nh_kv=8`）有三处需要特殊处理：

| 组件 | MHA | GQA | 影响 |
|---|---|---|---|
| K/V 投影 matmul | `h × h` | `h × (nh_kv·hd)` | K/V 计算量小 4× |
| FlashAttention HBM 读写 | K,V 用 `nh` 头 | K,V 用 `nh_kv` 头 | KV cache 带宽省 4× |
| RoPE | Q 和 K 各 `nh` 头 | K 只有 `nh_kv` 头 | K 的 RoPE 操作量小 4× |

### 2.2 修复细节

**`projections()`** ([perf_predict/predict.py](perf_predict/predict.py#L64-L81))：

```python
# 修复前：仅检查 nh*hd == h，Qwen3-8B (32×128=4096=h) 走 MHA 分支
if nh is None or nh * hd == h:
    t = 4 * matmul_time(M, h, h, ...)  # K/V 也按 h×h 算，实际应为 h×1024

# 修复后：同时检查 nh_kv
if nh is None or (nh * hd == h and (nh_kv or nh) == nh):
    t = 4 * matmul_time(M, h, h, ...)  # 真正 MHA
else:
    # GQA 分支：K/V 用 nh_kv·hd 维度
```

**`attention_fused()`** ([perf_predict/predict.py](perf_predict/predict.py#L84-L100))：

```python
# 修复前：所有 4 个张量 (Q,K,V,O) 均用 nh 头数量
bytes_moved = 4 * b * nh * max(s_q, s_kv) * hd * DTYPE_BYTES

# 修复后：Q/O 用 nh，K/V 用 nh_kv
bytes_moved = b * hd * DTYPE_BYTES * (nh·s_q + nh_kv·s_kv + nh_kv·s_kv + nh·s_q)
```

**`elementwise_ops()`** ([perf_predict/predict.py](perf_predict/predict.py#L103-L116))：

```python
# 修复前：RoPE 只算 Q
t += elem_time("rope", b * nh * s * hd, ...)

# 修复后：Q（nh 头）和 K（nh_kv 头）分别计算
t += elem_time("rope", b * nh * s * hd, b_effs, overheads)
t += elem_time("rope", b * nh_kv * s * hd, b_effs, overheads)
```

**`executor.py`** 同步修复（[sim/executor.py](sim/executor.py#L102-L110)）：传入 `nh_kv`。

### 2.3 GQA 效果

Qwen3-8B 在 2048→512 场景下的修复对比：

| 阶段 | Prefill | Decode | E2E |
|---|---|---|---|
| 无 GQA 支持 | 505ms | 15226ms | 15731ms |
| + attention_fused 修复 | 507ms | 14265ms | 14772ms |
| + projections 分支修复 | 459ms | 13073ms | 13532ms |

GQA 节省随 context 长度增加而放大（Qwen3-8B）：

| input_len | GQA decode | 同模型若为 MHA | 节省 |
|---:|:---:|:---:|---:|
| 2048 | 13073ms | 14856ms | 12% |
| 4096 | 13278ms | 15650ms | 15% |
| 8192 | 13687ms | 17239ms | 21% |
| 16384 | 14505ms | 20417ms | 29% |
| 32768 | 16141ms | 26772ms | **40%** |

## 3. GPU 显存建模

### 3.1 显存预算公式

```
usable_vram = total_vram × gpu_memory_utilization (默认 0.85)

kv_cache_pool = usable_vram - model_weight - activation
```

### 3.2 权重计算

`sim/config.py::model_weight_gb()`：

```
weight_gb = total_params_b × 2 / 1e9   (fp16 = 2 bytes/param)
```

`total_params_b` 由 `models/__init__.py::_compute_params()` 从架构维度精确计算（见 §1.4）。

### 3.3 激活值计算

`sim/config.py::activation_memory_gb()`，不再使用硬编码的 2GB：

```
activation = batch_tokens × (2h + 3·inter) × 2 bytes  +  0.5 GB (CUDA)
             \_______________ ______________/             \_ ___/
               FFN 块峰值 5 个共存张量                  allocator 固定开销
```

**推导**：一层 forward 的 FFN 块峰值时刻，以下 5 个张量同时在 HBM 中存活：

```
residual [S, h]  →  RMSNorm [S, h]  →  gate [S, inter]  ─┐
                                                           ├→ SiLU(gate)×up [S, inter]  →  down proj
                                         up   [S, inter]  ─┘
```

每 token 共存元素 = `2h + 3·inter`。中间张量逐层释放，只看一层的峰值。FlashAttention 的 S×S 矩阵在 SRAM 中，不计入 HBM。

各模型激活值对比（8192 tokens, tp=1）：

| 模型 | 旧值（固定） | 新值（计算） |
|---|---|---|
| Qwen3.5-2B | 2.0 GB | **0.9 GB** |
| Qwen3.5-9B | 2.0 GB | **1.3 GB** |
| Llama-2-13b-hf | 2.0 GB | **1.4 GB** |

配置 `activation_memory_gb: null` 自动计算，填数字覆盖。

### 3.4 TP 校验

`valid_tp_sizes()` 检查 TP 的可行性：

1. `num_heads % tp == 0` — 注意力头整除
2. `num_kv_heads % tp == 0` — GQA 的 KV 头整除
3. `weight/tp + activation(tp) < usable_vram` — 权重放得下
4. KV cache（完整 context）放得下

激活值随 TP 动态计算：`activation_memory_gb(model_spec, max_batch_tokens, tp)`。

## 4. PD 分离仿真 (`sim/`)

### 4.1 事件驱动引擎

`engine.py::SimulationEngine` 支持两种模式：

- **Colocated**：P 和 D 在同一 GPU 上，`dp` 参数控制数据并行度
- **Disaggregated**：P 侧 GPU 处理 prefill，D 侧 GPU 处理 decode，KV cache 经网络传输

#### 4.1.1 时钟推进与空闲检测

事件循环只在 **GPU 真正空闲**（无 running / waiting 请求）时才将时钟跳至下一个事件的到达时间。如果 GPU 有工作要做，时钟按 step_time 正常推进，不跳。这避免了之前无条件跳钟导致的"GPU 空转等请求"——例如 prefill 完成后本该立刻解码，时钟却跳到半秒后的下一个请求到达时间。

```python
# 调度/执行一 step 后
self.clock += step_time  # 推进时钟

# 下一轮循环开始
if event_queue and not sched.has_requests():  # ← 只有真空闲才跳
    self.clock = max(self.clock, event_queue[0].time)
```

Disaggregated 模式的 D 侧同样有死锁保护：连续 1000 个 idle tick（无 token 被调度）触发 `RuntimeError`，指明是 block pool 耗尽还是调度器卡死。

#### 4.1.2 数据并行（DP）

DP 按照 vLLM 架构建模：每个 DP rank 是**独立的 EngineCore**，拥有自己的 KV cache pool、scheduler 和模型权重。所有 rank 并行 step，wall-clock 取各 rank step time 的最大值。

请求通过 **least-loaded 路由**分发：每次到达事件触发时，选择 `running + waiting` 最小的 rank：

```python
def _pick_rank() -> int:
    loads = [len(s.running) + len(s.waiting) for s in scheds]
    return loads.index(min(loads))
```

与 vLLM 源码的 `DPCoordinator` 一致：前端根据各 rank 的队列长度做负载均衡路由。

DP 的吞吐量是自然涌现的——各 rank 并行处理不同请求子集，所有 rank 的 output tokens 之和除以 wall-clock 即为总吞吐。无需手动乘以 DP 因子，TTFT / TPOT 也是真实的 per-rank 值。

DP 模式下各 rank 的 pool 大小为 `num_blocks × tp_size`（TP 切分权重后每 GPU 的可用 KV cache）。

### 4.2 调度器

调度器复刻 vLLM v1 `core/sched/scheduler.py` 的两阶段结构：

```
Phase 1: 遍历 running 队列
  - decode 请求：num_new = 1（固定生成 1 token）
  - prefill 请求：num_new ≤ long_prefill_token_threshold（chunked prefill）
  - OOM 时：colocated 抢占 running 末尾 + recompute
             D 侧（swap 模式）：swap victim 到 CPU，保留 num_computed_tokens

Phase 2: 遍历 waiting 队列（仅在 Phase 1 无抢占时执行）
  - 最多准入 max_num_seqs 个请求
  - 每个请求受 prefill_threshold 和 token_budget 双重限制
  - 支持 prefix cache 命中（共享已缓存的 KV cache block）
```

#### 两级流控

| 参数 | 作用域 | 含义 |
|------|--------|------|
| `max_num_batched_tokens`（batch） | 全 step | 整 step 所有请求的 token 总和上限 |
| `long_prefill_token_threshold`（thr） | 单请求 | 单个请求单 step 最多拿多少 token |

实际分配到 `= min(prompt剩余, threshold, 剩余budget)`。当 `enable_chunked_prefill = false` 且请求需要的 token 超过剩余 budget 时，直接 break 等待下个 step。

与 vLLM 的区别：
- vLLM 抢占是 `while True` 循环（一直抢到够），模拟器最多抢 1 次
- vLLM Phase 2 交替遍历 waiting / skipped_waiting，模拟器先清空 waiting 再清空 skipped_waiting
- vLLM 可从 Phase 1 已调度的请求中回滚，模拟器不会出现此场景

### 4.3 KV Cache 管理

#### 4.3.1 Block Pool（PagedAttention）

`memory.py::BlockPool` 管理 PagedAttention 风格的块池。每个 block = 16 tokens（`block_size`），每 block 字节：

```
bytes_per_block = 2(K+V) × nl × nh_kv × hd × block_size × 2(bytes)
                = 2 × 32 × 32 × 128 × 16 × 2 = 8,388,608（≈ 8 MiB）
```

`num_blocks = kv_cache_memory_gb × 1024³ / bytes_per_block`。`kv_cache_memory_gb` 默认为 `usable_vram - model_weights - activation`（每 GPU 独立计算），可在 `search.yaml` 中覆盖。

#### 4.3.2 Prefix Caching（APC）

实现 vLLM 的自动前缀缓存。block 填满 16 token 时计算 SHA-256 hash 写入 `cached_block_hash_to_block`。后续请求的 prompt block_hashes 若命中，`touch()` 增加引用计数复用物理 block，节省 prefill token。

缓存淘汰：block 的 ref_cnt 归零后回到 `free_block_ids` 但 **hash 保留**。只在 `get_new_block()` 被调用需要回收物理 block 时才 `_evict_block()` 删除 hash。缓存容量 = 当前空闲 block 数，负载轻重自适应。

缓存命中率通过 `BlockPool.cache_hit_rate` 暴露，写入 xlsx 输出。开关为 `simulation.enable_prefix_caching`（默认 true）。

#### 4.3.3 GPU↔CPU Swap（D 侧 OOM 处理）

D 侧在 disaggregated 模式下支持 vLLM 的 **swap 预抢占**。当 block pool 耗尽且请求无法分配新 block 时：

```
swap_out(victim):
  - 释放 victim 的物理 block
  - 记录 block 数量到 _swapped_out[request_id]
  - 时间成本 = num_blocks × bytes_per_block / cpu_swap_bw_gb_s
  - victim 回到 waiting 队列，num_computed_tokens 保留

swap_in(request):
  - 从 _swapped_out 读取 block 数量
  - 分配等量新 block 填充 block_table
  - 时间成本同上
  - request 从断点继续 decode，无需 recompute
```

Swap 带宽配置为 `communication.cpu_swap_bw_gb_s`（默认 32 GB/s，PCIe 3.0 x16）。Swap 时间计入 `SchedulerOutput.swap_time`，被引擎加到 step time 中。

colocated 模式不使用 swap（有 `reset_on_preempt=True` 的 recompute 路径）。D 侧 scheduler 通过 `use_swap` 属性自动判断（`not reset_on_preempt and pool has cpu_swap_bw`）。

### 4.4 Roofline 步长时间预测

`executor.py::predict_step()` 计算单次 GPU forward pass 的时间：

```
step_time = projections + attention + elementwise + lm_head
```

#### 4.4.1 参数分离

**Projections 和 LM head** 用基于 `total_new_tokens` 的统一 F/B/p（`_select_roofline_params` 以 M_SPLIT=256 区分）。因为所有 token 合并为一次批量 matmul，有效带宽取决于总 M。

**Attention** 逐请求分别使用参数：
- Prefill 请求（`is_prefill_chunk=True`）：`F_peak_prefill / B_peak_prefill`（large s_q，compute-bound）
- Decode 请求（`s_q=1`）：`F_peak_decode / B_peak_decode`（memory-bound，KV cache 读取主导）

之前全员共用一组参数会导致：total < 256 时 prefill attention 估计偏慢 1.4×，total ≥ 256 时 decode attention 估计偏快 5×。

**Elementwise ops** 用 `b_effs` / `overheads`（独立于 F/B/p，因 elementwise 无 compute 项只有带宽和固定开销）。

#### 4.4.2 KV cache 长度

`num_computed_tokens` 在 `_update_after_schedule()` 中已递增，`predict_step()` 中直接使用：

```python
kv_len_after = req.num_computed_tokens  # 已含本 step 的 num_new
```

之前使用 `req.num_computed_tokens + num_new` 导致 prefill 的 KV 长度翻倍。

#### 4.4.3 TP All-Reduce

`predict_step_tp()` 在单 GPU 时间基础上 `÷ num_gpus + all_reduce_time`：

```
all_reduce_time = 2 × total_new_tokens × hidden_dim × 2(bytes) / intra_node_bw_gb_s
```

### 4.5 策略搜索

`strategy.py::search()` 在以下空间的笛卡尔积上搜索：

```
tp_sizes × max_batched_tokens × prefill_thresholds × pd_ratios × decode_tp_sizes
```

- **Colocated**：`mode in ("colocated", "auto")`。TP=tp, DP=total_gpus/tp。
- **Disaggregated**：`mode in ("disaggregated", "auto")`。P 侧 TP=tp, D 侧 TP=d_tp。`decode_tp_sizes` 独立控制 D 侧的 TP 并行度；D 侧 data-parallel 度 = d / d_tp。

搜索过程中使用 tqdm 进度条，详细结果写入 xlsx（不在终端逐条打印）。

#### 4.5.1 SLO 评分

```
compliance = ttft_pass_rate × tpot_pass_rate × (p99_pass ? 1 : 0)
score = throughput × compliance
```

- `ttft_pass_rate`：TTFT ≤ `slo.ttft_ms` 的请求比例
- `tpot_pass_rate`：TPOT ≤ `slo.tpot_ms` 的请求比例
- `p99_pass`：P99 总延迟 ≤ `slo.p99_latency_ms` → 1.0，否则 → 0.0（一票否决）

结果按 score 降序排列。p99 超标直接归零排到最后。

#### 4.5.2 输出

xlsx 包含以下字段（`report.py::export_xlsx()`）：

```
strategy_type, batch, thr, throughput_tok_s, ttft_mean/p50/p90/p99_ms,
tpot_mean/p50/p90/p99_ms, latency_p50/p90/p95/p99_ms, num_requests,
total_input_tokens, total_output_tokens, total_time_s, cache_hit_rate,
score, elapsed_s
```

### 4.6 `max_num_seqs` 配置

`max_num_seqs` 是用户指定的调度参数（`search.yaml` 的 `simulation` 节），**不做自动估算**。过大的值会导致 block pool 耗尽（每个请求在 decode 阶段逐步消耗 block）。1024 block 的池加 32 并发意味着平均每请求只能用 32 block ≈ 512 token，对应平均序列长度刚好到极限。长请求出现时可能触发 swap 或死锁。

### 4.7 请求到达速率与参数敏感性

`max_batched_tokens` 只在它是瓶颈时影响结果。瓶颈判断条件：

```
Phase1 decode 消耗 + Phase2 prefill 消耗 ≥ max_batched_tokens
```

如果到达速率低（如 light trace 2 req/s），每步只有 ~1 个新请求需要 prefill，实际消耗远小于 budget，参数从 512 改到 8192 不会改变调度决策，结果完全相同——这是正确行为，非 bug。

## 5. 关键设计决策

1. **Roofline 优于线性拟合**：3 参数 (F_peak, B_peak, p) 同时捕获 compute-bound 和 memory-bound 行为
2. **Prefill/decode 分离拟合**：M < 256 vs M ≥ 256 使用不同的 roofline 参数
3. **Attention 逐请求参数分离**：prefill 用 prefill 参数，decode 用 decode 参数。Projections 用统一参数（批量 matmul 有效性由总 M 决定）
4. **FlashAttention 建模**：S×S 矩阵不计入 HBM 带宽，只计 Q、K、V 读和 O 写
5. **GQA 贯穿全线**：projections、attention_fused、elementwise_ops、KV cache 块大小、KV 传输时间全部考虑 `nh_kv`
6. **config.json 为唯一数据源**：模型架构参数从 HF config 直接读取，参数量由公式计算，不再手工维护 `model_specs.yaml`
7. **激活值按负载计算**：从模型架构和 `max_batch_tokens` 推导，非固定常量
8. **DP 独立调度器**：每 rank 独立 scheduler + block pool + 模型权重，least-loaded 路由，并行 step。与 vLLM `DPCoordinator` 架构一致
9. **D 侧 swap 替代死等**：block pool 耗尽时 swap victim 到 CPU 保留进度，空间释放后 swap 回来继续 decode，避免死锁
10. **前缀缓存自调节容量**：缓存占用的 block 数 = 空闲 block 数，负载重时自动缩小，负载轻时自动扩大
