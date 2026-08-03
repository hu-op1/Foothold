# Foothold

LLM 推理性能工具链：**GPU 微基准 → Roofline 拟合 → PD 分离仿真**。

## 环境

- Python ≥ 3.12
- NVIDIA GPU（8GB+ VRAM），CUDA ≥ 12.8
- 包管理器：`uv`

```bash
uv sync --all-extras --no-build-isolation   # install ALL deps incl. CUDA kernels (flash-attn, fla, causal-conv1d, vllm) + dev; bare `uv sync` uninstalls extras
```

## 整体流程

```
                    config/bench.yaml
                          │
                          ▼
                      bench/
                          │
          ┌───────┬───────┼───────┬──────────┐
          ▼       ▼       ▼       ▼          ▼
       matmul  elemwise  flashattn  memcpy  cudagraph/launch
          │       │       │          │          │
          └───────┴───────┴──────────┴──────────┘
                          │
                          ▼
                      fit/  →  fit/results/<gpu>.json
                                    (F_peak, B_peak, p, memcpy LUT)
                                          │
                                          ▼
                                     sim/
                                     (PD disaggregation sim)
                                     config/search.yaml
                                     config/sim.yaml
                                     config/validate.yaml
```

## 0. 模型规格加载

模型架构参数从 **HuggingFace Hub** 动态加载，无需本地模型文件。

```bash
uv run python -c "from sim.config import load_model_spec; print(load_model_spec('meta-llama/Llama-2-7b-hf'))"
```

`sim/config.py:load_model_spec()` 调用 `transformers.AutoConfig.from_pretrained()` 获取 HF 配置，自动映射为内部 `model_spec` 字典。参数总量按架构公式精确计算，无需手工维护。

### 字段映射

| HF config.json 字段 | model_spec 字段 | 说明 |
|---|---|---|
| `hidden_size` | `hidden_dim` | |
| `intermediate_size` | `intermediate_dim` | |
| `num_attention_heads` | `num_heads` | |
| `num_key_value_heads` | `num_kv_heads` | 仅当 `< num_heads` 时输出（GQA） |
| `head_dim` | `head_dim` | 或 `hidden_size / num_heads` |
| `num_hidden_layers` | `num_layers` | |
| `vocab_size` | `vocab_size` | |
| `max_position_embeddings` | `max_model_len` | |
| `rms_norm_eps` 存在 | `norm_type = "rmsnorm"` | |
| `layer_types` | `attn_layers` | Qwen3.5 混合架构：统计 `"full_attention"` 个数 |
| — | `total_params_b` | **按架构公式精确计算** |

参数总量计算公式（无偏置、tie_word_embeddings 时 lm_head 与 embedding 共享）：

```python
per_layer = 2·h·nh·hd + 2·h·nkv·hd + 3·h·inter + 2·h
total = vocab·h + nl·per_layer + (vocab·h if not tied) + h
```

关键处理：
- **GQA**：`num_key_value_heads < num_attention_heads` → `num_kv_heads`
- **Qwen3.5 嵌套**：展开多模态 `text_config`
- **混合架构**：`layer_types` → `attn_layers` 统计全 attention 层数
- 任何 HF 模型 ID（`Qwen/Qwen3-8B`、`meta-llama/Llama-2-7b-hf` 等）可直接在 config YAML 中使用

## 1. 硬件标定 — `bench/`

```bash
uv run python main.py bench
```

### 配置 — [config/bench.yaml](config/bench.yaml)

```yaml
gpu: "3090"
dtype: ["float16", "bfloat16"]
matmul:
  M: [1, 2, 4, ..., 32768]         # 1=memory-bound, 32768=prefill-bound
  K: [4096, 8192]
  N: [4096, 8192]
elementwise:
  N: [1024, 4096, ..., 268435456]
  operators: [residual_add, rmsnorm, softmax, swiglu, rope]
flashattn:
  s_q: [1, 4, 16, ..., 32768]
  s_kv: [128, 512, ..., 262144]
  batch: [1, 2, 4, 6, 8]
  num_heads: 32
  num_kv_heads: 32
  head_dim: 128
cudagraph:
  matmul: { M: [...], K: [...], N: [...] }
  elementwise: { N: [...], operators: [...] }
  flashattn: { s_q: [...], s_kv: [...], batch: [...] }
launch_overhead:
  n_values: [1, 2, 4, 8, 16, 32, 64, 128]
  trials: 50
  warmup: 5
memcpy:
  bytes: [256, 512, 1024, ..., 4294967296]  # D2H/H2D 传输
```

### 原理

不按 LLM 算子名逐个测，直接测底层 kernel：

| 类别 | bench 内容 | 用途 |
|------|-----------|------|
| matmul | `torch.mm` 的 M×K×N 网格 | 拟合 roofline 参数 |
| elementwise | residual_add / rmsnorm / softmax / swiglu / rope | 拟合 B_eff + overhead |
| flashattn | `F.scaled_dot_product_attention` 的 (s_q, s_kv) 网格 | 拟合 FA 专用 roofline 参数 |
| memcpy | GPU↔CPU D2H/H2D 传输带宽 | 构建通信 LUT 查表 |
| cudagraph | 各算子 under CUDA Graph replay | 消除 kernel launch overhead 的独立参数 |
| launch_overhead | CPU wall-clock vs GPU event 斜率分析 | kernel dispatch 延迟 |

输出：`bench/results/<gpu>/matmul.csv`, `elementwise.csv`, `flashattn.csv`, `memcpy.csv`, `cudagraph_*.csv`, `launch_overhead.csv`

## 2. Roofline 拟合 — `fit/`

从 benchmark 结果拟合平滑 Roofline 模型的硬件参数及 memcpy LUT：

```python
time = ((flops/F_peak)^p + (bytes/B_peak)^p)^(1/p)
```

```bash
uv run python main.py fit
uv run python main.py fit --dir <path>   # 覆盖 bench 结果目录
```

输出 `fit/results/<gpu>.json`：

```json
{
  "type": "roofline",
  "F_peak_prefill": 80e12,
  "B_peak_prefill": 2.6e12,
  "p_prefill": 1.02,
  "F_peak_decode": null,
  "B_peak_decode": 700e9,
  "p_decode": 1.01,
  "elementwise": {
    "residual_add": { "B_eff": 839e9, "overhead_us": 37 },
    "softmax": { "B_eff": 550e9, "overhead_us": 45 },
    "swiglu": { "B_eff": 500e9, "overhead_us": 57 }
  },
  "kernel_launch_overhead_us": 5.2,
  "memcpy_d2h_bytes": [256, 512, ...],
  "memcpy_d2h_time_s": [1.2e-7, 2.1e-7, ...],
  "memcpy_h2d_bytes": [256, 512, ...],
  "memcpy_h2d_time_s": [1.1e-7, 2.0e-7, ...]
}
```

拟合策略（`fit/matmul.py`）：
- **M ≥ 256**（prefill 大 batch）：拟合 F_peak、B_peak、p
- **M < 256**（decode 小 batch）：固定 F_peak，拟合 B_peak 和 p

elementwise 模型：`time = bytes / B_eff + overhead`，B_eff 从大 N 点拟合，overhead 从小 N 点拟合。未实测的 op（`rope`、`layernorm`、`causal_mask`、`fused_residual_norm`）通过 proxy 映射到 `residual_add` 继承参数。

CUDA Graph 参数：独立命名空间（键含 `_cudagraph` 后缀），拟合自 `bench/cudagraph.py` 数据。

Memcpy LUT：`fit/memcpy.py` 从 benchmark 数据构建 byte-size → transfer-time 查表，替代简单的 BW+latency 线性模型，用于仿真中的通信建模。

可选线性拟合后端（无参数假设，直接用查表插值）：
```python
from fit import fit_all
params = fit_all(results, backend="linear")
```

换 GPU 只需重跑 bench → fit，换模型架构无需改动硬件参数。

## 3. PD 分离仿真 — `sim/`

事件驱动的 vLLM 推理服务仿真器，支持 colocated（共址）和 disaggregated（P/D 分离 GPU 池）两种部署模式。

```bash
uv run python main.py search                     # 策略网格搜索（从 config/search.yaml）
uv run python main.py search --config <path>     # 指定配置
uv run python main.py sim                        # 单次模拟（从 config/sim.yaml）
uv run python main.py validate -o <out>          # 可视化 sim 输出（CDF、throughput、对比）
```

### 配置 — [config/search.yaml](config/search.yaml)

通信模型基于实测 memcpy LUT（替代旧的 BW+latency 模型），运行 `--bench` + `--fit` 即可生成。

```yaml
gpu: "3090"
model: "Qwen/Qwen3-4B"
dtype: "bfloat16"
simulation:
  block_size: 16
  max_num_seqs: 32
  kv_cache_memory_gb: null           # null = 自动计算
  activation_memory_gb: null         # null = 自动从模型架构计算
  gpu_memory_utilization: 0.85
  enable_prefix_caching: true
  enable_chunked_prefill: true
  use_cudagraph: false
  async_scheduling: false
  scheduler_reserve_full_isl: true   # 门控：完整 prompt 必须能放入 KV cache
strategy:
  mode: both                         # both / colocated / disaggregated
  gpus_per_node: 8
  search:
    max_batched_tokens: [8192]
    prefill_thresholds: [8192]
    tp: true
    pp: true
    dp: true
    gpu_sweep: [1, 2, 4, 8, 16, 32, 64, 128]
  max_workers: 36                    # 并行策略评估进程数
trace:
  path: "traces/agent_trace_test.jsonl"
  max_requests: 5
  format: "agentic"                  # "sharegpt" 或 "agentic"
slo:
  p90_ttft_ms: 500
  p90_tpot_ms: 50
```

### 模块结构

| 模块 | 职责 |
|------|------|
| `engine.py` | 事件驱动主循环，colocated + disaggregated 双模式 |
| `scheduler.py` | vLLM v1 两阶段调度器（chunked prefill、抢占、swap） |
| `memory.py` | PagedAttention BlockPool、前缀缓存（SHA-256）、GPU↔CPU swap |
| `executor.py` | Roofline 步长时间预测（TP/PP 通信叠加） |
| `roofline.py` | 核心 roofline 数学：`roofline_time()`、`matmul_time()`、`attention_fused()` |
| `trace.py` | JSONL trace 加载（ShareGPT / agentic 会话链） |
| `strategy.py` | 网格策略搜索，CSV checkpoint/resume |
| `metrics.py` | TTFT / TPOT / latency 分布统计（p50/p90/p99） |
| `config.py` | 配置加载、显存预算计算、TP/PP 校验 |
| `communication.py` | KV cache 网络传输开销建模（overlap 支持） |
| `request.py` | Request dataclass（WAITING→RUNNING→FINISHED 生命周期） |
| `pipeline.py` | CPU schedule / GPU execute 时间重叠流水线 |
| `recorder.py` | 时序记录（meta.json + requests.jsonl + timeseries.csv） |
| `report.py` | 终端表格、Matplotlib 图表、CSV 导出 |
| `validate.py` | 对比 LLMServingSim 基线生成 CDF/chart |

### 显存建模

```
usable_vram = total_vram × gpu_memory_utilization (默认 0.85)
activation  = batch_tokens × (2h + 3·inter) × 2 + 0.5 GB (CUDA 开销)
kv_cache    = usable_vram - model_weight - activation
```

激活值从模型架构和 `max_batched_tokens` 自动计算，不再硬编码。

### DP 数据并行

每个 DP rank 有独立的 scheduler + block pool + 模型权重。请求通过 least-loaded 路由分发，所有 rank 并行 step，wall-clock 取最大值。与 vLLM `DPCoordinator` 架构一致。

### D 侧 Swap

Block pool 耗尽时，D 侧将 victim 请求的 KV cache swap 到 CPU（保留 `num_computed_tokens`），空间释放后 swap 回来继续 decode，避免完整 recompute。

### SLO 评分

SLO 是 p90 二元门——两个指标都必须达标，否则该策略得分为 0。达标后按 throughput 排序。

```
slo_pass = (p90_ttft_ms ≤ threshold) AND (p90_tpot_ms ≤ threshold)
score = throughput if slo_pass else 0
```

- `p90_ttft_ms`：p90 Time-To-First-Token（首 token 延迟）
- `p90_tpot_ms`：p90 Time-Per-Output-Token（每输出 token 延迟）
- 总延迟**不参与** SLO（随输出长度线性增长，不适合固定阈值）

## Trace 生成工具 — `tools/`

### Agent Trace

从 HuggingFace agent 会话数据集生成 agentic JSONL trace，适用于模拟 agentic 工作负载（含 tool calling 停顿）。

```bash
uv run python tools/generate_agent_trace.py --model Qwen/Qwen3-8B --sps 0.05
uv run python tools/generate_agent_trace.py --model Qwen/Qwen3-8B --sps 0.2 --max-sessions 500 --output traces/my.jsonl
```

### Conversation Trace

从对话数据集（HF 或本地 JSONL）生成 ShareGPT 格式 JSONL trace。

```bash
uv run python tools/generate_conversation_trace.py --dataset <hf_dataset> --model Qwen/Qwen3-8B --sps 1.0
```

### Trace 格式

- **ShareGPT**：每行一个独立请求，`{"conversations": [...], "arrival_time": ...}`
- **Agentic**：每行一个会话链，`{"session_id": "...", "sub_requests": [{...}, ...]}`，每子请求包含 `input_tok_ids`、`output_tok_ids`、`tool_duration_ns`

## 测试

```bash
uv run python -m pytest test/                        # 全部测试
uv run python -m pytest test/test_sim.py -v       # PD sim 测试
```

## 目录结构

```
foothold/
├── main.py                     # CLI 入口：bench / fit / search / sim / validate
├── AGENTS.md                   # AI 代理指令（快速参考）
├── CLAUDE.md                   # 完整架构与命令文档
├── .github/
│   └── instructions/           # 模块级细粒度指令（auto-attach）
│       ├── bench.instructions.md
│       ├── fit.instructions.md
│       ├── sim.instructions.md
│       ├── config.instructions.md
│       └── test.instructions.md
├── config/
│   ├── bench.yaml              # 硬件标定配置（bench/fit）
│   ├── search.yaml             # PD 策略搜索配置
│   ├── sim.yaml                # 单次模拟配置
│   └── validate.yaml           # 可视化对比配置
├── bench/                      # GPU kernel 基准测试
│   ├── matmul.py               # torch.mm  M×K×N 网格
│   ├── elementwise.py          # residual_add / rmsnorm / softmax / swiglu / rope
│   ├── flashattn.py            # F.scaled_dot_product_attention
│   ├── memcpy.py               # GPU↔CPU D2H/H2D 传输带宽
│   ├── cudagraph.py            # CUDA Graph replay 下各算子
│   ├── launch_overhead.py      # CPU→GPU kernel dispatch 延迟
│   └── utils.py                # CudaTimer, warmup, benchmark, checkpoint/resume
├── fit/                        # Roofline / 线性拟合
│   ├── __init__.py             # fit_all() 入口
│   ├── matmul.py               # prefill/decode 分裂拟合
│   ├── elementwise.py          # per-op B_eff + overhead + proxy 映射
│   ├── flashattn.py            # FA 专用 roofline 参数
│   ├── memcpy.py               # memcpy LUT 构建（替代 BW+latency 模型）
│   ├── cudagraph.py            # CUDA Graph 专用参数（_cudagraph 后缀）
│   ├── launch_overhead.py      # kernel launch 开销提取
│   ├── linear.py               # 无参数线性插值后端
│   └── utils.py                # roofline_time, roofline_fit, load/save
├── sim/                        # PD 分离仿真
│   ├── engine.py               # 事件驱动主循环
│   ├── scheduler.py            # vLLM v1 两阶段调度器
│   ├── memory.py               # PagedAttention BlockPool + 前缀缓存
│   ├── executor.py             # Roofline 步长时间预测
│   ├── roofline.py             # 核心 roofline 数学
│   ├── config.py               # 配置加载 + 显存预算
│   ├── trace.py                # JSONL trace 加载
│   ├── request.py              # Request dataclass
│   ├── strategy.py             # 网格策略搜索
│   ├── metrics.py              # TTFT/TPOT/latency 分布
│   ├── communication.py        # KV 传输建模
│   ├── pipeline.py             # Schedule/Execute 重叠流水线
│   ├── recorder.py             # 时序记录（LLMServingSim 兼容）
│   ├── report.py               # 终端表格 + CSV 导出
│   ├── validate.py             # 对比 LLMServingSim 基线
│   └── run_single.py           # 单次仿真运行辅助
├── test/                       # 测试
│   ├── test_sim.py             # sim 集成测试（pytest）
│   ├── analyze.py              # 时间分解分析
│   ├── compare.py              # 结果比较
│   └── gemm.py                 # GEMM 测试
├── tools/                      # 辅助脚本
│   ├── generate_agent_trace.py # 从 HF agent 数据集生成 agentic JSONL trace
│   └── generate_conversation_trace.py  # 从对话数据集生成 ShareGPT JSONL trace
├── traces/                     # 请求 trace 文件（JSONL）
├── bench/results/<gpu>/        # [gitignored] benchmark 输出
├── fit/results/                # [gitignored] 拟合结果
├── sim/output/                 # [gitignored] 仿真输出
├── docs/                       # 技术文档
│   ├── vllm-simulator-gaps.md  # vLLM vs 模拟器差异分析
│   ├── accuracy-improvements.md# 模拟精度提升跟踪
│   └── foothold-competitive-analysis.md  # 同类工具全面对比矩阵
├── pyproject.toml
└── README.md
```

## 详细参考

- [CLAUDE.md](CLAUDE.md) — 完整架构与命令参考
- `.github/instructions/*.instructions.md` — 各模块细粒度编码指南
- [docs/vllm-simulator-gaps.md](docs/vllm-simulator-gaps.md) — vLLM v0.19.0 vs 模拟器差异分析及已修复项
- [docs/accuracy-improvements.md](docs/accuracy-improvements.md) — 模拟精度提升路线图与回归日志
