# Foothold

LLM 推理性能工具链：**GPU 微基准 → Roofline 拟合 → PD 分离仿真**。

## 环境

- Python ≥ 3.14
- NVIDIA GPU（8GB+ VRAM），CUDA ≥ 12.6

```bash
uv sync
```

## 整体流程

```
models/<vendor>/<family>/<model>/config.json    ──→  model_spec dict (auto-discovered)
                                                          │
                                                          ▼
config/bench.yaml  →  bench/   →  bench/results/<gpu>/*.xlsx
                              │                           │
                              ▼                           │
config/bench.yaml  →  fit/  →  fit/results/<gpu>.json   │
                                    (F_peak, B_peak, p)    │
                                         │                 │
                                         └──────┬──────────┘
                                                ▼
                                    hw_params dict
                                          │
                                          ▼
                                     sim/
                                     (PD disaggregation sim)
                                     config/search.yaml
                                     config/sim.yaml
```

## 0. 模型建模 — `models/`

从 HuggingFace `config.json` 自动发现模型架构参数，无需手工维护模型列表。

```bash
uv run python -c "from models import load_model_specs; print(load_model_specs()['models'])"
```

### 目录约定

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

### 映射规则

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

嵌套 `text_config`（Qwen3.5 多模态）自动展开；无 `config.json` 的模型可回退到 `config/model_specs.yaml`。

## 1. 硬件标定 — `bench/`

```bash
uv run python main.py bench
uv run python main.py --bench        # 旧版 flag
```

### 配置 — [config/bench.yaml](config/bench.yaml)

```yaml
matmul:
  M: [1, 2, 4, ..., 32768]         # 1=memory-bound, 32768=prefill-bound
  K: [512, 1024, 2048, 4096, 8192]
  N: [512, 1024, 2048, 4096, 8192]
elementwise:
  N: [1024, 4096, ..., 134217728]
  operators: [residual_add, rmsnorm, softmax]
dtype: "float16"
warmup_iters: 200
bench_iters: 2000
max_memory_gb: 24
```

### 原理

不按 LLM 算子名逐个测，直接测底层 kernel：

| 类别 | bench 内容 | 用途 |
|------|-----------|------|
| matmul | `torch.mm` 的 M×K×N 网格 | 拟合 roofline 参数 |
| elementwise | residual_add / rmsnorm / softmax | 验证带宽一致性 |

输出：`bench/results/<gpu>/matmul.xlsx`, `elementwise.xlsx`

## 2. Roofline 拟合 — `fit/`

从 matmul benchmark 结果拟合平滑 Roofline 模型的三个硬件参数：

```python
time = ((flops/F_peak)^p + (bytes/B_peak)^p)^(1/p)
```

```bash
uv run python main.py fit
uv run python main.py --fit          # 旧版 flag
```

输出 `fit/results/<gpu>.json`：

```json
{
  "F_peak_prefill": 2.76e13,
  "B_peak_prefill": 6.18e11,
  "p_prefill": 1.01,
  "F_peak_decode": 2.76e13,
  "B_peak_decode": 4.91e11,
  "p_decode": 1.12,
  "elem_b_effs": { "residual_add": 4.2e11, "rmsnorm": 4.0e11, ... },
  "elem_overheads": { "residual_add": 0.0012, ... }
}
```

拟合策略（`fit/matmul.py`）：
- **M ≥ 256**（prefill 大 batch）：拟合 F_peak、B_peak、p
- **M < 256**（decode 小 batch）：固定 F_peak，拟合 B_peak 和 p

elementwise 模型：`time = bytes / B_eff + overhead`，B_eff 从大 N 点拟合，overhead 从小 N 点拟合。未实测的 op（swiglu、rope、layernorm）通过 proxy 映射继承。

可选线性拟合后端（无参数假设，直接用查表插值）：
```python
from fit import fit_all
params = fit_all(results, backend="linear")
```

换 GPU 只需重跑 bench → fit，换模型架构无需改动硬件参数。

## 3. PD 分离仿真 — `sim/`

事件驱动的 vLLM 推理服务仿真器，支持 colocated（共址）和 disaggregated（P/D 分离 GPU 池）两种部署模式。

```bash
uv run python main.py search                     # 从 config/search.yaml
uv run python main.py search --config <path>     # 指定配置
```

### 配置 — [config/search.yaml](config/search.yaml)

```yaml
gpu: "3090"
model: "Qwen3-8B"
simulation:
  block_size: 16
  max_num_seqs: 32
  gpu_memory_utilization: 0.85
  enable_prefix_caching: true
  enable_chunked_prefill: true
strategy:
  mode: auto         # colocated / disaggregated / auto
  total_gpus: 4
  search:
    pd_ratios: [[3,1], [1,3], [2,2]]
    max_batched_tokens: [256, 512, 1024, 2048, 4096, 8192]
    prefill_thresholds: [256, 512, 1024, 2048]
    tp_sizes: [1, 2, 4]
```

### 模块结构

| 模块 | 职责 |
|------|------|
| `engine.py` | 事件驱动主循环，colocated + disaggregated 双模式 |
| `scheduler.py` | vLLM v1 两阶段调度器（chunked prefill、抢占、swap） |
| `memory.py` | PagedAttention BlockPool、前缀缓存（SHA-256）、GPU↔CPU swap |
| `executor.py` | Roofline 步长时间预测（逐请求 attention 参数分离） |
| `trace.py` | JSONL 请求 trace 加载 |
| `strategy.py` | 网格策略搜索，score = throughput × SLO_compliance |
| `metrics.py` | TTFT / TPOT / latency 分布统计 |
| `config.py` | 配置加载、显存预算计算、TP 校验 |
| `communication.py` | KV cache 网络传输开销建模 |
| `report.py` | xlsx 结果导出 |

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

```
compliance = ttft_pass_rate × tpot_pass_rate × (p99_pass ? 1 : 0)
score = throughput × compliance
```

- ttft_pass_rate: TTFT ≤ `slo.ttft_ms` 的请求比例
- tpot_pass_rate: TPOT ≤ `slo.tpot_ms` 的请求比例
- p99_pass: P99 总延迟 ≤ `slo.p99_latency_ms` 一票否决

## 测试

```bash
uv run python -m pytest test/                        # 全部测试
uv run python -m pytest test/test_sim.py -v       # PD sim 测试
```

## 目录结构

```
foothold/
├── main.py                     # CLI 入口：bench / fit / search / sim
├── models/                     # HF config.json 自动发现 → model_spec
│   ├── __init__.py             # 映射、解析、参数计算
│   ├── Qwen/Qwen3/             # Qwen3-4B, -8B
│   ├── Qwen/Qwen3.5/           # Qwen3.5-2B, -4B, -9B
│   └── meta-llama/             # Llama-2-7b, -13b
├── config/
│   ├── bench.yaml              # 硬件标定配置（bench/fit）
│   ├── search.yaml             # PD 策略搜索配置
│   ├── sim.yaml                # 单次模拟配置
│   └── model_specs.yaml        # 模型参数（仅 fallback）
├── bench/                      # GPU kernel 基准测试
│   ├── matmul.py, elementwise.py
│   └── utils.py                # CUDA 计时、保存、内存估算
├── fit/                        # Roofline / 线性拟合
│   ├── __init__.py             # fit_all() 入口
│   ├── matmul.py, elementwise.py, linear.py
│   └── utils.py                # roofline_time, roofline_fit, load/save
├── sim/                     # PD 分离仿真
│   ├── engine.py, scheduler.py, memory.py, executor.py
│   ├── config.py, trace.py, strategy.py
│   ├── metrics.py, report.py, communication.py, request.py
├── test/                       # 测试
│   ├── test_sim.py             # sim 集成测试
│   └── ...
├── bench/results/<gpu>/        # [gitignored] benchmark 输出
├── fit/results/                # [gitignored] 拟合结果
├── docs/                       # 技术文档
│   ├── architecture.md         # 完整设计文档（中文）
│   └── superpowers/            # 开发计划/specs
└── pyproject.toml
```

## 详细参考

见 [docs/architecture.md](docs/architecture.md)（中文）— 包含 GQA 支持细节、显存建模推导、DP 架构、前缀缓存策略等完整技术说明。
