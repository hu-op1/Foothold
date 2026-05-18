# Foothold

LLM 推理性能工具链：硬件标定 → Roofline 建模 → 模型吞吐预测。

## 环境

- Python ≥ 3.10
- NVIDIA GPU（8GB+ VRAM），CUDA ≥ 12.6

```bash
uv sync
```

## 整体流程

```
config/default.yaml  →  bench/   →  results/<gpu>/*.xlsx    (原始数据)
                                    ↓
                                 fit/   →  fitted_params.json  (F_peak, B_peak, p)
                                             ↓
perf_predict/model_specs.yaml  →  perf_predict/  →  模型吞吐预测 (tokens/s)
```

## 1. 硬件标定 — `bench/`

```bash
uv run python main.py                          # 默认 config/default.yaml
uv run python main.py --output results/5060    # 指定输出目录
```

### 配置 — [config/default.yaml](config/default.yaml)

```yaml
matmul:
  M: [1, 2, 4, ..., 8192]         # 1=memory-bound, 8192=compute-bound
  K: [512, 1024, 2048, 4096, 8192]
  N: [512, 1024, 2048, 4096, 8192]
elementwise:
  N: [1024, 4096, ..., 16777216]
  operators: [residual_add, rmsnorm, softmax]
dtype: "float16"
warmup_iters: 200
bench_iters: 500
max_memory_gb: 6
```

### 原理

不再按 LLM 算子名（q_proj, qk_matmul…）逐个测，而是直接测底层 kernel：

| 类别 | bench 内容 | 用途 |
|------|-----------|------|
| matmul | `torch.mm` 的 M×K×N 网格 | 拟合 roofline 参数 |
| elementwise | residual_add / rmsnorm / softmax | 验证带宽一致性 |

输出：`results/<gpu>/matmul.xlsx`, `elementwise.xlsx`

## 2. Roofline 拟合 — `fit/`

从 matmul benchmark 结果拟合平滑 Roofline 模型的三个硬件参数：

```
time = ( (flops/F_peak)^p + (bytes/B_peak)^p )^(1/p)
```

```bash
uv run python -m fit results/5060                     # 打印拟合结果
uv run python -m fit results/5060 --save fitted.json   # 导出 JSON
```

输出 `fitted_params.json`：

```json
{
  "F_peak": 2.76e13,     // 有效峰值算力 (FLOP/s)
  "B_peak": 6.18e11,     // 有效显存带宽 (bytes/s)
  "p": 1.01,             // 计算/访存重叠度
  "r2": 0.9997
}
```

换 GPU 只需重跑 bench → fit，换模型架构无需改动硬件参数。

## 3. 吞吐预测 — `perf_predict/`

根据模型 shape 计算每个算子的 FLOPs 和 Bytes，套用 Roofline 公式得耗时，汇总得到 prefill/decode 端到端吞吐：

```bash
uv run python perf_predict/predict.py --list                # 列出已有模型
uv run python perf_predict/predict.py --model "Llama-2-7B" \
    --input-len 2048 --output-len 512 --batch 4             # 单模型预测
uv run python perf_predict/predict.py --predict-all \
    --input-len 2048 --output-len 512                       # 全部模型
```

### 预测原理

- 所有 matmul 类算子（投影、QK^T、score×V）走完整 Roofline 模型
- 所有 elementwise 算子（norm、softmax、swiglu、rope…）按 `bytes / B_peak` 计算
- **Prefill**: 所有 token 并行，M = b·s
- **Decode**: 每次 1 token，M = b·1，attention 按 s_kv 计算
- 兼容混合架构：`attn_layers` 控制 full attention 层数，其余层跳过 O(s²) attention
- 窗口注意力 / 线性注意力等新架构：只改 FLOPs/Bytes 公式，无需重新 bench

### 模型参数 — [model_specs.yaml](perf_predict/model_specs.yaml)

```yaml
- name: "Llama-2-7B"
  hidden_dim: 4096
  intermediate_dim: 11008
  num_heads: 32
  head_dim: 128
  num_layers: 32
  vocab_size: 32000
  norm_type: rmsnorm

- name: "Qwen3.5-2B"
  ...
  attn_layers: 6     # 只有 6/24 层用 full attention，其余 DeltaNet
```

## 目录结构

```
foothold/
├── main.py                     # 入口：benchmark / --fit
├── config/default.yaml         # 硬件标定配置
├── bench/                      # GPU kernel 基准测试
│   ├── matmul.py, elementwise.py
│   └── utils.py                # CUDA 计时、保存、内存估算
├── fit/                        # Roofline 模型拟合
│   ├── __main__.py             # uv run python -m fit <dir> --save <out>
│   ├── matmul.py, elementwise.py
│   └── utils.py                # roofline_time, roofline_fit, load/save
├── perf_predict/               # 模型吞吐预测
│   ├── model_specs.yaml        # 模型架构参数
│   └── predict.py              # 预测主程序
├── results/                    # [gitignored] benchmark 输出
└── pyproject.toml
```
