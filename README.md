# Foothold

LLM 推理性能工具链：基准测试 → 算子拟合 → 模型吞吐预测。

## 环境

- Python ≥ 3.10
- NVIDIA GPU（8GB+ VRAM），CUDA ≥ 12.6

```bash
uv sync
```

## 整体流程

```
config/default.yaml  →  bench/   →  results/<gpu>/*.xlsx     (原始数据)
                                     ↓
                                  fit/   →  fitted_params.json  (算子拟合参数)
                                              ↓
perf_predict/model_specs.yaml  →  perf_predict/  →  模型吞吐预测 (tokens/s)
```

## 1. 基准测试 — `bench/`

```bash
uv run python main.py                          # 默认 config/default.yaml
uv run python main.py --output results/3090    # 指定输出目录
```

### 配置 — [config/default.yaml](config/default.yaml)

```yaml
batch_sizes: [1, 8]
seq_lens: [1, 512, 2048]         # 1 覆盖 decode，512 中等，2048 长序列
hidden_dims: [512, 1024, 2048]
num_heads: [8, 32]
vocab_sizes: [128256]
dtype: "float16"
warmup_iters: 200
bench_iters: 500
max_memory_gb: 6                  # 保护 8GB 显存
```

### 算子覆盖

| 类别 | 算子 | work 公式 |
|------|------|----------|
| GEMM | q/k/v/o_proj | M·K·N, M=b·s, K=N=h |
| GEMM | ffn_up/gate | M·K·N, N=4h |
| GEMM | ffn_down | M·K·N, K=4h |
| GEMM | lm_head | M·K·N, N=vocab |
| Attention | qk_matmul | b·nh·s²·hd |
| Attention | softmax | b·nh·s² |
| Attention | score_v_matmul | b·nh·s²·hd |
| Norm | layernorm / rmsnorm | b·s·h |
| Activation | swiglu / rope / residual_add / causal_mask | 各不同 |

输出格式：`results/<gpu>/*.xlsx`，含 `all_operators.xlsx` 汇总。

## 2. 算子拟合 — `fit/`

从 benchmark 结果拟合每个算子的线性模型 `time = a·work + b`：

```bash
uv run python -m fit results/3090                        # 打印拟合结果
uv run python -m fit results/3090 --save fitted_params.json  # 导出 JSON
```

输出 `fitted_params.json`：

```json
{
  "q_proj":    {"a": 3.01e-11, "b": 0.038, "r2": 0.997, "type": "gemm"},
  "qk_matmul": {"a": 6.48e-11, "b": 0.231, "r2": 0.622, "type": "attention"},
  ...
}
```

## 3. 吞吐预测 — `perf_predict/`

根据模型架构参数 + 算子拟合结果，预测 prefill 和 decode 的端到端延迟与吞吐：

```bash
uv run python perf_predict/predict.py --list                # 列出已有模型
uv run python perf_predict/predict.py --model "Llama-2-7B" \
    --input-len 2048 --output-len 512 --batch 4             # 单模型预测
uv run python perf_predict/predict.py --predict-all \
    --input-len 2048 --output-len 512                       # 全部模型
```

### 预测原理

- **Prefill**: 所有 token 并行处理，attention 为 O(s²)
- **Decode**: 每次只算 1 个 token + KV cache，GEMM 的 M=1，attention 为 O(s_kv)
- 支持混合架构（如 Qwen3.5 的 DeltaNet + full attention）

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
  # attn_layers: 32   # 默认全层 attention；混合架构用此字段
```

### 实测对比

在 RTX 3090 + Llama-2-7B 上与 vLLM 真实吞吐（100 组不同 batch/input/output 组合）对比：

```
MAPE:         6.9%
within 10%:   80/100 (80%)
within 20%:   96/100 (96%)
```

## 目录结构

```
foothold/
├── main.py                     # 入口：benchmark / --fit
├── config/default.yaml         # benchmark 配置
├── bench/                      # GPU 算子基准测试
│   ├── gemm.py, attention.py, norm.py, activation.py
│   └── utils.py                # CUDA 计时、保存、内存估算
├── fit/                        # 算子级线性拟合
│   ├── __main__.py             # uv run python -m fit <dir> --save <out>
│   ├── gemm.py, attention.py, norm.py, activation.py
│   └── utils.py                # lstsq_fit, load_results, save/load_fitted_params
├── perf_predict/               # 模型吞吐预测
│   ├── model_specs.yaml        # 模型架构参数
│   └── predict.py              # 预测主程序
├── results/                    # [gitignored] benchmark 输出
└── pyproject.toml
```
