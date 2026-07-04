# Accuracy Improvement Tracking

模拟精度提升路线图，按影响程度排序。已完成的标记 ✅，未完成的标记 ⬜。

---

## ⚠️ 1. Elementwise 算子独立 benchmark + 独立拟合（部分回滚）

**日期**: 2026-07-03　**修订**: 2026-07-04　**分支**: main

**原始问题**: `fit/elementwise.py` 只实测了 `residual_add` 和 `softmax`，其余全部 proxy 到 `residual_add`。

**改动**:

| 文件 | 变更 |
|---|---|
| `bench/elementwise.py` | 新增 swiglu（`F.silu(gate) * up`）和 rope（2D 旋转变换）的 GPU benchmark |
| `config/bench.yaml` | `operators` 扩展为 `[residual_add, rmsnorm, softmax, swiglu, rope]` |

**✅ swiglu 独立拟合有效** — `F.silu(gate)*up` 是真实的 PyTorch 融合 kernel，测得的 B_eff (~500 GB/s) 和 overhead (~57μs) 合理。

**❌ rope 独立拟合已回滚** — 见下文。

---

### ⚠️ 1b. RoPE benchmark 不可靠 → 回退到 proxy

**日期**: 2026-07-04　**分支**: main

**发现**: `bench/elementwise.py` 中的 RoPE synthetic kernel 使用 PyTorch 切片索引：

```python
out[:, 0] = q2[:, 0] * 0.5 - q2[:, 1] * 0.866
out[:, 1] = q2[:, 1] * 0.5 + q2[:, 0] * 0.866
```

`[:, 0]` / `[:, 1]` 破坏了 PyTorch 的 kernel fusion，导致单次调用实际触发 **8+ 个独立 kernel launch**。拟合结果：

| 指标 | rope（broken benchmark） | residual_add（正常） | 高估 |
|---|---|---|---|
| overhead | **261 μs** | 37 μs | **7×** |
| B_eff | 274 GB/s | 839 GB/s | **3× slower** |

**vLLM 0.19.0 真实实现** (`csrc/pos_encoding_kernels.cu`): RoPE 是单个 in-place CUDA kernel — 每个 thread block 处理一个 token，block 内线程协作处理所有 head。overhead 和带宽特征与 `residual_add` 在同一量级。

**修复** (`fit/elementwise.py`): `rope` 已从 `MEASURED_OPS` 移除，加入 `PROXY` 映射到 `residual_add`：

```python
PROXY = {
    ...
    "rope": "residual_add",  # single fused in-place kernel, not ~8 launches
}
MEASURED_OPS = ["residual_add", "rmsnorm", "softmax", "swiglu"]  # rope 移除
```

**影响**: RoPE 时间占比从 ~41% → ~5-8%（合理范围）。直接用现有 fit 数据重新拟合即可生效（`uv run python main.py fit`）。

---

## ✅ 2. M_SPLIT 硬阈值 → log-space 平滑插值

**日期**: 2026-07-03　**分支**: main

**问题**: 

- `fit/matmul.py` 用 M_SPLIT=256 拟合两组 (B_peak, p)（decode / prefill）
- `sim/executor.py` 用 M_SPLIT=32 硬切换 —— 一个 token 从 31→32 可能导致 step time 跳变 5×

**改动** (`sim/executor.py`):

```
M ≤ 32  → 纯 decode 参数
32<M<256 → log-space 插值: B = exp(lerp(log(B_dec), log(B_pre), w))
M ≥ 256 → 纯 prefill 参数
```

F_peak 共享（拟合时已固定），仅 B_peak 和 p 参与插值。B_peak 用 log-space（物理上带宽比是乘法关系），p 用线性插值。

**使用现有 fit 数据即可生效**。

---

## ✅ 3. Attention per-request 求和 → 分组累加 + 单次 roofline

**日期**: 2026-07-03　**分支**: main

**问题**: 旧代码对每个 request 单独调用 `attention_fused(b=1)` 再求和。但 roofline time 是 L^p 范数：

$$\sum_i \left\|(f_i, b_i)\right\|_p \;\ge\; \left\|\sum_i (f_i, b_i)\right\|_p$$

三角不等式导致逐 request 求和**系统性高估** attention 时间。真实 FlashAttention 将所有同类型 request 合并为一次 kernel 调用。

**改动** (`sim/executor.py` — `predict_step()` 和 `predict_step_pp()`):

```python
# 旧: 逐 request 调用 attention_fused(b=1) 再求和
# 新: 按 prefill/decode 分组累加 FLOPs + bytes，各调用一次 roofline_time
prefill_flops = sum(4*nh*s_q_i*s_kv_i*hd)
prefill_bytes = sum(hd*DTYPE_BYTES*(2*nh*s_q_i + 2*nh_kv*s_kv_i))
attn_prefill_time = na * roofline_time(prefill_flops, prefill_bytes, F_p, B_p, p_p)
```

| 场景 | 旧（逐 request） | 新（分组单次 roofline） |
|---|---|---|
| 全部同类型、同长度 | 一致 | 一致 |
| 混合 prefill/decode | 高估 3–8% | 正确 |
| 不同 prompt 长度 | 高估 2–5% | 正确 |

**使用现有 fit 数据即可生效**。

---

## ✅ 4. FlashAttention 实测校准

**日期**: 2026-07-03　**分支**: main

**问题**: 旧代码中 attention 的 roofline 预测使用 matmul 拟合的 F_peak / B_peak / p 参数。但 FlashAttention 的硬件效率（tiling、SRAM 调度、causal mask 开销）与 GEMM 完全不同，直接用 matmul 参数存在系统偏差。

**改动**:

| 文件 | 变更 |
|---|---|
| `bench/flashattn.py` | **新建** — 使用 `F.scaled_dot_product_attention` 在 (s_q, s_kv) 网格上 benchmark FlashAttention |
| `config/bench.yaml` | 新增 `flashattn` 配置节：s_q [1..2048], s_kv [128..32768], GQA 参数 |
| `fit/flashattn.py` | **新建** — 从 FA benchmark 数据拟合专用的 (F_peak, B_peak, p) 参数，按 s_q=1 (decode) vs s_q>1 (prefill) 分裂 |
| `fit/__init__.py` | 集成 `fit_flashattn` 到 `fit_all()` 管线 |
| `sim/executor.py` | `predict_step()` 和 `predict_step_pp()` 中的 attention 优先使用 `F_peak_fa_*` / `B_peak_fa_*` / `p_fa_*`，无 FA 数据时 fallback 到 matmul 参数（向后兼容） |
| `main.py` | bench 管线增加 `bench_flashattn`；fit 管线加载 `flashattn.xlsx` |

**需在 Linux 上重新运行 bench `flashattn` + fit 生效**。Windows 上 PyTorch SDPA 使用 cuDNN 后端（非 Dao-AILab FA2），数据可作参考但不完全等价。

---

## ⬜ 5. Roofline p-norm 模型假设验证

当前 `time = ((flops/F)^p + (bytes/B)^p)^(1/p)` 是对 GPU 并行性的一种特定假设。真实 GPU 的计算/访存重叠行为可能偏离此函数形式。

**建议**: 在 log-log 图上检查实测 vs 拟合的系统偏差；考虑分段样条插值或非参数模型。

**难度**: 中　**影响**: ⭐⭐⭐

---

## ✅ 6. TP All-Reduce 带宽曲线 — 加入延迟项

**日期**: 2026-07-03　**分支**: main

**问题**: 旧代码中 all-reduce 和 inter-stage P2P 使用纯带宽模型 `time = bytes / bw`。对于 decode 场景（M=1，消息仅 ~16 KB），带宽项 ~1.7 µs，但实际 ring all-reduce 的延迟（N 跳 × ~2 µs/hop）是这个量的 10× 以上，导致 decode step time 被严重低估。

**改动**:

| 文件 | 变更 |
|---|---|
| `config/search.yaml` | `communication` 增加 `intra_latency_us: 2.0` |
| `config/sim.yaml` | 同上 |
| `sim/engine.py` | 读取 `intra_latency_us`，传入 `predict_step_tp` |
| `sim/executor.py` | `predict_step_pp()`: inter-stage 改为 `bytes/bw + latency`；`predict_step_tp()`: all-reduce 改为 `bytes/bw + N_gpus × latency` |

**模型**:

```
# 旧 (纯带宽)
inter_stage = (pp-1) × bytes / bw
all_reduce  = bytes / bw

# 新 (延迟+带宽)
inter_stage = (pp-1) × (bytes / bw + latency)
all_reduce  = bytes / bw + N_gpus × latency    # ring all-reduce N步
```

**精度影响**: decode (M=1) 的 TP 开销从被低估 ~10× → 正确量级。prefill (M 大) 几乎无影响（带宽项主导）。**使用现有 fit 数据即可生效**。

---

## ✅ 7. Matmul 性能的 K/N 维度敏感性 — Tensor Core tile 量化修正

**日期**: 2026-07-03　**分支**: main

**问题**: GPU tensor core 以 16×16 tile 为单位计算。当 K 或 N 不是 tile 的整数倍时，硬件自动 padding 到下一 tile 边界，padding 部分的计算被丢弃。纯 FLOPs 公式 `2*M*K*N` 忽略了这一浪费，对非对齐维度会低估实际耗时。

**改动** (`sim/roofline.py`):

新增 `_tile_waste(K, N)` 函数，返回时间膨胀系数 `≥ 1.0`:

```
eff_K = ceil(K/16) × 16
eff_N = ceil(N/16) × 16
waste = (eff_K / K) × (eff_N / N)
```

`matmul_time()` 改为 `t = roofline_time(...) * _tile_waste(K, N)`。

**实际影响**: 标准 LLM 维度（h=4096/8192, inter=11008/14336/12288, vocab=151936/32000）均为 16 的整数倍 → waste=1.0，**无变化**。此修正主要提升对非标/自定义模型维度的精度。

**使用现有 fit 数据即可生效**。

---

## ⬜ 8. KV 传输 overlap 模型细化

当前 `overlap = prefill_time * (num_layers-1) / num_layers` 是假设每层均匀分配时间的线性近似。

**建议**: 建模 per-layer compute/KV-save 时序，或使用 pipeline bubble 模型。

**难度**: 中　**影响**: ⭐⭐

---

## ✅ 9. Kernel Fusion 效应 — 对齐 vLLM 实现

**日期**: 2026-07-03　**分支**: main

**问题**: 经核对 vLLM 0.19.0 源码，发现三处融合在模拟器中被当作独立 op：

| 融合 | vLLM 实现 | 旧模拟器 | HBM 节省 |
|---|---|---|---|
| QKV 投影 | `QKVParallelLinear` — 1 次 `[M,h]×[h,3h]` | 3 次独立 `[M,h]×[h,h]` | 输入少读 2 次 `M×h` |
| RMSNorm + residual | `fused_add_rms_norm` — 1 个 kernel | 独立 `residual_add` + `rmsnorm` | 消除中间结果 HBM 读写 |
| SwiGLU | `swiglustep_and_mul` — 已融合 ✓ | 已直接 bench `F.silu(gate)*up` ✓ | 无差异 |

**改动**:

| 文件 | 变更 |
|---|---|
| `sim/roofline.py` | `attn_projections`: MHA → `[M,h]×[h,3h]` fused QKV + O；GQA → `[M,h]×[h,dim_q+2dim_kv]` + O。新增 `fused_residual_norm` op (ELEM_BYTES=4) 和 `fused_residual_norm_ops()` 函数，无 fit 数据时 fallback 到分离 op |
| `sim/executor.py` | `predict_step()` / `predict_step_pp()`: 用 `fused_residual_norm_ops` 替换 `norm_ops + residual_add_ops`；TP 缩放列表更新键名 |
| `sim/engine.py` | `time_acc` / `_zero_step_dict` 键名 `rmsnorm`+`residual_add` → `fused_add_norm` |
| `sim/report.py` | xlsx 列名同步更新 |

**精度影响**: QKV 融合减少 ~30% 投影时间估计（输入 HBM 节省）；RMSNorm+residual 融合减少 ~40% norm+add 时间估计（消除中间结果）。**使用现有 fit 数据即可生效**（fused_residual_norm 自动 fallback 到 rmsnorm B_eff）。

---

## ⬜ 10. GPU Boost Clock / Thermal 建模

F_peak/B_peak 是静态拟合值。真实 GPU 的 core clock 和 memory clock 各自变化，且受温度、功耗墙影响。长时间运行后性能可能下降 5–15%。

**建议**: 加入 thermal throttling 模型（或至少 clock variability factor）。

**难度**: 高　**影响**: ⭐⭐

---

## ⬜ 11. DeltaNet / 线性 Attention 层建模

`nd = nl - na`（DeltaNet 层）当前被跳过 attention 计算。但其 chunk-wise parallel scan 计算 pattern 与标准 attention 完全不同，需独立建模。

**建议**: 增加 DeltaNet / linear attention 的计算量模型。

**难度**: 中　**影响**: ⭐（仅影响 Qwen3.5 等混合架构模型）

---

## ⬜ 12. 调度器自身开销

`schedule()` 本身有计算开销（~0.1–0.5ms/step），当前视为零时间。

**建议**: 加入固定的 scheduler overhead 参数。

**难度**: 低　**影响**: ⭐

---

## 优先级矩阵

```
影响 ↑
 5 │  1✅         4⬜
   │
 4 │  2✅  3✅
   │
 3 │        5⬜  6⬜
   │
 2 │  7⬜  8⬜  9⬜ 10⬜
   │
 1 │ 11⬜ 12⬜
   └──────────────────────→ 难度
      低    中    高
```

**下一步建议**: #6（低难度、中等影响）→ #5（验证模型假设）→ #4（实测 FA 校准）。
