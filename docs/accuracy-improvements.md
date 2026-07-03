# Accuracy Improvement Tracking

模拟精度提升路线图，按影响程度排序。已完成的标记 ✅，未完成的标记 ⬜。

---

## ✅ 1. Elementwise 算子独立 benchmark + 独立拟合

**日期**: 2026-07-03　**分支**: main

**问题**: `fit/elementwise.py` 只实测了 `residual_add` 和 `softmax`，其余全部 proxy 到 `residual_add`：

| 算子 | 旧 proxy | 真实访存/计算模式 |
|---|---|---|
| swiglu | residual_add | 读gate+读up+SiLU+乘+写（少算 50% 访存量） |
| rope | residual_add | 读Q/K+三角函数+写（计算量不可忽略） |
| rmsnorm | residual_add | 读+归约+rsqrt+写（reduction 型，已 benchmark 但未拟合） |

**改动**:

| 文件 | 变更 |
|---|---|
| `bench/elementwise.py` | 新增 swiglu（`F.silu(gate) * up`）和 rope（2D 旋转变换）的 GPU benchmark |
| `config/bench.yaml` | `operators` 扩展为 `[residual_add, rmsnorm, softmax, swiglu, rope]` |
| `fit/elementwise.py` | 移除全局 PROXY → `MEASURED_OPS` 独立拟合 5 个算子；仅保留语义合理的 proxy（`layernorm→rmsnorm`，`causal_mask→residual_add`） |

**需重新运行 bench + fit 生效**。

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

## ⬜ 7. Matmul 性能的 K/N 维度敏感性

当前 roofline 只关心 M×K×N 的总 FLOPs/bytes。但相同 FLOPs 下，不同 K/N 形状在 tensor core 上的 tile 利用率不同。例如 `[M,4096]×[4096,4096]` vs `[M,11008]×[11008,4096]` 性能不同。

**建议**: benchmark 使用模型真实投影维度，或按 `K%16` / `N%16` 做 tile 利用率修正。

**难度**: 中　**影响**: ⭐⭐

---

## ⬜ 8. KV 传输 overlap 模型细化

当前 `overlap = prefill_time * (num_layers-1) / num_layers` 是假设每层均匀分配时间的线性近似。

**建议**: 建模 per-layer compute/KV-save 时序，或使用 pipeline bubble 模型。

**难度**: 中　**影响**: ⭐⭐

---

## ⬜ 9. Kernel Fusion 效应

RMSNorm+residual 融合、QKV 投影融合等在真实推理引擎中普遍存在，但 roofline 将其当作独立 op 求和。

**建议**: 增加 fused kernel benchmark（如 fused RMSNorm+residual），或在 roofline 中加入 fusion factor。

**难度**: 高　**影响**: ⭐⭐

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
