---
description: "Use when modifying roofline model fitting in fit/. Covers matmul prefill/decode split, elementwise proxy mapping, curve fitting conventions, and output format."
applyTo: "fit/**/*.py"
---

# fit/ — Roofline Model Fitting

## Entry Point

`fit/__init__.py` exposes `fit_all(results, backend="roofline")`:

```python
def fit_all(results, backend="roofline"):
    if backend == "linear":
        params = {"type": "linear"}
        params.update(fit_matmul_linear(results))
        params.update(fit_elementwise_linear(results))
        return params
    # roofline backend
    params = {"type": "roofline"}
    params.update(_fit_matmul_roofline(results))
    params.update(_fit_elementwise_roofline(results))
    params.update(_fit_flashattn_roofline(results))
    # ... plus cudagraph and launch_overhead
```

## Roofline Model

Core math in `fit/utils.py`:

```
time = ((flops / F_peak)^p + (bytes / B_peak)^p)^(1/p)
```

Fitting uses `scipy.optimize.curve_fit`. Initial guesses from median FLOP/s and byte/s ratios. `p` clamped to `[1.0, 10.0]`.

## Matmul Fitting (`fit/matmul.py`)

- **Split at M=256**: Prefill (M ≥ 256) vs decode (M < 256).
- Fit **shared F_peak** on prefill data first, then fix it and fit B_peak + p for each region.
- Estimate per-kernel launch overhead from smallest-M data points.
- Guard: `if len(data) < 5: print(f"  {label}: too few points, skipping")`.

## Elementwise Fitting (`fit/elementwise.py`)

Model: `time = bytes / B_eff + overhead`.

- **B_eff** from large-N points (memory-bandwidth dominated).
- **Overhead** from small-N points (kernel launch dominated).
- **Proxy map** for unmeasured ops — defined in `PROXY` dict with documented rationale:

```python
PROXY = {
    "rope": "residual_add",  # single fused in-place CUDA kernel
    "rmsnorm": "residual_add",
    "layernorm": "residual_add",
    "causal_mask": "residual_add",
    "fused_residual_norm": "residual_add",
}
MEASURED_OPS = ["residual_add", "rmsnorm", "softmax", "swiglu"]
```

When adding a new proxy: benchmark the real kernel first, then map if the overhead/B_eff are within ±20% of the proxy.

## CUDA Graph Fitting (`fit/cudagraph.py`)

- Keys namespaced with `_cudagraph` suffix (e.g. `F_peak_prefill_cudagraph`).
- Same split strategy, no F_peak sharing needed for elementwise.

## GatedDelta Fitting (`fit/gateddelta.py`)

`fit_gateddelta(results)` fits hybrid-architecture kernels:

- **Scan** uses `ls_`-prefixed dedicated roofline params: decode/prefill split + per-batch B_peak curve + `nvh` anchor. Keys like `ls_F_peak_prefill`, `ls_B_peak_decode`, … — these are what `StepContext.precompute()` in `sim/graph.py` detects to select scan params.
- **Conv1d** is fit as an elementwise op (`B_eff` + `overhead`) and merged into `elem_b_effs` / `elem_overheads`.

Keep the `ls_` prefix convention — renaming breaks the graph wiring in `sim/graph.py`.

## Loading Bench Results

```python
from fit.utils import load_results
results = load_results("bench/results/3090/")
# Returns list[dict], skips OOM rows
```

## Output

`save_fitted_params(params, path)` writes JSON to `fit/results/<gpu>.json`. Key structure:

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
    "residual_add": {"B_eff": 839e9, "overhead_us": 37}
  }
}
```
