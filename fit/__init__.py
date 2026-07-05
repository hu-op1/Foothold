from fit.utils import load_results, save_fitted_params, roofline_time
from fit.matmul import fit_matmul as _fit_matmul_roofline
from fit.elementwise import fit_elementwise as _fit_elementwise_roofline
from fit.flashattn import fit_flashattn as _fit_flashattn_roofline
from fit.cudagraph import fit_cudagraph_all as _fit_cudagraph_all
from fit.launch_overhead import fit_launch_overhead as _fit_launch_overhead
from fit.linear import fit_matmul_linear, fit_elementwise_linear


def fit_all(results, backend="roofline"):
    """Run all fits, return combined params dict.

    Args:
        results: list of benchmark result dicts.
        backend: "roofline" (default) or "linear".
    """
    if backend == "linear":
        params = {"type": "linear"}
        params.update(fit_matmul_linear(results))
        params.update(fit_elementwise_linear(results))
        return params

    # Default: roofline
    params = {"type": "roofline"}
    params.update(_fit_matmul_roofline(results))
    params.update(_fit_elementwise_roofline(results))
    params.update(_fit_flashattn_roofline(results))
    # CUDA Graph fit — separate params with _cudagraph suffix
    cg_results = [r for r in results if r.get("op_name", "").startswith("cudagraph_")]
    if cg_results:
        params.update(_fit_cudagraph_all(cg_results))
    # Kernel launch overhead (CPU→GPU dispatch) — from CPU-wall-clock bench
    lo_results = [r for r in results if r.get("op_name", "").startswith("launch_")]
    if lo_results:
        params.update(_fit_launch_overhead(lo_results))
    return params
