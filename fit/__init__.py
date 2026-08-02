from fit.utils import load_results, save_fitted_params, roofline_time
from fit.matmul import fit_matmul as _fit_matmul_roofline
from fit.elementwise import fit_elementwise as _fit_elementwise_roofline
from fit.flashattn import fit_flashattn as _fit_flashattn_roofline
from fit.gateddelta import fit_gateddelta as _fit_gateddelta_roofline
from fit.cudagraph import fit_cudagraph_all as _fit_cudagraph_all
from fit.launch_overhead import fit_launch_overhead as _fit_launch_overhead
from fit.memcpy import fit_memcpy as _fit_memcpy
def fit_all(results):
    params = {"type": "roofline"}
    params.update(_fit_matmul_roofline(results))
    params.update(_fit_elementwise_roofline(results))
    params.update(_fit_flashattn_roofline(results))
    gd = _fit_gateddelta_roofline(results)
    for k in list(gd):
        if k.startswith("elem_b_effs") or k.startswith("elem_overheads"):
            params.setdefault(k, {}).update(gd.pop(k))
    params.update(gd)
    # CUDA Graph fit — separate params with _cudagraph suffix
    cg_results = [r for r in results if r.get("op_name", "").startswith("cudagraph_")]
    if cg_results:
        params.update(_fit_cudagraph_all(cg_results))
    # Kernel launch overhead (CPU→GPU dispatch) — from CPU-wall-clock bench
    lo_results = [r for r in results if r.get("op_name", "").startswith("launch_")]
    if lo_results:
        params.update(_fit_launch_overhead(lo_results))
    # GPU↔CPU memory copy LUT — replaces simple BW+latency model
    params.update(_fit_memcpy(results))
    return params
