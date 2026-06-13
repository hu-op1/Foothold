from fit.utils import load_results, save_fitted_params, roofline_time
from fit.matmul import fit_matmul as _fit_matmul_roofline
from fit.elementwise import fit_elementwise as _fit_elementwise_roofline
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
    return params
