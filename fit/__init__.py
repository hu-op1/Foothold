from fit.utils import load_results, save_fitted_params, roofline_time
from fit.matmul import fit_matmul
from fit.elementwise import fit_elementwise


def fit_all(results):
    """Run all fits, return combined params dict."""
    params = {}
    params.update(fit_matmul(results))
    params.update(fit_elementwise(results))
    return params
