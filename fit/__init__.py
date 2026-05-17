import os
from fit.utils import load_results, save_fitted_params
from fit.gemm import fit_gemm
from fit.attention import fit_attention
from fit.norm import fit_norm
from fit.activation import fit_activation


def fit_all(results):
    """Run all fits, return combined params dict."""
    params = {}
    params.update(fit_gemm(results))
    params.update(fit_attention(results))
    params.update(fit_norm(results))
    params.update(fit_activation(results))
    return params


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fit performance models to benchmark results")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Directory containing xlsx result files")
    parser.add_argument("--save", type=str, default=None,
                        help="Save fitted params to JSON file")
    args = parser.parse_args()

    all_path = os.path.join(args.results_dir, "all_operators.xlsx")
    if not os.path.exists(all_path):
        print(f"Not found: {all_path}")
        return

    results = load_results(all_path)
    if not results:
        print("No valid results to fit.")
        return

    print(f"Loaded {len(results)} rows from {all_path}")
    params = fit_all(results)

    if args.save:
        save_fitted_params(params, args.save)
