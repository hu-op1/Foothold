from fit.utils import load_results, save_fitted_params, roofline_time
from fit.matmul import fit_matmul
from fit.elementwise import fit_elementwise


def fit_all(results):
    """Run all fits, return combined params dict."""
    params = {}
    params.update(fit_matmul(results))
    params.update(fit_elementwise(results))
    return params


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Fit roofline model to benchmark results")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Directory containing xlsx result files")
    parser.add_argument("--save", type=str, default=None,
                        help="Save fitted params to JSON file")
    args = parser.parse_args()

    matmul_path = os.path.join(args.results_dir, "matmul.xlsx")
    elem_path = os.path.join(args.results_dir, "elementwise.xlsx")

    results = []
    for path in [matmul_path, elem_path]:
        if os.path.exists(path):
            results.extend(load_results(path))
        else:
            print(f"Warning: not found: {path}")

    if not results:
        print("No valid results to fit.")
        return

    print(f"Loaded {len(results)} rows from {args.results_dir}")
    params = fit_all(results)

    if args.save:
        save_fitted_params(params, args.save)
