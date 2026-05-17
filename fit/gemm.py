import numpy as np
from fit.utils import lstsq_fit, lstsq_log_fit


def fit_gemm(results):
    """Fit GEMM operators, return {op_name: {a, b, r2}}."""
    gemm_ops = {"q_proj", "k_proj", "v_proj", "o_proj", "ffn_up", "ffn_gate", "ffn_down", "lm_head"}
    ops = sorted(set(r["op_name"] for r in results) & gemm_ops)
    params = {}

    print("=" * 60)
    print("GEMM Operators")
    print("=" * 60)

    for op_name in ops:
        op_results = [r for r in results if r["op_name"] == op_name]
        work = np.array([r["M"] * r["K"] * r["N"] for r in op_results])
        time_ms = np.array([r["time_ms"] for r in op_results])

        a, b, r2 = lstsq_fit(work, time_ms)
        c1, _, r2_log = lstsq_log_fit(work, time_ms)

        flops = np.sum(2 * work)
        total_ms = np.sum(time_ms)
        avg_tflops = (flops / (total_ms / 1000)) / 1e12 if total_ms > 0 else 0

        print(f"\n{op_name}")
        print(f"  work = M*K*N")
        print(f"  time = {a:.3e} * work + {b:.4f}   R2={r2:.4f}")
        print(f"  power-law exponent: {c1:.3f}   R2_log={r2_log:.4f}")
        print(f"  effective TFLOPS: {avg_tflops:.1f}")

        params[op_name] = {"a": float(a), "b": float(b), "r2": float(r2), "type": "gemm"}

    return params
