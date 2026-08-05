"""Fit Gated Delta Rule scan and causal conv1d roofline params.

Scan: dedicated (F_peak, B_peak, p) per regime (decode s_q=1 / prefill
s_q>1), with per-batch B_peak curves and an nvh scale anchor — mirrors
fit/flashattn.py.  Keys use the `ls_` (linear scan) prefix.

Conv1d: fitted as an elementwise-style (B_eff, overhead) pair, merged into
elem_b_effs / elem_overheads so sim/graph.py charges it per element.

Both must be re-run after installing fla / causal-conv1d to capture the
fused kernels' higher bandwidth.
"""

import numpy as np
from fit.utils import roofline_fit

S_Q_SPLIT = 1


def _fit_subset(results, label, F_fixed=None, min_points=5):
    flops = np.array([r["flops"] for r in results])
    bytes_moved = np.array([r["bytes"] for r in results])
    times = np.array([r["time_ms"] for r in results]) / 1000.0

    if len(results) < min_points:
        print(f"  {label}: too few points ({len(results)}), skipping")
        return {}

    if F_fixed is not None:
        from scipy.optimize import curve_fit

        def model(X, B, p):
            f, b_arr = X
            return (f / F_fixed + b_arr / B) ** (1 / p)

        B0 = float(np.median(bytes_moved / times))
        popt, _ = curve_fit(model, (flops, bytes_moved), times,
                            p0=[B0 * 0.5, 2.0],
                            bounds=([1e8, 1.0], [1e13, 10.0]),
                            maxfev=20000)
        B_peak, p = popt
        F_peak = F_fixed
        predicted = model((flops, bytes_moved), B_peak, p)
        ss_res = float(np.sum((times - predicted) ** 2))
        ss_tot = float(np.sum((times - np.mean(times)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        F_peak, B_peak, p, r2 = roofline_fit(flops, bytes_moved, times)

    print(f"  {label}: F={F_peak / 1e12:.1f} TF  B={B_peak / 1e12:.2f} TB  "
          f"p={p:.3f}  R2={r2:.4f}")
    return {"F_peak": float(F_peak), "B_peak": float(B_peak),
            "p": float(p), "r2": float(r2)}


def _fit_scan_dtype(results, label_suffix=""):
    """Fit scan roofline for a single dtype subset (mirrors FA fit)."""
    batch_sizes = sorted(set(r["b"] for r in results))

    if len(batch_sizes) <= 1:
        decode = [r for r in results if r["s_q"] <= S_Q_SPLIT]
        prefill = [r for r in results if r["s_q"] > S_Q_SPLIT]
        p_prefill = _fit_subset(prefill, f"prefill (s_q > {S_Q_SPLIT})",
                                min_points=4)
        if not p_prefill:
            raise ValueError(
                f"Too few prefill points ({len(prefill)}) to fit the scan roofline "
                "— gateddelta.csv is likely contaminated by OOM rows. Restore a "
                "clean CSV and re-run the fit."
            )
        F_shared = p_prefill["F_peak"]
        # decode has no s_kv dimension (state is fixed size): only batch
        # sweeps it, so 3 points suffice for the (B, p) fit with fixed F.
        p_decode = _fit_subset(decode, f"decode (s_q = {S_Q_SPLIT})",
                               F_fixed=F_shared, min_points=3)
        params = {}
        params.update({f"{k}_ls_decode": v for k, v in p_decode.items()})
        params.update({f"{k}_ls_prefill": v for k, v in p_prefill.items()})
        if label_suffix:
            return {f"{k}{label_suffix}": v for k, v in params.items()}
        return params

    # ── Step 1: fit F_peak from the largest batch with enough prefill
    # points (the torch-reference bench may OOM the largest batches) ──
    # Also require a healthy anchor fit (R2 ≥ 0.9): partially OOM-polluted
    # batches fit with garbage R2 and extrapolate F absurdly high.
    largest_b = None
    p_largest = {}
    for b_val in reversed(batch_sizes):
        prefill_b = [r for r in results
                     if r["b"] == b_val and r["s_q"] > S_Q_SPLIT]
        if len(prefill_b) < 4:
            continue
        p_candidate = _fit_subset(prefill_b,
                                  f"prefill b={b_val} (F_peak candidate){label_suffix}",
                                  min_points=4)
        if p_candidate.get("r2", 0.0) < 0.9:
            print(f"  → b={b_val} anchor fit R2={p_candidate['r2']:.3f} < 0.9 "
                  f"(likely polluted), trying smaller batch")
            continue
        largest_b = b_val
        p_largest = p_candidate
        break
    if largest_b is None:
        per_batch = {b_val: len([r for r in results
                                 if r["b"] == b_val and r["s_q"] > S_Q_SPLIT])
                     for b_val in batch_sizes}
        raise ValueError(
            "No batch has ≥4 valid prefill points with a healthy anchor fit "
            f"(R2≥0.9; per-batch prefill point counts: {per_batch}). "
            "gateddelta.csv is likely contaminated by OOM rows — restore a "
            "clean CSV and re-run the fit."
        )
    F_shared = p_largest["F_peak"]
    print(f"  → shared F_peak = {F_shared / 1e12:.1f} TF (from b={largest_b} prefill)")

    decode_B, decode_p, prefill_B, prefill_p = [], [], [], []
    for b_val in batch_sizes:
        batch_results = [r for r in results if r["b"] == b_val]
        decode = [r for r in batch_results if r["s_q"] <= S_Q_SPLIT]
        prefill_b = [r for r in batch_results if r["s_q"] > S_Q_SPLIT]

        p_p = _fit_subset(prefill_b, f"prefill b={b_val}{label_suffix}",
                          F_fixed=F_shared, min_points=4)
        # Decode has exactly one point per batch (s_q=1), so a per-batch
        # (B, p) fit is impossible; skip it instead of spamming the
        # misleading "too few points" log.  The decode batch curve comes
        # from the all-batch fit below (the batch sweep provides the
        # 3+ points).  Kept as a guard so extra decode s_q values in the
        # grid would re-enable the per-batch fit automatically.
        p_d = (_fit_subset(decode, f"decode  b={b_val}{label_suffix}",
                           F_fixed=F_shared, min_points=3)
               if len(decode) >= 3 else {})

        prefill_B.append(p_p.get("B_peak", 0.0))
        prefill_p.append(p_p.get("p", 1.0))
        decode_B.append(p_d.get("B_peak", 0.0))
        decode_p.append(p_d.get("p", 1.0))

    prefill_all = [r for r in results if r["s_q"] > S_Q_SPLIT]
    decode_all = [r for r in results if r["s_q"] <= S_Q_SPLIT]
    p_prefill_u = _fit_subset(prefill_all, f"prefill (all batches){label_suffix}",
                              F_fixed=F_shared, min_points=4)
    p_decode_u = _fit_subset(decode_all, f"decode  (all batches){label_suffix}",
                             F_fixed=F_shared, min_points=3)

    params = {}
    params.update({f"{k}_ls_decode{label_suffix}": v for k, v in p_decode_u.items()})
    params.update({f"{k}_ls_prefill{label_suffix}": v for k, v in p_prefill_u.items()})
    params[f"ls_batch_sizes{label_suffix}"] = batch_sizes
    params[f"ls_decode_B{label_suffix}"] = decode_B
    params[f"ls_decode_p{label_suffix}"] = decode_p
    params[f"ls_prefill_B{label_suffix}"] = prefill_B
    params[f"ls_prefill_p{label_suffix}"] = prefill_p
    params[f"ls_bench_nvh{label_suffix}"] = results[0].get("nvh", 32)

    print(f"  decode  B(TB/s): {[f'{v/1e12:.2f}' for v in decode_B]}")
    print(f"  prefill B(TB/s): {[f'{v/1e12:.2f}' for v in prefill_B]}")

    return params


def _fit_conv1d_dtype(results, label_suffix=""):
    """Fit conv1d as elementwise (B_eff, overhead) — mirrors fit/elementwise."""
    bytes_moved = np.array([r["bytes"] for r in results])
    times = np.array([r["time_ms"] for r in results]) / 1000.0
    if len(results) < 3:
        print(f"  conv1d{label_suffix}: too few points ({len(results)}), skipping")
        return {}

    idx = np.argsort(bytes_moved)
    bs = bytes_moved[idx]
    ts = times[idx]
    bw = bs[-3:] / ts[-3:]
    B_eff = float(np.median(bw))
    overhead = float(np.median(ts[:3] - bs[:3] / B_eff))
    overhead = max(overhead, 0.0)
    print(f"  conv1d{label_suffix}: B={B_eff / 1e12:.2f} TB/s  "
          f"overhead={overhead * 1e6:.1f} us")
    return {
        f"elem_b_effs{label_suffix}": {"conv1d": B_eff},
        f"elem_overheads{label_suffix}": {"conv1d": overhead},
    }


def fit_gateddelta(results):
    """Fit scan (ls_* roofline) + conv1d (elem profile) from benchmark data.

    Returns dict with keys:
      - F_peak_ls_prefill / F_peak_ls_decode, B_peak_ls_*, p_ls_*, r2_ls_*
      - ls_batch_sizes, ls_decode_B/p, ls_prefill_B/p, ls_bench_nvh
      - elem_b_effs{_dtype}["conv1d"], elem_overheads{_dtype}["conv1d"]
    """
    scan_results = [r for r in results if r["op_name"] == "gated_delta_rule"]
    conv_results = [r for r in results if r["op_name"] == "conv1d"]

    # Anchor the per-request -> batch interpolation to the smallest nvh in
    # the pool (flops/bytes already scale linearly with nvh, so the fit is
    # nvh-normalized; the anchor only sets the effective-batch mapping).
    scan_results = sorted(scan_results, key=lambda r: r.get("nvh", 0))

    print("\n" + "=" * 60)
    print("Roofline Fit (Gated Delta Rule)")
    print(f"  split at s_q = {S_Q_SPLIT + 1} (decode: s_q=1, prefill: s_q>1)")
    print("=" * 60)

    params = {}

    if scan_results:
        dtypes = sorted(set(r.get("dtype", "float16") for r in scan_results))
        if len(dtypes) <= 1:
            params.update(_fit_scan_dtype(scan_results))
        else:
            for dt in dtypes:
                subset = [r for r in scan_results if r.get("dtype", "float16") == dt]
                print(f"\n  ── {dt} ──")
                params.update(_fit_scan_dtype(subset, label_suffix=f"_{dt}"))
    else:
        print("\n[GatedDelta Fit] No gated_delta_rule data — skipping")

    if conv_results:
        print("\n" + "=" * 60)
        print("Roofline Fit (causal conv1d, elementwise)")
        print("=" * 60)
        dtypes = sorted(set(r.get("dtype", "float16") for r in conv_results))
        if len(dtypes) <= 1:
            params.update(_fit_conv1d_dtype(conv_results))
        else:
            for dt in dtypes:
                subset = [r for r in conv_results if r.get("dtype", "float16") == dt]
                print(f"\n  ── {dt} ──")
                params.update(_fit_conv1d_dtype(subset, label_suffix=f"_{dt}"))
    else:
        print("\n[GatedDelta Fit] No conv1d data — skipping")

    return params
