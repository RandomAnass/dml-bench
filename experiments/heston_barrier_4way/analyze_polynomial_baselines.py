#!/usr/bin/env python3
"""
Polynomial baseline comparison for Heston barrier multi-seed results.

Per the deep-analysis review (multi_seed_deep_analysis.md §2): vanilla NN is
worse than a quadratic in S_0 on this barrier-pricing problem. Reviewers
will run polynomial regression themselves and find this; we must surface it
in any results table.

This script:
1. Regenerates the multi-seed ground truth deterministically (seed=999).
2. Fits polynomial regressors of degrees 0..6 on S_0 alone (no NN).
3. Reports val_mse for each polynomial vs the GT.
4. Compares polynomials to the multi-seed NN results (loaded from JSON).
5. Computes R² and val_mse-relative-to-quintic-polynomial for context.

Usage:
    python experiments/heston_barrier_4way/analyze_polynomial_baselines.py
    python experiments/heston_barrier_4way/analyze_polynomial_baselines.py \\
        --results-dir results/heston_barrier_4way/multi_seed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference
from experiments.heston_barrier_4way.run_pilot import HESTON_PARAMS


def regenerate_ground_truth(n_test=200, n_paths=100_000, seed=999):
    """Regenerate the same ground truth used by multi-seed runs."""
    rng = np.random.RandomState(seed)
    S0_test = rng.uniform(
        HESTON_PARAMS["strike"] * 0.7, HESTON_PARAMS["strike"] * 1.3, n_test,
    )
    gt = heston_barrier_doc_mc_reference(
        S0=S0_test, n_paths=n_paths, seed=seed, **HESTON_PARAMS,
    )
    return gt["x"].flatten(), gt["y"].flatten(), gt["dydx"].flatten()


def polynomial_baseline_val_mse(x_train, y_train, x_test, y_test, degree):
    """Fit a polynomial of given degree on (x_train, y_train), evaluate on test."""
    coeffs = np.polyfit(x_train, y_train, degree)
    y_pred = np.polyval(coeffs, x_test)
    return np.mean((y_pred - y_test) ** 2)


def constant_baseline_val_mse(y_test):
    """Constant predictor = E[y_test]. val_mse = Var(y_test)."""
    return np.var(y_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/heston_barrier_4way/multi_seed")
    ap.add_argument("--n_train_poly", type=int, default=1024,
                    help="Number of training points for polynomial fits (must split GT)")
    args = ap.parse_args()

    print("=" * 80)
    print("POLYNOMIAL BASELINE ANALYSIS — Heston barrier")
    print("=" * 80)

    print("\n1. Regenerating multi-seed ground truth (seed=999, n_test=200, n_paths=100k)...")
    x_gt, y_gt, dydx_gt = regenerate_ground_truth()
    print(f"   x range: [{x_gt.min():.3f}, {x_gt.max():.3f}]")
    print(f"   y range: [{y_gt.min():.4e}, {y_gt.max():.4e}]")
    print(f"   y mean:  {y_gt.mean():.4e}, var: {np.var(y_gt):.4e}")

    # Polynomial fit: use full GT to fit (poly is data-cheap), evaluate on same GT
    # (in-sample evaluation gives the OPTIMISTIC bound on what poly can achieve).
    # This is the right comparison: "best possible polynomial fit" vs "DML method".
    # If we want held-out poly fit, split GT 80/20.
    print("\n2. Polynomial baselines (in-sample fit on full GT — best-case for poly):")
    print(f"   {'Degree':>8}  {'val_mse':>14}  {'R^2':>10}")
    poly_results = {}
    var_y = np.var(y_gt)
    poly_results[0] = constant_baseline_val_mse(y_gt)
    print(f"   {0:>8}  {poly_results[0]:14.4e}  {1 - poly_results[0]/var_y:10.6f}")
    for deg in range(1, 7):
        mse = polynomial_baseline_val_mse(x_gt, y_gt, x_gt, y_gt, deg)
        poly_results[deg] = mse
        r2 = 1 - mse / var_y
        print(f"   {deg:>8}  {mse:14.4e}  {r2:10.6f}")

    # Also do held-out polynomial evaluation (split 80/20)
    print("\n3. Polynomial baselines (held-out 80/20 split — fair comparison):")
    print(f"   {'Degree':>8}  {'val_mse':>14}  {'R^2':>10}")
    rng = np.random.RandomState(42)
    n = len(x_gt)
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    x_tr, y_tr = x_gt[train_idx], y_gt[train_idx]
    x_te, y_te = x_gt[test_idx], y_gt[test_idx]
    var_y_te = np.var(y_te)
    print(f"   {0:>8}  {np.var(y_te):14.4e}  {0:10.6f}  (constant predictor on test)")
    for deg in range(1, 7):
        mse = polynomial_baseline_val_mse(x_tr, y_tr, x_te, y_te, deg)
        r2 = 1 - mse / var_y_te if var_y_te > 0 else 0
        print(f"   {deg:>8}  {mse:14.4e}  {r2:10.6f}")

    # Load DML methods from JSON
    print(f"\n4. DML methods from {args.results_dir} (means across seeds):")
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"   {results_dir} not found — skipping DML comparison")
        return

    by_method = {}
    for f in results_dir.glob("*.json"):
        if f.name in ("summary.json", "analysis.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            by_method.setdefault(data["method"], []).append(data)
        except Exception:
            pass

    if not by_method:
        print(f"   No JSON results in {results_dir}")
        return

    # Sort methods by mean val_mse
    sortable = []
    for method, vals in by_method.items():
        v_mean = np.mean([r["test_value_mse"] for r in vals])
        v_std = np.std([r["test_value_mse"] for r in vals])
        sortable.append((v_mean, method, v_std, len(vals)))
    sortable.sort()

    quintic_inS = poly_results[5]
    quad_inS = poly_results[2]
    print(f"   Comparison anchors: quadratic-in-S_0 = {quad_inS:.4e}, "
          f"quintic = {quintic_inS:.4e}")
    print(f"\n   {'Method':<28}  {'val_mse':>14}  {'rel-quad':>10}  {'rel-quintic':>12}  {'beats quad?':>12}")
    for (v_mean, method, v_std, n) in sortable:
        rq = v_mean / quad_inS
        rqu = v_mean / quintic_inS
        beats_quad = "YES" if v_mean < quad_inS else "NO"
        print(f"   {method:<28}  {v_mean:14.4e}  {rq:10.2f}x  {rqu:12.2f}x  {beats_quad:>12}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_methods = len(sortable)
    n_beat_quad = sum(1 for v, _, _, _ in sortable if v < quad_inS)
    n_beat_quintic = sum(1 for v, _, _, _ in sortable if v < quintic_inS)
    print(f"  {n_methods} DML methods total")
    print(f"  {n_beat_quad} methods beat the quadratic baseline ({quad_inS:.4e})")
    print(f"  {n_beat_quintic} methods beat the quintic baseline ({quintic_inS:.4e})")

    if n_beat_quintic == 0:
        print("\n  CONCLUSION (PRICE): NO DML method beats a quintic polynomial in S_0.")
        print("  The barrier price is sufficiently smooth in S_0 that polynomial regression")
        print("  with 6 parameters captures it more accurately than a 263k-parameter NN.")
        print("  This needs to be reported transparently in any paper.")

    # ========================================================================
    # CRITICAL: also check polynomial DELTA baseline (DML's actual value-add)
    # ========================================================================
    print("\n" + "=" * 80)
    print("DELTA COMPARISON — does polynomial help on Greeks?")
    print("=" * 80)
    print("Polynomial delta = derivative of price polynomial (analytical, 1D).")
    print("Each polynomial fit on full GT, then derivative evaluated.\n")
    var_d = np.var(dydx_gt)
    print(f"   {'Degree':>8}  {'price_mse':>14}  {'delta_mse':>14}  {'delta R^2':>10}")
    poly_delta_results = {}
    for deg in range(1, 7):
        coeffs = np.polyfit(x_gt, y_gt, deg)
        # Derivative coefficients
        dcoeffs = np.polyder(coeffs)
        d_pred = np.polyval(dcoeffs, x_gt)
        d_mse = np.mean((d_pred - dydx_gt) ** 2)
        poly_delta_results[deg] = d_mse
        p_mse = poly_results[deg]
        d_r2 = 1 - d_mse / var_d if var_d > 0 else 0
        print(f"   {deg:>8}  {p_mse:14.4e}  {d_mse:14.4e}  {d_r2:10.6f}")

    quintic_delta = poly_delta_results[5]
    quad_delta = poly_delta_results[2]
    print(f"\n   Anchors: quadratic-derivative-in-S_0 delta_mse = {quad_delta:.4e}, "
          f"quintic = {quintic_delta:.4e}")
    print(f"\n   {'Method':<28}  {'delta_mse':>14}  {'rel-quad-d':>10}  {'rel-quintic-d':>14}  {'beats quad-d?':>14}")

    sortable_d = []
    for method, vals in by_method.items():
        d_mean = np.mean([r["test_grad_mse"] for r in vals])
        sortable_d.append((d_mean, method))
    sortable_d.sort()
    for (d_mean, method) in sortable_d:
        rqd = d_mean / quad_delta
        rqud = d_mean / quintic_delta
        beats = "YES" if d_mean < quad_delta else "NO"
        print(f"   {method:<28}  {d_mean:14.4e}  {rqd:10.2f}x  {rqud:14.2f}x  {beats:>14}")

    # Summary for delta
    n_beat_quad_d = sum(1 for d, _ in sortable_d if d < quad_delta)
    n_beat_quintic_d = sum(1 for d, _ in sortable_d if d < quintic_delta)
    print(f"\n  {n_beat_quad_d}/{len(sortable_d)} methods beat the quadratic-derivative baseline ({quad_delta:.4e})")
    print(f"  {n_beat_quintic_d}/{len(sortable_d)} methods beat the quintic-derivative baseline ({quintic_delta:.4e})")
    if n_beat_quintic_d > 0:
        print("\n  CONCLUSION (DELTA): some DML methods DO beat polynomial-derivative on Greeks.")
        print("  DML's value-add is in delta estimation, not in price prediction.")
        print("  This is the right framing for the paper.")
    else:
        print("\n  CONCLUSION (DELTA): NO DML method beats polynomial-derivative on Greeks either.")
        print("  Polynomial regression with analytical derivative beats every NN.")
        print("  This is a strong negative finding.")


if __name__ == "__main__":
    main()
