#!/usr/bin/env python3
"""Polynomial-in-inputs baseline on the SPY-options BS-target panel.

Extends the polynomial baseline protocol from the Heston-barrier domain
(`experiments/heston_barrier_4way/analyze_polynomial_baselines.py`) to
the 4-D SPY-options regression cell of `results/spy_options_temporal/`.

Inputs: (moneyness = S/K, T, r, σ_IV)  → 4-D
Target: BS-formula call price normalised by K (target_mode="bs_price",
        the same setup as the cited DML run; see
        `experiments/real_data_spy/run_spy_experiment.py`).
Greeks: analytical BS first partials (delta, theta, rho, vega).

Caveat — total polynomial parameter count vs. data:
  d=4, multivariate polynomial of total degree p has C(d+p, p) features.
  p=2 →  15  params
  p=3 →  35  params
  p=4 →  70  params
  p=5 → 126  params
  p=6 → 210  params
At n_train = 10 000 (the canonical SPY temporal-split size) all degrees
≤ 6 are well-conditioned in OLS.  We add a small Ridge regularisation
(alpha=1e-8) to keep the high-degree fits numerically stable on
near-collinear monomials.

Protocol:
  1. Load SPY data with `load_spy_data(n_train=10000, n_test=10000,
     split_mode="temporal", target_mode="bs_price")` — identical to the
     cited DML run (see `results/spy_options_temporal/spy_n10000_*.json`).
  2. Fit a multivariate polynomial of total degrees 1..6 on x_train.
  3. Evaluate price MSE on (x_test, y_test).
     For gradient MSE we differentiate the polynomial analytically by
     hand-rolling derivatives of `PolynomialFeatures` columns (sklearn
     does not expose that), evaluated at x_test. Compare to the
     analytical BS Greeks `dydx_test`.
  4. Save one JSON per (n_train, seed) with all degrees.

Usage:
    python experiments/polynomial_baselines/run_spy.py
    python experiments/polynomial_baselines/run_spy.py --resume
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.real_data_spy.spy_data_loader import load_spy_data


# Match `experiments/real_data_spy/run_spy_experiment.py` configuration
# (SEEDS / TRAIN_SIZES / TEST_SIZE).
SEEDS = [42, 123, 456, 789, 1337]  # 5 seeds, sufficient for paired comparison
TRAIN_SIZES = [10_000]              # primary cell cited in §5.1
TEST_SIZE = 10_000
DEGREES = list(range(1, 7))         # 1..6 — quintic = 126 params, hexic = 210
RIDGE_ALPHA = 1e-8                  # near-OLS — only for numerical stability

RESULTS_DIR = Path("results/polynomial_baselines")


def polynomial_derivative(poly_features: PolynomialFeatures, x: np.ndarray) -> np.ndarray:
    """Compute the Jacobian of PolynomialFeatures(x) w.r.t. each input column.

    PolynomialFeatures stores each output as a monomial via `powers_` —
    a (n_output_features, n_input_features) integer-power matrix. The
    derivative of x_1^{p_1} x_2^{p_2} ... x_d^{p_d} w.r.t. x_j is
    p_j * x_j^{p_j-1} * prod_{k!=j} x_k^{p_k}.

    Returns: array of shape (n_samples, n_output_features, n_input_features).
    """
    powers = poly_features.powers_  # (n_out, n_in)
    n_samples, n_in = x.shape
    n_out = powers.shape[0]
    jac = np.zeros((n_samples, n_out, n_in))

    for j in range(n_in):
        for k in range(n_out):
            p = powers[k]
            if p[j] == 0:
                continue  # derivative of constant w.r.t. x_j is 0
            new_p = p.copy()
            new_p[j] -= 1
            coef = float(p[j])
            term = coef * np.prod(x ** new_p, axis=1)
            jac[:, k, j] = term
    return jac


def fit_and_eval(x_train, y_train, x_test, y_test, dydx_test, degree):
    """Fit a multivariate polynomial of given total degree and evaluate.

    Inputs are per-feature standardised (subtract train-mean, divide by
    train-std) before generating polynomial features.  Gradient
    predictions are mapped back to the original-input scale via the
    chain rule (divide each output column by sd[j]).

    Returns dict with price_mse, grad_mse, n_params (per pricer head).
    """
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0  # guard against degenerate columns
    x_tr = (x_train - mu) / sd
    x_te = (x_test - mu) / sd

    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Phi_train = pf.fit_transform(x_tr)
    Phi_test = pf.transform(x_te)
    n_features = Phi_train.shape[1]

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)  # bias is in Phi already
    model.fit(Phi_train, y_train.ravel())
    y_pred = model.predict(Phi_test)
    price_mse = float(np.mean((y_pred - y_test.ravel()) ** 2))

    # Gradient prediction: derivative of fitted polynomial at x_te.
    # NOTE: derivative is w.r.t. STANDARDISED inputs; chain rule
    # divides by sd to recover dydx in the original input units.
    jac_test = polynomial_derivative(pf, x_te)  # (n_test, n_feat, d)
    coef = model.coef_  # shape (n_features,)
    dydx_pred_std = np.einsum("k,nkj->nj", coef, jac_test)
    dydx_pred = dydx_pred_std / sd.ravel()      # (n_test, d)
    # dydx_test shape: (n_test, 1, d)
    dydx_true = np.asarray(dydx_test).reshape(dydx_pred.shape)
    grad_mse = float(np.mean((dydx_pred - dydx_true) ** 2))

    return {
        "price_mse": price_mse,
        "grad_mse": grad_mse,
        "n_params": int(n_features),
        "x_mean_train": mu.ravel().tolist(),
        "x_std_train": sd.ravel().tolist(),
    }


def run_one_cell(n_train: int, seed: int) -> dict:
    """Polynomial baseline on a single (n_train, seed) SPY temporal cell."""
    t0 = time.time()
    spy = load_spy_data(
        n_train=n_train, n_test=TEST_SIZE,
        include_volume=False, stratify_by_moneyness=True,
        seed=seed, split_mode="temporal",
        target_mode="bs_price",
    )
    x_train = spy["x_train"]   # (n_train, 4)
    y_train = spy["y_train"]   # (n_train, 1)
    x_test = spy["x_test"]     # (n_test, 4)
    y_test = spy["y_test"]     # (n_test, 1)
    dydx_test = spy["dydx_test"]  # (n_test, 1, 4)

    poly_results = {}
    for deg in DEGREES:
        try:
            poly_results[deg] = fit_and_eval(x_train, y_train, x_test, y_test, dydx_test, deg)
        except Exception as e:
            poly_results[deg] = {"error": str(e)}

    return {
        "domain": "spy_bs_target",
        "dataset": "spy_options",
        "split_mode": "temporal",
        "target_mode": "bs_price",
        "dim": x_train.shape[1],
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "seed": seed,
        "polynomial_results": {str(k): v for k, v in poly_results.items()},
        "elapsed_s": time.time() - t0,
    }


def make_key(n_train, seed):
    return f"spy_n{n_train}_s{seed}_polynomial"


def save_result(out_dir: Path, key: str, result: dict):
    result["key"] = key
    result["timestamp"] = datetime.now().isoformat()
    path = out_dir / f"{key}.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    tmp_path.rename(path)


def load_existing(out_dir: Path) -> set:
    keys = set()
    if out_dir.exists():
        for f in out_dir.glob("spy_n*_polynomial.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                keys.add(data.get("key", f.stem))
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser(description="SPY polynomial baseline")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-train", nargs="+", type=int, default=TRAIN_SIZES)
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_dir) if args.resume else set()

    cells = [(n, s) for n in args.n_train for s in args.seeds]
    print(f"SPY polynomial cells: {len(cells)}  ({len(args.n_train)} sizes × {len(args.seeds)} seeds)")
    if args.resume and existing:
        print(f"Resume: skipping {len(existing)} existing cells")

    completed = 0
    skipped = 0
    t0 = time.time()
    for (n_train, seed) in cells:
        key = make_key(n_train, seed)
        if key in existing:
            skipped += 1
            continue
        try:
            result = run_one_cell(n_train, seed)
            save_result(out_dir, key, result)
            completed += 1
            best_deg = min(
                DEGREES,
                key=lambda d: result["polynomial_results"][str(d)].get("price_mse", float("inf")),
            )
            best_price = result["polynomial_results"][str(best_deg)].get("price_mse", float("nan"))
            print(f"  {key}  best_deg={best_deg} price_mse={best_price:.3e}  "
                  f"({result['elapsed_s']:.1f}s)")
        except Exception as e:
            print(f"  {key}  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {completed} completed, {skipped} skipped, "
          f"total wall-clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
