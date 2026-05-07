#!/usr/bin/env python3
r"""Polynomial-in-inputs baseline on the 1-D synthetic block.

Reviewer R-NeurIPS-Sim flagged that the polynomial baseline currently
appears only on the Heston path-dependent barrier
(`experiments/heston_barrier_4way/analyze_polynomial_baselines.py` →
\Cref{tab:heston-barrier-multi-seed}).  We extend that protocol to the
1-D synthetic functions where polynomial regression is informative:

  - poly_trig     (smooth + low-amplitude trig perturbation, [-1, 1])
  - trig          (high-frequency sinusoid, [-π, π])
  - black_scholes (Black–Scholes call price, smooth call surface)

Why d=1 only:
  At d ≥ 2 the polynomial parameter count blows up (d=20 quintic has
  C(25,5)=53,130 params on n=1024 data — overfit-bound).  The d=1 cells
  are matched to the Tier-3 grid (see `run_full_benchmark.py`,
  build_tier3_experiments §A) where the existing DML/baselines results
  live in `results/tier3_benchmark/*_d1_*.json`.  We compare against the
  same (seed, n_samples) splits.

Protocol (mirrors the Heston-barrier polynomial baseline):
  1. Regenerate (x, y, dydx) deterministically with `generate_data` and
     split via `train_test_split(train_ratio=0.8, seed=seed)`.  Identical
     to the existing tier3 NN/baseline runs.
  2. Fit a univariate polynomial of degrees 1..6 on (x_train, y_train).
  3. Evaluate price MSE and gradient MSE on (x_test, y_test, dydx_test).
     Gradient is the derivative of the fitted polynomial, evaluated
     analytically (numpy.polyder).
  4. Save one JSON per (domain, n_samples, seed) with all degrees.
  5. Aggregate the best-degree polynomial against the existing best DML
     test_value_mse / test_grad_mse from `results/tier3_benchmark/`.

Usage:
    python experiments/polynomial_baselines/run_synthetic.py
    python experiments/polynomial_baselines/run_synthetic.py --resume
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.functions import generate_data, train_test_split


# Match the canonical Tier-3 grid for d=1 noise=0 cells (see
# run_full_benchmark.py build_tier3_experiments).
SEEDS = [42, 123, 456, 789, 1000]
N_SAMPLES_LIST = [256, 512, 1024, 2048, 4096, 8192]
DOMAINS = ["poly_trig", "trig", "black_scholes"]
# Heston-barrier baseline used degrees 1..6 because S_0 ∈ [0.7K, 1.3K]
# is well-conditioned and the price is near-linear; here we extend to
# degree 12 because (a) trig is high-frequency and benefits from higher
# order, (b) Black–Scholes inputs span [50, 150] (call surface saturates
# in S/K, so the polynomial converges only at higher order even after
# standardisation).  We standardise inputs before fitting (chain rule
# applied to the gradient prediction) — this is the universal safe
# choice for univariate polyfit at large degree.
DEGREES = list(range(1, 13))  # 1..12

RESULTS_DIR = Path("results/polynomial_baselines")
TIER3_DIR = Path("results/tier3_benchmark")


def fit_and_eval(x_train, y_train, x_test, y_test, dydx_test, degree):
    """Fit a univariate polynomial of given degree and evaluate.

    Inputs are standardised (subtract train-mean, divide by train-std)
    before fitting.  The chain rule is applied to the gradient
    prediction so that the polynomial derivative is reported in the
    original input units (matching dydx_test).

    Returns dict with price_mse, grad_mse, n_params.
    """
    x_tr_raw = np.asarray(x_train).flatten()
    y_tr = np.asarray(y_train).flatten()
    x_te_raw = np.asarray(x_test).flatten()
    y_te = np.asarray(y_test).flatten()
    d_te = np.asarray(dydx_test).flatten()

    mu = float(x_tr_raw.mean())
    sd = float(x_tr_raw.std())
    if sd < 1e-12:  # degenerate constant input
        sd = 1.0
    x_tr = (x_tr_raw - mu) / sd
    x_te = (x_te_raw - mu) / sd

    # numpy.polyfit returns coefficients in descending order; polyder
    # gives the derivative coefficients analytically.
    coeffs = np.polyfit(x_tr, y_tr, degree)
    y_pred = np.polyval(coeffs, x_te)
    dcoeffs = np.polyder(coeffs)
    # chain rule: d/dx_raw = (1/sd) * d/dx_std
    d_pred = np.polyval(dcoeffs, x_te) / sd

    return {
        "price_mse": float(np.mean((y_pred - y_te) ** 2)),
        "grad_mse": float(np.mean((d_pred - d_te) ** 2)),
        "n_params": degree + 1,
        "x_mean_train": mu,
        "x_std_train": sd,
    }


def run_one_cell(domain: str, n_samples: int, seed: int) -> dict:
    """Run polynomial baseline on a single (domain, n_samples, seed) cell."""
    t0 = time.time()
    data = generate_data(domain, n_dim=1, n_samples=n_samples, seed=seed)
    train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)

    x_tr = train_data.x  # (n_train, 1)
    y_tr = train_data.y  # (n_train, 1)
    x_te = test_data.x   # (n_test, 1)
    y_te = test_data.y   # (n_test, 1)
    d_te = test_data.dydx  # (n_test, 1, 1)

    poly_results = {}
    for deg in DEGREES:
        poly_results[deg] = fit_and_eval(x_tr, y_tr, x_te, y_te, d_te, deg)

    return {
        "domain": domain,
        "func_type": domain,  # alias used by tier3 schema
        "dim": 1,
        "n_samples": n_samples,
        "n_train": x_tr.shape[0],
        "n_test": x_te.shape[0],
        "noise_level": 0.0,
        "seed": seed,
        "polynomial_results": {str(k): v for k, v in poly_results.items()},
        "elapsed_s": time.time() - t0,
    }


def make_key(domain, n_samples, seed):
    return f"{domain}_d1_n{n_samples}_noise0.0_s{seed}_polynomial"


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
        for f in out_dir.glob("*_polynomial.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                keys.add(data.get("key", f.stem))
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser(description="Polynomial baselines on 1D synthetic block")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already saved in results/polynomial_baselines")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--domains", nargs="+", default=DOMAINS,
                    help="Subset of domains to run")
    ap.add_argument("--n-samples", nargs="+", type=int, default=N_SAMPLES_LIST,
                    help="Subset of n_samples to run")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                    help="Subset of seeds to run")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_dir) if args.resume else set()

    cells = [(d, n, s) for d in args.domains
                       for n in args.n_samples
                       for s in args.seeds]
    print(f"Total cells: {len(cells)} ({len(args.domains)} domains × "
          f"{len(args.n_samples)} sizes × {len(args.seeds)} seeds)")
    if args.resume and existing:
        print(f"Resume: skipping {len(existing)} existing cells")

    completed = 0
    skipped = 0
    t0 = time.time()
    for (domain, n_samples, seed) in cells:
        key = make_key(domain, n_samples, seed)
        if key in existing:
            skipped += 1
            continue
        try:
            result = run_one_cell(domain, n_samples, seed)
            save_result(out_dir, key, result)
            completed += 1
            best_deg = min(DEGREES, key=lambda d: result["polynomial_results"][str(d)]["price_mse"])
            best_price = result["polynomial_results"][str(best_deg)]["price_mse"]
            print(f"  {key}  best_deg={best_deg} price_mse={best_price:.3e}  "
                  f"({result['elapsed_s']:.2f}s)")
        except Exception as e:
            print(f"  {key}  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {completed} completed, {skipped} skipped, "
          f"total wall-clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
