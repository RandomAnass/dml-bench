#!/usr/bin/env python3
"""
G&K v2 §3.4 BS barrier replication WITH polynomial baseline.

Setup matches G&K v2 Table 2 row n=2 EXACTLY:
    - BS GBM, σ=0.20, r=0
    - K=100 (we use K=100; G&K uses K=1.0 normalized — equivalent up to scale)
    - B=85 = 0.85·K
    - T_{i+1} - T_i = 1/3, n_monitor=2 → T_total = 2/3
    - Barrier monitored at intermediate dates ONLY (not expiry)
    - m=1024 training spots, k=10 paths/spot, 5 seeds

Methods: vanilla, dml_fixed (pathwise), dml_lrm_fixed, dml_fuzzy_fixed.
Ground truth: 100k-path MC with discrete-monitoring (matches training discretization).

Then applies polynomial baseline analysis to compare DML vs polynomial regression.
This determines whether our "polynomial beats DML" finding on Heston is also true
on G&K's BS setup — which would imply the entire DML-on-barriers literature has
a polynomial-baseline blind spot.

Usage:
    python experiments/heston_barrier_4way/run_gk_replication.py --gpu 1
    python experiments/heston_barrier_4way/run_gk_replication.py --analyze-only

References:
    - Glasserman & Karmarkar 2025/2026 v2 §3.4, Table 2.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.lrm_labels_bs_barrier import (
    lrm_barrier_bs_intermediate_only,
    pathwise_barrier_bs_intermediate_only,
    fuzzy_barrier_bs_intermediate_only,
    bs_barrier_doc_mc_reference,
)


GK_PARAMS = {
    "strike": 100.0,
    "barrier": 85.0,
    "vol": 0.20,
    "r": 0.0,
    "T_total": 2.0 / 3.0,
    "n_monitor": 2,
}

HPARAMS = {
    "n_epochs": 500,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

METHODS = ["vanilla", "dml_fixed", "dml_lrm_fixed", "dml_fuzzy_fixed"]
SEEDS = [42, 123, 456, 789, 1337]
N_SAMPLES = 1024
K_PATHS = 10
GROUND_TRUTH_N_TEST = 200
GROUND_TRUTH_N_PATHS = 100_000
GROUND_TRUTH_SEED = 999
FINITE_DIFF_BUMP = 0.01

TRAINER_METHOD_MAP = {
    "vanilla":         "vanilla",
    "dml_fixed":       "dml_fixed",
    "dml_lrm_fixed":   "dml_fixed",
    "dml_fuzzy_fixed": "dml_fixed",
}


def make_key(method, seed):
    return f"gk_repro_bs_n2_{method}_s{seed}"


def load_existing(results_dir):
    out = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
                out[d.get("key", f.stem)] = d
            except Exception:
                pass
    return out


def save_result(results_dir, key, d):
    d["key"] = key
    d["timestamp"] = datetime.now().isoformat()
    p = results_dir / f"{key}.json"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=2, default=str)
    tmp.rename(p)


def generate_data(method, seed):
    if method in ("vanilla", "dml_fixed"):
        return pathwise_barrier_bs_intermediate_only(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **GK_PARAMS,
        )
    elif method == "dml_lrm_fixed":
        return lrm_barrier_bs_intermediate_only(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **GK_PARAMS,
        )
    elif method == "dml_fuzzy_fixed":
        return fuzzy_barrier_bs_intermediate_only(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **GK_PARAMS,
        )
    else:
        raise ValueError(method)


def prepare(data, method, gt):
    if method in ("vanilla", "dml_fixed"):
        dydx_key = "dydx_pw"
    elif method == "dml_lrm_fixed":
        dydx_key = "dydx_lrm"
    elif method == "dml_fuzzy_fixed":
        dydx_key = "dydx_fuzzy"
    return {
        "x_train": data["x"], "y_train": data["y"], "dydx_train": data[dydx_key],
        "x_test": gt["x"], "y_test": gt["y"], "dydx_test": gt["dydx"],
    }


def polynomial_baselines(x_gt, y_gt, dydx_gt, max_deg=6):
    """Fit polynomials of degrees 1..max_deg and return val_mse + delta_mse."""
    out = {}
    for deg in range(1, max_deg + 1):
        coeffs = np.polyfit(x_gt, y_gt, deg)
        dcoeffs = np.polyder(coeffs)
        y_pred = np.polyval(coeffs, x_gt)
        d_pred = np.polyval(dcoeffs, x_gt)
        out[deg] = {
            "val_mse": float(np.mean((y_pred - y_gt) ** 2)),
            "delta_mse": float(np.mean((d_pred - dydx_gt) ** 2)),
        }
    return out


def run_gk(results_dir, existing, resume):
    print("\n" + "=" * 70)
    print("G&K v2 §3.4 BS BARRIER REPLICATION (n_monitor=2 setup)")
    print("=" * 70)
    print(f"GK_PARAMS: {GK_PARAMS}")
    print(f"Methods: {METHODS}")
    print(f"Seeds: {SEEDS}")

    print(f"\nGenerating BS barrier MC ground truth at matched discretization "
          f"(n_monitor=2, n_paths={GROUND_TRUTH_N_PATHS})...")
    t_gt = time.time()
    rng_gt = np.random.RandomState(GROUND_TRUTH_SEED)
    S0_test = rng_gt.uniform(
        GK_PARAMS["strike"] * 0.5, GK_PARAMS["strike"] * 1.5, GROUND_TRUTH_N_TEST,
    )
    gt = bs_barrier_doc_mc_reference(
        S0=S0_test, n_paths=GROUND_TRUTH_N_PATHS, seed=GROUND_TRUTH_SEED,
        finite_diff_bump=FINITE_DIFF_BUMP, **GK_PARAMS,
    )
    print(f"  Ready in {time.time() - t_gt:.1f}s. mean SE_price={gt['std_err_price'].mean():.4e}, "
          f"mean SE_delta={gt['std_err_delta'].mean():.4e}")

    # Polynomial baseline on the SAME ground truth used for DML eval
    poly_results = polynomial_baselines(
        gt["x"].flatten(), gt["y"].flatten(), gt["dydx"].flatten(),
    )
    save_result(results_dir, "polynomial_baselines", {
        "method": "polynomial_baselines",
        "results_per_degree": poly_results,
        "model": "black_scholes_gbm",
        "gk_params": dict(GK_PARAMS),
    })
    print(f"\nPolynomial baselines on the SAME GT:")
    print(f"  {'Degree':>8}  {'val_mse':>14}  {'delta_mse':>14}")
    for deg, r in poly_results.items():
        print(f"  {deg:>8}  {r['val_mse']:14.4e}  {r['delta_mse']:14.4e}")

    for seed in SEEDS:
        for method in METHODS:
            key = make_key(method, seed)
            if resume and key in existing:
                print(f"  SKIP: {method} seed={seed}")
                continue
            print(f"\n--- {method} seed={seed} ---")
            data = generate_data(method, seed)
            data_split = prepare(data, method, gt)

            t0 = time.time()
            try:
                trainer_method = TRAINER_METHOD_MAP[method]
                result = train_single_experiment(
                    x_train=data_split["x_train"], y_train=data_split["y_train"],
                    dydx_train=data_split["dydx_train"],
                    x_test=data_split["x_test"], y_test=data_split["y_test"],
                    dydx_test=data_split["dydx_test"],
                    method=trainer_method, seed=seed, pbar=False, **HPARAMS,
                )
                elapsed = time.time() - t0
                save_result(results_dir, key, {
                    "method": method, "trainer_method": trainer_method, "seed": seed,
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "hparams": dict(HPARAMS), "gk_params": dict(GK_PARAMS),
                })
                print(f"  val_mse={result.test_value_mse:.4e}, "
                      f"grad_mse={result.test_grad_mse:.4e}, ep={result.best_epoch}, t={elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()


def analyze(results_dir):
    existing = load_existing(results_dir)
    if not existing:
        print("No results.")
        return
    print("\n" + "=" * 80)
    print("G&K REPLICATION RESULTS")
    print("=" * 80)
    poly = existing.get("polynomial_baselines", {}).get("results_per_degree", {})
    if poly:
        print(f"  Polynomial baselines (in-sample on GT, n_test=200):")
        for deg, r in poly.items():
            print(f"    deg={deg}: price_mse={r['val_mse']:.4e}  delta_mse={r['delta_mse']:.4e}")

    by_method = {}
    for k, r in existing.items():
        if r.get("method") == "polynomial_baselines":
            continue
        by_method.setdefault(r["method"], []).append(r)

    print(f"\n  DML methods (mean ± std across {len(SEEDS)} seeds):")
    print(f"  {'Method':<25} {'val_mse mean':>14} {'val_mse std':>14} "
          f"{'grad_mse mean':>14} {'grad_mse std':>14} {'n':>3}")
    for method in METHODS:
        if method not in by_method:
            continue
        vals = by_method[method]
        v = [r["test_value_mse"] for r in vals]
        g = [r["test_grad_mse"] for r in vals]
        print(f"  {method:<25} {np.mean(v):14.4e} {np.std(v):14.4e} "
              f"{np.mean(g):14.4e} {np.std(g):14.4e} {len(vals):>3}")

    if poly:
        # Compare DML vs polynomial
        quintic = poly.get("5", poly.get(5, {})).get("val_mse")
        quintic_d = poly.get("5", poly.get(5, {})).get("delta_mse")
        if quintic is not None:
            print(f"\n  vs quintic baseline (val_mse={quintic:.4e}, delta_mse={quintic_d:.4e}):")
            for method in METHODS:
                if method not in by_method:
                    continue
                v = np.mean([r["test_value_mse"] for r in by_method[method]])
                d = np.mean([r["test_grad_mse"] for r in by_method[method]])
                print(f"    {method:<25} val: {v/quintic:.2f}x  delta: {d/quintic_d:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = Path("results/heston_barrier_4way/gk_replication")
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir)
        return
    existing = load_existing(results_dir) if args.resume else {}
    run_gk(results_dir, existing, args.resume)
    analyze(results_dir)
    print("\nG&K replication complete!")


if __name__ == "__main__":
    main()
