#!/usr/bin/env python3
"""
BS down-and-out barrier — n-monitor sweep matching G&K v2 Table 2.

Runs the full method × seed grid for n_monitor ∈ {2, 4, 8, 16} and adds
polynomial baseline + warmup variants. The BS LRM score under GBM Markov
collapses to single-step (only S_0-dependent transition is the first), so
all n_monitor values use the same single-step score — the n-sweep tests
whether label paradigms (pathwise / LRM / fuzzy) and balancers (fixed-λ /
warmup) preserve their ranking as the number of discrete Dirac sources
grows.

Methods (7 total):
    vanilla, dml_fixed (pathwise), dml_warmup (pathwise),
    dml_lrm_fixed, dml_lrm_warmup,
    dml_fuzzy_fixed, dml_fuzzy_warmup

Output: results/heston_barrier_4way/bs_n_sweep/n{n}/
    bs_repro_n{n}_{method}_s{seed}.json
    polynomial_baselines.json
    analysis.json (per-n)
plus a top-level summary at:
    results/heston_barrier_4way/bs_n_sweep/n_sweep_summary.json

Usage:
    python experiments/heston_barrier_4way/run_bs_n_sweep.py --gpu 0
    python experiments/heston_barrier_4way/run_bs_n_sweep.py --gpu 0 --n-list 2 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.lrm_labels_bs_barrier import (
    lrm_barrier_bs_intermediate_only,
    pathwise_barrier_bs_intermediate_only,
    fuzzy_barrier_bs_intermediate_only,
    bs_barrier_doc_mc_reference,
)


GK_PARAMS_BASE = {
    "strike": 100.0,
    "barrier": 85.0,
    "vol": 0.20,
    "r": 0.0,
}

# T_total scales with n_monitor at G&K's convention (Δt = 1/3 fixed)
DT_PER_STEP = 1.0 / 3.0

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
    "early_stopping_patience": 200,
}

METHODS = [
    "vanilla", "dml_fixed", "dml_warmup",
    "dml_lrm_fixed", "dml_lrm_warmup",
    "dml_fuzzy_fixed", "dml_fuzzy_warmup",
]
TRAINER_METHOD_MAP = {
    "vanilla":           "vanilla",
    "dml_fixed":         "dml_fixed",
    "dml_warmup":        "dml_warmup",
    "dml_lrm_fixed":     "dml_fixed",
    "dml_lrm_warmup":    "dml_warmup",
    "dml_fuzzy_fixed":   "dml_fixed",
    "dml_fuzzy_warmup":  "dml_warmup",
}
LABEL_PARADIGM = {
    "vanilla":          "pathwise",
    "dml_fixed":        "pathwise",
    "dml_warmup":       "pathwise",
    "dml_lrm_fixed":    "lrm",
    "dml_lrm_warmup":   "lrm",
    "dml_fuzzy_fixed":  "fuzzy",
    "dml_fuzzy_warmup": "fuzzy",
}

SEEDS = [42, 123, 456, 789, 1337]
N_SAMPLES = 1024
K_PATHS = 10
GROUND_TRUTH_N_TEST = 200
GROUND_TRUTH_N_PATHS = 100_000
GROUND_TRUTH_SEED = 999
FINITE_DIFF_BUMP = 0.01


def gk_params(n_monitor: int) -> dict:
    return {**GK_PARAMS_BASE,
            "T_total": n_monitor * DT_PER_STEP,
            "n_monitor": n_monitor}


def gen_data(method: str, seed: int, n_monitor: int) -> dict:
    paradigm = LABEL_PARADIGM[method]
    params = gk_params(n_monitor)
    if paradigm == "pathwise":
        return pathwise_barrier_bs_intermediate_only(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **params,
        )
    if paradigm == "lrm":
        return lrm_barrier_bs_intermediate_only(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **params,
        )
    if paradigm == "fuzzy":
        return fuzzy_barrier_bs_intermediate_only(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **params,
        )
    raise ValueError(paradigm)


def get_dydx_key(method: str) -> str:
    return {"pathwise": "dydx_pw", "lrm": "dydx_lrm", "fuzzy": "dydx_fuzzy"}[LABEL_PARADIGM[method]]


def polynomial_baselines(x_gt: np.ndarray, y_gt: np.ndarray, d_gt: np.ndarray, max_deg: int = 6):
    out = {}
    for deg in range(1, max_deg + 1):
        coeffs = np.polyfit(x_gt, y_gt, deg)
        dcoeffs = np.polyder(coeffs)
        out[deg] = {
            "val_mse": float(np.mean((np.polyval(coeffs, x_gt) - y_gt) ** 2)),
            "delta_mse": float(np.mean((np.polyval(dcoeffs, x_gt) - d_gt) ** 2)),
        }
    return out


def run_one_n(n_monitor: int, results_root: Path) -> dict:
    out_dir = results_root / f"n{n_monitor}"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = gk_params(n_monitor)
    print(f"\n{'=' * 70}\nBS n-monitor = {n_monitor}  (T_total={params['T_total']:.4f})\n{'=' * 70}")

    # GT
    gt_npz = out_dir / "gt.npz"
    if gt_npz.exists():
        print(f"  GT cached: {gt_npz}")
        gt_npz_d = np.load(gt_npz)
        x_test = gt_npz_d["x"].reshape(-1, 1)
        y_test = gt_npz_d["y"].reshape(-1, 1)
        dydx_test = gt_npz_d["dydx"].reshape(-1, 1, 1)
        gt_x_flat = gt_npz_d["x"].flatten()
        gt_y_flat = gt_npz_d["y"].flatten()
        gt_d_flat = gt_npz_d["dydx"].flatten()
    else:
        print(f"  Computing GT (200 spots × 100k MC paths)…")
        rng_gt = np.random.RandomState(GROUND_TRUTH_SEED)
        S0_test = rng_gt.uniform(params["strike"] * 0.5,
                                  params["strike"] * 1.5,
                                  GROUND_TRUTH_N_TEST)
        t0 = time.time()
        gt = bs_barrier_doc_mc_reference(
            S0=S0_test, n_paths=GROUND_TRUTH_N_PATHS, seed=GROUND_TRUTH_SEED,
            finite_diff_bump=FINITE_DIFF_BUMP, **params,
        )
        np.savez(gt_npz,
                 x=gt["x"].flatten(), y=gt["y"].flatten(),
                 dydx=gt["dydx"].flatten() if gt["dydx"].ndim == 1 else gt["dydx"].flatten(),
                 std_err_price=gt.get("std_err_price", np.zeros_like(gt["y"].flatten())),
                 std_err_delta=gt.get("std_err_delta", np.zeros_like(gt["y"].flatten())))
        print(f"  GT saved in {time.time() - t0:.1f}s")
        gt_x_flat = gt["x"].flatten()
        gt_y_flat = gt["y"].flatten()
        gt_d_flat = gt["dydx"].flatten() if gt["dydx"].ndim == 1 else gt["dydx"].flatten()
        x_test = gt_x_flat.reshape(-1, 1)
        y_test = gt_y_flat.reshape(-1, 1)
        dydx_test = gt_d_flat.reshape(-1, 1, 1)

    # Polynomial baseline
    poly = polynomial_baselines(gt_x_flat, gt_y_flat, gt_d_flat)
    with open(out_dir / "polynomial_baselines.json", "w") as f:
        json.dump(poly, f, indent=2)
    print(f"  poly quintic: val={poly[5]['val_mse']:.3e} delta={poly[5]['delta_mse']:.3e}")

    # DML methods
    for method in METHODS:
        for seed in SEEDS:
            out_json = out_dir / f"bs_repro_n{n_monitor}_{method}_s{seed}.json"
            if out_json.exists():
                continue
            t_gen = time.time()
            data = gen_data(method, seed, n_monitor)
            dydx_key = get_dydx_key(method)
            gen_time = time.time() - t_gen

            t0 = time.time()
            try:
                result = train_single_experiment(
                    x_train=data["x"], y_train=data["y"],
                    dydx_train=data[dydx_key],
                    x_test=x_test, y_test=y_test, dydx_test=dydx_test,
                    method=TRAINER_METHOD_MAP[method], seed=seed, pbar=False,
                    **HPARAMS,
                )
                elapsed = time.time() - t0
                rec = {
                    "method": method, "seed": seed, "n_monitor": n_monitor,
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "data_gen_s": round(gen_time, 2),
                }
                with open(out_json, "w") as f:
                    json.dump(rec, f, indent=2, default=str)
                print(f"  n={n_monitor} {method} seed={seed}: "
                      f"val={result.test_value_mse:.3e} grad={result.test_grad_mse:.3e} "
                      f"({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"  ERROR n={n_monitor} {method} seed={seed}: {e}", flush=True)
                traceback.print_exc()
                with open(out_json, "w") as f:
                    json.dump({"method": method, "seed": seed, "n_monitor": n_monitor,
                                "error": str(e)}, f, indent=2)

    # Aggregate
    summary = {}
    for method in METHODS:
        v = []; g = []
        for seed in SEEDS:
            f = out_dir / f"bs_repro_n{n_monitor}_{method}_s{seed}.json"
            if not f.exists(): continue
            with open(f) as fh:
                d = json.load(fh)
            if "error" in d: continue
            v.append(d["test_value_mse"]); g.append(d["test_grad_mse"])
        if v:
            summary[method] = {
                "n_seeds": len(v),
                "value_mse_mean": float(np.mean(v)),
                "value_mse_std":  float(np.std(v)),
                "grad_mse_mean":  float(np.mean(g)),
                "grad_mse_std":   float(np.std(g)),
            }
    with open(out_dir / "analysis.json", "w") as f:
        json.dump({"n_monitor": n_monitor, "summary": summary,
                    "polynomial": poly}, f, indent=2)
    return {"n_monitor": n_monitor, "methods": summary, "polynomial": poly}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n-list", type=int, nargs="+", default=[2, 4, 8, 16])
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results_root = REPO_ROOT / "results" / "heston_barrier_4way" / "bs_n_sweep"
    results_root.mkdir(parents=True, exist_ok=True)

    overall = []
    for n in args.n_list:
        overall.append(run_one_n(n_monitor=n, results_root=results_root))

    with open(results_root / "n_sweep_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print("\n=== n-SWEEP SUMMARY ===")
    print(f"  {'n':>3}  {'method':<22}  {'val mean':>11}  {'val std':>11}  {'rel quintic':>11}")
    for o in overall:
        n = o["n_monitor"]
        qv = o["polynomial"][5]["val_mse"]
        for m, s in o["methods"].items():
            print(f"  {n:>3}  {m:<22}  {s['value_mse_mean']:>11.3e}  "
                  f"{s['value_mse_std']:>11.3e}  {s['value_mse_mean'] / qv:>10.2f}x")


if __name__ == "__main__":
    main()
