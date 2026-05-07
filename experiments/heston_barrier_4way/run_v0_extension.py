#!/usr/bin/env python3
"""
Heston barrier with 2-D inputs (S_0, V_0) — V_0 extension experiment.

Tests whether the polynomial-baseline-beats-DML finding survives when V_0
is also a function input. In 2-D the polynomial in (S_0, V_0) has many more
parameters and the function f(S_0, V_0) is significantly less smooth
(vol-of-vol curvature kicks in).

Methods:
    vanilla, dml_pathwise_warmup, dml_fuzzy_warmup, dml_bel_warmup
    (a tight subset; we already established the warmup balancer's universal
     dominance in the 1-D experiment.)

Polynomial baseline: degree-d 2-D polynomial in (S_0, V_0), d ∈ {1..6}.
Compute as np.polynomial.polynomial.polyvander2d / lstsq.

Output:
    results/heston_barrier_4way/v0_extension/
        gt.npz                         # 2-D GT on (S0, V0) uniform grid
        {method}_seed{s}.json          # per-seed test metrics
        polynomial_2d_baselines.json   # 2-D polynomial fit at each degree
        analysis.json                  # aggregated mean/std

Usage:
    python experiments/heston_barrier_4way/run_v0_extension.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dml_benchmark.heston_2d_inputs import (
    pathwise_barrier_heston_2d,
    fuzzy_barrier_heston_2d,
    bel_barrier_heston_2d,
)
from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference
from dml_benchmark.trainer import train_single_experiment

# Heston params (same as 1-D pilot, except v0 now varies per spot)
HESTON_PARAMS = {
    "strike": 1.0,
    "barrier": 0.85,
    "kappa": 1.0,
    "theta": 0.04,
    "sigma_v": 0.15,
    "rho": -0.7,
    "r": 0.0,
    "T1": 1.0 / 3.0,
    "T2": 2.0 / 3.0,
    "n_substeps_to_T1": 84,
    "n_substeps_T1_to_T2": 84,
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
    "early_stopping_patience": 200,
}

N_SAMPLES = 1024
K_PATHS = 10
SEEDS = [42, 123, 456, 789, 1337]

# Wide spot range (matches G&K v2; we showed wide is where DML can win in 1-D)
SPOT_LOW_MULT = 0.5
SPOT_HIGH_MULT = 1.5
V0_LOW_MULT = 0.5
V0_HIGH_MULT = 1.5

# Grid for the 2-D ground truth
N_GT_S = 30
N_GT_V = 30
GT_PATHS = 50_000
GT_SEED = 999

METHODS = {
    "vanilla":              ("pathwise", "vanilla"),
    "dml_pathwise_warmup":  ("pathwise", "dml_warmup"),
    "dml_fuzzy_warmup":     ("fuzzy",    "dml_warmup"),
    "dml_bel_warmup":       ("bel",      "dml_warmup"),
}


def gen_data(label_method: str, seed: int) -> dict:
    if label_method == "pathwise":
        return pathwise_barrier_heston_2d(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            v0_low_mult=V0_LOW_MULT, v0_high_mult=V0_HIGH_MULT,
            **HESTON_PARAMS,
        )
    if label_method == "fuzzy":
        return fuzzy_barrier_heston_2d(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            v0_low_mult=V0_LOW_MULT, v0_high_mult=V0_HIGH_MULT,
            **HESTON_PARAMS,
        )
    if label_method == "bel":
        return bel_barrier_heston_2d(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            v0_low_mult=V0_LOW_MULT, v0_high_mult=V0_HIGH_MULT,
            **HESTON_PARAMS,
        )
    raise ValueError(label_method)


def get_dydx_key(label_method: str) -> str:
    return {"pathwise": "dydx_pw", "fuzzy": "dydx_fuzzy", "bel": "dydx_lrm"}[label_method]


def compute_gt_2d(out: Path) -> dict:
    if out.exists():
        print(f"  GT cached: {out}")
        return dict(np.load(out))
    print(f"  Computing 2-D GT: {N_GT_S} S0 × {N_GT_V} V0 = {N_GT_S * N_GT_V} spots, "
          f"{GT_PATHS} paths each…")
    K = HESTON_PARAMS["strike"]
    theta = HESTON_PARAMS["theta"]
    S0_grid = np.linspace(K * SPOT_LOW_MULT, K * SPOT_HIGH_MULT, N_GT_S)
    V0_grid = np.linspace(theta * V0_LOW_MULT, theta * V0_HIGH_MULT, N_GT_V)
    S0_mesh, V0_mesh = np.meshgrid(S0_grid, V0_grid, indexing="ij")
    S0_flat = S0_mesh.flatten()
    V0_flat = V0_mesh.flatten()

    t0 = time.time()
    gt = heston_barrier_doc_mc_reference(
        S0=S0_flat,
        v0=V0_flat,                  # array → 2-D mode
        n_paths=GT_PATHS, seed=GT_SEED,
        **{k: v for k, v in HESTON_PARAMS.items() if k != "v0"},
    )
    np.savez(
        out,
        x=gt["x"],                   # (n, 2)
        y=gt["y"].flatten(),
        dydx=gt["dydx"][:, 0, 0],
        std_err_price=gt["std_err_price"],
        std_err_delta=gt["std_err_delta"],
        S0_grid=S0_grid, V0_grid=V0_grid,
    )
    print(f"  GT saved in {time.time() - t0:.1f}s -> {out}")
    return dict(np.load(out))


def fit_2d_polynomial(x: np.ndarray, y: np.ndarray, deg: int):
    """Fit a 2-D polynomial in (S_0, V_0) of total degree deg via least squares."""
    S0 = x[:, 0]; V0 = x[:, 1]
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            cols.append((S0 ** i) * (V0 ** j))
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs, A


def eval_2d_polynomial(x: np.ndarray, coeffs: np.ndarray, deg: int) -> np.ndarray:
    S0 = x[:, 0]; V0 = x[:, 1]
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            cols.append((S0 ** i) * (V0 ** j))
    A = np.column_stack(cols)
    return A @ coeffs


def eval_2d_polynomial_dS(x: np.ndarray, coeffs: np.ndarray, deg: int) -> np.ndarray:
    """∂P/∂S_0 of the 2-D polynomial fit."""
    S0 = x[:, 0]; V0 = x[:, 1]
    cols = []
    idx = 0
    grad_cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            if i == 0:
                grad_cols.append(np.zeros_like(S0))
            else:
                grad_cols.append(i * (S0 ** (i - 1)) * (V0 ** j))
            idx += 1
    A_dS = np.column_stack(grad_cols)
    return A_dS @ coeffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = REPO_ROOT / "results" / "heston_barrier_4way" / "v0_extension"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = compute_gt_2d(out_dir / "gt.npz")
    x_test = gt["x"]                # (n, 2)
    y_test = gt["y"].reshape(-1, 1)
    dydx_test = np.zeros((x_test.shape[0], 1, 2))
    dydx_test[:, 0, 0] = gt["dydx"]

    # ---------- Polynomial baseline (2-D) ----------
    poly_results = {}
    for deg in range(1, 7):
        coeffs, _ = fit_2d_polynomial(x_test, gt["y"], deg)
        y_pred = eval_2d_polynomial(x_test, coeffs, deg)
        d_pred = eval_2d_polynomial_dS(x_test, coeffs, deg)
        p_mse = float(np.mean((y_pred - gt["y"]) ** 2))
        d_mse = float(np.mean((d_pred - gt["dydx"]) ** 2))
        n_params = sum(1 for i in range(deg + 1) for j in range(deg + 1 - i))
        poly_results[deg] = {"price_mse": p_mse, "delta_mse": d_mse, "n_params": n_params}
        print(f"  poly2d deg={deg}: n_params={n_params}, price_mse={p_mse:.4e}, delta_mse={d_mse:.4e}")
    with open(out_dir / "polynomial_2d_baselines.json", "w") as f:
        json.dump(poly_results, f, indent=2)

    if args.analyze_only:
        return

    # ---------- DML methods ----------
    for method, (label_method, trainer_method) in METHODS.items():
        for seed in SEEDS:
            out_json = out_dir / f"{method}_seed{seed}.json"
            if out_json.exists():
                print(f"  SKIP (cached): {method} seed={seed}")
                continue
            t_gen = time.time()
            data = gen_data(label_method, seed)
            dydx_key = get_dydx_key(label_method)
            gen_time = time.time() - t_gen

            t0 = time.time()
            try:
                result = train_single_experiment(
                    x_train=data["x"],
                    y_train=data["y"],
                    dydx_train=data[dydx_key],
                    x_test=x_test,
                    y_test=y_test,
                    dydx_test=dydx_test,
                    method=trainer_method,
                    seed=seed,
                    pbar=False,
                    **HPARAMS,
                )
                elapsed = time.time() - t0
                rec = {
                    "method": method,
                    "label_method": label_method,
                    "trainer_method": trainer_method,
                    "seed": seed,
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "data_gen_s": round(gen_time, 2),
                    "hparams": HPARAMS,
                    "heston_params": HESTON_PARAMS,
                    "spot_range": [SPOT_LOW_MULT, SPOT_HIGH_MULT],
                    "v0_range": [V0_LOW_MULT, V0_HIGH_MULT],
                }
                with open(out_json, "w") as f:
                    json.dump(rec, f, indent=2, default=str)
                print(f"  {method} seed={seed}: val_mse={result.test_value_mse:.4e}, "
                      f"grad_mse={result.test_grad_mse:.4e} ({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"  ERROR: {method} seed={seed}: {e}", flush=True)
                with open(out_json, "w") as f:
                    json.dump({"method": method, "seed": seed, "error": str(e)}, f, indent=2)

    # ---------- Aggregation ----------
    summary = {"methods": {}, "polynomial_2d": poly_results}
    for method in METHODS:
        rows = []
        for seed in SEEDS:
            f = out_dir / f"{method}_seed{seed}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                d = json.load(fh)
            if "error" in d: continue
            rows.append(d)
        if rows:
            v = [r["test_value_mse"] for r in rows]
            g = [r["test_grad_mse"] for r in rows]
            summary["methods"][method] = {
                "n_seeds": len(rows),
                "test_value_mse_mean": float(np.mean(v)),
                "test_value_mse_std": float(np.std(v)),
                "test_grad_mse_mean": float(np.mean(g)),
                "test_grad_mse_std": float(np.std(g)),
            }
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {out_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
