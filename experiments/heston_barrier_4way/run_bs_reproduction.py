#!/usr/bin/env python3
"""
BS barrier reproduction (= run #2 / VALIDATION).

Reproduces G&K v2 §3.4 / Table 2 row n=2: BS down-and-out call with single
intermediate barrier check at T_1=1/3, expiry T_2=2/3. Compare our LRM
single-step price RMSE to G&K's reported value (~0.022).

If our pipeline matches G&K within ~20%, this confirms:
  (a) the LRM single-step formula is implemented correctly under BS
  (b) the Heston-barrier pathwise-NOT-catastrophic finding (vs G&K's
      BS-barrier pathwise-catastrophic finding) is a real BS-vs-Heston regime
      difference, not a pipeline bug

Setup (matches G&K v2 §3.4 + Table 2 row n=2):
    - σ = 0.20 (constant, BS GBM)
    - T = 1.0 (full window; barrier check at T_1=1/3, expiry T_2=2/3)
    - K = 1.0 spot, B = 0.85
    - m = 1024 training spots, k = 10 paths per spot
    - 5 seeds: [42, 123, 456, 789, 1337]
    - 4 methods: vanilla, dml_fixed (pathwise), dml_lrm_fixed (LRM single-step), dml_fuzzy_fixed
    - Total: 4 × 5 = 20 runs

Note: G&K v2 used k=100 paths per spot for barrier (vs our k=10 for consistency
with the existing benchmark). At k=10 our LRM variance will be ~10x higher
than G&K's, so we expect Price RMSE ~ 0.022 × √10 ≈ 0.07 — verify this is
consistent with their Table 2 trend.

Usage:
    python experiments/heston_barrier_4way/run_bs_reproduction.py --gpu 1
    python experiments/heston_barrier_4way/run_bs_reproduction.py --gpu 1 --resume

Expected runtime: ~30-45 min on 1 A6000.

References:
    - Glasserman & Karmarkar (2025/2026) v2 §3.4, Table 2.
    - Reiner & Rubinstein (1991) — closed-form barrier price (used as ground truth).
    - Black & Scholes (1973), Hull (2018) Ch. 26.
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
from dml_benchmark.lrm_labels import lrm_barrier_bs
from dml_benchmark.fuzzy_smoothing import fuzzy_barrier_bs
from dml_benchmark.high_fidelity_references import (
    barrier_bs_analytical_delta,
    _barrier_bs_price,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# G&K v2 Table 2 row n=2 setup (BS down-and-out call)
BS_PARAMS = {
    "strike": 100.0,
    "barrier": 85.0,
    "vol": 0.20,
    "r": 0.0,
    "T": 1.0,
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
GROUND_TRUTH_SEED = 999

TRAINER_METHOD_MAP = {
    "vanilla":          "vanilla",
    "dml_fixed":        "dml_fixed",
    "dml_lrm_fixed":    "dml_fixed",
    "dml_fuzzy_fixed":  "dml_fixed",
}


# ============================================================================
# UTILITIES (same pattern as run_pilot.py)
# ============================================================================

def make_key(method: str, seed: int) -> str:
    return f"bs_barrier_doc_{method}_s{seed}"


def load_existing(results_dir: Path) -> dict:
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            if f.name in ("summary.json", "analysis.json", "ground_truth.json"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    existing[data.get("key", f.stem)] = data
            except Exception:
                pass
    return existing


def save_result(results_dir: Path, key: str, result_dict: dict):
    result_dict["key"] = key
    result_dict["timestamp"] = datetime.now().isoformat()
    path = results_dir / f"{key}.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp_path.rename(path)


def generate_bs_ground_truth(n_test: int, seed: int) -> dict:
    """Reiner-Rubinstein closed-form ground truth (Hull Ch.26, our existing
    high_fidelity_references.barrier_bs_analytical_delta)."""
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(BS_PARAMS["strike"] * 0.5, BS_PARAMS["strike"] * 1.5, n_test)
    price = _barrier_bs_price(
        S0, K=BS_PARAMS["strike"], B=BS_PARAMS["barrier"],
        sigma=BS_PARAMS["vol"], r=BS_PARAMS["r"], T=BS_PARAMS["T"],
    )
    delta = barrier_bs_analytical_delta(
        S0, strike=BS_PARAMS["strike"], barrier=BS_PARAMS["barrier"],
        vol=BS_PARAMS["vol"], r=BS_PARAMS["r"], T=BS_PARAMS["T"],
    )
    return {
        "x": S0.reshape(n_test, 1),
        "y": price.reshape(n_test, 1),
        "dydx": delta.reshape(n_test, 1, 1),
        "method": "reiner_rubinstein_closed_form",
    }


def _pathwise_barrier_bs(seed: int) -> dict:
    """Pathwise (biased) labels for BS barrier — differentiate through the
    discontinuous payoff. Misses the Dirac at the barrier."""
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(BS_PARAMS["strike"] * 0.5, BS_PARAMS["strike"] * 1.5, (N_SAMPLES, 1))
    n_steps = 2  # match lrm_barrier_bs default monitoring
    dt = BS_PARAMS["T"] / n_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-BS_PARAMS["r"] * BS_PARAMS["T"])

    y_all = np.zeros((N_SAMPLES, K_PATHS))
    dydx_all = np.zeros((N_SAMPLES, K_PATHS))

    for p in range(K_PATHS):
        Z_all = rng.standard_normal((N_SAMPLES, n_steps))
        S = S0.copy()
        alive = np.ones(N_SAMPLES, dtype=bool)
        for step in range(n_steps):
            Z_step = Z_all[:, step:step + 1]
            S = S * np.exp(
                (BS_PARAMS["r"] - 0.5 * BS_PARAMS["vol"] ** 2) * dt
                + BS_PARAMS["vol"] * sqrt_dt * Z_step
            )
            alive &= (S.flatten() > BS_PARAMS["barrier"])
        S_T = S.flatten()
        call_payoff = np.maximum(S_T - BS_PARAMS["strike"], 0)
        call_indicator = (S_T > BS_PARAMS["strike"]).astype(np.float64)
        S0_flat = S0.flatten()
        # Pathwise: ∂π/∂S_0 = 1{alive} · 1{S_T>K} · S_T/S_0 · discount (biased)
        pathwise_delta = alive.astype(np.float64) * call_indicator * (S_T / S0_flat) * discount
        payoff = call_payoff * alive.astype(np.float64) * discount
        y_all[:, p] = payoff
        dydx_all[:, p] = pathwise_delta

    return {
        "x": S0,
        "y": y_all.mean(axis=1, keepdims=True),
        "dydx_pw": dydx_all.mean(axis=1).reshape(N_SAMPLES, 1, 1),
    }


def generate_data(method: str, seed: int) -> dict:
    """Dispatch to the right BS data generator."""
    if method == "vanilla" or method == "dml_fixed":
        return _pathwise_barrier_bs(seed=seed)
    elif method == "dml_lrm_fixed":
        # n_steps=2 matches G&K v2 Table 2 row n=2 (single intermediate barrier
        # check). Without this, defaults to 252 (daily-monitor) — different
        # experimental setup. (Bug fix 2026-05-04.)
        return lrm_barrier_bs(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, n_steps=2,
            strike=BS_PARAMS["strike"], barrier=BS_PARAMS["barrier"],
            vol=BS_PARAMS["vol"], r=BS_PARAMS["r"], T=BS_PARAMS["T"],
        )
    elif method == "dml_fuzzy_fixed":
        # Same n_steps=2 explicit-arg fix.
        return fuzzy_barrier_bs(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, n_steps=2,
            strike=BS_PARAMS["strike"], barrier=BS_PARAMS["barrier"],
            vol=BS_PARAMS["vol"], r=BS_PARAMS["r"], T=BS_PARAMS["T"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def prepare_data_dict(data: dict, method: str, seed: int, ground_truth: dict) -> dict:
    if method in ("vanilla", "dml_fixed"):
        dydx_key = "dydx_pw"
    elif method == "dml_lrm_fixed":
        dydx_key = "dydx_lrm"
    elif method == "dml_fuzzy_fixed":
        dydx_key = "dydx_fuzzy"
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "x_train": data["x"],
        "y_train": data["y"],
        "dydx_train": data[dydx_key],
        "x_test": ground_truth["x"],
        "y_test": ground_truth["y"],
        "dydx_test": ground_truth["dydx"],
    }


# ============================================================================
# MAIN
# ============================================================================

def run_bs_reproduction(results_dir: Path, existing: dict, resume: bool):
    print("\n" + "=" * 70)
    print("BS BARRIER REPRODUCTION (vs G&K v2 §3.4 / Table 2 row n=2)")
    print("=" * 70)
    print(f"Methods:  {len(METHODS)} ({', '.join(METHODS)})")
    print(f"Seeds:    {SEEDS}")
    print(f"BS:       σ={BS_PARAMS['vol']}, T={BS_PARAMS['T']}, "
          f"K={BS_PARAMS['strike']}, B={BS_PARAMS['barrier']}")
    print(f"Output:   {results_dir}")

    print(f"\nGenerating BS ground truth: {GROUND_TRUTH_N_TEST} spots, "
          f"Reiner-Rubinstein closed form...")
    t_gt = time.time()
    ground_truth = generate_bs_ground_truth(GROUND_TRUTH_N_TEST, GROUND_TRUTH_SEED)
    print(f"  Ground truth ready in {time.time() - t_gt:.1f}s")

    for seed in SEEDS:
        for method in METHODS:
            key = make_key(method, seed)
            if resume and key in existing:
                print(f"  SKIP (exists): {method} seed={seed}")
                continue

            print(f"\n--- {method} seed={seed} ---")
            t_gen = time.time()
            data = generate_data(method, seed)
            data_split = prepare_data_dict(data, method, seed, ground_truth)
            gen_time = time.time() - t_gen

            t0 = time.time()
            try:
                trainer_method = TRAINER_METHOD_MAP[method]
                result = train_single_experiment(
                    x_train=data_split["x_train"],
                    y_train=data_split["y_train"],
                    dydx_train=data_split["dydx_train"],
                    x_test=data_split["x_test"],
                    y_test=data_split["y_test"],
                    dydx_test=data_split["dydx_test"],
                    method=trainer_method, seed=seed, pbar=False,
                    **HPARAMS,
                )
                elapsed = time.time() - t0
                result_dict = {
                    "method": method,
                    "trainer_method": trainer_method,
                    "seed": seed,
                    "payoff": "barrier_doc_call",
                    "model": "black_scholes",
                    "n_samples": N_SAMPLES,
                    "k_paths": K_PATHS,
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "data_gen_s": round(gen_time, 2),
                    "hparams": dict(HPARAMS),
                    "bs_params": dict(BS_PARAMS),
                }
                save_result(results_dir, key, result_dict)
                print(f"  val_mse={result.test_value_mse:.6e}, "
                      f"grad_mse={result.test_grad_mse:.6e}, t={elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()


def analyze(results_dir: Path):
    existing = load_existing(results_dir)
    if not existing:
        print("No results found.")
        return
    print("\n" + "=" * 80)
    print("BS BARRIER REPRODUCTION — RESULTS (means across seeds)")
    print("=" * 80)
    by_method = {}
    for key, res in existing.items():
        by_method.setdefault(res["method"], []).append(res)
    print(f"  {'Method':<25} {'val_mse mean':>14} {'val_mse std':>14} "
          f"{'grad_mse mean':>14} {'n':>4}")
    for method in METHODS:
        if method not in by_method:
            continue
        vals = by_method[method]
        v_mean = np.mean([r["test_value_mse"] for r in vals])
        v_std = np.std([r["test_value_mse"] for r in vals])
        g_mean = np.mean([r["test_grad_mse"] for r in vals])
        n = len(vals)
        print(f"  {method:<25} {v_mean:14.4e} {v_std:14.4e} "
              f"{g_mean:14.4e} {n:>4}")
    print("\nG&K v2 Table 2 row n=2: LRM Price RMSE = 0.022 (k=100, m=1024)")
    print("Our k=10 → expected ~3x larger SE; sqrt(LRM val_mse) is comparable.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = Path("results/heston_barrier_4way/bs_reproduction")
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir)
        return
    existing = load_existing(results_dir) if args.resume else {}
    run_bs_reproduction(results_dir, existing, args.resume)
    analyze(results_dir)
    print("\nBS barrier reproduction complete!")


if __name__ == "__main__":
    main()
