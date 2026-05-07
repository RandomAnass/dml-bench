#!/usr/bin/env python3
"""
Experiment A/B/D: LRM Labels + Adaptive Balancing vs. Fixed-λ (G&K Baseline).

Head-to-head comparison: Glasserman & Karmarkar (2025) fixed-λ approach
versus our GradNorm/ReLoBRaLo adaptive balancing, both using LRM derivative
labels for discontinuous payoffs.

Experiments combined in this script:
    A. LRM + GradNorm vs fixed-λ + LRM  (digital, barrier, basket digital)
    B. LRM variance scaling with dimension (d=1..50 basket digital)
    D. Network size sensitivity (4×20 vs 4×256)

Key hypothesis: GradNorm automatically down-weights noisy LRM labels,
outperforming G&K's fixed 50-50 weighting. The advantage grows with
dimension d due to O(d) LRM variance scaling.

Usage:
    python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0
    python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0 --resume
    python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0 --only digital
    python experiments/lrm_comparison/run_lrm_vs_adaptive.py --gpu 0 --only dimscaling

Expected runtime: ~2-4 hours on 1 GPU
"""

import sys
import os
import time
import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.lrm_labels import (
    lrm_digital_bs,
    lrm_barrier_bs,
    lrm_basket_bachelier,
    prepare_for_training,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Methods to compare
METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

# Training hyperparameters — G&K baseline
GK_HPARAMS = {
    "n_epochs": 100,          # G&K use 100 epochs
    "batch_size": 256,        # G&K min batch 256
    "n_layers": 4,            # G&K: 4 hidden layers
    "hidden_size": 20,        # G&K: 20 units (tiny!)
    "activation": "softplus", # G&K: softplus
    "lr": 0.005,              # We use our scheduler instead of their one-cycle
    "lambda_": 1.0,           # Fixed λ=1 (50-50 value-deriv weight)
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

# Our standard hyperparameters
OUR_HPARAMS = {
    "n_epochs": 500,          # More epochs with early stopping
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,       # Our standard: 4×256
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999]
N_SEEDS = 10  # Use all 10 seeds

# Sample sizes to test
SAMPLE_SIZES = [1024, 4096, 8192]

# Dimensions for basket digital scaling experiment
DIMS = [1, 5, 10, 20, 50]

# Network sizes for sensitivity experiment
NETWORK_SIZES = [20, 64, 128, 256]


# ============================================================================
# UTILITIES
# ============================================================================

def make_key(experiment, payoff, dim, n_samples, method, hidden_size, seed):
    return f"lrm_{experiment}_{payoff}_d{dim}_n{n_samples}_h{hidden_size}_s{seed}_{method}"


def load_existing(results_dir):
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            if f.name in ("summary.json", "analysis.json"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    existing[data.get("key", f.stem)] = data
            except Exception:
                pass
    return existing


def save_result(results_dir, key, result_dict):
    result_dict["key"] = key
    result_dict["timestamp"] = datetime.now().isoformat()
    path = results_dir / f"{key}.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp_path.rename(path)


def run_single(data_split, method, hparams, key, results_dir, existing, resume, seed):
    """Run a single training experiment and save results."""
    if resume and key in existing:
        print(f"  SKIP (exists): {method}")
        return existing[key]

    print(f"  Training: {method} (h={hparams['hidden_size']})...", end=" ", flush=True)
    t0 = time.time()

    try:
        result = train_single_experiment(
            x_train=data_split["x_train"],
            y_train=data_split["y_train"],
            dydx_train=data_split["dydx_train"],
            x_test=data_split["x_test"],
            y_test=data_split["y_test"],
            dydx_test=data_split["dydx_test"],
            method=method,
            seed=seed,
            pbar=False,
            **hparams,
        )
        elapsed = time.time() - t0

        result_dict = {
            "method": method,
            "seed": seed,
            "test_value_mse": float(result.test_value_mse),
            "test_grad_mse": float(result.test_grad_mse),
            "best_epoch": int(result.best_epoch),
            "time_s": round(elapsed, 2),
            "hparams": {k: v for k, v in hparams.items()},
        }

        save_result(results_dir, key, result_dict)
        print(
            f"val={result.test_value_mse:.6e}, "
            f"grad={result.test_grad_mse:.6e}, "
            f"ep={result.best_epoch}, "
            f"t={elapsed:.1f}s"
        )
        return result_dict

    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return None


# ============================================================================
# EXPERIMENT A: DIGITAL & BARRIER OPTIONS — LRM vs ADAPTIVE
# ============================================================================

def run_digital_experiments(results_dir, existing, resume, n_seeds=N_SEEDS):
    """Experiment A: Digital call (BS), LRM labels, fixed vs adaptive."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A.1: Black-Scholes Digital Call — LRM Labels")
    print("=" * 70)
    print("Payoff: 1{S_T > K}  (pathwise ≡ 0, LRM = unbiased)")
    print(f"Methods: {METHODS}")
    print(f"Seeds: {n_seeds},  Sample sizes: {SAMPLE_SIZES}")
    print()

    results = {}

    for n_samples in SAMPLE_SIZES:
        for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
            print(f"\n--- digital n={n_samples}, seed={seed} ({seed_idx+1}/{n_seeds}) ---")

            # Generate LRM data
            data_raw = lrm_digital_bs(
                n_samples=n_samples, k_paths=10, seed=seed
            )
            data_split = prepare_for_training(data_raw, test_frac=0.2, seed=seed)

            # Also track LRM variance for analysis
            lrm_var_mean = float(data_raw["lrm_var"].mean())

            for hparams, hname in [(GK_HPARAMS, "gk"), (OUR_HPARAMS, "ours")]:
                for method in METHODS:
                    key = make_key(
                        "A1", "digital", 1, n_samples, method,
                        hparams["hidden_size"], seed
                    )
                    # Add hparams identifier to key
                    key = f"{key}_{hname}"

                    rd = run_single(
                        data_split, method, hparams, key,
                        results_dir, existing, resume, seed
                    )
                    if rd is not None:
                        rd["payoff"] = "digital_call"
                        rd["model"] = "black_scholes"
                        rd["dim"] = 1
                        rd["n_samples"] = n_samples
                        rd["lrm_var_mean"] = lrm_var_mean
                        rd["hparam_set"] = hname
                        save_result(results_dir, key, rd)
                        results[key] = rd

    return results


def run_barrier_experiments(results_dir, existing, resume, n_seeds=N_SEEDS):
    """Experiment A.2: Barrier call (BS), LRM labels."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A.2: Black-Scholes Barrier Call — LRM Labels")
    print("=" * 70)
    print("Payoff: max(S_T-K,0) · 1{min S_t > B}  (discontinuous at barrier)")
    print(f"Methods: {METHODS}")
    print()

    results = {}

    for n_samples in [1024, 4096]:
        for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
            print(f"\n--- barrier n={n_samples}, seed={seed} ({seed_idx+1}/{n_seeds}) ---")

            data_raw = lrm_barrier_bs(
                n_samples=n_samples, barrier=80.0, n_steps=50,
                k_paths=10, seed=seed
            )
            data_split = prepare_for_training(data_raw, test_frac=0.2, seed=seed)
            lrm_var_mean = float(data_raw["lrm_var"].mean())

            for method in METHODS:
                key = make_key(
                    "A2", "barrier", 1, n_samples, method,
                    OUR_HPARAMS["hidden_size"], seed
                )

                rd = run_single(
                    data_split, method, OUR_HPARAMS, key,
                    results_dir, existing, resume, seed
                )
                if rd is not None:
                    rd["payoff"] = "barrier_knock_out"
                    rd["dim"] = 1
                    rd["n_samples"] = n_samples
                    rd["lrm_var_mean"] = lrm_var_mean
                    save_result(results_dir, key, rd)
                    results[key] = rd

    return results


# ============================================================================
# EXPERIMENT B: DIMENSIONAL SCALING OF LRM VARIANCE
# ============================================================================

def run_dim_scaling_experiments(results_dir, existing, resume, n_seeds=N_SEEDS):
    """Experiment B: Basket digital, dimension sweep d=1..50."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: LRM Variance Scaling with Dimension")
    print("=" * 70)
    print("Bachelier basket digital: 1{Basket > K}")
    print(f"Dimensions: {DIMS}")
    print(f"Methods: {METHODS}")
    print()

    results = {}
    n_samples = 4096  # Fixed sample size for clean comparison

    for d in DIMS:
        for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
            print(f"\n--- basket d={d}, seed={seed} ({seed_idx+1}/{n_seeds}) ---")

            data_raw = lrm_basket_bachelier(
                n_samples=n_samples, d=d, k_paths=10, seed=seed
            )
            data_split = prepare_for_training(data_raw, test_frac=0.2, seed=seed)
            lrm_var_mean = float(data_raw["lrm_var"].mean())

            print(f"  LRM label variance: {lrm_var_mean:.6f}")

            for method in METHODS:
                key = make_key(
                    "B", "basket_digital", d, n_samples, method,
                    OUR_HPARAMS["hidden_size"], seed
                )

                rd = run_single(
                    data_split, method, OUR_HPARAMS, key,
                    results_dir, existing, resume, seed
                )
                if rd is not None:
                    rd["payoff"] = "digital_basket"
                    rd["model"] = "bachelier"
                    rd["dim"] = d
                    rd["n_samples"] = n_samples
                    rd["lrm_var_mean"] = lrm_var_mean
                    save_result(results_dir, key, rd)
                    results[key] = rd

    return results


# ============================================================================
# EXPERIMENT D: NETWORK SIZE SENSITIVITY
# ============================================================================

def run_network_size_experiments(results_dir, existing, resume):
    """Experiment D: Network size sweep (4×20 to 4×256)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Network Size Sensitivity")
    print("=" * 70)
    print("Digital call (d=1) + basket digital (d=7)")
    print(f"Network sizes: {NETWORK_SIZES}")
    print(f"Methods: {METHODS}")
    print()

    results = {}
    n_samples = 4096

    for payoff_name, gen_fn, gen_kwargs in [
        ("digital", lrm_digital_bs, {"n_samples": n_samples, "k_paths": 10}),
        ("basket7", lrm_basket_bachelier, {"n_samples": n_samples, "d": 7, "k_paths": 10}),
    ]:
        for hidden_size in NETWORK_SIZES:
            hp = dict(OUR_HPARAMS)
            hp["hidden_size"] = hidden_size

            d = 1 if payoff_name == "digital" else 7

            for seed_idx, seed in enumerate(SEEDS[:5]):  # 5 seeds for network sweep
                print(f"\n--- {payoff_name} h={hidden_size}, seed={seed} ({seed_idx+1}/5) ---")

                data_raw = gen_fn(seed=seed, **gen_kwargs)
                data_split = prepare_for_training(data_raw, test_frac=0.2, seed=seed)

                for method in METHODS:
                    key = make_key(
                        "D", payoff_name, d, n_samples, method, hidden_size, seed
                    )

                    rd = run_single(
                        data_split, method, hp, key,
                        results_dir, existing, resume, seed
                    )
                    if rd is not None:
                        rd["payoff"] = payoff_name
                        rd["dim"] = d
                        rd["n_samples"] = n_samples
                        rd["hidden_size"] = hidden_size
                        save_result(results_dir, key, rd)
                        results[key] = rd

    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(results_dir):
    """Load all results and print summary analysis."""
    existing = load_existing(results_dir)
    if not existing:
        print("No results found.")
        return

    print("\n" + "=" * 90)
    print("RESULTS ANALYSIS: LRM + Adaptive Balancing")
    print("=" * 90)

    # Group by experiment
    experiments = {}
    for key, res in existing.items():
        exp = key.split("_")[1] if "_" in key else "unknown"
        if exp not in experiments:
            experiments[exp] = []
        experiments[exp].append(res)

    for exp_name in sorted(experiments.keys()):
        exp_results = experiments[exp_name]

        # Group by method
        by_method = {}
        for r in exp_results:
            m = r.get("method", "?")
            if m not in by_method:
                by_method[m] = []
            by_method[m].append(r)

        print(f"\n--- Experiment {exp_name} ({len(exp_results)} results) ---")
        print(f"  {'Method':<25} {'Mean Val MSE':>14} {'Std Val MSE':>13} {'Count':>6}")
        print(f"  {'-'*25} {'-'*14} {'-'*13} {'-'*6}")

        for method in METHODS:
            if method in by_method:
                vals = [r["test_value_mse"] for r in by_method[method]]
                print(
                    f"  {method:<25} {np.mean(vals):14.6e} "
                    f"{np.std(vals):13.6e} {len(vals):6d}"
                )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LRM Labels + Adaptive Balancing Experiments (vs G&K)"
    )
    parser.add_argument("--gpu", type=int, default=None, help="Pin to GPU")
    parser.add_argument("--resume", action="store_true", help="Skip completed")
    parser.add_argument(
        "--only",
        choices=["digital", "barrier", "dimscaling", "netsize", "analyze"],
        default=None,
        help="Run only one experiment type",
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of seeds to use (max {len(SEEDS)})"
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    n_seeds = min(args.seeds, len(SEEDS))

    results_dir = Path("results/lrm_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing(results_dir) if args.resume else {}

    if args.only == "analyze":
        analyze_results(results_dir)
        return

    print("=" * 70)
    print("LRM + ADAPTIVE BALANCING EXPERIMENTS")
    print("  Competing with Glasserman & Karmarkar (2025)")
    print("=" * 70)
    print(f"Methods:     {METHODS}")
    print(f"Seeds:       {SEEDS[:n_seeds]}")
    print(f"Results dir: {results_dir}")
    if args.gpu is not None:
        print(f"GPU:         {args.gpu}")
    print()

    all_results = {}

    if args.only is None or args.only == "digital":
        all_results.update(run_digital_experiments(results_dir, existing, args.resume, n_seeds))

    if args.only is None or args.only == "barrier":
        all_results.update(run_barrier_experiments(results_dir, existing, args.resume, n_seeds))

    if args.only is None or args.only == "dimscaling":
        all_results.update(run_dim_scaling_experiments(results_dir, existing, args.resume, n_seeds))

    if args.only is None or args.only == "netsize":
        all_results.update(run_network_size_experiments(results_dir, existing, args.resume))

    # Summary
    if all_results:
        analyze_results(results_dir)

    print(f"\n{len(all_results)} experiments completed. Results in {results_dir}")


if __name__ == "__main__":
    main()
