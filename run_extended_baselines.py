#!/usr/bin/env python3
"""
Extended Baselines at High Dimensions (d=50, 100).

Demonstrates where kernel baselines (KRR, RF) break down and NNs become
essential. GP is excluded — O(n³) is infeasible at these sizes.

This addresses a reviewer concern: "Baselines only tested at d≤20."

Usage:
    python run_extended_baselines.py --gpu 0           # GPU 0
    python run_extended_baselines.py --gpu 1           # GPU 1
    python run_extended_baselines.py --dry-run          # Print plan only
"""

import sys
import os
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.functions import generate_data, train_test_split, corrupt_derivatives
from dml_benchmark.baselines import run_baseline_experiment
from dml_benchmark.trainer import train_single_experiment

# ============================================================================
# CONFIGURATION
# ============================================================================

# Functions to test (subset — these show the clearest dimension scaling)
FUNCTIONS = ["poly_trig", "trig"]

# High dimensions where baselines weren't previously tested
DIMS = [50, 100]

# Sample sizes — must be large enough for d=100
SAMPLE_SIZES = [1024, 4096]

# 5 seeds for statistical validity
SEEDS = [42, 123, 456, 789, 1024]

# Baselines to run (NO GP — O(n³) infeasible)
BASELINE_METHODS = ["krr", "rf"]

# NN methods for comparison
NN_METHODS = ["vanilla", "dml_fixed"]

RESULTS_DIR = Path("results/tier5_extended_baselines")


def make_key(func_type, dim, n_samples, noise_level, seed, method):
    """Generate unique result key."""
    return f"{func_type}_d{dim}_n{n_samples}_noise{noise_level}_s{seed}_{method}"


def load_existing_results(results_dir):
    """Load all existing results for resume support."""
    results = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    results[f.stem] = data
            except Exception:
                pass
    return results


def save_result(results_dir, key, result_dict):
    """Save a single result to JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / f"{key}.json"
    with open(filepath, "w") as f:
        json.dump(result_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extended baselines at d=50,100")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Build experiment list
    experiments = []
    for func in FUNCTIONS:
        for dim in DIMS:
            for ns in SAMPLE_SIZES:
                for seed in SEEDS:
                    for method in BASELINE_METHODS:
                        experiments.append({
                            "func": func, "dim": dim, "n_samples": ns,
                            "seed": seed, "method": f"baseline_{method}",
                            "is_baseline": True,
                        })
                    for method in NN_METHODS:
                        experiments.append({
                            "func": func, "dim": dim, "n_samples": ns,
                            "seed": seed, "method": method,
                            "is_baseline": False,
                        })

    total = len(experiments)
    print(f"Extended Baselines Experiment Plan")
    print(f"  Functions:   {FUNCTIONS}")
    print(f"  Dimensions:  {DIMS}")
    print(f"  Samples:     {SAMPLE_SIZES}")
    print(f"  Seeds:       {SEEDS}")
    print(f"  Baselines:   {BASELINE_METHODS} (GP excluded — O(n³) infeasible)")
    print(f"  NN methods:  {NN_METHODS}")
    print(f"  Total:       {total} experiments")
    print(f"  Results dir: {RESULTS_DIR}")

    if args.dry_run:
        print("\nDry run — no experiments executed.")
        return

    existing = load_existing_results(RESULTS_DIR)
    print(f"\nExisting results: {len(existing)}")

    completed = 0
    skipped = 0
    errors = []
    total_start = time.time()

    # Group by data config to reuse generated data
    from collections import defaultdict
    data_groups = defaultdict(list)
    for exp in experiments:
        key = (exp["func"], exp["dim"], exp["n_samples"], exp["seed"])
        data_groups[key].append(exp)

    for gi, ((func, dim, ns, seed), group) in enumerate(sorted(data_groups.items())):
        elapsed = time.time() - total_start
        print(f"\n[{gi+1}/{len(data_groups)}] {func} d={dim} n={ns} s={seed} "
              f"[{elapsed/60:.0f}m elapsed, {completed} done]")

        # Generate data once per group
        try:
            data = generate_data(func, n_dim=dim, n_samples=ns, seed=seed)
            train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
        except Exception as e:
            msg = f"DATA GEN FAILED: {func} d={dim} n={ns} s={seed}: {e}"
            print(f"  ❌ {msg}")
            errors.append(msg)
            continue

        x_train = train_data.x
        y_train = train_data.y
        dydx_train = train_data.dydx
        x_test = test_data.x
        y_test = test_data.y
        dydx_test = test_data.dydx

        for exp in group:
            method = exp["method"]
            result_key = make_key(func, dim, ns, 0.0, seed, method)

            if args.resume and result_key in existing:
                skipped += 1
                continue

            print(f"  {method} ...", end=" ", flush=True)
            t0 = time.time()

            try:
                if exp["is_baseline"]:
                    bl_name = method.replace("baseline_", "")
                    bl_result = run_baseline_experiment(
                        bl_name, x_train, y_train, dydx_train,
                        x_test, y_test, dydx_test,
                    )
                    elapsed_exp = time.time() - t0
                    result_dict = {
                        "method": method,
                        "func_type": func,
                        "dim": dim,
                        "n_samples": ns,
                        "noise_level": 0.0,
                        "seed": seed,
                        "lambda": 1.0,
                        "test_value_mse": bl_result["value_mse"],
                        "test_grad_mse": bl_result["grad_mse"],
                        "time_s": elapsed_exp,
                    }
                else:
                    result = train_single_experiment(
                        x_train=x_train, y_train=y_train,
                        dydx_train=dydx_train,
                        x_test=x_test, y_test=y_test, dydx_test=dydx_test,
                        lambda_=1.0, n_epochs=500, batch_size=256,
                        n_layers=4, hidden_size=256, lr=0.005,
                        method=method, seed=seed, pbar=False,
                    )
                    elapsed_exp = time.time() - t0
                    result_dict = {
                        "method": method,
                        "func_type": func,
                        "dim": dim,
                        "n_samples": ns,
                        "noise_level": 0.0,
                        "seed": seed,
                        "lambda": 1.0,
                        "test_value_mse": result.test_value_mse,
                        "test_grad_mse": result.test_grad_mse,
                        "best_epoch": result.best_epoch,
                        "n_epochs_actual": getattr(result, 'n_epochs_actual', 0),
                        "time_s": elapsed_exp,
                    }

                save_result(RESULTS_DIR, result_key, result_dict)
                print(f"✅ MSE={result_dict['test_value_mse']:.6f} ({elapsed_exp:.1f}s)")
                completed += 1

            except Exception as e:
                elapsed_exp = time.time() - t0
                msg = f"FAILED: {result_key}: {e}"
                print(f"❌ ({elapsed_exp:.1f}s) {msg}")
                errors.append(msg)
                traceback.print_exc()

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"EXTENDED BASELINES COMPLETE")
    print(f"{'='*70}")
    print(f"  Time:      {total_time/60:.0f} min")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {len(errors)}")

    if errors:
        print("\nFailed:")
        for e in errors[:10]:
            print(f"  ❌ {e}")


if __name__ == "__main__":
    main()
