#!/usr/bin/env python3
"""
Architecture Ablation Study.

Tests whether DML's advantage is robust across different network architectures.
Compares 3 architectures on poly_trig d=10 with 5 seeds:
  - Small:    2 layers × 128 hidden
  - Default:  4 layers × 256 hidden  (main paper architecture)
  - Large:    6 layers × 512 hidden

Total: 3 archs × 2 methods × 5 seeds × 1 config = 30 experiments (minimal).

Usage:
    python run_architecture_ablation.py --gpu 0
    python run_architecture_ablation.py --dry-run
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

from dml_benchmark.functions import generate_data, train_test_split
from dml_benchmark.trainer import train_single_experiment

# ============================================================================
# CONFIGURATION
# ============================================================================

ARCHITECTURES = [
    {"name": "small",   "n_layers": 2, "hidden_size": 128},
    {"name": "default", "n_layers": 4, "hidden_size": 256},
    {"name": "large",   "n_layers": 6, "hidden_size": 512},
]

# Fixed test config — poly_trig is the flagship function
FUNC = "poly_trig"
DIM = 10
N_SAMPLES = 1024
SEEDS = [42, 123, 456, 789, 1024]
METHODS = ["vanilla", "dml_fixed"]

RESULTS_DIR = Path("results/tier5_arch_ablation")


def make_key(arch_name, method, seed):
    return f"{FUNC}_d{DIM}_n{N_SAMPLES}_{arch_name}_{method}_s{seed}"


def load_existing(results_dir):
    results = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    results[f.stem] = json.load(fh)
            except Exception:
                pass
    return results


def save_result(results_dir, key, result_dict):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{key}.json", "w") as f:
        json.dump(result_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Architecture ablation study")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    total = len(ARCHITECTURES) * len(METHODS) * len(SEEDS)
    print(f"Architecture Ablation Study")
    print(f"  Config:  {FUNC} d={DIM} n={N_SAMPLES}")
    print(f"  Archs:   {[a['name'] for a in ARCHITECTURES]}")
    print(f"  Methods: {METHODS}")
    print(f"  Seeds:   {SEEDS}")
    print(f"  Total:   {total} experiments")

    if args.dry_run:
        for arch in ARCHITECTURES:
            print(f"\n  {arch['name']}: {arch['n_layers']}L × {arch['hidden_size']}H")
            for method in METHODS:
                print(f"    × {method} × {len(SEEDS)} seeds")
        print("\nDry run — no experiments executed.")
        return

    existing = load_existing(RESULTS_DIR)
    print(f"\nExisting results: {len(existing)}")

    completed = 0
    skipped = 0
    errors = []
    t_start = time.time()

    for seed in SEEDS:
        # Generate data once per seed
        try:
            data = generate_data(FUNC, n_dim=DIM, n_samples=N_SAMPLES, seed=seed)
            train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
        except Exception as e:
            print(f"  ❌ Data gen failed s={seed}: {e}")
            errors.append(str(e))
            continue

        x_train, y_train = train_data.x, train_data.y
        dydx_train = train_data.dydx
        x_test, y_test = test_data.x, test_data.y
        dydx_test = test_data.dydx

        for arch in ARCHITECTURES:
            for method in METHODS:
                key = make_key(arch["name"], method, seed)

                if args.resume and key in existing:
                    skipped += 1
                    continue

                print(f"  {arch['name']} {method} s={seed} ...", end=" ", flush=True)
                t0 = time.time()

                try:
                    result = train_single_experiment(
                        x_train=x_train, y_train=y_train,
                        dydx_train=dydx_train,
                        x_test=x_test, y_test=y_test, dydx_test=dydx_test,
                        lambda_=1.0, n_epochs=500, batch_size=256,
                        n_layers=arch["n_layers"],
                        hidden_size=arch["hidden_size"],
                        lr=0.005, method=method, seed=seed, pbar=False,
                    )
                    elapsed = time.time() - t0

                    result_dict = {
                        "method": method,
                        "func_type": FUNC,
                        "dim": DIM,
                        "n_samples": N_SAMPLES,
                        "noise_level": 0.0,
                        "seed": seed,
                        "lambda": 1.0,
                        "arch_name": arch["name"],
                        "n_layers": arch["n_layers"],
                        "hidden_size": arch["hidden_size"],
                        "test_value_mse": result.test_value_mse,
                        "test_grad_mse": result.test_grad_mse,
                        "best_epoch": result.best_epoch,
                        "n_epochs_actual": getattr(result, 'n_epochs_actual', 0),
                        "time_s": elapsed,
                    }

                    save_result(RESULTS_DIR, key, result_dict)
                    print(f"✅ MSE={result_dict['test_value_mse']:.6f} ({elapsed:.1f}s)")
                    completed += 1

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"❌ ({elapsed:.1f}s) {e}")
                    errors.append(f"{key}: {e}")
                    traceback.print_exc()

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ARCHITECTURE ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Time:      {total_time/60:.1f} min")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {len(errors)}")

    # Quick analysis if we have results
    all_results = load_existing(RESULTS_DIR)
    if all_results:
        print(f"\n  Quick analysis ({len(all_results)} results):")
        for arch in ARCHITECTURES:
            print(f"\n  {arch['name']} ({arch['n_layers']}L×{arch['hidden_size']}H):")
            for method in METHODS:
                vals = [r["test_value_mse"] for r in all_results.values()
                        if r.get("arch_name") == arch["name"]
                        and r["method"] == method]
                if vals:
                    print(f"    {method:<12}: MSE = {np.mean(vals):.6f} ± {np.std(vals):.6f} "
                          f"(n={len(vals)})")

        # DML advantage per arch
        print(f"\n  DML advantage (% MSE reduction vs vanilla):")
        for arch in ARCHITECTURES:
            van_vals = [r["test_value_mse"] for r in all_results.values()
                        if r.get("arch_name") == arch["name"] and r["method"] == "vanilla"]
            dml_vals = [r["test_value_mse"] for r in all_results.values()
                        if r.get("arch_name") == arch["name"] and r["method"] == "dml_fixed"]
            if van_vals and dml_vals:
                van_m, dml_m = np.mean(van_vals), np.mean(dml_vals)
                adv = 100 * (van_m - dml_m) / van_m if van_m > 0 else 0
                print(f"    {arch['name']:<10}: {adv:+.1f}%  "
                      f"(van={van_m:.6f}, dml={dml_m:.6f})")


if __name__ == "__main__":
    main()
