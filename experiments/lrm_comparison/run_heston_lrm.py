#!/usr/bin/env python3
"""
Experiment C: Heston Euler-Scheme LRM — Novel Experiment.

Glasserman & Karmarkar (2025) discuss Euler-scheme LRM in §3.5 of their
paper but never implement it. This experiment fills that gap and shows
where Euler-LRM variance explodes — exactly the regime where our adaptive
loss balancing (GradNorm/ReLoBRaLo) provides the most benefit.

Key comparisons:
    1. Euler-LRM + fixed-λ  (G&K's theoretical approach, never tested)
    2. Euler-LRM + GradNorm (our contribution)
    3. Bump-and-reprice (existing Heston in functions.py, for reference)
    4. Vanilla (no derivatives)

Sweep over:
    - n_steps ∈ {50, 100, 252} — more steps → more LRM variance
    - payoff_type ∈ {'call', 'digital'} — smooth vs discontinuous
    - n_samples ∈ {1024, 4096}

Expected findings:
    - Euler-LRM variance grows ~O(N_steps) for the score accumulation
    - Fixed-λ degrades for large N_steps; GradNorm compensates
    - For 'call' payoff, PW (bump-and-reprice) is cheaper and lower-variance

Usage:
    python experiments/lrm_comparison/run_heston_lrm.py --gpu 0
    python experiments/lrm_comparison/run_heston_lrm.py --gpu 0 --resume

Expected runtime: ~1-2 hours on 1 GPU
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.lrm_labels import lrm_euler_heston, prepare_for_training


# ============================================================================
# CONFIGURATION
# ============================================================================

METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

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

SEEDS = [42, 123, 456, 789, 1337]
N_STEPS_LIST = [50, 100, 252]
PAYOFF_TYPES = ["call", "digital"]
SAMPLE_SIZES = [1024, 4096]


# ============================================================================
# UTILITIES
# ============================================================================

def make_key(payoff, n_steps, n_samples, method, seed):
    return f"heston_lrm_{payoff}_steps{n_steps}_n{n_samples}_s{seed}_{method}"


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


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_heston_lrm_experiments(results_dir, existing, resume):
    """Run Heston Euler-LRM experiments across all configurations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Heston Euler-Scheme LRM (NOVEL)")
    print("=" * 70)
    print("G&K discuss this in §3.5 but never implement it.")
    print("We show where Euler-LRM variance explodes and GradNorm compensates.")
    print(f"Steps:   {N_STEPS_LIST}")
    print(f"Payoffs: {PAYOFF_TYPES}")
    print(f"Samples: {SAMPLE_SIZES}")
    print(f"Methods: {METHODS}")
    print(f"Seeds:   {SEEDS}")
    print()

    # First: measure LRM variance across configurations
    print("--- LRM Variance Analysis (before training) ---")
    for payoff in PAYOFF_TYPES:
        for n_steps in N_STEPS_LIST:
            data_raw = lrm_euler_heston(
                n_samples=4096, n_steps=n_steps, k_paths=10,
                payoff_type=payoff, seed=42
            )
            var = data_raw["lrm_var"].mean()
            print(f"  {payoff:>8} steps={n_steps:3d}: LRM var = {var:.6f}")
    print()

    results = {}
    total_count = (
        len(PAYOFF_TYPES) * len(N_STEPS_LIST) * len(SAMPLE_SIZES)
        * len(METHODS) * len(SEEDS)
    )
    done = 0

    for payoff in PAYOFF_TYPES:
        for n_steps in N_STEPS_LIST:
            for n_samples in SAMPLE_SIZES:
                for seed_idx, seed in enumerate(SEEDS):
                    print(
                        f"\n--- {payoff} steps={n_steps} n={n_samples} "
                        f"seed={seed} ({seed_idx+1}/{len(SEEDS)}) ---"
                    )

                    # Generate Heston LRM data
                    t_gen = time.time()
                    data_raw = lrm_euler_heston(
                        n_samples=n_samples,
                        n_steps=n_steps,
                        k_paths=10,
                        payoff_type=payoff,
                        seed=seed,
                    )
                    data_split = prepare_for_training(data_raw, test_frac=0.2, seed=seed)
                    lrm_var = float(data_raw["lrm_var"].mean())
                    gen_time = time.time() - t_gen
                    print(f"  Data gen: {gen_time:.1f}s, LRM var: {lrm_var:.6f}")

                    for method in METHODS:
                        key = make_key(payoff, n_steps, n_samples, method, seed)

                        if resume and key in existing:
                            print(f"  SKIP (exists): {method}")
                            results[key] = existing[key]
                            done += 1
                            continue

                        print(f"  Training: {method}...", end=" ", flush=True)
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
                                **HPARAMS,
                            )
                            elapsed = time.time() - t0

                            result_dict = {
                                "method": method,
                                "seed": seed,
                                "payoff": payoff,
                                "model": "heston_euler",
                                "n_steps": n_steps,
                                "n_samples": n_samples,
                                "dim": 1,
                                "lrm_var_mean": lrm_var,
                                "test_value_mse": float(result.test_value_mse),
                                "test_grad_mse": float(result.test_grad_mse),
                                "best_epoch": int(result.best_epoch),
                                "time_s": round(elapsed, 2),
                                "hparams": {k: v for k, v in HPARAMS.items()},
                            }

                            save_result(results_dir, key, result_dict)
                            results[key] = result_dict
                            done += 1

                            print(
                                f"val={result.test_value_mse:.6e}, "
                                f"grad={result.test_grad_mse:.6e}, "
                                f"ep={result.best_epoch}, t={elapsed:.1f}s"
                            )

                        except Exception as e:
                            elapsed = time.time() - t0
                            print(f"FAILED ({elapsed:.1f}s): {e}")
                            traceback.print_exc()
                            done += 1

    return results


def analyze(results_dir):
    """Print analysis of Heston LRM results."""
    existing = load_existing(results_dir)
    heston = {k: v for k, v in existing.items() if "heston" in k}

    if not heston:
        print("No Heston LRM results found.")
        return

    print("\n" + "=" * 90)
    print("HESTON EULER-LRM ANALYSIS")
    print("=" * 90)

    # Group by (payoff, n_steps)
    groups = {}
    for key, res in heston.items():
        gkey = (res.get("payoff", "?"), res.get("n_steps", 0))
        if gkey not in groups:
            groups[gkey] = {}
        method = res["method"]
        if method not in groups[gkey]:
            groups[gkey][method] = []
        groups[gkey][method].append(res)

    for (payoff, n_steps) in sorted(groups.keys()):
        group = groups[(payoff, n_steps)]
        lrm_vars = [r.get("lrm_var_mean", 0) for methods in group.values() for r in methods]
        avg_lrm_var = np.mean(lrm_vars) if lrm_vars else 0

        print(f"\n--- {payoff} steps={n_steps} (LRM var={avg_lrm_var:.6f}) ---")
        print(f"  {'Method':<25} {'Mean Val MSE':>14} {'Std':>13} {'Count':>6}")

        for method in METHODS:
            if method in group:
                vals = [r["test_value_mse"] for r in group[method]]
                print(
                    f"  {method:<25} {np.mean(vals):14.6e} "
                    f"{np.std(vals):13.6e} {len(vals):6d}"
                )


def main():
    parser = argparse.ArgumentParser(description="Heston Euler-LRM Experiments")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results_dir = Path("results/lrm_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analyze(results_dir)
        return

    existing = load_existing(results_dir) if args.resume else {}

    run_heston_lrm_experiments(results_dir, existing, args.resume)
    analyze(results_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
