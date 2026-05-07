#!/usr/bin/env python3
"""
Phase 2: Quick Validation Runner
Runs the QuickExplorationConfig grid: dims [2,10,50], samples [512,2048],
3 seeds, 4 NN methods + baselines (dim<=20), 200 epochs.

Expected time: ~30-60 min on 1 GPU, ~15-30 min on 2 GPUs.

Usage:
    python run_phase2_validation.py                  # run everything
    python run_phase2_validation.py --gpu 0          # pin to GPU 0
    python run_phase2_validation.py --resume          # skip already-saved results
    python run_phase2_validation.py --baselines-only  # only baselines (CPU)
    python run_phase2_validation.py --nn-only         # only NN methods (GPU)
"""

import sys
import os
import time
import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from itertools import product

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.functions import generate_data, train_test_split
from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.baselines import run_baseline_experiment
from dml_benchmark.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Quick Validation")
    parser.add_argument("--gpu", type=int, default=None, help="Pin to specific GPU")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    parser.add_argument("--baselines-only", action="store_true", help="Only run baselines")
    parser.add_argument("--nn-only", action="store_true", help="Only run NN methods")
    return parser.parse_args()


def make_key(func_type, dim, n_samples, seed, method):
    return f"{func_type}_d{dim}_n{n_samples}_s{seed}_{method}"


def load_existing_results(results_path):
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return data.get("results", {})
    return {}


def save_results(results_path, config, all_results, errors, total_time):
    with open(results_path, "w") as f:
        json.dump({
            "config": config,
            "results": all_results,
            "errors": errors,
            "total_time_s": total_time,
            "n_experiments": len(all_results),
        }, f, indent=2, default=str)


def run_phase2():
    args = parse_args()
    
    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Pinned to GPU {args.gpu}")
    
    config = get_config("quick")
    results_dir = Path("results/phase2_validation")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "phase2_results.json"
    
    print("=" * 70)
    print("PHASE 2: QUICK VALIDATION")
    print("=" * 70)
    print(f"Dimensions:    {config['dimensions']}")
    print(f"Sample sizes:  {config['sample_sizes']}")
    print(f"Seeds:         {config['seeds']}")
    print(f"Epochs:        {config['n_epochs']}")
    print(f"Function types: {config['function_types']}")
    print(f"Architecture:  {config['n_layers']}L x {config['hidden_size']}H, {config['activation']}")
    
    nn_methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
    baseline_methods = ["gp", "krr", "rf"]
    
    # Build experiment grid
    grid = list(product(
        config["function_types"],
        config["dimensions"],
        config["sample_sizes"],
        config["seeds"]
    ))
    
    total_nn = len(grid) * len(nn_methods)
    total_bl = sum(1 for _, d, _, _ in grid if d <= 20) * len(baseline_methods)
    
    if args.baselines_only:
        print(f"\nRunning BASELINES ONLY: {total_bl} experiments")
    elif args.nn_only:
        print(f"\nRunning NN ONLY: {total_nn} experiments")
    else:
        print(f"\nTotal experiments: {total_nn} NN + {total_bl} baselines = {total_nn + total_bl}")
    print()
    
    # Resume support
    all_results = load_existing_results(results_path) if args.resume else {}
    if args.resume and all_results:
        print(f"Resuming: {len(all_results)} experiments already completed\n")
    
    errors = []
    total_start = time.time()
    completed = 0
    skipped = 0
    
    for i, (func_type, dim, n_samples, seed) in enumerate(grid):
        
        # Generate data once per (func_type, dim, n_samples, seed)
        banner = f"[{i+1}/{len(grid)}] {func_type} | dim={dim} | n={n_samples} | seed={seed}"
        print(f"\n{'='*70}")
        print(banner)
        print(f"{'='*70}")
        
        try:
            data = generate_data(func_type, n_dim=dim, n_samples=n_samples, seed=seed)
            train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
        except Exception as e:
            msg = f"DATA GEN FAILED: {banner}: {e}"
            print(f"  ❌ {msg}")
            errors.append(msg)
            traceback.print_exc()
            continue
        
        x_train, y_train, dydx_train = train_data.x, train_data.y, train_data.dydx
        x_test, y_test, dydx_test = test_data.x, test_data.y, test_data.dydx
        
        # ---- NN Methods ----
        if not args.baselines_only:
            for method in nn_methods:
                key = make_key(func_type, dim, n_samples, seed, method)
                
                if args.resume and key in all_results:
                    skipped += 1
                    print(f"  [skip] {method}")
                    continue
                
                print(f"  Training: {method}...", end=" ", flush=True)
                t0 = time.time()
                
                try:
                    result = train_single_experiment(
                        x_train=x_train, y_train=y_train, dydx_train=dydx_train,
                        x_test=x_test, y_test=y_test, dydx_test=dydx_test,
                        lambda_=1.0,
                        n_epochs=config["n_epochs"],
                        batch_size=config["batch_size"],
                        n_layers=config["n_layers"],
                        hidden_size=config["hidden_size"],
                        lr=config["lr"],
                        activation=config["activation"],
                        seed=seed,
                        method=method,
                        pbar=False
                    )
                    elapsed = time.time() - t0
                    
                    all_results[key] = {
                        "method": method,
                        "func_type": func_type,
                        "dim": dim,
                        "n_samples": n_samples,
                        "seed": seed,
                        "test_value_mse": result.test_value_mse,
                        "test_grad_mse": result.test_grad_mse,
                        "best_epoch": result.best_epoch,
                        "time_s": elapsed,
                    }
                    
                    loss_ok = result.training_logs[-1]["train_loss"] < result.training_logs[0]["train_loss"]
                    status = "✅" if loss_ok else "⚠️"
                    
                    print(f"{status} val_MSE={result.test_value_mse:.6f} "
                          f"grad_MSE={result.test_grad_mse:.6f} "
                          f"best@{result.best_epoch} "
                          f"({elapsed:.1f}s)")
                    completed += 1
                    
                except Exception as e:
                    elapsed = time.time() - t0
                    msg = f"TRAIN FAILED: {key}: {e}"
                    print(f"❌ ({elapsed:.1f}s) {msg}")
                    errors.append(msg)
                    traceback.print_exc()
        
        # ---- Baselines (dim <= 20 only) ----
        if not args.nn_only and dim <= 20:
            for bl_name in baseline_methods:
                key = make_key(func_type, dim, n_samples, seed, f"baseline_{bl_name}")
                
                if args.resume and key in all_results:
                    skipped += 1
                    print(f"  [skip] baseline_{bl_name}")
                    continue
                
                print(f"  Baseline: {bl_name}...", end=" ", flush=True)
                t0 = time.time()
                
                try:
                    bl_result = run_baseline_experiment(
                        bl_name,
                        x_train, y_train, dydx_train,
                        x_test, y_test, dydx_test
                    )
                    elapsed = time.time() - t0
                    
                    all_results[key] = {
                        "method": f"baseline_{bl_name}",
                        "func_type": func_type,
                        "dim": dim,
                        "n_samples": n_samples,
                        "seed": seed,
                        "test_value_mse": bl_result["value_mse"],
                        "test_grad_mse": bl_result["grad_mse"],
                        "time_s": elapsed,
                    }
                    
                    print(f"✅ val_MSE={bl_result['value_mse']:.6f} "
                          f"grad_MSE={bl_result['grad_mse']:.6f} "
                          f"({elapsed:.1f}s)")
                    completed += 1
                    
                except Exception as e:
                    elapsed = time.time() - t0
                    msg = f"BASELINE FAILED: {key}: {e}"
                    print(f"❌ ({elapsed:.1f}s) {msg}")
                    errors.append(msg)
                    traceback.print_exc()
        
        # Save after every grid point (crash resilience)
        save_results(results_path, config, all_results, errors, time.time() - total_start)
    
    # ---- Final Summary ----
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"Total time:     {total_time/60:.1f} min")
    print(f"Completed:      {completed}")
    print(f"Skipped (resume): {skipped}")
    print(f"Errors:         {len(errors)}")
    print(f"Results saved:  {results_path}")
    
    if errors:
        print("\nFAILED EXPERIMENTS:")
        for e in errors:
            print(f"  ❌ {e}")
    
    # Per-method summary
    print(f"\n{'Method':<25} {'Mean Val MSE':>14} {'Mean Grad MSE':>14} {'Count':>6}")
    print("-" * 65)
    
    method_groups = {}
    for key, r in all_results.items():
        m = r["method"]
        if m not in method_groups:
            method_groups[m] = {"val": [], "grad": []}
        method_groups[m]["val"].append(r["test_value_mse"])
        method_groups[m]["grad"].append(r["test_grad_mse"])
    
    for m in sorted(method_groups.keys()):
        vals = method_groups[m]["val"]
        grads = method_groups[m]["grad"]
        print(f"{m:<25} {np.mean(vals):>14.6f} {np.mean(grads):>14.6f} {len(vals):>6}")
    
    # Save final
    save_results(results_path, config, all_results, errors, total_time)
    
    if len(errors) == 0:
        print(f"\n🟢 PHASE 2 COMPLETE — All {completed} experiments passed")
        return True
    else:
        print(f"\n🟡 PHASE 2 DONE with {len(errors)} error(s)")
        return False


if __name__ == "__main__":
    success = run_phase2()
    sys.exit(0 if success else 1)
