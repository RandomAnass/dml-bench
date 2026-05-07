#!/usr/bin/env python3
"""
Phase 1: Smoke Test Runner
Run this on the server to validate the pipeline before expensive experiments.
Expected time: <2 minutes on CPU, <1 minute on GPU.

Usage:
    python run_smoke_test.py
"""

import sys
import time
import json
import traceback
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.functions import generate_data, train_test_split
from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.baselines import run_baseline_experiment
from dml_benchmark.config import get_config


def run_smoke_test():
    """Run minimal smoke test across all methods and baselines."""
    
    config = get_config("smoke")
    results_dir = Path("results/smoke_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SMOKE TEST — Phase 1")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2, default=str)}")
    print()
    
    all_results = {}
    errors = []
    total_start = time.time()
    
    for func_type in config["function_types"]:
        for dim in config["dimensions"]:
            for n_samples in config["sample_sizes"]:
                for seed in config["seeds"]:
                    
                    # Generate data
                    print(f"\n--- {func_type} | dim={dim} | n={n_samples} | seed={seed} ---")
                    try:
                        data = generate_data(func_type, n_dim=dim, n_samples=n_samples, seed=seed)
                        train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
                    except Exception as e:
                        msg = f"DATA GEN FAILED: {func_type} dim={dim} n={n_samples}: {e}"
                        print(f"  ❌ {msg}")
                        errors.append(msg)
                        continue
                    
                    x_train, y_train, dydx_train = train_data.x, train_data.y, train_data.dydx
                    x_test, y_test, dydx_test = test_data.x, test_data.y, test_data.dydx
                    
                    # ---- NN Methods ----
                    methods = config.get("methods", ["vanilla", "dml_fixed"])
                    
                    for method in methods:
                        key = f"{func_type}_d{dim}_n{n_samples}_s{seed}_{method}"
                        print(f"  Training: {method}...", end=" ", flush=True)
                        
                        t0 = time.time()
                        try:
                            result = train_single_experiment(
                                x_train=x_train, y_train=y_train, dydx_train=dydx_train,
                                x_test=x_test, y_test=y_test, dydx_test=dydx_test,
                                lambda_=config["lambda_values"][-1],  # Use largest lambda for DML
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
                                "test_value_mse": result.test_value_mse,
                                "test_grad_mse": result.test_grad_mse,
                                "time_s": elapsed,
                                "best_epoch": result.best_epoch,
                                "first_train_loss": result.training_logs[0]["train_loss"],
                                "last_train_loss": result.training_logs[-1]["train_loss"],
                            }
                            
                            loss_decreased = result.training_logs[-1]["train_loss"] < result.training_logs[0]["train_loss"]
                            status = "✅" if loss_decreased else "⚠️ LOSS DID NOT DECREASE"
                            
                            print(f"{status} MSE={result.test_value_mse:.6f} "
                                  f"| grad_MSE={result.test_grad_mse:.6f} "
                                  f"| {elapsed:.1f}s")
                            
                        except Exception as e:
                            elapsed = time.time() - t0
                            msg = f"TRAIN FAILED: {key}: {e}"
                            print(f"❌ {msg}")
                            errors.append(msg)
                            traceback.print_exc()
                    
                    # ---- Baselines (only for small dim) ----
                    if dim <= 20:
                        for baseline_name in ["gp", "krr", "rf"]:
                            key = f"{func_type}_d{dim}_n{n_samples}_s{seed}_baseline_{baseline_name}"
                            print(f"  Baseline: {baseline_name}...", end=" ", flush=True)
                            
                            t0 = time.time()
                            try:
                                bl_result = run_baseline_experiment(
                                    baseline_name,
                                    x_train, y_train, dydx_train,
                                    x_test, y_test, dydx_test
                                )
                                elapsed = time.time() - t0
                                
                                all_results[key] = {
                                    "method": f"baseline_{baseline_name}",
                                    "test_value_mse": bl_result["value_mse"],
                                    "test_grad_mse": bl_result["grad_mse"],
                                    "time_s": elapsed,
                                }
                                
                                print(f"✅ MSE={bl_result['value_mse']:.6f} "
                                      f"| grad_MSE={bl_result['grad_mse']:.6f} "
                                      f"| {elapsed:.1f}s")
                                
                            except Exception as e:
                                elapsed = time.time() - t0
                                msg = f"BASELINE FAILED: {key}: {e}"
                                print(f"❌ {msg}")
                                errors.append(msg)
                                traceback.print_exc()
    
    # ---- Summary ----
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Experiments run: {len(all_results)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  ❌ {e}")
    
    # Print results table
    print(f"\n{'Method':<25} {'Value MSE':>12} {'Grad MSE':>12} {'Time(s)':>8}")
    print("-" * 60)
    for key, r in sorted(all_results.items()):
        print(f"{r['method']:<25} {r['test_value_mse']:>12.6f} {r['test_grad_mse']:>12.6f} {r['time_s']:>8.1f}")
    
    # Save results
    results_path = results_dir / "smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config,
            "results": all_results,
            "errors": errors,
            "total_time_s": total_time
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    # Go/No-Go
    if len(errors) == 0:
        print("\n🟢 SMOKE TEST PASSED — Safe to proceed to Phase 2")
        return True
    else:
        print(f"\n🔴 SMOKE TEST FAILED — {len(errors)} error(s). Fix before proceeding.")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
