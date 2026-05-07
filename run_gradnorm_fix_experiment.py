#!/usr/bin/env python3
"""
GradNorm Dimension Fix — Minimal Experiment Runner

Tests the dimension-normalized GradNorm fix against standard GradNorm
and DML-fixed across the full dimension range.

Usage:
    python run_gradnorm_fix_experiment.py --gpu 0
    python run_gradnorm_fix_experiment.py --gpu 0 --resume

Expected runtime: ~30 minutes on 1 GPU (72 experiments × ~25s each)

See GRADNORM_DIMENSION_FIX.md for full research context.
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

sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.functions import generate_data, train_test_split
from dml_benchmark.trainer import train_single_experiment


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_CONFIG = {
    "func_type": "poly_trig",       # Strongest DML advantage → clearest signal
    "dims": [2, 5, 10, 20, 50, 100],
    "n_samples": 1024,
    "noise_level": 0.0,
    "seeds": [42, 123, 456],
    "methods": [
        "dml_fixed",                # Baseline: hand-tuned λ=1
        "dml_gradnorm",             # Standard GradNorm (fails at high d)
        "dml_dimnorm_gradnorm",     # FIX: normalize by d
        "dml_sqrtdimnorm_gradnorm", # FIX variant: normalize by √d
    ],
}

TRAIN_HPARAMS = {
    "n_epochs": 500,
    "batch_size": 512,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
}


def make_key(dim, seed, method):
    return f"gradnorm_fix_poly_trig_d{dim}_n1024_s{seed}_{method}"


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
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp_path.rename(path)


def analyze_results(all_results):
    """Analyze and print results grouped by dimension."""
    print("\n" + "=" * 90)
    print("RESULTS ANALYSIS: GradNorm Dimension Fix")
    print("=" * 90)
    
    methods = EXPERIMENT_CONFIG["methods"]
    dims = EXPERIMENT_CONFIG["dims"]
    
    # Header
    header = f"{'dim':>5} | "
    header += " | ".join(f"{m:>25}" for m in methods)
    header += " | fix_vs_GN"
    print(f"\n{header}")
    print("-" * len(header))
    
    analysis = {}
    for dim in dims:
        row = f"{dim:>5} | "
        dim_vals = {}
        for method in methods:
            vals = []
            for seed in EXPERIMENT_CONFIG["seeds"]:
                key = make_key(dim, seed, method)
                if key in all_results:
                    vals.append(all_results[key]["test_value_mse"])
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                dim_vals[method] = (mean_v, std_v, vals)
                row += f" {mean_v:10.6f}±{std_v:.6f} |"
            else:
                row += f" {'N/A':>25} |"
        
        # Compute improvement ratios
        gn = dim_vals.get("dml_gradnorm", (None,))[0]
        fix_d = dim_vals.get("dml_dimnorm_gradnorm", (None,))[0]
        if gn and fix_d and gn > 0:
            ratio = gn / fix_d
            row += f" {ratio:7.2f}x"
        else:
            row += f" {'N/A':>8}"
        
        print(row)
        analysis[dim] = dim_vals
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    for method in methods:
        wins = 0
        total = 0
        for dim in dims:
            if method in analysis.get(dim, {}) and "dml_fixed" in analysis.get(dim, {}):
                m_val = analysis[dim][method][0]
                fixed_val = analysis[dim]["dml_fixed"][0]
                total += 1
                if m_val <= fixed_val * 1.05:  # within 5% of dml_fixed
                    wins += 1
        if total > 0:
            print(f"  {method:>30}: {wins}/{total} dims within 5% of dml_fixed")
    
    # Critical comparison: high-dim performance
    print("\n  HIGH-DIM (d≥50) comparison:")
    for dim in [50, 100]:
        if dim in analysis:
            for method in methods:
                if method in analysis[dim]:
                    v = analysis[dim][method][0]
                    print(f"    d={dim} {method:>30}: {v:.6f}")
    
    return analysis


def run_experiment():
    parser = argparse.ArgumentParser(description="GradNorm Dimension Fix Experiment")
    parser.add_argument("--gpu", type=int, default=None, help="Pin to specific GPU")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    results_dir = Path("results/gradnorm_fix")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = EXPERIMENT_CONFIG
    n_total = len(cfg["dims"]) * len(cfg["seeds"]) * len(cfg["methods"])
    
    print("=" * 70)
    print("GRADNORM DIMENSION FIX EXPERIMENT")
    print("=" * 70)
    print(f"Function:      {cfg['func_type']}")
    print(f"Dimensions:    {cfg['dims']}")
    print(f"Sample size:   {cfg['n_samples']}")
    print(f"Seeds:         {cfg['seeds']}")
    print(f"Methods:       {cfg['methods']}")
    print(f"Total exps:    {n_total}")
    print(f"Est. runtime:  ~{n_total * 25 / 60:.0f} minutes")
    if args.gpu is not None:
        print(f"GPU:           {args.gpu}")
    print()
    
    # Load existing results
    all_results = {}
    if args.resume:
        all_results = load_existing(results_dir)
        if all_results:
            print(f"Resuming: {len(all_results)} experiments found")
    
    errors = []
    completed = 0
    skipped = 0
    total_start = time.time()
    
    for di, dim in enumerate(cfg["dims"]):
        print(f"\n{'='*70}")
        print(f"[{di+1}/{len(cfg['dims'])}] dim={dim}")
        print(f"{'='*70}")
        
        for seed in cfg["seeds"]:
            # Generate data once per (dim, seed)
            try:
                data = generate_data(
                    cfg["func_type"], n_dim=dim,
                    n_samples=cfg["n_samples"], seed=seed
                )
                train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
            except Exception as e:
                msg = f"DATA GEN FAILED: dim={dim} seed={seed}: {e}"
                print(f"  ❌ {msg}")
                errors.append(msg)
                traceback.print_exc()
                continue
            
            x_train = train_data.x
            y_train = train_data.y
            dydx_train = train_data.dydx
            x_test = test_data.x
            y_test = test_data.y
            dydx_test = test_data.dydx
            
            for method in cfg["methods"]:
                key = make_key(dim, seed, method)
                
                if args.resume and key in all_results:
                    skipped += 1
                    continue
                
                elapsed_so_far = time.time() - total_start
                print(f"  {method} s={seed}...", end=" ", flush=True)
                t0 = time.time()
                
                try:
                    result = train_single_experiment(
                        x_train=x_train,
                        y_train=y_train,
                        dydx_train=dydx_train,
                        x_test=x_test,
                        y_test=y_test,
                        dydx_test=dydx_test,
                        method=method,
                        seed=seed,
                        pbar=False,
                        **TRAIN_HPARAMS,
                    )
                    elapsed = time.time() - t0
                    
                    result_dict = {
                        "method": method,
                        "func_type": cfg["func_type"],
                        "dim": dim,
                        "n_samples": cfg["n_samples"],
                        "noise_level": cfg["noise_level"],
                        "seed": seed,
                        "lambda": TRAIN_HPARAMS["lambda_"],
                        "test_value_mse": result.test_value_mse,
                        "test_grad_mse": result.test_grad_mse,
                        "best_epoch": result.best_epoch,
                        "time_s": elapsed,
                        "n_epochs_actual": len(result.training_logs),
                    }
                    
                    all_results[key] = result_dict
                    save_result(results_dir, key, result_dict)
                    
                    print(f"✅ MSE={result.test_value_mse:.6f} "
                          f"grad={result.test_grad_mse:.6f} "
                          f"best@{result.best_epoch} ({elapsed:.1f}s)")
                    completed += 1
                    
                except Exception as e:
                    elapsed = time.time() - t0
                    msg = f"TRAIN FAILED: {key}: {e}"
                    print(f"❌ ({elapsed:.1f}s) {msg}")
                    errors.append(msg)
                    traceback.print_exc()
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"DONE — {completed} completed, {skipped} skipped, {len(errors)} errors")
    print(f"Total time: {total_time/60:.1f} minutes")
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  - {e}")
    
    # Analyze results
    if all_results:
        analysis = analyze_results(all_results)
        
        # Save analysis
        analysis_path = results_dir / "analysis.json"
        analysis_data = {
            "config": {
                "experiment": EXPERIMENT_CONFIG,
                "hparams": TRAIN_HPARAMS,
            },
            "n_completed": completed,
            "n_skipped": skipped,
            "n_errors": len(errors),
            "total_time_s": total_time,
            "results": {k: v for k, v in all_results.items()},
            "timestamp": datetime.now().isoformat(),
        }
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    run_experiment()
