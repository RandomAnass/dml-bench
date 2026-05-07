#!/usr/bin/env python3
"""
Re-run ALL vanilla experiments with autodiff gradient evaluation.

The original trainer.py evaluated vanilla gradient MSE by predicting zeros.
The original Huge & Savine DML paper evaluates vanilla gradients via backprop/autodiff.
Our code has been fixed (trainer.py FIX-5), but all existing vanilla result JSONs
have incorrect test_grad_mse values. This script re-runs them.

Strategy:
  - Read each existing vanilla JSON to get its config (func_type, dim, n_samples, seed, etc.)
  - Re-train the vanilla model with identical config
  - Evaluate with the fixed trainer (autodiff gradients)
  - Write corrected JSON to the same path (backup originals first)

Usage:
    # GPU 0 handles tier1+tier2+lrm_comparison
    CUDA_VISIBLE_DEVICES=0 conda run -n dml-bench-env python3 scripts/rerun_vanilla_autodiff.py --partition 0

    # GPU 1 handles tier3+tier4+spy+unified+rest
    CUDA_VISIBLE_DEVICES=1 conda run -n dml-bench-env python3 scripts/rerun_vanilla_autodiff.py --partition 1
"""
import sys
import os
import argparse
import json
import glob
import shutil
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

from dml_benchmark.trainer import train_single_experiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BACKUP_DIR = os.path.join(RESULTS_DIR, "_vanilla_zeros_backup")


def find_vanilla_files():
    """Find all vanilla result JSONs across the benchmark."""
    patterns = [
        "tier1_benchmark/*.json", "tier2_benchmark/*.json",
        "tier3_benchmark/*.json", "tier4_benchmark/*.json",
        "unified_comparison/multi_seed/*.json",
        "spy_options_temporal/*.json", "spy_options/*.json",
        "warmup_experiments/*.json", "compute_matched_controls/*.json",
        "realworld/*.json", "lrm_comparison/*.json",
        "gradient_noise_sweep/*.json", "gradnorm_fix/*.json",
    ]
    vanilla_files = []
    for pat in patterns:
        for f in glob.glob(os.path.join(RESULTS_DIR, pat)):
            try:
                d = json.load(open(f))
                if d.get("method") == "vanilla":
                    vanilla_files.append(f)
            except Exception:
                pass
    return sorted(vanilla_files)


def rerun_tier_vanilla(json_path):
    """
    Re-run a single tier-benchmark vanilla experiment.
    Reads config from the existing JSON, re-trains, writes corrected result.
    """
    d = json.load(open(json_path))

    func_type = d.get("func_type", "")
    dim = d.get("dim", 2)
    n_samples = d.get("n_samples", 1024)
    noise_level = d.get("noise_level", 0.0)
    seed = d.get("seed", 42)
    hparams = d.get("hparams", {})

    n_epochs = hparams.get("n_epochs", 500)
    batch_size = hparams.get("batch_size", 256)
    n_layers = hparams.get("n_layers", 4)
    hidden_size = hparams.get("hidden_size", 256)
    lr = hparams.get("lr", 0.005)
    activation = hparams.get("activation", "softplus")

    # Generate data (same pipeline as run_full_benchmark.py)
    from dml_benchmark.functions import generate_data, train_test_split, corrupt_derivatives

    data = generate_data(func_type, n_dim=dim, n_samples=n_samples, seed=seed)

    # Add noise if specified (same as run_full_benchmark.py)
    if noise_level > 0:
        data = corrupt_derivatives(data, noise_level=noise_level, seed=seed)

    # Train/test split (same function used by run_full_benchmark.py)
    train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
    x_train, y_train, dydx_train = train_data.x, train_data.y, train_data.dydx
    x_test, y_test, dydx_test = test_data.x, test_data.y, test_data.dydx

    # Re-train with fixed trainer (autodiff eval)
    result = train_single_experiment(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        method="vanilla",
        n_epochs=n_epochs, batch_size=batch_size,
        n_layers=n_layers, hidden_size=hidden_size,
        lr=lr, activation=activation, seed=seed,
        pbar=False
    )

    # Build corrected JSON
    corrected = dict(d)  # keep all original fields
    corrected["test_value_mse"] = result.test_value_mse
    corrected["test_grad_mse"] = result.test_grad_mse
    corrected["time_s"] = result.total_time_s
    corrected["best_epoch"] = result.best_epoch
    corrected["eval_mode"] = "autodiff"  # mark as corrected
    corrected["rerun_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=int, required=True, choices=[0, 1],
                        help="0=tier1+tier2+lrm, 1=tier3+tier4+spy+unified+rest")
    parser.add_argument("--dry-run", action="store_true", help="Count files only")
    args = parser.parse_args()

    vanilla_files = find_vanilla_files()
    print(f"Total vanilla files found: {len(vanilla_files)}")

    # Partition across GPUs
    tier12 = [f for f in vanilla_files if "/tier1_" in f or "/tier2_" in f or "/lrm_" in f]
    rest = [f for f in vanilla_files if f not in tier12]

    if args.partition == 0:
        files = tier12
        label = "GPU0 (tier1+tier2+lrm)"
    else:
        files = rest
        label = "GPU1 (tier3+tier4+spy+unified+rest)"

    print(f"Partition {args.partition} ({label}): {len(files)} files")

    if args.dry_run:
        return

    # Backup originals
    os.makedirs(BACKUP_DIR, exist_ok=True)

    done = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for i, fpath in enumerate(files):
        # Skip if already re-run
        try:
            existing = json.load(open(fpath))
            if existing.get("eval_mode") == "autodiff":
                skipped += 1
                continue
        except Exception:
            pass

        # Backup
        rel = os.path.relpath(fpath, RESULTS_DIR)
        backup_path = os.path.join(BACKUP_DIR, rel)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        if not os.path.exists(backup_path):
            shutil.copy2(fpath, backup_path)

        # Re-run
        try:
            corrected = rerun_tier_vanilla(fpath)
            with open(fpath, "w") as f:
                json.dump(corrected, f, indent=2)
            done += 1

            if done % 10 == 0 or done <= 3:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done - skipped) / rate / 3600 if rate > 0 else 0
                print(f"  [{done}/{len(files)}] {os.path.basename(fpath)}: "
                      f"val={corrected['test_value_mse']:.4e}, "
                      f"grad={corrected['test_grad_mse']:.4e} "
                      f"({rate:.1f}/s, ETA {eta:.1f}h)")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {os.path.basename(fpath)}: {e}")

    elapsed = time.time() - t_start
    print(f"\nDone: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total time: {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
