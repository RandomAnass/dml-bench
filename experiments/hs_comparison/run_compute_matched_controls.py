#!/usr/bin/env python3
"""
B3: Compute-Matched Controls for Warmup Fairness Analysis

The warmup method uses 2 phases (250 + 250 epochs), with Phase 1 having
early stopping disabled (patience=251). This gives warmup ~4.35× more
wall-clock compute than vanilla (which early-stops around epoch 39-40).

This script runs compute-matched controls to isolate the warmup paradigm's
contribution from the extra-compute effect:

Controls:
  1. dml_fixed_no_es: dml_fixed with 500 epochs, NO early stopping
  2. vanilla_500_no_es: vanilla with 500 epochs, NO early stopping
  3. dml_gradnorm_no_es: dml_gradnorm with 500 epochs, NO early stopping

If warmup still outperforms these controls, the advantage is from the
TWO-PHASE PARADIGM, not just from running more epochs.

Usage:
  python experiments/hs_comparison/run_compute_matched_controls.py --gpu 0
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from dml_benchmark.trainer import (
    train_single_experiment, set_deterministic, create_data_loaders,
    DmlTrainer, DmlFeedForward, DmlLoss, VanillaLoss, TrainingResult
)
from dml_benchmark.loss_balancing import GradNormDmlLoss

# Import the unified experiment's data pipeline
sys.path.insert(0, str(ROOT / "experiments" / "unified_comparison"))
from run_unified_experiment import (
    generate_dataset, HPARAMS, SEEDS, DATA_CONFIG,
    train_warmup, WARMUP_FRACTION,
)

RESULTS_DIR = ROOT / "results" / "compute_matched_controls"

# Datasets to test (same as unified)
DATASETS = list(DATA_CONFIG.keys())

# Methods: baseline (with early stopping) + compute-matched controls (no ES)
CONTROL_METHODS = [
    "vanilla",            # baseline: 500 epochs, patience=50
    "dml_fixed",          # baseline: 500 epochs, patience=50
    "dml_gradnorm",       # baseline: 500 epochs, patience=50
    "dml_warmup",         # 250+250 epochs, Phase 1 no ES, Phase 2 patience=50
    "vanilla_no_es",      # control: 500 epochs, no early stopping
    "dml_fixed_no_es",    # control: 500 epochs, no early stopping
    "dml_gradnorm_no_es", # control: 500 epochs, no early stopping
]

# Use 5 seeds for this analysis
SEEDS_SUBSET = SEEDS[:5]


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_no_early_stopping(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_test,
    method="dml_fixed",
    seed=42,
    pbar=False,
    **hparams
):
    """
    Train with early stopping DISABLED (patience = n_epochs + 1).
    
    This matches the compute budget of warmup Phase 1, which also
    disables early stopping.
    """
    set_deterministic(seed)
    input_dim = x_train.shape[1]
    n_epochs = hparams.get("n_epochs", 500)
    
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        batch_size=hparams.get("batch_size", 256),
        seed=seed,
    )
    
    model = DmlFeedForward(
        input_dim=input_dim, output_dim=1,
        n_layers=hparams.get("n_layers", 4),
        hidden_size=hparams.get("hidden_size", 256),
        activation=hparams.get("activation", "softplus"),
    )
    
    if method == "vanilla_no_es":
        loss_fn = VanillaLoss()
        use_dml = False
    elif method == "dml_fixed_no_es":
        loss_fn = DmlLoss(
            lambda_=1.0,
            input_dim=input_dim,
            lambda_j=normalizer.lambda_j,
        )
        use_dml = True
    elif method == "dml_gradnorm_no_es":
        loss_fn = GradNormDmlLoss(input_dim=input_dim)
        use_dml = True
    else:
        raise ValueError(f"Unknown no-ES method: {method}")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.get("lr", 0.005),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20),
        min_lr=1e-6,
    )
    
    trainer = DmlTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        normalizer=normalizer,
        scheduler=scheduler,
        use_dml=use_dml,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )
    
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        config={"method": method, "early_stopping": False},
        pbar=pbar,
        early_stopping_patience=n_epochs + 1,  # DISABLE early stopping
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    result.test_value_mse = test_metrics["value_mse"]
    result.test_grad_mse = test_metrics["grad_mse"]
    
    return result


def run_single_control(
    method: str,
    data: dict,
    seed: int,
    hparams: dict,
    pbar: bool = False,
) -> dict:
    """Run a single method with compute tracking."""
    t0 = time.time()
    
    eval_y_test = data["y_test"]
    eval_dydx_test = data["dydx_eval_test"]
    
    if method in ("vanilla", "dml_fixed", "dml_gradnorm"):
        # Standard methods with early stopping
        dml_method_map = {
            "vanilla": "vanilla",
            "dml_fixed": "dml_fixed",
            "dml_gradnorm": "dml_gradnorm",
        }
        label_key = "dydx_pw_train" if "dydx_pw_train" in data else "dydx_lrm_train"
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data[label_key],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method=dml_method_map[method],
            lambda_=1.0 if method == "dml_fixed" else None,
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_warmup":
        label_key = "dydx_pw_train" if "dydx_pw_train" in data else "dydx_lrm_train"
        result = train_warmup(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data[label_key],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            warmup_fraction=WARMUP_FRACTION,
            seed=seed, pbar=pbar, **hparams,
        )
    elif method.endswith("_no_es"):
        label_key = "dydx_pw_train" if "dydx_pw_train" in data else "dydx_lrm_train"
        result = train_no_early_stopping(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data[label_key],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method=method,
            seed=seed, pbar=pbar, **hparams,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    time_s = time.time() - t0
    
    return {
        "test_value_mse": result.test_value_mse,
        "test_grad_mse": result.test_grad_mse,
        "time_s": time_s,
        "best_epoch": result.best_epoch,
        "total_epochs": result.config.get("n_epochs", hparams.get("n_epochs", 500)) if hasattr(result, "config") else 500,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run specific dataset only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = [args.dataset] if args.dataset else DATASETS
    
    print("=" * 70)
    print("B3: Compute-Matched Controls for Warmup Fairness")
    print("=" * 70)
    
    all_results = {}
    
    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")
        
        for seed in SEEDS_SUBSET:
            print(f"\n  Seed: {seed}")
            data = generate_dataset(ds_name, seed)
            
            for method in CONTROL_METHODS:
                key = f"{ds_name}_{method}_s{seed}"
                result_path = RESULTS_DIR / f"{key}.json"
                
                if args.resume and result_path.exists():
                    print(f"    {method:25s}: exists, skipping")
                    with open(result_path) as f:
                        all_results[key] = json.load(f)
                    continue
                
                print(f"    {method:25s}: ", end="", flush=True)
                try:
                    result = run_single_control(
                        method, data, seed, HPARAMS, pbar=False
                    )
                    
                    record = {
                        "key": key,
                        "dataset": ds_name,
                        "method": method,
                        "seed": seed,
                        "test_value_mse": result["test_value_mse"],
                        "test_grad_mse": result["test_grad_mse"],
                        "time_s": result["time_s"],
                        "best_epoch": result["best_epoch"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    all_results[key] = record
                    with open(result_path, "w") as f:
                        json.dump(record, f, indent=2)
                    
                    print(f"val_mse={result['test_value_mse']:.6f} "
                          f"grad_mse={result['test_grad_mse']:.6f} "
                          f"({result['time_s']:.1f}s, ep={result['best_epoch']})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPUTE FAIRNESS SUMMARY")
    print("=" * 70)
    
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    
    for key, record in all_results.items():
        group = f"{record['dataset']}_{record['method']}"
        summary[group]["value_mse"].append(record["test_value_mse"])
        summary[group]["grad_mse"].append(record["test_grad_mse"])
        summary[group]["time_s"].append(record["time_s"])
    
    print(f"\n{'Group':40s} | {'Value MSE':>14s} | {'Grad MSE':>14s} | {'Time':>8s}")
    print("-" * 85)
    
    for group in sorted(summary.keys()):
        vm = np.array(summary[group]["value_mse"])
        gm = np.array(summary[group]["grad_mse"])
        ts = np.array(summary[group]["time_s"])
        print(f"{group:40s} | {vm.mean():.6f}±{vm.std():.4f} | "
              f"{gm.mean():.6f}±{gm.std():.4f} | {ts.mean():.1f}s")
    
    # Save summary
    summary_data = {
        group: {
            "value_mse_mean": float(np.mean(summary[group]["value_mse"])),
            "value_mse_std": float(np.std(summary[group]["value_mse"])),
            "grad_mse_mean": float(np.mean(summary[group]["grad_mse"])),
            "grad_mse_std": float(np.std(summary[group]["grad_mse"])),
            "time_s_mean": float(np.mean(summary[group]["time_s"])),
            "n_seeds": len(summary[group]["value_mse"]),
        }
        for group in sorted(summary.keys())
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
