#!/usr/bin/env python3
"""
B2: Original H&S Code Comparison — Lambda_j Ablation

Instead of running H&S's slow manual-backprop code, this script isolates
the KEY algorithmic difference: lambda_j computation.

H&S lambda_j = 1 / sqrt(mean(dydx_normalized²))  [data-driven per-dim]
Ours lambda_j = x_std / y_std                       [simple ratio]

Both share: z-score normalization, 1/(1+λd) weighting, MSE loss.
The lambda_j difference can be up to 88× (for digital options).

This script runs our fast autograd framework with BOTH lambda_j formulas
and compares results, isolating the effect of this design choice.

Usage:
  python experiments/hs_comparison/run_lambda_j_ablation.py --gpu 0
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from copy import deepcopy

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "differential-ml"))

import torch
from dml_benchmark.trainer import (
    train_single_experiment, set_deterministic, create_data_loaders,
    DmlTrainer, DmlFeedForward, DmlLoss, VanillaLoss,
)
from dml_benchmark.model import DataNormalizer

# Import unified experiment data pipeline
sys.path.insert(0, str(ROOT / "experiments" / "unified_comparison"))
from run_unified_experiment import (
    generate_dataset, HPARAMS, SEEDS, DATA_CONFIG,
)

# H&S normalizer for comparison
from differential_ml.util.data_util import DataNormalizer as HsDataNormalizer

RESULTS_DIR = ROOT / "results" / "lambda_j_ablation"
SEEDS_SUBSET = SEEDS[:5]
DATASETS = list(DATA_CONFIG.keys())


def compute_hs_lambda_j(x_train, y_train, dydx_train):
    """Compute H&S lambda_j using their DataNormalizer."""
    hs_norm = HsDataNormalizer()
    # Ensure correct shapes for H&S normalizer
    y = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    if dydx_train.ndim == 2:
        dydx = dydx_train[:, np.newaxis, :]  # (n, 1, d)
    else:
        dydx = dydx_train
    hs_norm.initialize_with_data(x_train, y, dydx)
    return hs_norm.lambda_j.flatten()  # (d,)


def compute_our_lambda_j(x_train, y_train, dydx_train):
    """Compute our lambda_j using our DataNormalizer."""
    our_norm = DataNormalizer()
    our_norm.initialize_with_data(x_train, y_train, dydx_train)
    return np.atleast_1d(our_norm.lambda_j)  # (d,)


def train_with_custom_lambda_j(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_test,
    lambda_j_override,
    seed=42,
    **hparams,
):
    """
    Train dml_fixed with a custom lambda_j value.
    
    Monkeypatches the normalizer's lambda_j before training.
    """
    set_deterministic(seed)
    input_dim = x_train.shape[1]
    
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        batch_size=hparams.get("batch_size", 256),
        seed=seed,
    )
    
    # Override lambda_j
    normalizer.lambda_j = lambda_j_override
    
    model = DmlFeedForward(
        input_dim=input_dim, output_dim=1,
        n_layers=hparams.get("n_layers", 4),
        hidden_size=hparams.get("hidden_size", 256),
        activation=hparams.get("activation", "softplus"),
    )
    
    loss_fn = DmlLoss(
        lambda_=1.0,
        input_dim=input_dim,
        lambda_j=lambda_j_override,
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.get("lr", 0.005)
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
        use_dml=True,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )
    
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=hparams.get("n_epochs", 500),
        config={"method": "dml_fixed_custom_lj"},
        pbar=False,
    )
    
    test_metrics = trainer.evaluate(test_loader)
    result.test_value_mse = test_metrics["value_mse"]
    result.test_grad_mse = test_metrics["grad_mse"]
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = [args.dataset] if args.dataset else DATASETS
    
    print("=" * 70)
    print("B2: Lambda_j Ablation — H&S vs Ours")
    print("=" * 70)
    
    all_results = {}
    
    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")
        
        for seed in SEEDS_SUBSET:
            data = generate_dataset(ds_name, seed)
            
            # Compute both lambda_j values
            label_key = "dydx_pw_train" if "dydx_pw_train" in data else "dydx_lrm_train"
            hs_lj = compute_hs_lambda_j(data["x_train"], data["y_train"], data[label_key])
            our_lj = compute_our_lambda_j(data["x_train"], data["y_train"], data[label_key])
            
            ratio = hs_lj / (our_lj + 1e-30)
            print(f"\n  Seed {seed}: lambda_j H&S={hs_lj.mean():.4f}, "
                  f"Ours={our_lj.mean():.4f}, ratio(H&S/ours)={ratio.mean():.4f}")
            
            eval_dydx = data["dydx_eval_test"]
            
            for lj_name, lj_value in [("our_lambda_j", our_lj), ("hs_lambda_j", hs_lj)]:
                key = f"{ds_name}_{lj_name}_s{seed}"
                result_path = RESULTS_DIR / f"{key}.json"
                
                if args.resume and result_path.exists():
                    print(f"    {lj_name:15s}: exists, skipping")
                    with open(result_path) as f:
                        all_results[key] = json.load(f)
                    continue
                
                print(f"    {lj_name:15s}: ", end="", flush=True)
                
                try:
                    t0 = time.time()
                    result = train_with_custom_lambda_j(
                        x_train=data["x_train"],
                        y_train=data["y_train"],
                        dydx_train=data[label_key],
                        x_test=data["x_test"],
                        y_test=data["y_test"],
                        dydx_test=eval_dydx,
                        lambda_j_override=lj_value,
                        seed=seed,
                        **HPARAMS,
                    )
                    time_s = time.time() - t0
                    
                    record = {
                        "key": key,
                        "dataset": ds_name,
                        "lambda_j_source": lj_name,
                        "lambda_j_values": lj_value.tolist(),
                        "seed": seed,
                        "test_value_mse": result.test_value_mse,
                        "test_grad_mse": result.test_grad_mse,
                        "best_epoch": result.best_epoch,
                        "time_s": time_s,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    all_results[key] = record
                    with open(result_path, "w") as f:
                        json.dump(record, f, indent=2)
                    
                    print(f"val_mse={result.test_value_mse:.6f} "
                          f"grad_mse={result.test_grad_mse:.6f} "
                          f"({time_s:.1f}s, ep={result.best_epoch})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("LAMBDA_J ABLATION SUMMARY")
    print("=" * 70)
    
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    
    for key, record in all_results.items():
        group = f"{record['dataset']}_{record['lambda_j_source']}"
        summary[group]["value_mse"].append(record["test_value_mse"])
        summary[group]["grad_mse"].append(record["test_grad_mse"])
    
    print(f"\n{'Group':40s} | {'Value MSE':>18s} | {'Grad MSE':>18s}")
    print("-" * 85)
    
    # Group by dataset and show side-by-side
    for ds in datasets:
        for lj_src in ["our_lambda_j", "hs_lambda_j"]:
            group = f"{ds}_{lj_src}"
            if group in summary:
                vm = np.array(summary[group]["value_mse"])
                gm = np.array(summary[group]["grad_mse"])
                print(f"{group:40s} | {vm.mean():.6f} ± {vm.std():.4f} | "
                      f"{gm.mean():.6f} ± {gm.std():.4f}")
        print()
    
    # Save summary
    summary_data = {}
    for ds in datasets:
        ours_g = f"{ds}_our_lambda_j"
        hs_g = f"{ds}_hs_lambda_j"
        if ours_g in summary and hs_g in summary:
            vm_ours = np.array(summary[ours_g]["value_mse"])
            vm_hs = np.array(summary[hs_g]["value_mse"])
            gm_ours = np.array(summary[ours_g]["grad_mse"])
            gm_hs = np.array(summary[hs_g]["grad_mse"])
            summary_data[ds] = {
                "our_value_mse": float(vm_ours.mean()),
                "hs_value_mse": float(vm_hs.mean()),
                "value_mse_ratio": float(vm_ours.mean() / (vm_hs.mean() + 1e-30)),
                "our_grad_mse": float(gm_ours.mean()),
                "hs_grad_mse": float(gm_hs.mean()),
                "grad_mse_ratio": float(gm_ours.mean() / (gm_hs.mean() + 1e-30)),
            }
    
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
