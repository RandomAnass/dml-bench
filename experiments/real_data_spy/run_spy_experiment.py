#!/usr/bin/env python3
"""
Experiment E: SPY Real-World Options Data — DML with Analytical Greeks.

The first DML benchmark experiment on real market data. Uses 1.57M SPY
options records (2020-2022) with exact Black-Scholes Greeks as derivative
labels. No simulation needed.

Methods:
    - Vanilla: predict mid_price from (moneyness, T, r, iv)
    - DML fixed: add BS Greeks, fixed lambda=1 weighting
    - DML GradNorm: adaptive weighting
    - DML ReLoBRaLo: robust alternative
    - DML Warmup: vanilla warmup -> GradNorm fine-tuning (proposed method)

Split modes:
    - temporal (default): Train on 2020-01 to 2021-06, test on 2021-07+.
      5-day embargo gap. Standard for financial ML (Lopez de Prado 2018).
    - random (deprecated): Legacy random split with temporal leakage.

Usage:
    python experiments/real_data_spy/run_spy_experiment.py --gpu 0
    python experiments/real_data_spy/run_spy_experiment.py --gpu 0 --split-mode temporal
    python experiments/real_data_spy/run_spy_experiment.py --gpu 0 --resume

Expected runtime: ~30-60 min on 1 GPU
"""

import sys
import os
import time
import json
import argparse
import traceback
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.trainer import (
    train_single_experiment,
    DmlTrainer, create_data_loaders, get_device, set_deterministic,
    TrainingResult,
)
from dml_benchmark.model import DmlFeedForward, VanillaLoss
from dml_benchmark.loss_balancing import GradNormDmlLoss
from experiments.real_data_spy.spy_data_loader import load_spy_data


# ============================================================================
# CONFIGURATION
# ============================================================================

METHODS = [
    "vanilla", "dml_fixed", "dml_fixed_half",
    "dml_gradnorm", "dml_relobralo", "dml_warmup",
]

HPARAMS = {
    "n_epochs": 500,
    "batch_size": 512,        # Larger batch for 50K+ samples
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

WARMUP_FRACTION = 0.5  # 50% vanilla warmup, 50% GradNorm fine-tuning

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999]

# Training set sizes to test (stratified subsample from 1.57M)
TRAIN_SIZES = [10000, 50000]

# Test set size (fixed)
TEST_SIZE = 10000


# ============================================================================
# UTILITIES
# ============================================================================

def make_key(n_train, method, seed):
    return f"spy_n{n_train}_s{seed}_{method}"


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
# WARMUP TRAINING
# ============================================================================

def train_warmup_spy(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_test,
    warmup_fraction=WARMUP_FRACTION,
    seed=42, pbar=False, **hparams
):
    """
    Two-phase training: vanilla warmup -> DML GradNorm fine-tuning.

    Phase 1: vanilla (value-only) to learn the price surface.
    Phase 2: GradNorm to refine derivative predictions.
    """
    set_deterministic(seed)
    input_dim = x_train.shape[1]
    n_epochs = hparams.get("n_epochs", 500)
    warmup_epochs = int(n_epochs * warmup_fraction)
    finetune_epochs = n_epochs - warmup_epochs

    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        batch_size=hparams.get("batch_size", 512), seed=seed,
    )
    device = get_device()

    # Phase 1: Vanilla warmup
    model = DmlFeedForward(
        input_dim=input_dim, output_dim=1,
        n_layers=hparams.get("n_layers", 4),
        hidden_size=hparams.get("hidden_size", 256),
        activation=hparams.get("activation", "softplus"),
    )
    vanilla_loss_fn = VanillaLoss()
    # G-H6 (2026-04-16): align SPY warmup with canonical spec per
    # EVIDENCE/warmup_definition.md: AdamW (wd=0) per D015, phase-1 ES
    # enabled with patience=50 per D033, phase-2 LR drop = /10 per D022.
    opt_p1 = torch.optim.AdamW(
        model.parameters(), lr=hparams.get("lr", 0.005), weight_decay=0.0
    )
    sched_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p1, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20), min_lr=1e-6,
    )
    trainer_p1 = DmlTrainer(
        model=model, loss_fn=vanilla_loss_fn, optimizer=opt_p1,
        normalizer=normalizer, scheduler=sched_p1, use_dml=False,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )
    result_p1 = trainer_p1.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=warmup_epochs, config={"phase": "vanilla_warmup"},
        pbar=pbar, early_stopping_patience=50,
    )
    if trainer_p1.best_model_state is not None:
        model.load_state_dict(trainer_p1.best_model_state)
        model = model.to(device)

    # Phase 2: DML GradNorm fine-tuning — fresh AdamW, LR = α/10 (D022).
    dml_loss_fn = GradNormDmlLoss(input_dim=input_dim)
    finetune_lr = hparams.get("lr", 0.005) / 10.0
    opt_p2 = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.0)
    sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p2, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20), min_lr=1e-7,
    )
    trainer_p2 = DmlTrainer(
        model=model, loss_fn=dml_loss_fn, optimizer=opt_p2,
        normalizer=normalizer, scheduler=sched_p2, use_dml=True,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )
    result_p2 = trainer_p2.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=finetune_epochs, config={"phase": "dml_gradnorm_finetune"},
        pbar=pbar, early_stopping_patience=50,
    )
    if trainer_p2.best_model_state is not None:
        model.load_state_dict(trainer_p2.best_model_state)
        model = model.to(device)

    # Evaluate
    eval_trainer = DmlTrainer(
        model=model, loss_fn=dml_loss_fn, optimizer=opt_p2,
        normalizer=normalizer, use_dml=True,
    )
    test_metrics = eval_trainer.evaluate(test_loader)

    total_time = result_p1.total_time_s + result_p2.total_time_s
    return TrainingResult(
        config={"method": "dml_warmup", "warmup_fraction": warmup_fraction},
        final_train_loss=result_p2.final_train_loss,
        final_val_loss=float(trainer_p2.best_val_loss),
        test_value_mse=test_metrics["value_mse"],
        test_grad_mse=test_metrics["grad_mse"],
        training_logs=result_p1.training_logs + result_p2.training_logs,
        total_time_s=total_time,
        # J-M1 (2026-04-16): report absolute epoch of the returned model
        # (phase-2 best offset by phase-1 epoch count), matching the unified
        # runner's J9 fix. Returned model is phase-2's best, not phase-1's.
        best_epoch=warmup_epochs + result_p2.best_epoch,
    )


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_spy_experiments(results_dir, existing, resume, n_seeds, train_sizes,
                        split_mode="temporal", target_mode="bs_price"):
    """Run SPY options experiments across configurations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E: SPY Real-World Options — DML with BS Greeks")
    print("=" * 70)
    print("Data: SPY EOD options 2020-2022 (Kaggle, CC0)")
    print("Input: (moneyness, T, r, iv) — 4D")
    _target_label = {
        "bs_price": "BS formula price at market IV (Option A, H&S ground-truth)",
        "svi":      "BS formula price at SVI-fitted IV (Option C, smile-coherent)",
        "market":   "market mid (v3 mode)",
    }.get(target_mode, target_mode)
    print(f"Target mode: {target_mode} ({_target_label})")
    print("Derivatives: BS Greeks (delta, theta, rho, vega) — exact analytical")
    print(f"Split mode: {split_mode}")
    print(f"Train sizes: {train_sizes}")
    print(f"Test size:   {TEST_SIZE}")
    print(f"Methods:     {METHODS}")
    print(f"Seeds:       {SEEDS[:n_seeds]}")
    print()

    results = {}

    for n_train in train_sizes:
        for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
            print(f"\n--- n_train={n_train}, seed={seed} ({seed_idx+1}/{n_seeds}) ---")

            # Load data with proper temporal split
            t_load = time.time()
            data = load_spy_data(
                n_train=n_train,
                n_test=TEST_SIZE,
                include_volume=False,  # 4D: features with analytical derivs
                stratify_by_moneyness=True,
                seed=seed,
                split_mode=split_mode,
                target_mode=target_mode,
            )
            load_time = time.time() - t_load

            meta = data["metadata"]
            print(f"  Loaded in {load_time:.1f}s")
            print(f"  x_train: {data['x_train'].shape}, "
                  f"y range: [{data['y_train'].min():.4f}, {data['y_train'].max():.4f}]")
            print(f"  BS vs mid RMSE: {meta['bs_vs_mid_rmse']:.6f}")
            if split_mode == "temporal":
                print(f"  Train period: <= {meta['train_end_date']}, "
                      f"Test period: >= {meta['test_start_date']} "
                      f"(embargo: {meta['embargo_days']}d, "
                      f"excluded: {meta['n_embargo_excluded']})")

            for method in METHODS:
                key = make_key(n_train, method, seed)

                if resume and key in existing:
                    print(f"  SKIP (exists): {method}")
                    results[key] = existing[key]
                    continue

                print(f"  Training: {method}...", end=" ", flush=True)
                t0 = time.time()

                try:
                    if method == "dml_warmup":
                        result = train_warmup_spy(
                            x_train=data["x_train"],
                            y_train=data["y_train"],
                            dydx_train=data["dydx_train"],
                            x_test=data["x_test"],
                            y_test=data["y_test"],
                            dydx_test=data["dydx_test"],
                            warmup_fraction=WARMUP_FRACTION,
                            seed=seed,
                            pbar=False,
                            **HPARAMS,
                        )
                    else:
                        result = train_single_experiment(
                            x_train=data["x_train"],
                            y_train=data["y_train"],
                            dydx_train=data["dydx_train"],
                            x_test=data["x_test"],
                            y_test=data["y_test"],
                            dydx_test=data["dydx_test"],
                            method=method,
                            seed=seed,
                            pbar=False,
                            **HPARAMS,
                        )
                    elapsed = time.time() - t0

                    result_dict = {
                        "method": method,
                        "seed": seed,
                        "dataset": "spy_options",
                        "n_train": n_train,
                        "n_test": TEST_SIZE,
                        "dim": 4,
                        "split_mode": split_mode,
                        "test_value_mse": float(result.test_value_mse),
                        "test_grad_mse": float(result.test_grad_mse),
                        "best_epoch": int(result.best_epoch),
                        "time_s": round(elapsed, 2),
                        "bs_vs_mid_rmse": meta["bs_vs_mid_rmse"],
                        "metadata": meta,
                        "hparams": {k: v for k, v in HPARAMS.items()},
                    }

                    save_result(results_dir, key, result_dict)
                    results[key] = result_dict

                    print(
                        f"val={result.test_value_mse:.6e}, "
                        f"grad={result.test_grad_mse:.6e}, "
                        f"ep={result.best_epoch}, t={elapsed:.1f}s"
                    )

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"FAILED ({elapsed:.1f}s): {e}")
                    traceback.print_exc()

    return results


def analyze_and_print(results_dir):
    """Analyze SPY results."""
    existing = load_existing(results_dir)
    spy = {k: v for k, v in existing.items() if "spy" in k}

    if not spy:
        print("No SPY results found.")
        return

    print("\n" + "=" * 90)
    print("SPY OPTIONS EXPERIMENT — RESULTS ANALYSIS")
    print("=" * 90)

    # Group by (n_train, method)
    by_ntrain = {}
    for key, res in spy.items():
        n_train = res.get("n_train", 0)
        method = res.get("method", "?")
        if n_train not in by_ntrain:
            by_ntrain[n_train] = {}
        if method not in by_ntrain[n_train]:
            by_ntrain[n_train][method] = []
        by_ntrain[n_train][method].append(res)

    for n_train in sorted(by_ntrain.keys()):
        group = by_ntrain[n_train]
        print(f"\n--- n_train = {n_train} ---")
        print(f"  {'Method':<25} {'Mean Val MSE':>14} {'Std':>13} {'Mean Grad MSE':>14} {'Count':>6}")
        print(f"  {'-'*25} {'-'*14} {'-'*13} {'-'*14} {'-'*6}")

        best_val = float("inf")
        best_method = None

        for method in METHODS:
            if method in group:
                vals = [r["test_value_mse"] for r in group[method]]
                grads = [r["test_grad_mse"] for r in group[method]]
                mean_val = np.mean(vals)

                if mean_val < best_val:
                    best_val = mean_val
                    best_method = method

                print(
                    f"  {method:<25} {mean_val:14.6e} "
                    f"{np.std(vals):13.6e} "
                    f"{np.mean(grads):14.6e} {len(vals):6d}"
                )

        if best_method:
            print(f"  BEST: {best_method}")

        # DML vs vanilla improvement
        if "vanilla" in group and best_method and best_method != "vanilla":
            vanilla_mean = np.mean([r["test_value_mse"] for r in group["vanilla"]])
            improvement = (vanilla_mean - best_val) / vanilla_mean * 100
            print(f"  Improvement over vanilla: {improvement:.1f}%")

    # Overall comparison: BS model vs our DML
    bs_rmses = [r.get("bs_vs_mid_rmse", 0) for r in spy.values() if r.get("bs_vs_mid_rmse")]
    if bs_rmses:
        print(f"\n  BS analytical RMSE vs market: {np.mean(bs_rmses):.6f}")
        print(f"  (Our DML should approach or beat this)")


def main():
    parser = argparse.ArgumentParser(description="SPY Real-World Options Experiments")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument(
        "--n-train", type=int, nargs="+", default=None,
        help="Training sizes (default: 10000 50000)"
    )
    parser.add_argument(
        "--split-mode", type=str, default="temporal",
        choices=["temporal", "random"],
        help="Train/test split mode (default: temporal). "
             "'random' is deprecated due to temporal leakage."
    )
    parser.add_argument(
        "--target-mode", type=str, default="bs_price",
        choices=["bs_price", "market", "svi"],
        help="Regression target. 'bs_price' (default, Option A) = BS-formula "
             "price computed from raw market IV; matches H&S 2020 ground-truth "
             "training. 'svi' (Option C) = BS-formula price recomputed at the "
             "SVI-fitted IV per (date, maturity) slice (Gatheral & Jacquier "
             "2014); requires svi_iv.npy cache built by calibrate_svi.py. "
             "'market' = observed market mid; v3 mode, retained for reproducibility."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override the default results directory."
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Results directory based on split mode + target mode
    target_suffix = {
        "bs_price": "_optionA",
        "svi":      "_optionC",
        "market":   "",
    }.get(args.target_mode, "")
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    elif args.split_mode == "temporal":
        results_dir = Path("results/spy_options_temporal" + target_suffix)
    else:
        results_dir = Path("results/spy_options" + target_suffix)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analyze_and_print(results_dir)
        return

    existing = load_existing(results_dir) if args.resume else {}
    train_sizes = args.n_train if args.n_train else TRAIN_SIZES

    run_spy_experiments(
        results_dir, existing, args.resume,
        args.seeds, train_sizes, args.split_mode,
        target_mode=args.target_mode,
    )
    analyze_and_print(results_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
