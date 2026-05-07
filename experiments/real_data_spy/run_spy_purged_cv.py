#!/usr/bin/env python3
"""
Experiment E-CV: SPY Real-World Options — Purged Walk-Forward Cross-Validation.

Extends the temporal split SPY experiment with purged walk-forward CV
following Lopez de Prado (2018), "Advances in Financial Machine Learning,"
Chapter 7. Provides k-fold temporal CV with embargo gaps to prevent
information leakage from overlapping option contracts.

Methods: vanilla, dml_fixed, dml_gradnorm, dml_relobralo, dml_warmup
Folds: 5 (expanding window, ~6-month test segments, 5-day embargo)

Results stored in results/spy_options_purged_cv/ with per-fold and
aggregated metrics.

Usage:
    python experiments/real_data_spy/run_spy_purged_cv.py --gpu 0
    python experiments/real_data_spy/run_spy_purged_cv.py --gpu 0 --resume
    python experiments/real_data_spy/run_spy_purged_cv.py --gpu 0 --n-folds 5

Expected runtime: ~5-10 hours on 1 GPU (5 folds × 5 methods × 2 sizes × 10 seeds)
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
from experiments.real_data_spy.spy_data_loader import load_spy_data_purged_walkforward


# ============================================================================
# CONFIGURATION
# ============================================================================

METHODS = [
    "vanilla", "dml_fixed", "dml_fixed_half",
    "dml_gradnorm", "dml_relobralo", "dml_warmup",
]

HPARAMS = {
    "n_epochs": 500,
    "batch_size": 512,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

WARMUP_FRACTION = 0.5

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999]
TRAIN_SIZES = [10000, 50000]
TEST_SIZE = 10000
N_FOLDS = 5


# ============================================================================
# UTILITIES
# ============================================================================

def make_key(n_train, method, seed, fold_idx):
    return f"spy_cv_n{n_train}_s{seed}_f{fold_idx}_{method}"


def load_existing(results_dir):
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            if f.name.startswith(("summary", "analysis", "ANALYSIS")):
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
# WARMUP TRAINING (same as run_spy_experiment.py)
# ============================================================================

def train_warmup_spy(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_test,
    warmup_fraction=WARMUP_FRACTION,
    seed=42, pbar=False, **hparams
):
    """Two-phase training: vanilla warmup -> DML GradNorm fine-tuning."""
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
    # J-M2 (2026-04-16): canonical warmup spec per EVIDENCE/warmup_definition.md
    # (AdamW wd=0 per D015, phase-1 ES patience=50 per D033, phase-2 lr/10 per D022).
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

    # Phase 2: DML GradNorm fine-tuning
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
        # L-H3 (2026-04-16): report absolute phase-2 best epoch (matches
        # unified runner + SPY main runner per J-M1).
        best_epoch=warmup_epochs + result_p2.best_epoch,
    )


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_purged_cv_experiments(results_dir, existing, resume, n_seeds,
                              train_sizes, n_folds, target_mode="bs_price"):
    """Run SPY purged walk-forward CV experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E-CV: SPY — Purged Walk-Forward Cross-Validation")
    print("=" * 70)
    print(f"Data:     SPY EOD options 2020-2022 (Kaggle, CC0)")
    print(f"Folds:    {n_folds} (expanding window, 5-day embargo)")
    print(f"Methods:  {METHODS}")
    print(f"Seeds:    {SEEDS[:n_seeds]}")
    print(f"Sizes:    {train_sizes}")
    total = len(train_sizes) * n_seeds * n_folds * len(METHODS)
    print(f"Total:    {total} experiments")
    print()

    results = {}
    done_count = 0

    for n_train in train_sizes:
        for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
            print(f"\n{'='*60}")
            print(f"n_train={n_train}, seed={seed} ({seed_idx+1}/{n_seeds})")
            print(f"{'='*60}")

            # Load all folds for this seed
            t_load = time.time()
            try:
                folds = load_spy_data_purged_walkforward(
                    n_train=n_train,
                    n_test=TEST_SIZE,
                    include_volume=False,
                    stratify_by_moneyness=True,
                    seed=seed,
                    n_folds=n_folds,
                    target_mode=target_mode,
                )
            except Exception as e:
                print(f"  FAILED to load folds: {e}")
                traceback.print_exc()
                continue
            load_time = time.time() - t_load
            print(f"  Loaded {n_folds} folds in {load_time:.1f}s")

            for fold_idx, fold_data in enumerate(folds):
                meta = fold_data["metadata"]
                print(f"\n  --- Fold {fold_idx}/{n_folds-1}: "
                      f"train<={meta['train_end_date']}, "
                      f"test=[{meta['test_start_date']}..{meta['test_end_date']}] "
                      f"(pool: {meta['train_pool_size']:,}/{meta['test_pool_size']:,}) ---")

                for method in METHODS:
                    key = make_key(n_train, method, seed, fold_idx)

                    if resume and key in existing:
                        print(f"    SKIP (exists): {method}")
                        results[key] = existing[key]
                        done_count += 1
                        continue

                    print(f"    Training: {method}...", end=" ", flush=True)
                    t0 = time.time()

                    try:
                        if method == "dml_warmup":
                            result = train_warmup_spy(
                                x_train=fold_data["x_train"],
                                y_train=fold_data["y_train"],
                                dydx_train=fold_data["dydx_train"],
                                x_test=fold_data["x_test"],
                                y_test=fold_data["y_test"],
                                dydx_test=fold_data["dydx_test"],
                                warmup_fraction=WARMUP_FRACTION,
                                seed=seed,
                                pbar=False,
                                **HPARAMS,
                            )
                        else:
                            result = train_single_experiment(
                                x_train=fold_data["x_train"],
                                y_train=fold_data["y_train"],
                                dydx_train=fold_data["dydx_train"],
                                x_test=fold_data["x_test"],
                                y_test=fold_data["y_test"],
                                dydx_test=fold_data["dydx_test"],
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
                            "split_mode": "purged_walkforward",
                            "fold_idx": fold_idx,
                            "n_folds": n_folds,
                            "test_value_mse": float(result.test_value_mse),
                            "test_grad_mse": float(result.test_grad_mse),
                            "best_epoch": int(result.best_epoch),
                            "time_s": round(elapsed, 2),
                            "bs_vs_mid_rmse": meta.get("bs_vs_mid_rmse", 0),
                            "metadata": meta,
                            "hparams": {k: v for k, v in HPARAMS.items()},
                        }

                        save_result(results_dir, key, result_dict)
                        results[key] = result_dict
                        done_count += 1

                        print(
                            f"val={result.test_value_mse:.6e}, "
                            f"grad={result.test_grad_mse:.6e}, "
                            f"ep={result.best_epoch}, t={elapsed:.1f}s "
                            f"[{done_count}/{total}]"
                        )

                    except Exception as e:
                        elapsed = time.time() - t0
                        done_count += 1
                        print(f"FAILED ({elapsed:.1f}s): {e}")
                        traceback.print_exc()

    return results


def analyze_cv_results(results_dir, n_folds):
    """Analyze and print purged CV results summary."""
    existing = load_existing(results_dir)
    if not existing:
        print("No results found.")
        return

    print("\n" + "=" * 90)
    print("SPY PURGED WALK-FORWARD CV — RESULTS SUMMARY")
    print("=" * 90)

    # Group: (n_train, method) -> list of (seed, fold_idx, result)
    grouped = {}
    for key, res in existing.items():
        n_train = res.get("n_train", 0)
        method = res.get("method", "?")
        seed = res.get("seed", 0)
        fold_idx = res.get("fold_idx", 0)
        gkey = (n_train, method)
        if gkey not in grouped:
            grouped[gkey] = []
        grouped[gkey].append(res)

    for n_train in sorted(set(k[0] for k in grouped)):
        print(f"\n{'='*70}")
        print(f"n_train = {n_train}")
        print(f"{'='*70}")
        print(f"  {'Method':<20} {'Mean Val MSE':>14} {'Std':>12} "
              f"{'Mean Grad MSE':>14} {'Std':>12} {'N':>4}")
        print(f"  {'-'*20} {'-'*14} {'-'*12} {'-'*14} {'-'*12} {'-'*4}")

        vanilla_mean = None
        for method in METHODS:
            gkey = (n_train, method)
            if gkey not in grouped:
                continue
            results = grouped[gkey]

            # Average across seeds and folds
            val_mses = [r["test_value_mse"] for r in results]
            grad_mses = [r["test_grad_mse"] for r in results]

            mean_val = np.mean(val_mses)
            std_val = np.std(val_mses)
            mean_grad = np.mean(grad_mses)
            std_grad = np.std(grad_mses)

            if method == "vanilla":
                vanilla_mean = mean_val
                vanilla_grad = mean_grad

            print(f"  {method:<20} {mean_val:14.6e} {std_val:12.4e} "
                  f"{mean_grad:14.6e} {std_grad:12.4e} {len(results):4d}")

        # Improvement summary
        if vanilla_mean is not None:
            print()
            print(f"  Improvement vs vanilla:")
            for method in METHODS:
                if method == "vanilla":
                    continue
                gkey = (n_train, method)
                if gkey not in grouped:
                    continue
                mean_val = np.mean([r["test_value_mse"] for r in grouped[gkey]])
                mean_grad = np.mean([r["test_grad_mse"] for r in grouped[gkey]])
                val_pct = (mean_val - vanilla_mean) / vanilla_mean * 100
                grad_imp = vanilla_grad / mean_grad if mean_grad > 0 else float("inf")
                sign = "+" if val_pct > 0 else ""
                print(f"    {method:<20}: val {sign}{val_pct:.1f}%, grad {grad_imp:.1f}x")

    # Per-fold breakdown
    print(f"\n{'='*70}")
    print("Per-fold breakdown (averaged across seeds):")
    for n_train in sorted(set(k[0] for k in grouped)):
        for fold_idx in range(n_folds):
            fold_results = {}
            for gkey, results in grouped.items():
                if gkey[0] != n_train:
                    continue
                method = gkey[1]
                fold_vals = [r["test_value_mse"] for r in results if r.get("fold_idx") == fold_idx]
                if fold_vals:
                    fold_results[method] = np.mean(fold_vals)

            if fold_results:
                vals_str = ", ".join(f"{m}: {v:.4e}" for m, v in fold_results.items())
                print(f"  n={n_train} fold {fold_idx}: {vals_str}")


def main():
    parser = argparse.ArgumentParser(
        description="SPY Purged Walk-Forward CV Experiments"
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument(
        "--n-train", type=int, nargs="+", default=None,
        help="Training sizes (default: 10000 50000)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=N_FOLDS,
        help=f"Number of CV folds (default: {N_FOLDS})"
    )
    parser.add_argument(
        "--target-mode", type=str, default="bs_price",
        choices=["bs_price", "market", "svi"],
        help="Regression target. 'bs_price' (default, Option A) = BS-formula "
             "price at raw market IV; 'svi' (Option C) = BS price at "
             "SVI-fitted IV (requires svi_iv.npy cache); 'market' = observed "
             "market mid (v3 mode)."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override the default results directory."
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    target_suffix = {
        "bs_price": "_optionA",
        "svi":      "_optionC",
        "market":   "",
    }.get(args.target_mode, "")
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path("results/spy_options_purged_cv" + target_suffix)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_folds = args.n_folds

    if args.analyze_only:
        analyze_cv_results(results_dir, n_folds)
        return

    existing = load_existing(results_dir) if args.resume else {}
    train_sizes = args.n_train if args.n_train else TRAIN_SIZES

    run_purged_cv_experiments(
        results_dir, existing, args.resume,
        args.seeds, train_sizes, n_folds,
        target_mode=args.target_mode,
    )
    analyze_cv_results(results_dir, n_folds)

    print("\nDone!")


if __name__ == "__main__":
    main()
