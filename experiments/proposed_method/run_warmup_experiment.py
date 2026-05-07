#!/usr/bin/env python3
"""
Experiment F: DML-Warmup — Vanilla pre-training + DML fine-tuning.

Hypothesis: GradNorm gives 68-106x better Greeks than vanilla on SPY with
only 2-3% worse value MSE (p>0.05 — not significant). The value penalty
comes from the derivative loss interfering EARLY in training before the
value function is learned. A warmup schedule that starts vanilla and
gradually introduces derivative supervision should achieve:
    1. Value MSE ≤ vanilla (because the first phase IS vanilla)
    2. Gradient MSE << vanilla (because the second phase uses DML)
    3. Better Pareto point than pure GradNorm

Method: dml_warmup
    Phase 1 (epochs 0..W):  Train vanilla (λ=0) → learns value function
    Phase 2 (epochs W..E):  Train DML with GradNorm → refines gradients
    The model checkpoint tracks best COMBINED val loss after phase 2
    begins, and best VALUE-ONLY val loss during phase 1.

Also tests: Heston Euler-LRM to confirm that warmup cannot fix bad labels.

Usage:
    python experiments/proposed_method/run_warmup_experiment.py --gpu 0
    python experiments/proposed_method/run_warmup_experiment.py --gpu 0 --spy
    python experiments/proposed_method/run_warmup_experiment.py --gpu 0 --heston
"""

import sys
import os
import time
import json
import copy
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from dml_benchmark.model import (
    DmlFeedForward, DmlLoss, VanillaLoss, DmlDataset,
    DataNormalizer, LossComponents, get_device
)
from dml_benchmark.loss_balancing import GradNormDmlLoss
from dml_benchmark.trainer import (
    DmlTrainer, set_deterministic, create_data_loaders, TrainingResult
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999]

SPY_HPARAMS = {
    "n_epochs": 500,
    "batch_size": 512,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

HESTON_HPARAMS = {
    "n_epochs": 500,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

# Warmup fractions to test: what fraction of epochs is vanilla-only
WARMUP_FRACTIONS = [0.3, 0.5, 0.7]


# ============================================================================
# TWO-PHASE TRAINING
# ============================================================================

def train_warmup_experiment(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_test,
    warmup_fraction=0.5,
    seed=42,
    pbar=False,
    **hparams,
):
    """
    Two-phase training: vanilla warmup → DML GradNorm fine-tuning.

    Phase 1 (vanilla): Learns value function without derivative noise.
    Phase 2 (DML GradNorm): Refines derivatives from a strong value init.

    Returns TrainingResult with combined metrics.
    """
    set_deterministic(seed)
    input_dim = x_train.shape[1]
    n_epochs = hparams.get("n_epochs", 500)
    warmup_epochs = int(n_epochs * warmup_fraction)
    finetune_epochs = n_epochs - warmup_epochs

    # Shared data loaders & normalizer
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        batch_size=hparams.get("batch_size", 256),
        seed=seed,
    )
    device = get_device()

    # ---- Phase 1: Vanilla warmup ----
    model = DmlFeedForward(
        input_dim=input_dim,
        output_dim=1,
        n_layers=hparams.get("n_layers", 4),
        hidden_size=hparams.get("hidden_size", 256),
        activation=hparams.get("activation", "softplus"),
    )

    vanilla_loss_fn = VanillaLoss()
    # J-M2 (2026-04-16): canonical warmup spec (AdamW wd=0 per D015,
    # phase-1 ES patience=50 per D033, phase-2 lr/10 per D022).
    optimizer_p1 = torch.optim.AdamW(
        model.parameters(), lr=hparams.get("lr", 0.005), weight_decay=0.0
    )
    scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20),
        min_lr=1e-6,
    )

    trainer_p1 = DmlTrainer(
        model=model,
        loss_fn=vanilla_loss_fn,
        optimizer=optimizer_p1,
        normalizer=normalizer,
        scheduler=scheduler_p1,
        use_dml=False,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )

    # Train phase 1
    result_p1 = trainer_p1.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=warmup_epochs,
        config={"phase": "vanilla_warmup"},
        pbar=pbar,
        early_stopping_patience=50,
    )

    # Restore best phase 1 model
    if trainer_p1.best_model_state is not None:
        model.load_state_dict(trainer_p1.best_model_state)
        model = model.to(device)

    phase1_val_loss = trainer_p1.best_val_loss

    # ---- Phase 2: DML GradNorm fine-tuning ----
    dml_loss_fn = GradNormDmlLoss(input_dim=input_dim)

    # J-M2: Use lower LR for fine-tuning (1/10 per 2026-04-14 ablation D022).
    finetune_lr = hparams.get("lr", 0.005) / 10.0
    optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.0)
    scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20),
        min_lr=1e-7,
    )

    trainer_p2 = DmlTrainer(
        model=model,
        loss_fn=dml_loss_fn,
        optimizer=optimizer_p2,
        normalizer=normalizer,
        scheduler=scheduler_p2,
        use_dml=True,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )

    # Train phase 2 with early stopping
    result_p2 = trainer_p2.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=finetune_epochs,
        config={"phase": "dml_gradnorm_finetune"},
        pbar=pbar,
        early_stopping_patience=50,
    )

    # Restore best phase 2 model
    if trainer_p2.best_model_state is not None:
        model.load_state_dict(trainer_p2.best_model_state)
        model = model.to(device)

    # ---- Evaluate ----
    # Need a temporary trainer for evaluation
    eval_trainer = DmlTrainer(
        model=model,
        loss_fn=dml_loss_fn,
        optimizer=optimizer_p2,
        normalizer=normalizer,
        use_dml=True,
    )
    test_metrics = eval_trainer.evaluate(test_loader)

    # Build combined result
    total_logs = result_p1.training_logs + result_p2.training_logs
    total_time = result_p1.total_time_s + result_p2.total_time_s
    best_epoch_overall = result_p1.best_epoch + warmup_epochs  # Relative to total

    result = TrainingResult(
        config={
            "method": "dml_warmup",
            "warmup_fraction": warmup_fraction,
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "finetune_lr": finetune_lr,
            "phase1_best_val_loss": float(phase1_val_loss),
        },
        final_train_loss=result_p2.final_train_loss,
        final_val_loss=float(trainer_p2.best_val_loss),
        test_value_mse=test_metrics["value_mse"],
        test_grad_mse=test_metrics["grad_mse"],
        training_logs=total_logs,
        total_time_s=total_time,
        best_epoch=best_epoch_overall,
    )

    return result


# ============================================================================
# UTILITIES
# ============================================================================

def make_key(experiment, warmup_frac, method, seed, extra=""):
    wf = f"w{int(warmup_frac*100)}"
    key = f"{experiment}_{wf}_{method}_s{seed}"
    if extra:
        key += f"_{extra}"
    return key


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
# SPY EXPERIMENTS
# ============================================================================

def run_spy_warmup(results_dir, resume=False, n_seeds=10):
    """Run warmup experiments on SPY data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F1: SPY — DML Warmup vs Vanilla vs GradNorm")
    print("=" * 70)

    from experiments.real_data_spy.spy_data_loader import load_spy_data

    existing = load_existing(results_dir) if resume else {}
    train_sizes = [10000, 50000]

    for n_train in train_sizes:
        for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
            print(f"\n--- n_train={n_train}, seed={seed} ({seed_idx+1}/{n_seeds}) ---")

            data = load_spy_data(
                n_train=n_train, n_test=10000,
                include_volume=False,
                stratify_by_moneyness=True, seed=seed,
            )

            # Run warmup variants
            for wf in WARMUP_FRACTIONS:
                key = make_key("spy", wf, "dml_warmup", seed, f"n{n_train}")

                if resume and key in existing:
                    print(f"  SKIP: warmup={wf} (exists)")
                    continue

                print(f"  warmup={wf:.0%} ...", end=" ", flush=True)
                t0 = time.time()

                try:
                    result = train_warmup_experiment(
                        x_train=data["x_train"],
                        y_train=data["y_train"],
                        dydx_train=data["dydx_train"],
                        x_test=data["x_test"],
                        y_test=data["y_test"],
                        dydx_test=data["dydx_test"],
                        warmup_fraction=wf,
                        seed=seed,
                        pbar=False,
                        **SPY_HPARAMS,
                    )
                    elapsed = time.time() - t0

                    result_dict = {
                        "method": "dml_warmup",
                        "warmup_fraction": wf,
                        "seed": seed,
                        "dataset": "spy_options",
                        "n_train": n_train,
                        "n_test": 10000,
                        "dim": 4,
                        "test_value_mse": float(result.test_value_mse),
                        "test_grad_mse": float(result.test_grad_mse),
                        "best_epoch": int(result.best_epoch),
                        "time_s": round(elapsed, 2),
                        "hparams": SPY_HPARAMS,
                        "phase1_val_loss": result.config.get("phase1_best_val_loss"),
                    }
                    save_result(results_dir, key, result_dict)

                    print(
                        f"val={result.test_value_mse:.6e}, "
                        f"grad={result.test_grad_mse:.6e}, "
                        f"t={elapsed:.1f}s"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()


# ============================================================================
# HESTON EXPERIMENTS
# ============================================================================

def run_heston_warmup(results_dir, resume=False, n_seeds=10):
    """Run warmup experiments on Heston Euler-LRM data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F2: HESTON — DML Warmup vs Vanilla (LRM labels)")
    print("=" * 70)

    from dml_benchmark.lrm_labels import lrm_euler_heston, prepare_for_training

    existing = load_existing(results_dir) if resume else {}

    payoffs = ["call", "digital"]
    n_steps_list = [50, 100, 252]
    n_samples = 1024
    k_paths = 10

    for payoff in payoffs:
        for n_steps in n_steps_list:
            for seed_idx, seed in enumerate(SEEDS[:n_seeds]):
                for wf in [0.5]:  # Only test 50% warmup for Heston
                    key = make_key(
                        "heston", wf, "dml_warmup", seed,
                        f"{payoff}_steps{n_steps}"
                    )

                    if resume and key in existing:
                        print(f"  SKIP: {key} (exists)")
                        continue

                    print(f"  {payoff} steps={n_steps} seed={seed} warmup={wf:.0%} ...",
                          end=" ", flush=True)
                    t0 = time.time()

                    try:
                        data = lrm_euler_heston(
                            n_samples=n_samples,
                            n_steps=n_steps,
                            k_paths=k_paths,
                            payoff_type=payoff,
                            seed=seed,
                        )
                        train_data = prepare_for_training(
                            data, test_frac=0.2, seed=seed
                        )

                        result = train_warmup_experiment(
                            x_train=train_data["x_train"],
                            y_train=train_data["y_train"],
                            dydx_train=train_data["dydx_train"],
                            x_test=train_data["x_test"],
                            y_test=train_data["y_test"],
                            dydx_test=train_data["dydx_test"],
                            warmup_fraction=wf,
                            seed=seed,
                            pbar=False,
                            **HESTON_HPARAMS,
                        )
                        elapsed = time.time() - t0

                        result_dict = {
                            "method": "dml_warmup",
                            "warmup_fraction": wf,
                            "seed": seed,
                            "payoff": payoff,
                            "model": "heston_euler_lrm",
                            "n_steps": n_steps,
                            "n_samples": n_samples,
                            "dim": 1,
                            "lrm_var_mean": float(np.mean(data["lrm_var"])),
                            "test_value_mse": float(result.test_value_mse),
                            "test_grad_mse": float(result.test_grad_mse),
                            "best_epoch": int(result.best_epoch),
                            "time_s": round(elapsed, 2),
                            "hparams": HESTON_HPARAMS,
                        }
                        save_result(results_dir, key, result_dict)

                        print(
                            f"val={result.test_value_mse:.6e}, "
                            f"grad={result.test_grad_mse:.6e}, "
                            f"t={elapsed:.1f}s"
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")
                        traceback.print_exc()


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_warmup_results(results_dir):
    """Analyze warmup results and compare to baselines."""
    print("\n" + "=" * 70)
    print("WARMUP EXPERIMENT — RESULTS ANALYSIS")
    print("=" * 70)

    results = load_existing(results_dir)
    if not results:
        print("No results found.")
        return

    # Also load baseline results for comparison
    from experiments.analyze_new_experiments import load_spy_results, load_lrm_results

    spy_baselines = load_spy_results()
    lrm_baselines = load_lrm_results()

    # ---- SPY Analysis ----
    spy_warmup = {k: v for k, v in results.items() if "spy" in k}
    if spy_warmup:
        print(f"\n--- SPY Warmup Results ({len(spy_warmup)} experiments) ---")

        # Group by (n_train, warmup_fraction)
        from collections import defaultdict
        groups = defaultdict(list)
        for r in spy_warmup.values():
            groups[(r["n_train"], r["warmup_fraction"])].append(r)

        # Get baselines
        spy_by_ntrain = defaultdict(lambda: defaultdict(list))
        for r in spy_baselines.values():
            spy_by_ntrain[r["n_train"]][r["method"]].append(r)

        for n_train in sorted(set(k[0] for k in groups)):
            print(f"\n  n_train = {n_train}:")
            print(f"    {'Method':<30} {'Mean Val MSE':>14} {'Std':>13} {'Mean Grad MSE':>14} {'Count':>6}")
            print(f"    {'-'*30} {'-'*14} {'-'*13} {'-'*14} {'-'*6}")

            # Baselines
            for m in ["vanilla", "dml_gradnorm", "dml_fixed"]:
                if m in spy_by_ntrain[n_train]:
                    vals = [r["test_value_mse"] for r in spy_by_ntrain[n_train][m]]
                    grads = [r["test_grad_mse"] for r in spy_by_ntrain[n_train][m]]
                    print(f"    {m:<30} {np.mean(vals):14.6e} {np.std(vals):13.6e} {np.mean(grads):14.6e} {len(vals):6d}")

            # Warmup variants
            for wf in WARMUP_FRACTIONS:
                key = (n_train, wf)
                if key in groups:
                    recs = groups[key]
                    vals = [r["test_value_mse"] for r in recs]
                    grads = [r["test_grad_mse"] for r in recs]
                    label = f"dml_warmup (w={wf:.0%})"
                    print(f"    {label:<30} {np.mean(vals):14.6e} {np.std(vals):13.6e} {np.mean(grads):14.6e} {len(vals):6d}")

            # Pareto analysis
            print(f"\n    Pareto (vs vanilla):")
            van_val = np.mean([r["test_value_mse"] for r in spy_by_ntrain[n_train].get("vanilla", [])])
            van_grad = np.mean([r["test_grad_mse"] for r in spy_by_ntrain[n_train].get("vanilla", [])])

            all_methods = []
            for m in ["dml_gradnorm", "dml_fixed"]:
                if m in spy_by_ntrain[n_train]:
                    mv = np.mean([r["test_value_mse"] for r in spy_by_ntrain[n_train][m]])
                    mg = np.mean([r["test_grad_mse"] for r in spy_by_ntrain[n_train][m]])
                    all_methods.append((m, mv, mg))

            for wf in WARMUP_FRACTIONS:
                key = (n_train, wf)
                if key in groups:
                    recs = groups[key]
                    mv = np.mean([r["test_value_mse"] for r in recs])
                    mg = np.mean([r["test_grad_mse"] for r in recs])
                    all_methods.append((f"warmup_w{int(wf*100)}", mv, mg))

            for name, mv, mg in all_methods:
                val_pen = (mv - van_val) / van_val * 100
                grad_imp = van_grad / mg if mg > 0 else float("inf")
                print(f"      {name:<25} val_penalty={val_pen:+.1f}%  grad_improvement={grad_imp:.0f}x")

    # ---- Heston Analysis ----
    heston_warmup = {k: v for k, v in results.items() if "heston" in k}
    if heston_warmup:
        print(f"\n--- Heston Warmup Results ({len(heston_warmup)} experiments) ---")
        from collections import defaultdict
        h_groups = defaultdict(list)
        for r in heston_warmup.values():
            h_groups[(r["payoff"], r["n_steps"])].append(r)

        # Get baselines
        h_baselines = defaultdict(lambda: defaultdict(list))
        for r in lrm_baselines.values():
            if "heston" in r.get("key", ""):
                h_baselines[(r["payoff"], r["n_steps"])][r["method"]].append(r)

        for config_key in sorted(h_groups.keys()):
            payoff, nsteps = config_key
            recs = h_groups[config_key]
            print(f"\n  {payoff} steps={nsteps}:")

            # Baselines
            for m in ["vanilla", "dml_fixed", "dml_gradnorm"]:
                if m in h_baselines[config_key]:
                    vals = [r["test_value_mse"] for r in h_baselines[config_key][m]]
                    print(f"    {m:<25} val_mse={np.mean(vals):12.4f}")

            # Warmup
            vals = [r["test_value_mse"] for r in recs]
            print(f"    {'dml_warmup (w=50%)':<25} val_mse={np.mean(vals):12.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DML Warmup Experiments")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--spy", action="store_true", help="Run SPY only")
    parser.add_argument("--heston", action="store_true", help="Run Heston only")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results_dir = Path("results/warmup_experiments")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analyze_warmup_results(results_dir)
        return

    run_spy_ = args.spy or (not args.spy and not args.heston)
    run_heston_ = args.heston or (not args.spy and not args.heston)

    if run_spy_:
        run_spy_warmup(results_dir, resume=args.resume, n_seeds=args.seeds)
    if run_heston_:
        run_heston_warmup(results_dir, resume=args.resume, n_seeds=args.seeds)

    analyze_warmup_results(results_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
