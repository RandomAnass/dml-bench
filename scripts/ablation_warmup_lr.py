#!/usr/bin/env python3
"""
Warmup phase-2 LR strategy ablation.

The dml_warmup method is a novel contribution. Phase 2 (after the vanilla
warmup) needs a learning rate that lets the model adapt to the new (E,F)
loss without knocking out phase-1 weights. Default convention is lr/5 (from
unified_comparison/run_unified_experiment.py:443), but this is a heuristic
choice with no published convention.

This ablation tests 7 strategies × 3 targets × 5 splits/seeds × 1 architecture
(MLP — fastest; results inform the choice for all 3 archs in the main grid).

Strategies:
  - lr_no_drop   : lr (no change at phase transition)
  - lr_div_2     : lr / 2
  - lr_div_5     : lr / 5  (current convention — baseline)
  - lr_div_10    : lr / 10
  - linear_warmup: linear ramp from 0 to lr over first 25 P2 epochs
  - cosine_decay : cosine from lr to lr/100 over phase 2
  - adaptive     : lr (no drop) + ReduceLROnPlateau patience=10

Targets:
  - aspirin (rMD17 molecular, 21 atoms)
  - ethanol (rMD17 molecular, 9 atoms)
  - barrier_bs (Tier 2 finance, discontinuous payoff)

Usage: python scripts/ablation_warmup_lr.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STRATEGIES = [
    "lr_no_drop", "lr_div_2", "lr_div_5", "lr_div_10",
    "linear_warmup", "cosine_decay", "adaptive",
]
MOLECULAR_TARGETS = ["aspirin", "ethanol"]
FINANCE_TARGET = "barrier_bs"


def _strategy_to_kwargs(strategy: str, base_lr: float):
    """Map strategy name to (phase_2_lr, scheduler_cls_name, scheduler_args)."""
    if strategy == "lr_no_drop":
        return base_lr, None, None
    if strategy == "lr_div_2":
        return base_lr / 2.0, None, None
    if strategy == "lr_div_5":
        return base_lr / 5.0, None, None
    if strategy == "lr_div_10":
        return base_lr / 10.0, None, None
    if strategy == "linear_warmup":
        # Caller will need to plug a LinearLR scheduler in phase 2
        return base_lr, "LinearLR", {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 25}
    if strategy == "cosine_decay":
        return base_lr, "CosineAnnealingLR", {"T_max": 250, "eta_min": base_lr / 100}
    if strategy == "adaptive":
        return base_lr, "ReduceLROnPlateau", {"factor": 0.5, "patience": 10, "min_lr": 1e-7}
    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--targets", nargs="+", default=MOLECULAR_TARGETS + [FINANCE_TARGET])
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--split_ids", type=str, default="1,2,3,4,5",
                        help="For molecular targets: canonical splits.")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000",
                        help="For finance targets: random seeds.")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="Total epochs (phase 1 = N/2, phase 2 = N/2)")
    parser.add_argument("--results_dir", default="results/ablation_warmup_lr")
    args = parser.parse_args()

    split_ids = [int(s) for s in args.split_ids.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(4)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Warmup-LR ablation — {len(args.strategies)} strategies × {len(args.targets)} targets")
    print(f"Strategies: {args.strategies}")
    print(f"Targets: {args.targets}")
    print(f"n_epochs: {args.n_epochs} (phase 1 = {args.n_epochs // 2}, phase 2 = {args.n_epochs - args.n_epochs // 2})")
    print("=" * 70)

    n_done = 0
    n_failed = 0

    for target in args.targets:
        is_molecular = target in MOLECULAR_TARGETS
        iterators = split_ids if is_molecular else seeds
        iterator_label = "split" if is_molecular else "seed"

        for it_value in iterators:
            for strategy in args.strategies:
                key = f"warmup_lr_ablation_{target}_{iterator_label}{it_value}_{strategy}"
                print(f"\n--- {key} ---")

                t0 = time.time()
                try:
                    if is_molecular:
                        result = _run_molecular_warmup_ablation(
                            target, it_value, strategy, args.n_epochs, args.gpu,
                        )
                    else:
                        result = _run_finance_warmup_ablation(
                            target, it_value, strategy, args.n_epochs, args.gpu,
                        )
                    result["key"] = key
                    result["target"] = target
                    result["strategy"] = strategy
                    result[iterator_label] = it_value
                    save_path = results_dir / f"{key}.json"
                    with open(save_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"  OK ({time.time() - t0:.1f}s)  "
                          f"val_MSE={result.get('test_value_mse', '?')}  "
                          f"grad_MSE={result.get('test_grad_mse', '?')}")
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    import traceback
                    print(f"  FAIL: {e}")
                    traceback.print_exc()

    print(f"\nDone. {n_done} succeeded, {n_failed} failed.")


def _run_molecular_warmup_ablation(target, split_id, strategy, n_epochs, gpu):
    """MLP molecular warmup with custom phase-2 LR strategy.

    NOTE: this script intentionally re-implements the warmup loop here so we
    can plug in arbitrary phase-2 schedulers. Production warmup uses the
    canonical lr/5 drop in dml_benchmark.train_warmup. Only the LR strategy
    differs — all other components match.
    """
    import torch
    # J-H1 (2026-04-16): load_rmd17_flat was removed in the MLP runner rewrite.
    from experiments.molecular.run_mlp_molecular import load_rmd17_pairwise, HPARAMS
    from dml_benchmark.trainer import DmlTrainer, create_data_loaders
    from dml_benchmark.model import DmlFeedForward, VanillaLoss
    from dml_benchmark.loss_balancing import GradNormDmlLoss

    # I-H1 (2026-04-16): 5-tuple return signature.
    (x_tr, y_tr, dy_tr), (x_va, y_va, dy_va), (x_te, y_te, dy_te), meta, _ = load_rmd17_pairwise(
        target, split_id=split_id,
        n_train=950, n_val=50, n_test=1000, verbose=False,
    )
    input_dim = x_tr.shape[1]

    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_tr, y_tr, dy_tr, x_te, y_te, dy_te,
        x_val=x_va, y_val=y_va, dydx_val=dy_va,
        batch_size=HPARAMS["batch_size"], seed=split_id,
    )

    base_lr = HPARAMS["lr"]
    n_phase1 = n_epochs // 2
    n_phase2 = n_epochs - n_phase1

    # Phase 1 — vanilla
    model = DmlFeedForward(
        input_dim=input_dim, output_dim=1,
        n_layers=HPARAMS["n_layers"], hidden_size=HPARAMS["hidden_size"],
        activation=HPARAMS["activation"],
    )
    opt_p1 = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)
    sched_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p1, mode="min", factor=0.5, patience=20, min_lr=1e-6,
    )
    trainer_p1 = DmlTrainer(
        model=model, loss_fn=VanillaLoss(), optimizer=opt_p1,
        normalizer=normalizer, scheduler=sched_p1, use_dml=False,
    )
    trainer_p1.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_phase1, config={"phase": "vanilla_warmup", "strategy": strategy},
        pbar=False, early_stopping_patience=n_phase1 + 1,
    )
    if trainer_p1.best_model_state is not None:
        model.load_state_dict(trainer_p1.best_model_state)

    # Phase 2 — gradnorm with LR strategy
    p2_lr, sched_name, sched_args = _strategy_to_kwargs(strategy, base_lr)
    opt_p2 = torch.optim.AdamW(model.parameters(), lr=p2_lr, weight_decay=0.0)
    if sched_name == "LinearLR":
        sched_p2 = torch.optim.lr_scheduler.LinearLR(opt_p2, **sched_args)
    elif sched_name == "CosineAnnealingLR":
        sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, **sched_args)
    elif sched_name == "ReduceLROnPlateau":
        sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_p2, mode="min", **sched_args)
    else:
        sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_p2, mode="min", factor=0.5, patience=20, min_lr=1e-7,
        )

    dml_loss = GradNormDmlLoss(input_dim=input_dim)
    trainer_p2 = DmlTrainer(
        model=model, loss_fn=dml_loss, optimizer=opt_p2,
        normalizer=normalizer, scheduler=sched_p2, use_dml=True,
    )
    p2_result = trainer_p2.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_phase2, config={"phase": "dml_phase2", "strategy": strategy},
        pbar=False, early_stopping_patience=50,
    )
    if trainer_p2.best_model_state is not None:
        model.load_state_dict(trainer_p2.best_model_state)

    test_metrics = trainer_p2.evaluate(test_loader)
    return {
        "model": "MLP_flat_coords",
        "molecule": target,
        "n_atoms": meta["n_atoms"],
        "test_value_mse": float(test_metrics.get("value_mse", float("nan"))),
        "test_grad_mse": float(test_metrics.get("grad_mse", float("nan"))),
        "n_phase1_epochs": n_phase1,
        "n_phase2_epochs": n_phase2,
        "phase2_lr": p2_lr,
        "phase2_scheduler": sched_name,
    }


def _run_finance_warmup_ablation(target, seed, strategy, n_epochs, gpu):
    """Finance (barrier_bs etc.) warmup ablation.

    Uses the unified_comparison dataset generator for parity with v2 results.
    Pathwise labels (dydx_pw_train) are the DML baseline for barrier options.
    """
    import torch
    from experiments.unified_comparison.run_unified_experiment import (
        generate_dataset, DATA_CONFIG,
    )
    from dml_benchmark.trainer import DmlTrainer, create_data_loaders
    from dml_benchmark.model import DmlFeedForward, VanillaLoss
    from dml_benchmark.loss_balancing import GradNormDmlLoss

    data = generate_dataset(target, seed)
    # Use LRM labels for barrier_bs (pathwise is all-zero for digital-type payoffs;
    # LRM gives non-zero gradients, analytically comparable to the closed-form delta).
    x_train = data["x_train"]
    y_train = data["y_train"]
    dydx_train = data["dydx_lrm_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    dydx_test = data["dydx_eval_test"]  # analytical ground truth
    input_dim = x_train.shape[1]

    # Train/val/test split: use pathwise train set for training, 20% for val,
    # keep the provided test set separate (matches v2 warmup convention).
    n_val = max(1, int(len(x_train) * 0.2))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(x_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr = x_train[tr_idx]; y_tr = y_train[tr_idx]; dy_tr = dydx_train[tr_idx]
    x_va = x_train[val_idx]; y_va = y_train[val_idx]; dy_va = dydx_train[val_idx]

    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_tr, y_tr, dy_tr, x_test, y_test, dydx_test,
        x_val=x_va, y_val=y_va, dydx_val=dy_va,
        batch_size=256, seed=seed,
    )

    base_lr = 0.005  # v2 finance MLP default
    n_phase1 = n_epochs // 2
    n_phase2 = n_epochs - n_phase1

    # Phase 1 — vanilla
    model = DmlFeedForward(
        input_dim=input_dim, output_dim=1,
        n_layers=4, hidden_size=256, activation="softplus",
    )
    opt_p1 = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)
    sched_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p1, mode="min", factor=0.5, patience=20, min_lr=1e-6,
    )
    trainer_p1 = DmlTrainer(
        model=model, loss_fn=VanillaLoss(), optimizer=opt_p1,
        normalizer=normalizer, scheduler=sched_p1, use_dml=False,
    )
    trainer_p1.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_phase1, config={"phase": "vanilla_warmup", "strategy": strategy},
        pbar=False, early_stopping_patience=n_phase1 + 1,
    )
    if trainer_p1.best_model_state is not None:
        model.load_state_dict(trainer_p1.best_model_state)

    # Phase 2 — gradnorm with LR strategy
    p2_lr, sched_name, sched_args = _strategy_to_kwargs(strategy, base_lr)
    opt_p2 = torch.optim.AdamW(model.parameters(), lr=p2_lr, weight_decay=0.0)
    if sched_name == "LinearLR":
        sched_p2 = torch.optim.lr_scheduler.LinearLR(opt_p2, **sched_args)
    elif sched_name == "CosineAnnealingLR":
        sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, **sched_args)
    elif sched_name == "ReduceLROnPlateau":
        sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_p2, mode="min", **sched_args)
    else:
        sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_p2, mode="min", factor=0.5, patience=20, min_lr=1e-7,
        )

    dml_loss = GradNormDmlLoss(input_dim=input_dim)
    trainer_p2 = DmlTrainer(
        model=model, loss_fn=dml_loss, optimizer=opt_p2,
        normalizer=normalizer, scheduler=sched_p2, use_dml=True,
    )
    trainer_p2.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_phase2, config={"phase": "dml_phase2", "strategy": strategy},
        pbar=False, early_stopping_patience=50,
    )
    if trainer_p2.best_model_state is not None:
        model.load_state_dict(trainer_p2.best_model_state)

    test_metrics = trainer_p2.evaluate(test_loader)
    return {
        "model": "MLP_flat",
        "dataset": target,
        "test_value_mse": float(test_metrics.get("value_mse", float("nan"))),
        "test_grad_mse": float(test_metrics.get("grad_mse", float("nan"))),
        "n_phase1_epochs": n_phase1,
        "n_phase2_epochs": n_phase2,
        "phase2_lr": p2_lr,
        "phase2_scheduler": sched_name,
        "label_type": "dydx_lrm",
        "eval_source": data.get("eval_source", "?"),
    }


if __name__ == "__main__":
    main()
