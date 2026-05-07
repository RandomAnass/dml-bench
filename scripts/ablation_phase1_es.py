#!/usr/bin/env python3
"""
Warmup phase-1 early-stopping ablation.

Question: should phase 1 of dml_warmup use early stopping, or run to a fixed
budget (current default)?

Argument for fixed: consistent starting point for phase 2, "best-model
tracking already restores best-val-MSE, so ES would only save compute".
Argument for ES: avoid wasted compute + potential overfitting drift if best-
model tracking misses (e.g., noisy val signal).

We test 3 phase-1 ES configurations across 5 targets covering both regimes
where DML works well (smooth payoffs) and where there are discontinuities
(barrier/digital):

ES configs:
  - es_off_p1            : phase 1 ES disabled (CURRENT default — patience > N/2)
  - es_on_p1_pat50       : phase 1 ES patience=50 on val total loss
  - es_on_p1_pat20       : phase 1 ES patience=20 (aggressive)

Targets:
  - digital_bs   (discontinuous payoff, 1D, DML/fuzzy wins big in v2)
  - barrier_bs   (discontinuous payoff with reflection, classic warmup target)
  - basket_d1    (smooth payoff, 1D)
  - ethanol      (molecular smooth, 9 atoms)
  - aspirin      (molecular smooth, 21 atoms)

Phase 2: ALWAYS uses lr/10 (post-warmup-LR-ablation default), GradNorm,
ES patience=50. Budget for phase 2 is FIXED regardless of when phase 1 stopped
(decoupled — so the metric isolates "did ES change the final model?").

Total: 3 configs × 5 targets × 5 splits/seeds = 75 runs MLP, ~3-5 hr.

Usage: python scripts/ablation_phase1_es.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

ES_CONFIGS = ["es_off_p1", "es_on_p1_pat50", "es_on_p1_pat20"]
MOLECULAR_TARGETS = ["ethanol", "aspirin"]
FINANCE_TARGETS = ["digital_bs", "barrier_bs", "basket_d1"]


def _phase1_patience(es_config: str, warmup_epochs: int) -> int:
    if es_config == "es_off_p1":
        return warmup_epochs + 1   # disable ES
    if es_config == "es_on_p1_pat50":
        return 50
    if es_config == "es_on_p1_pat20":
        return 20
    raise ValueError(f"Unknown es_config: {es_config}")


def _run_target_warmup(target, split_id_or_seed, es_config, n_epochs, gpu, is_molecular):
    """Run one (target, ES config, split/seed) warmup experiment.

    Phase 1: vanilla, with patience set per `es_config`.
    Phase 2: dml_gradnorm, lr/10, patience=50 (fixed regardless of phase 1).
    """
    import torch
    from dml_benchmark.trainer import DmlTrainer, create_data_loaders
    from dml_benchmark.model import DmlFeedForward, VanillaLoss
    from dml_benchmark.loss_balancing import GradNormDmlLoss

    if is_molecular:
        # J-H1 (2026-04-16): load_rmd17_flat was removed in the MLP runner
        # rewrite (flat XYZ → pairwise distances). Port to load_rmd17_pairwise.
        from experiments.molecular.run_mlp_molecular import load_rmd17_pairwise
        # I-H1 (2026-04-16): 5-tuple return signature.
        (x_tr, y_tr, dy_tr), (x_va, y_va, dy_va), (x_te, y_te, dy_te), meta, _ = load_rmd17_pairwise(
            target, split_id=split_id_or_seed,
            n_train=950, n_val=50, n_test=1000, verbose=False,
        )
    else:
        from experiments.unified_comparison.run_unified_experiment import generate_dataset
        data = generate_dataset(target, split_id_or_seed)
        x_tr_full = data["x_train"]
        y_tr_full = data["y_train"]
        dy_tr_full = data["dydx_lrm_train"]
        x_te = data["x_test"]
        y_te = data["y_test"]
        dy_te = data["dydx_eval_test"]
        # Carve a 20% val out of training for finance targets
        n_val = max(1, int(len(x_tr_full) * 0.2))
        rng = np.random.RandomState(split_id_or_seed)
        perm = rng.permutation(len(x_tr_full))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        x_tr = x_tr_full[tr_idx]; y_tr = y_tr_full[tr_idx]; dy_tr = dy_tr_full[tr_idx]
        x_va = x_tr_full[val_idx]; y_va = y_tr_full[val_idx]; dy_va = dy_tr_full[val_idx]
        meta = {"n_atoms": None, "input_dim": x_tr.shape[1]}

    input_dim = x_tr.shape[1]
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_tr, y_tr, dy_tr, x_te, y_te, dy_te,
        x_val=x_va, y_val=y_va, dydx_val=dy_va,
        batch_size=256, seed=split_id_or_seed,
    )

    base_lr = 0.005
    n_phase1_max = n_epochs // 2
    n_phase2 = n_epochs - n_phase1_max
    p1_patience = _phase1_patience(es_config, n_phase1_max)

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
    p1_result = trainer_p1.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_phase1_max,
        config={"phase": "vanilla_warmup", "es_config": es_config, "patience": p1_patience},
        pbar=False, early_stopping_patience=p1_patience,
    )
    p1_actual_epochs = len(p1_result.training_logs)
    if trainer_p1.best_model_state is not None:
        model.load_state_dict(trainer_p1.best_model_state)
    p1_best_val = float(trainer_p1.best_val_loss)
    p1_best_epoch = int(trainer_p1.best_epoch)

    # Phase 2 — gradnorm with lr/10 (post-warmup-LR-ablation default)
    p2_lr = base_lr / 10.0
    opt_p2 = torch.optim.AdamW(model.parameters(), lr=p2_lr, weight_decay=0.0)
    sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p2, mode="min", factor=0.5, patience=20, min_lr=1e-7,
    )
    # K-L4 (2026-04-16): pass explicit shared_layer_name for parity with
    # the main trainer (J4) and molecular runners (D031). For a 4-layer MLP
    # (layers.0..layers.4), layers.3 is the last hidden linear.
    n_layers = 4  # matches DmlFeedForward construction at line 118
    dml_loss = GradNormDmlLoss(
        input_dim=input_dim, shared_layer_name=f"layers.{n_layers - 1}.weight",
    )
    trainer_p2 = DmlTrainer(
        model=model, loss_fn=dml_loss, optimizer=opt_p2,
        normalizer=normalizer, scheduler=sched_p2, use_dml=True,
    )
    p2_result = trainer_p2.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_phase2,
        config={"phase": "dml_phase2", "es_config": es_config},
        pbar=False, early_stopping_patience=50,
    )
    p2_actual_epochs = len(p2_result.training_logs)
    if trainer_p2.best_model_state is not None:
        model.load_state_dict(trainer_p2.best_model_state)

    test_metrics = trainer_p2.evaluate(test_loader)

    return {
        "model": "MLP_flat",
        "target": target,
        "is_molecular": is_molecular,
        "es_config": es_config,
        "phase1_patience": p1_patience,
        "n_atoms": meta.get("n_atoms"),
        "input_dim": input_dim,
        "test_value_mse": float(test_metrics.get("value_mse", float("nan"))),
        "test_grad_mse": float(test_metrics.get("grad_mse", float("nan"))),
        "phase1_actual_epochs": p1_actual_epochs,
        "phase1_max_epochs": n_phase1_max,
        "phase1_best_val_loss": p1_best_val,
        "phase1_best_epoch": p1_best_epoch,
        "phase1_compute_saved_pct": round(100 * (1 - p1_actual_epochs / n_phase1_max), 1),
        "phase2_actual_epochs": p2_actual_epochs,
        "phase2_max_epochs": n_phase2,
        "phase2_lr": p2_lr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--targets", nargs="+",
                        default=MOLECULAR_TARGETS + FINANCE_TARGETS)
    parser.add_argument("--es_configs", nargs="+", default=ES_CONFIGS)
    parser.add_argument("--split_ids", type=str, default="1,2,3,4,5",
                        help="For molecular: canonical Figshare splits.")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000",
                        help="For finance: random seeds.")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--results_dir", default="results/ablation_phase1_es")
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

    print("=" * 80)
    print(f"Phase-1 ES ablation — {len(args.es_configs)} configs × {len(args.targets)} targets")
    print(f"ES configs: {args.es_configs}")
    print(f"Targets: {args.targets}")
    print(f"n_epochs: {args.n_epochs} (phase 1 max = {args.n_epochs // 2}, phase 2 = {args.n_epochs - args.n_epochs // 2})")
    print("=" * 80)

    n_done = 0; n_failed = 0
    for target in args.targets:
        is_mol = target in MOLECULAR_TARGETS
        iterators = split_ids if is_mol else seeds
        label = "split" if is_mol else "seed"
        for it in iterators:
            for es_cfg in args.es_configs:
                key = f"phase1_es_{target}_{label}{it}_{es_cfg}"
                print(f"\n--- {key} ---")
                t0 = time.time()
                try:
                    res = _run_target_warmup(target, it, es_cfg, args.n_epochs, args.gpu, is_mol)
                    res["key"] = key
                    res[label] = it
                    save_path = results_dir / f"{key}.json"
                    with open(save_path, "w") as f:
                        json.dump(res, f, indent=2, default=str)
                    print(f"  OK ({time.time()-t0:.1f}s)  val_MSE={res['test_value_mse']:.4e}  "
                          f"grad_MSE={res['test_grad_mse']:.4e}  "
                          f"p1_epochs={res['phase1_actual_epochs']}/{res['phase1_max_epochs']} "
                          f"({res['phase1_compute_saved_pct']:.0f}% saved)")
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    print(f"  FAIL: {e}")
                    traceback.print_exc()

    print(f"\nDone. {n_done} succeeded, {n_failed} failed.")


if __name__ == "__main__":
    main()
