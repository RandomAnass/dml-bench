#!/usr/bin/env python3
"""
Retrain vanilla and dml_fuzzy_warmup × 5 seeds × 2 spot ranges, save
fine-grid predictions for the narrow-vs-wide comparison figure.

We bypass `train_single_experiment` and use `DmlTrainer` + the helper
`_train_warmup` directly so we can keep the trained model and predict
on a uniform spot grid.

Usage:
    python experiments/heston_barrier_4way/retrain_for_predictions.py --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dml_benchmark.model import DmlFeedForward, DmlLoss, VanillaLoss
from dml_benchmark.trainer import DmlTrainer, set_deterministic, create_data_loaders
from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference

import experiments.heston_barrier_4way.run_pilot as rp
from dml_benchmark.fuzzy_smoothing import fuzzy_barrier_heston

PRED_DIR = REPO_ROOT / "results" / "heston_barrier_4way" / "predictions"

SEEDS = [42, 123, 456, 789, 1337]
HPARAMS = dict(rp.HPARAMS)
HPARAMS["early_stopping_patience"] = 200

N_FINE = 200
GT_PATHS = 100_000
GT_SEED = 999


def gen_uniform_grid(spot_low_mult: float, spot_high_mult: float) -> np.ndarray:
    return np.linspace(
        rp.HESTON_PARAMS["strike"] * spot_low_mult,
        rp.HESTON_PARAMS["strike"] * spot_high_mult,
        N_FINE,
    ).reshape(-1, 1)


def compute_gt(setup_dir: Path, S0_grid: np.ndarray):
    out = setup_dir / "gt.npz"
    if out.exists():
        print(f"  GT cached: {out}")
        return dict(np.load(out))
    print(f"  Computing GT on {N_FINE}-point uniform grid (100k MC paths)...")
    t0 = time.time()
    gt = heston_barrier_doc_mc_reference(
        S0=S0_grid.flatten(),
        n_paths=GT_PATHS,
        seed=GT_SEED,
        **rp.HESTON_PARAMS,
    )
    np.savez(
        out,
        x=gt["x"].flatten(),
        y=gt["y"].flatten(),
        dydx=gt["dydx"].flatten(),
        std_err_price=gt["std_err_price"].flatten(),
        std_err_delta=gt["std_err_delta"].flatten(),
    )
    print(f"  GT saved in {time.time() - t0:.1f}s -> {out}")
    return dict(np.load(out))


def gen_training_data(method: str, seed: int) -> dict:
    """Use existing run_pilot helpers (respects SPOT_LOW/HIGH set on rp)."""
    if method == "vanilla":
        return rp._pathwise_barrier_heston(seed=seed)
    elif method == "dml_fuzzy_warmup":
        return fuzzy_barrier_heston(
            n_samples=rp.N_SAMPLES, k_paths=rp.K_PATHS, seed=seed,
            spot_low_mult=rp.SPOT_LOW_MULT, spot_high_mult=rp.SPOT_HIGH_MULT,
            **rp.HESTON_PARAMS,
        )
    else:
        raise ValueError(method)


def predict_grid(model, normalizer, x_grid: np.ndarray, device):
    """Predict (y, dy/dx) on raw x_grid by normalising → forward → unnormalising."""
    model.eval()
    x_norm_np = normalizer.normalize_x(x_grid)
    x_t = torch.tensor(x_norm_np, dtype=torch.float32, device=device)
    x_t.requires_grad_(True)
    y_norm_t, dydx_norm_t = model.forward_with_greek(x_t)
    y_norm = y_norm_t.detach().cpu().numpy()
    dydx_norm = dydx_norm_t.detach().cpu().numpy()
    y_pred = normalizer.unscale_y(y_norm).flatten()
    dydx_pred = normalizer.unscale_dydx(dydx_norm).flatten()
    return y_pred, dydx_pred


def train_vanilla_and_predict(seed: int, x_grid: np.ndarray, gt: dict) -> dict:
    set_deterministic(seed)
    data = gen_training_data("vanilla", seed)
    x_train = data["x"]; y_train = data["y"]; dydx_train = data["dydx_pw"]
    x_test = gt["x"].reshape(-1, 1)
    y_test = gt["y"].reshape(-1, 1)
    dydx_test = gt["dydx"].reshape(-1, 1, 1)

    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train, x_test, y_test, dydx_test,
        batch_size=HPARAMS["batch_size"], seed=seed,
    )
    model = DmlFeedForward(
        input_dim=1, output_dim=1,
        n_layers=HPARAMS["n_layers"], hidden_size=HPARAMS["hidden_size"],
        activation=HPARAMS["activation"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=HPARAMS["lr"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=HPARAMS["scheduler_factor"],
        patience=HPARAMS["scheduler_patience"], min_lr=1e-6,
    )
    trainer = DmlTrainer(
        model=model, loss_fn=VanillaLoss(),
        optimizer=optimizer, normalizer=normalizer, scheduler=scheduler,
        use_dml=False, max_grad_norm=HPARAMS["max_grad_norm"],
    )
    config = {"method": "vanilla", "seed": seed, "n_train": len(x_train),
              "n_test": len(x_test), "input_dim": 1}
    trainer.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=HPARAMS["n_epochs"], config=config, pbar=False,
        early_stopping_patience=HPARAMS["early_stopping_patience"],
    )
    test_metrics = trainer.evaluate(test_loader)
    device = next(model.parameters()).device
    y_pred, dydx_pred = predict_grid(model, normalizer, x_grid, device)
    return {
        "x": x_grid.flatten(), "y_pred": y_pred, "dydx_pred": dydx_pred,
        "test_value_mse": float(test_metrics["value_mse"]),
        "test_grad_mse": float(test_metrics["grad_mse"]),
    }


def train_warmup_and_predict(seed: int, x_grid: np.ndarray, gt: dict) -> dict:
    """dml_fuzzy_warmup: fuzzy labels + 2-stage warmup. Inline the trainer
    so we can keep the model and predict on x_grid afterwards."""
    set_deterministic(seed)
    data = gen_training_data("dml_fuzzy_warmup", seed)
    x_train = data["x"]; y_train = data["y"]; dydx_train = data["dydx_fuzzy"]
    x_test = gt["x"].reshape(-1, 1)
    y_test = gt["y"].reshape(-1, 1)
    dydx_test = gt["dydx"].reshape(-1, 1, 1)

    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train, x_test, y_test, dydx_test,
        batch_size=HPARAMS["batch_size"], seed=seed,
    )
    model = DmlFeedForward(
        input_dim=1, output_dim=1,
        n_layers=HPARAMS["n_layers"], hidden_size=HPARAMS["hidden_size"],
        activation=HPARAMS["activation"],
    )

    # Warmup: stage 1 (price-only), stage 2 (price + grad). We follow the
    # in-trainer recipe: stage 1 uses VanillaLoss; stage 2 uses DmlLoss(λ=1).
    optimizer = torch.optim.AdamW(model.parameters(), lr=HPARAMS["lr"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=HPARAMS["scheduler_factor"],
        patience=HPARAMS["scheduler_patience"], min_lr=1e-6,
    )

    warmup_fraction = 0.2  # match other DML-warmup runs
    warmup_epochs = max(1, int(round(HPARAMS["n_epochs"] * warmup_fraction)))
    full_epochs = HPARAMS["n_epochs"] - warmup_epochs

    # Stage 1: vanilla
    trainer = DmlTrainer(
        model=model, loss_fn=VanillaLoss(),
        optimizer=optimizer, normalizer=normalizer, scheduler=scheduler,
        use_dml=False, max_grad_norm=HPARAMS["max_grad_norm"],
    )
    cfg1 = {"method": "warmup_stage1", "seed": seed, "n_train": len(x_train),
            "n_test": len(x_test), "input_dim": 1}
    trainer.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=warmup_epochs, config=cfg1, pbar=False,
        early_stopping_patience=HPARAMS["early_stopping_patience"],
    )

    # Stage 2: switch to DML loss; reuse the same model/optimizer/scheduler
    dml_loss = DmlLoss(lambda_=HPARAMS["lambda_"], input_dim=1,
                       lambda_j=normalizer.lambda_j)
    trainer2 = DmlTrainer(
        model=model, loss_fn=dml_loss,
        optimizer=optimizer, normalizer=normalizer, scheduler=scheduler,
        use_dml=True, max_grad_norm=HPARAMS["max_grad_norm"],
    )
    cfg2 = {"method": "warmup_stage2", "seed": seed, "n_train": len(x_train),
            "n_test": len(x_test), "input_dim": 1}
    trainer2.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=full_epochs, config=cfg2, pbar=False,
        early_stopping_patience=HPARAMS["early_stopping_patience"],
    )

    test_metrics = trainer2.evaluate(test_loader)
    device = next(model.parameters()).device
    y_pred, dydx_pred = predict_grid(model, normalizer, x_grid, device)
    return {
        "x": x_grid.flatten(), "y_pred": y_pred, "dydx_pred": dydx_pred,
        "test_value_mse": float(test_metrics["value_mse"]),
        "test_grad_mse": float(test_metrics["grad_mse"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    setup_specs = {
        "narrow": (0.7, 1.3),
        "wide":   (0.5, 1.5),
    }

    for setup, (spot_low, spot_high) in setup_specs.items():
        setup_dir = PRED_DIR / setup
        setup_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"SETUP: {setup} -> spot range [{spot_low}K, {spot_high}K]")
        print(f"{'=' * 70}")

        rp.SPOT_LOW_MULT = spot_low
        rp.SPOT_HIGH_MULT = spot_high
        rp.HPARAMS = HPARAMS
        rp.SEEDS = SEEDS

        x_grid = gen_uniform_grid(spot_low, spot_high)
        gt = compute_gt(setup_dir, x_grid)

        for method, fn in [
            ("vanilla", train_vanilla_and_predict),
            ("dml_fuzzy_warmup", train_warmup_and_predict),
        ]:
            for seed in SEEDS:
                out = setup_dir / f"{method}_seed{seed}.npz"
                if out.exists():
                    print(f"  SKIP (cached): {method} seed={seed}")
                    continue
                t0 = time.time()
                print(f"  Training {method} seed={seed}...", flush=True)
                preds = fn(seed, x_grid, gt)
                np.savez(out, **preds)
                print(f"    val_mse={preds['test_value_mse']:.4e}  "
                      f"grad_mse={preds['test_grad_mse']:.4e}  "
                      f"({time.time() - t0:.1f}s)", flush=True)

    print(f"\nAll predictions saved under {PRED_DIR}")


if __name__ == "__main__":
    main()
