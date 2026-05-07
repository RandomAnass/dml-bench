#!/usr/bin/env python3
"""
B2: Original Huge & Savine (2020) Code Comparison

Runs the H&S original PyTorch implementation on our benchmark datasets
and compares against our dml_fixed reimplementation.

Approach:
  1. Use H&S code from repos/differential-ml/pt/ (their official PyTorch port)
  2. Adapt to accept our data format
  3. Run on digital_bs & basket_d1 (their paper's primary datasets)
  4. Compare: value MSE, gradient MSE, training time
  5. Also test with G&K-like architecture (4×20 softplus)

Key differences being tested:
  - Architecture: H&S default (2×100 sigmoid) vs our (4×256 softplus)
  - lambda_j: H&S (1/RMS(dydx)) vs ours (x_std/y_std)
  - Optimizer: H&S (Adam lr=0.1) vs ours (Adam lr=0.005)
  - Epochs: H&S (100) vs ours (500 w/ early stopping)
  - Greeks: H&S (manual backprop) vs ours (autograd)

Usage:
  python experiments/hs_comparison/run_hs_comparison.py --gpu 0
"""

import sys
import os
import json
import time
import copy
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
# H&S code needs both: ROOT for differential_ml symlink, and repo root for 'pt.*' imports
sys.path.insert(0, str(ROOT / "repos" / "differential-ml"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# H&S imports
from differential_ml.pt.DmlTrainer import DmlTrainer as HsDmlTrainer
from differential_ml.pt.modules.DmlFeedForward import DmlFeedForward as HsDmlFeedForward
from differential_ml.pt.modules.DmlLoss import DmlLoss as HsDmlLoss
from differential_ml.pt.modules.DmlDataset import DmlDataset as HsDmlDataset
from differential_ml.util.data_util import DataNormalizer as HsDataNormalizer

# Our imports
from dml_benchmark.trainer import train_single_experiment, set_deterministic
from dml_benchmark.high_fidelity_references import (
    barrier_bs_analytical_delta,
    heston_digital_cos_delta,
    basket_high_k_lrm_delta,
)

# Data generator imports (same datasets as unified comparison)
from dml_benchmark.lrm_labels import (
    lrm_digital_bs,
    lrm_basket_bachelier,
)
from dml_benchmark.functions import generate_data


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = ROOT / "results" / "hs_comparison"
SEEDS = [42, 123, 456]  # 3 seeds for comparison

# Datasets to test (H&S paper focuses on these)
DATASETS = {
    "digital_bs": {
        "n_samples": 2048,
        "k_paths": 10,
        "strike": 100.0,
        "vol": 0.2,
        "r": 0.05,
        "T": 1.0,
    },
    "basket_d1": {
        "n_samples": 2048,
        "k_paths": 10,
        "n_assets": 1,
        "strike": 0.0,
        "vol": 0.2,
        "T": 1.0,
    },
}

# H&S architecture configurations to test
HS_CONFIGS = {
    "hs_default": {
        # H&S example defaults with larger batch (batch_size=32 is too slow
        # for their manual Greek computation; 256 is data-proportional)
        "n_layers": 2,
        "hidden_size": 100,
        "activation": "softplus",
        "lr": 0.1,
        "n_epochs": 100,
        "batch_size": 256,
        "lambda_": 1.0,
    },
    "our_arch": {
        # Our benchmark architecture for apple-to-apple comparison
        "n_layers": 4,
        "hidden_size": 256,
        "activation": "softplus",
        "lr": 0.005,
        "n_epochs": 100,  # Match epoch count for fair comparison
        "batch_size": 256,
        "lambda_": 1.0,
    },
}


# ============================================================================
# DATA GENERATION
# ============================================================================

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (n, d). Squeeze 3D (n, 1, d) → (n, d)."""
    if arr.ndim == 3:
        assert arr.shape[1] == 1
        return arr.squeeze(1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (n, 1, d) for H&S format."""
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    return arr


def generate_dataset(dataset_name: str, seed: int):
    """Generate train/test data for a given dataset."""
    set_deterministic(seed)
    config = DATASETS[dataset_name]

    if dataset_name == "digital_bs":
        data = lrm_digital_bs(
            n_samples=config["n_samples"],
            k_paths=config["k_paths"],
            seed=seed,
        )
        x = ensure_2d(data["x"])
        y = ensure_2d(data["y"])
        dydx = ensure_2d(data["dydx_lrm"])
        dydx_eval = ensure_2d(data["dydx_exact"])

    elif dataset_name == "basket_d1":
        data = lrm_basket_bachelier(
            n_samples=config["n_samples"],
            n_assets=config["n_assets"],
            k_paths=config["k_paths"],
            seed=seed,
        )
        x = ensure_2d(data["x"])
        y = ensure_2d(data["y"])
        dydx = ensure_2d(data["dydx_lrm"])
        dydx_eval = ensure_2d(data.get("dydx_exact", data["dydx_lrm"]))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split into train/test (80/20)
    n = len(x)
    n_train = int(0.8 * n)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "dydx_train": dydx[train_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
        "dydx_test": dydx[test_idx],
        "dydx_eval_test": dydx_eval[test_idx],
    }


# ============================================================================
# H&S ORIGINAL CODE RUNNER
# ============================================================================

def run_hs_original(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_eval_test,
    config: dict,
    seed: int,
) -> dict:
    """
    Run the H&S original PyTorch implementation.

    Uses their code from repos/differential-ml/pt/:
      - DmlFeedForward: manual backprop network
      - DmlLoss: 1/(1+λd) value + λd/(1+λd) deriv weighting, with lambda_j IN loss
      - DataNormalizer: z-score + lambda_j = 1/RMS(dydx_norm)
      - DmlTrainer: simple SGD step

    Key: H&S applies lambda_j = 1/RMS(scaled_dydx) INSIDE the loss function,
    whereas our code uses lambda_j = x_std/y_std and pre-scales in the normalizer.
    """
    set_deterministic(seed)
    t0 = time.time()

    # H&S expects dydx as 3D: (n, output_dim, input_dim)
    dydx_train_3d = ensure_3d(dydx_train)

    # Use H&S DataNormalizer for their lambda_j computation
    normalizer = HsDataNormalizer()
    normalizer.initialize_with_data(
        x_raw=x_train, y_raw=y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train,
        dydx_raw=dydx_train_3d
    )
    # Normalize train
    x_n = normalizer.scale_x(x_train)
    y_n = normalizer.scale_y(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train)
    dydx_n = normalizer.scale_dy_dx(dydx_train_3d)[0]

    # Normalize test
    x_test_n = normalizer.scale_x(x_test)
    y_test_2d = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test

    # Map activation
    act_map = {
        "softplus": torch.nn.Softplus(),
        "sigmoid": torch.nn.Sigmoid(),
        "relu": torch.nn.ReLU(),
    }
    activation = act_map[config["activation"]]

    # Build H&S network
    net = HsDmlFeedForward(
        input_dimension=normalizer.input_dimension,
        output_dimension=normalizer.output_dimension,
        number_of_hidden_layers=config["n_layers"],
        hidden_layer_dimension=config["hidden_size"],
        activation=activation,
    )

    loss_fn = HsDmlLoss(
        _lambda=config["lambda_"],
        _input_dim=normalizer.input_dimension,
        _lambda_j=normalizer.lambda_j,
    )
    optimizer = torch.optim.Adam(lr=config["lr"], params=net.parameters())
    trainer = HsDmlTrainer(net=net, loss=loss_fn, optimizer=optimizer)

    # Create dataset and dataloader
    dataset = HsDmlDataset(x_n, y_n, dydx_n)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(seed)
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=g)
    dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

    # Training loop (no early stopping — H&S uses fixed epochs)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(config["n_epochs"]):
        net.train()
        for batch in dataloader:
            inputs = batch["x"]
            targets = batch["y"]
            gradients = batch["dydx"]
            trainer.step(inputs, targets, gradients)

        # Validate
        net.eval()
        with torch.no_grad():
            valid_loader = DataLoader(valid_set, batch_size=len(valid_set))
            for vbatch in valid_loader:
                vx, vy, vdydx = vbatch["x"], vbatch["y"], vbatch["dydx"]
                vy_out, vgreek_out = net.forward_with_greek(vx)
                val_loss = loss_fn(vy_out, vy, vgreek_out, vdydx, net)
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = copy.deepcopy(net.state_dict())
                    best_epoch = epoch

    time_s = time.time() - t0

    # Restore best and evaluate
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    x_test_t = torch.tensor(x_test_n, dtype=torch.float32)
    with torch.no_grad():
        y_pred_n, dydx_pred_n = net.forward_with_greek(x_test_t)

    # Unscale predictions to original space
    y_pred = normalizer.unscale_y(y_pred_n.numpy())  # (n_test, 1)
    # Greek output is (n_test, 1, d) — unscale to raw space
    dydx_pred_raw = normalizer.unscale_dy_dx(dydx_pred_n.numpy().squeeze(1))  # (n_test, d)

    # Compute metrics in original space
    value_mse = float(np.mean((y_pred.flatten() - y_test.flatten()) ** 2))
    grad_mse = float(np.mean((dydx_pred_raw - dydx_eval_test) ** 2))

    return {
        "value_mse": value_mse,
        "grad_mse": grad_mse,
        "time_s": time_s,
        "best_epoch": best_epoch,
        "lambda_j_hs": normalizer.lambda_j.tolist() if normalizer.lambda_j is not None else None,
    }


# ============================================================================
# OUR dml_fixed RUNNER
# ============================================================================

def run_our_dml_fixed(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_eval_test,
    config: dict,
    seed: int,
) -> dict:
    """Run our benchmark's dml_fixed method."""
    set_deterministic(seed)
    t0 = time.time()

    result = train_single_experiment(
        x_train=x_train, y_train=y_train, dydx_train=dydx_train,
        x_test=x_test, y_test=y_test, dydx_test=dydx_eval_test,
        lambda_=config["lambda_"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        n_layers=config["n_layers"],
        hidden_size=config["hidden_size"],
        lr=config["lr"],
        activation=config["activation"],
        seed=seed,
        pbar=False,
        method="dml_fixed",
    )

    time_s = time.time() - t0

    return {
        "value_mse": result.test_value_mse,
        "grad_mse": result.test_grad_mse,
        "time_s": time_s,
        "best_epoch": result.best_epoch,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("B2: H&S Original Code Comparison")
    print("=" * 70)

    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        for config_name, config in HS_CONFIGS.items():
            print(f"\n  Config: {config_name}")
            print(f"  Arch: {config['n_layers']}×{config['hidden_size']} {config['activation']}")
            print(f"  LR: {config['lr']}, Epochs: {config['n_epochs']}, BS: {config['batch_size']}")

            for seed in SEEDS:
                key = f"{dataset_name}_{config_name}_s{seed}"

                # Check if result exists (resume support)
                result_path = RESULTS_DIR / f"{key}.json"
                if args.resume and result_path.exists():
                    print(f"    Seed {seed}: exists, skipping")
                    with open(result_path) as f:
                        all_results[key] = json.load(f)
                    continue

                print(f"    Seed {seed}: ", end="", flush=True)
                data = generate_dataset(dataset_name, seed)

                try:
                    # Run H&S original
                    hs_result = run_hs_original(
                        data["x_train"], data["y_train"], data["dydx_train"],
                        data["x_test"], data["y_test"], data["dydx_eval_test"],
                        config, seed,
                    )

                    # Run our dml_fixed with same architecture
                    our_result = run_our_dml_fixed(
                        data["x_train"], data["y_train"], data["dydx_train"],
                        data["x_test"], data["y_test"], data["dydx_eval_test"],
                        config, seed,
                    )

                    result = {
                        "key": key,
                        "dataset": dataset_name,
                        "config": config_name,
                        "seed": seed,
                        "architecture": f"{config['n_layers']}x{config['hidden_size']}",
                        "hs_original": hs_result,
                        "our_dml_fixed": our_result,
                        "value_mse_ratio": our_result["value_mse"] / max(hs_result["value_mse"], 1e-30),
                        "grad_mse_ratio": our_result["grad_mse"] / max(hs_result["grad_mse"], 1e-30),
                        "timestamp": datetime.now().isoformat(),
                    }

                    all_results[key] = result
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)

                    v_ratio = result["value_mse_ratio"]
                    g_ratio = result["grad_mse_ratio"]
                    print(f"val_ratio={v_ratio:.3f} grad_ratio={g_ratio:.3f} "
                          f"(hs={hs_result['time_s']:.1f}s ours={our_result['time_s']:.1f}s)")

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))

    for key, result in all_results.items():
        group = f"{result['dataset']}_{result['config']}"
        summary[group]["value_ratio"].append(result["value_mse_ratio"])
        summary[group]["grad_ratio"].append(result["grad_mse_ratio"])

    print(f"\n{'Group':45s} | {'Val MSE (ours/hs)':>18s} | {'Grad MSE (ours/hs)':>18s}")
    print("-" * 90)
    for group in sorted(summary.keys()):
        vr = np.array(summary[group]["value_ratio"])
        gr = np.array(summary[group]["grad_ratio"])
        print(f"{group:45s} | {vr.mean():.3f} ± {vr.std():.3f} | {gr.mean():.3f} ± {gr.std():.3f}")

    # Save summary
    summary_path = RESULTS_DIR / "comparison_summary.json"
    summary_data = {
        group: {
            "value_ratio_mean": float(np.mean(summary[group]["value_ratio"])),
            "value_ratio_std": float(np.std(summary[group]["value_ratio"])),
            "grad_ratio_mean": float(np.mean(summary[group]["grad_ratio"])),
            "grad_ratio_std": float(np.std(summary[group]["grad_ratio"])),
            "n_seeds": len(summary[group]["value_ratio"]),
        }
        for group in sorted(summary.keys())
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
