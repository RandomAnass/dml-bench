#!/usr/bin/env python3
"""
Revision experiments for DML-Bench paper.
Three runs:
  RUN-1: Warmup with fixed lambda Phase 2 (vs GradNorm Phase 2)
  RUN-2: Re-evaluate unified comparison with autodiff vanilla gradients
  RUN-3: Early stopping ablation (total loss vs value-only)

Usage:
    # RUN-1: Warmup fixed-lambda Phase 2
    CUDA_VISIBLE_DEVICES=0 conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 1

    # RUN-2: Re-evaluate unified comparison with autodiff vanilla
    CUDA_VISIBLE_DEVICES=1 conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 2

    # RUN-3: Early stopping ablation
    CUDA_VISIBLE_DEVICES=0 conda run -n dml-bench-env python3 scripts/run_revision_experiments.py --run 3
"""
import sys
import os
import argparse
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dml_benchmark.model import DmlFeedForward, DmlLoss, VanillaLoss, DataNormalizer, DmlDataset
from dml_benchmark.loss_balancing import GradNormDmlLoss
from dml_benchmark.fuzzy_smoothing import (
    fuzzy_digital_bs, fuzzy_barrier_bs, fuzzy_basket_bachelier, fuzzy_euler_heston,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999]


def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_dataset(dataset_name, seed, n_samples=4096):
    """Generate one of the 5 unified comparison datasets."""
    generators = {
        "digital_bs": lambda: fuzzy_digital_bs(n_samples=n_samples, seed=seed),
        "barrier_bs": lambda: fuzzy_barrier_bs(n_samples=n_samples, seed=seed),
        "basket_d1": lambda: fuzzy_basket_bachelier(n_samples=n_samples, d=1, seed=seed),
        "basket_d7": lambda: fuzzy_basket_bachelier(n_samples=n_samples, d=7, seed=seed),
        "heston_digital": lambda: fuzzy_euler_heston(n_samples=n_samples, seed=seed),
    }
    if dataset_name not in generators:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return generators[dataset_name]()


def train_two_phase(data, phase2_mode="gradnorm", seed=42, n_epochs=500,
                    warmup_fraction=0.5, early_stop_metric="loss"):
    """
    Two-phase warmup training.
    Phase 1: value-only (vanilla)
    Phase 2: DML with either GradNorm or fixed lambda=1.0

    Args:
        phase2_mode: "gradnorm" or "fixed" — determines Phase 2 loss
        early_stop_metric: "loss" (total) or "value_loss" (value-only)
    """
    set_deterministic(seed)
    device = get_device()

    x = data["x"]
    y_fuzzy = data["y"]
    dydx_fuzzy = data["dydx_fuzzy"]
    y_exact = data.get("y_exact", y_fuzzy)
    dydx_exact = data.get("dydx_exact", dydx_fuzzy)

    # Train/val/test split: 60/20/20
    n = len(x)
    idx = np.random.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    x_train, y_train, dydx_train = x[idx[:n_train]], y_fuzzy[idx[:n_train]], dydx_fuzzy[idx[:n_train]]
    x_val, y_val, dydx_val = x[idx[n_train:n_train+n_val]], y_exact[idx[n_train:n_train+n_val]], dydx_exact[idx[n_train:n_train+n_val]]
    x_test, y_test, dydx_test = x[idx[n_train+n_val:]], y_exact[idx[n_train+n_val:]], dydx_exact[idx[n_train+n_val:]]

    # Ensure shapes
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    if dydx_train.ndim == 2:
        dydx_train = dydx_train.reshape(-1, 1, x.shape[1])
        dydx_val = dydx_val.reshape(-1, 1, x.shape[1])
        dydx_test = dydx_test.reshape(-1, 1, x.shape[1])

    # Normalize
    normalizer = DataNormalizer()
    normalizer.initialize_with_data(x_train, y_train, dydx_train)
    xn, yn, dn = normalizer.normalize_all(x_train, y_train, dydx_train)
    xvn = normalizer.normalize_x(x_val)
    yvn = normalizer.normalize_y(y_val)
    dvn = normalizer.normalize_dydx(dydx_val)

    dim = x.shape[1]
    model = DmlFeedForward(dim, 1, 4, 256, "softplus").to(device)

    # Phase 1: Vanilla
    warmup_epochs = int(n_epochs * warmup_fraction)
    finetune_epochs = n_epochs - warmup_epochs

    vanilla_loss = VanillaLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    train_dataset = DmlDataset(xn, yn, dn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataset = DmlDataset(xvn, yvn, dvn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    best_val = float("inf")
    best_state = None

    # Phase 1 training
    for epoch in range(warmup_epochs):
        model.train()
        for batch in train_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            lc = vanilla_loss(y_pred, yb)
            lc.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                y_pred = model(xb)
                lc = vanilla_loss(y_pred, yb)
                val_loss += lc.total.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best Phase 1 model
    model.load_state_dict(best_state)

    # Phase 2: DML
    if phase2_mode == "gradnorm":
        dml_loss = GradNormDmlLoss(input_dim=dim)
    elif phase2_mode == "fixed":
        dml_loss = DmlLoss(lambda_=1.0, input_dim=dim, lambda_j=normalizer.lambda_j)
    else:
        raise ValueError(f"Unknown phase2_mode: {phase2_mode}")

    optimizer2 = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="min", factor=0.5, patience=20, min_lr=1e-7
    )

    best_val2 = float("inf")
    best_state2 = None
    patience_counter = 0

    for epoch in range(finetune_epochs):
        model.train()
        for batch in train_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            db = batch["dydx"].to(device)
            optimizer2.zero_grad()

            with torch.enable_grad():
                y_pred, dydx_pred = model.forward_with_greek(xb)

            if phase2_mode == "gradnorm":
                lc = dml_loss(y_pred, yb, dydx_pred, db, model)
            else:
                lc = dml_loss(y_pred, yb, dydx_pred, db, model)
            lc.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()

        # Validate with autodiff gradients
        model.eval()
        val_vloss = 0
        val_gloss = 0
        n_val_batches = 0
        for batch in val_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            db = batch["dydx"].to(device)
            with torch.enable_grad():
                y_pred, dydx_pred = model.forward_with_greek(xb)
            val_vloss += torch.mean((y_pred - yb) ** 2).item()
            val_gloss += torch.mean((dydx_pred - db) ** 2).item()
            n_val_batches += 1

        val_vloss /= n_val_batches
        val_gloss /= n_val_batches
        val_total = val_vloss + val_gloss

        monitor = val_total if early_stop_metric == "loss" else val_vloss
        scheduler2.step(monitor)

        if monitor < best_val2:
            best_val2 = monitor
            best_state2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    # Load best Phase 2 model
    model.load_state_dict(best_state2)

    # Evaluate on test set with autodiff for ALL methods
    model.eval()
    x_test_n = normalizer.normalize_x(x_test)
    x_test_t = torch.tensor(x_test_n, dtype=torch.float32).to(device)

    with torch.enable_grad():
        y_pred, dydx_pred = model.forward_with_greek(x_test_t)
    y_pred = normalizer.unscale_y(y_pred.detach().cpu().numpy())
    dydx_pred = normalizer.unscale_dydx(dydx_pred.detach().cpu().numpy())

    value_mse = float(np.mean((y_test - y_pred) ** 2))
    grad_mse = float(np.mean((dydx_test - dydx_pred) ** 2))

    return {"test_value_mse": value_mse, "test_grad_mse": grad_mse, "best_epoch": epoch}


def train_vanilla(data, seed=42, n_epochs=500, early_stop_metric="loss"):
    """Train vanilla model and evaluate gradients via autodiff."""
    set_deterministic(seed)
    device = get_device()

    x = data["x"]
    y = data.get("y_exact", data["y"])
    dydx = data.get("dydx_exact", data["dydx_fuzzy"])

    n = len(x)
    idx = np.random.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    x_train = x[idx[:n_train]]
    y_train = y[idx[:n_train]].reshape(-1, 1) if y.ndim == 1 else y[idx[:n_train]]
    x_val = x[idx[n_train:n_train+n_val]]
    y_val = y[idx[n_train:n_train+n_val]].reshape(-1, 1) if y.ndim == 1 else y[idx[n_train:n_train+n_val]]
    x_test = x[idx[n_train+n_val:]]
    y_test = y[idx[n_train+n_val:]].reshape(-1, 1) if y.ndim == 1 else y[idx[n_train+n_val:]]
    dydx_test = dydx[idx[n_train+n_val:]]
    if dydx_test.ndim == 2:
        dydx_test = dydx_test.reshape(-1, 1, x.shape[1])

    # Dummy dydx for DataNormalizer
    dydx_train_dummy = np.zeros((len(x_train), 1, x.shape[1]))

    normalizer = DataNormalizer()
    normalizer.initialize_with_data(x_train, y_train, dydx_train_dummy)
    xn = normalizer.normalize_x(x_train)
    yn = normalizer.normalize_y(y_train)
    dn = np.zeros_like(dydx_train_dummy)

    dim = x.shape[1]
    model = DmlFeedForward(dim, 1, 4, 256, "softplus").to(device)

    loss_fn = VanillaLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    dataset = DmlDataset(xn, yn, dn)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    xvn = normalizer.normalize_x(x_val)
    yvn = normalizer.normalize_y(y_val)
    dvn = np.zeros((len(x_val), 1, dim))
    val_dataset = DmlDataset(xvn, yvn, dvn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            lc = loss_fn(y_pred, yb)
            lc.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                y_pred = model(xb)
                lc = loss_fn(y_pred, yb)
                val_loss += lc.total.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate with AUTODIFF gradients (not zeros)
    x_test_n = normalizer.normalize_x(x_test)
    x_test_t = torch.tensor(x_test_n, dtype=torch.float32).to(device)

    with torch.enable_grad():
        y_pred, dydx_pred = model.forward_with_greek(x_test_t)
    y_pred = normalizer.unscale_y(y_pred.detach().cpu().numpy())
    dydx_pred = normalizer.unscale_dydx(dydx_pred.detach().cpu().numpy())

    value_mse = float(np.mean((y_test - y_pred) ** 2))
    grad_mse = float(np.mean((dydx_test - dydx_pred) ** 2))

    return {"test_value_mse": value_mse, "test_grad_mse": grad_mse, "best_epoch": epoch}


# ============================================================================
# RUN-1: Warmup with fixed lambda Phase 2
# ============================================================================

def run_1_warmup_fixed_lambda():
    """Compare warmup+GradNorm vs warmup+fixed-lambda on unified comparison datasets."""
    datasets = ["digital_bs", "barrier_bs", "basket_d1", "basket_d7", "heston_digital"]
    results = []

    for ds in datasets:
        for seed in SEEDS:
            for phase2 in ["gradnorm", "fixed"]:
                key = f"{ds}_warmup_{phase2}_s{seed}"
                out_file = os.path.join(RESULTS_DIR, "revision", f"{key}.json")
                if os.path.exists(out_file):
                    print(f"SKIP (exists): {key}")
                    continue

                print(f"Training: {key}...", end=" ", flush=True)
                t0 = time.time()
                data = generate_dataset(ds, seed)
                result = train_two_phase(data, phase2_mode=phase2, seed=seed)
                result.update({
                    "dataset": ds, "method": f"warmup_{phase2}", "seed": seed,
                    "phase2_mode": phase2, "time_s": time.time() - t0,
                    "key": key, "eval_mode": "autodiff",
                })
                results.append(result)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"value={result['test_value_mse']:.4e}, grad={result['test_grad_mse']:.4e} ({result['time_s']:.0f}s)")

    # Summary
    print("\n" + "=" * 80)
    print("RUN-1 SUMMARY: Warmup Phase 2 Comparison")
    print("=" * 80)
    for phase2 in ["gradnorm", "fixed"]:
        ph_results = [r for r in results if r["phase2_mode"] == phase2]
        if ph_results:
            vm = np.mean([r["test_value_mse"] for r in ph_results])
            gm = np.mean([r["test_grad_mse"] for r in ph_results])
            print(f"  warmup_{phase2}: value={vm:.4e}, grad={gm:.4e} (n={len(ph_results)})")


# ============================================================================
# RUN-2: Re-evaluate with autodiff vanilla gradients
# ============================================================================

def run_2_autodiff_vanilla():
    """Re-run vanilla + key DML methods with autodiff gradient evaluation."""
    datasets = ["digital_bs", "barrier_bs", "basket_d1", "basket_d7", "heston_digital"]
    methods_to_run = ["vanilla", "warmup_fuzzy", "warmup_gradnorm"]
    results = []

    for ds in datasets:
        for seed in SEEDS:
            for method in methods_to_run:
                key = f"{ds}_{method}_autodiff_s{seed}"
                out_file = os.path.join(RESULTS_DIR, "revision", f"{key}.json")
                if os.path.exists(out_file):
                    print(f"SKIP (exists): {key}")
                    continue

                print(f"Training: {key}...", end=" ", flush=True)
                t0 = time.time()
                data = generate_dataset(ds, seed)

                if method == "vanilla":
                    result = train_vanilla(data, seed=seed)
                elif method == "warmup_fuzzy":
                    result = train_two_phase(data, phase2_mode="gradnorm", seed=seed)
                elif method == "warmup_gradnorm":
                    result = train_two_phase(data, phase2_mode="gradnorm", seed=seed)

                result.update({
                    "dataset": ds, "method": method, "seed": seed,
                    "time_s": time.time() - t0, "key": key, "eval_mode": "autodiff",
                })
                results.append(result)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"value={result['test_value_mse']:.4e}, grad={result['test_grad_mse']:.4e} ({result['time_s']:.0f}s)")

    # Summary
    print("\n" + "=" * 80)
    print("RUN-2 SUMMARY: Autodiff Vanilla Gradient Evaluation")
    print("=" * 80)
    for method in methods_to_run:
        m_results = [r for r in results if r["method"] == method]
        if m_results:
            vm = np.mean([r["test_value_mse"] for r in m_results])
            gm = np.mean([r["test_grad_mse"] for r in m_results])
            print(f"  {method}: value={vm:.4e}, grad={gm:.4e} (n={len(m_results)})")

    # Compute improvement ratios
    v_results = [r for r in results if r["method"] == "vanilla"]
    if v_results:
        v_gm = np.mean([r["test_grad_mse"] for r in v_results])
        for method in methods_to_run[1:]:
            m_results = [r for r in results if r["method"] == method]
            if m_results:
                m_gm = np.mean([r["test_grad_mse"] for r in m_results])
                print(f"  {method} vs vanilla (autodiff): {v_gm/m_gm:.1f}x gradient improvement")


# ============================================================================
# RUN-3: Early stopping ablation
# ============================================================================

def run_3_early_stopping_ablation():
    """Compare total-loss vs value-only early stopping on 2 datasets."""
    datasets = ["digital_bs", "barrier_bs"]
    seeds_short = SEEDS[:5]
    results = []

    for ds in datasets:
        for seed in seeds_short:
            for es_metric in ["loss", "value_loss"]:
                key = f"{ds}_warmup_es_{es_metric}_s{seed}"
                out_file = os.path.join(RESULTS_DIR, "revision", f"{key}.json")
                if os.path.exists(out_file):
                    print(f"SKIP (exists): {key}")
                    continue

                print(f"Training: {key}...", end=" ", flush=True)
                t0 = time.time()
                data = generate_dataset(ds, seed)
                result = train_two_phase(
                    data, phase2_mode="gradnorm", seed=seed,
                    early_stop_metric=es_metric
                )
                result.update({
                    "dataset": ds, "method": f"warmup_es_{es_metric}", "seed": seed,
                    "early_stop_metric": es_metric, "time_s": time.time() - t0,
                    "key": key, "eval_mode": "autodiff",
                })
                results.append(result)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"value={result['test_value_mse']:.4e}, grad={result['test_grad_mse']:.4e} ({result['time_s']:.0f}s)")

    # Summary
    print("\n" + "=" * 80)
    print("RUN-3 SUMMARY: Early Stopping Ablation")
    print("=" * 80)
    for es in ["loss", "value_loss"]:
        es_results = [r for r in results if r.get("early_stop_metric") == es]
        if es_results:
            vm = np.mean([r["test_value_mse"] for r in es_results])
            gm = np.mean([r["test_grad_mse"] for r in es_results])
            print(f"  ES on {es}: value={vm:.4e}, grad={gm:.4e} (n={len(es_results)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    os.makedirs(os.path.join(RESULTS_DIR, "revision"), exist_ok=True)

    if args.run == 1:
        run_1_warmup_fixed_lambda()
    elif args.run == 2:
        run_2_autodiff_vanilla()
    elif args.run == 3:
        run_3_early_stopping_ablation()


if __name__ == "__main__":
    main()
