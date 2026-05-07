#!/usr/bin/env python3
"""
Fuzzy-label sensitivity experiment.
Sweeps eps_mult ∈ {0.1, 0.25, 0.5, 1.0, 2.0} on discontinuous datasets.
Reports whether the ranking is robust or depends on careful tuning.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
from collections import defaultdict

from dml_benchmark.model import DmlFeedForward, DmlLoss, VanillaLoss, DataNormalizer, DmlDataset
from dml_benchmark.fuzzy_smoothing import fuzzy_digital_bs, fuzzy_barrier_bs


def run_single(dataset_fn, eps_mult, seed=42, n_samples=4096, n_epochs=500):
    """Run one experiment with a specific eps_mult."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate data with specified eps_mult
    data = dataset_fn(n_samples=n_samples, seed=seed, eps_mult=eps_mult)
    x, y = data["x"], data["y"]
    dydx_fuzzy = data["dydx_fuzzy"]
    y_exact = data.get("y_exact", y)
    dydx_exact = data.get("dydx_exact", dydx_fuzzy)
    epsilon = data.get("epsilon", eps_mult)

    # Train/test split (80/20)
    n_train = int(0.8 * len(x))
    x_train, y_train, dydx_train = x[:n_train], y[:n_train], dydx_fuzzy[:n_train]
    x_test, y_test, dydx_test = x[n_train:], y_exact[n_train:], dydx_exact[n_train:]

    # Normalize
    normalizer = DataNormalizer()
    y_tr = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    dydx_tr = dydx_train.reshape(-1, 1, x_train.shape[1]) if dydx_train.ndim == 2 else dydx_train
    normalizer.initialize_with_data(x_train, y_tr, dydx_tr)

    x_n, y_n, dydx_n = normalizer.normalize_all(x_train, y_tr, dydx_tr)

    dim = x_train.shape[1]
    model = DmlFeedForward(dim, 1, 4, 256, "softplus")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = DmlLoss(lambda_=1.0, input_dim=dim, lambda_j=normalizer.lambda_j)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dataset = DmlDataset(x_n, y_n, dydx_n)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            db = batch["dydx"].to(device)

            optimizer.zero_grad()
            y_pred, dydx_pred = model.forward_with_greek(xb)
            loss_comp = loss_fn(y_pred, yb, dydx_pred, db, model)
            loss_comp.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_comp.total.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on exact test targets
    x_test_n = normalizer.normalize_x(x_test)
    x_test_t = torch.tensor(x_test_n, dtype=torch.float32).to(device)
    y_pred, dydx_pred = model.forward_with_greek(x_test_t)
    y_pred = y_pred.detach().cpu().numpy()
    dydx_pred = dydx_pred.detach().cpu().numpy()

    y_pred_orig = normalizer.unscale_y(y_pred)
    dydx_pred_orig = normalizer.unscale_dydx(dydx_pred)

    y_test_r = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    dydx_test_r = dydx_test.reshape(-1, 1, dim) if dydx_test.ndim == 2 else dydx_test

    value_mse = float(np.mean((y_test_r - y_pred_orig)**2))
    grad_mse = float(np.mean((dydx_test_r - dydx_pred_orig)**2))

    return {
        "eps_mult": eps_mult,
        "epsilon": float(epsilon) if np.isscalar(epsilon) else float(np.mean(epsilon)),
        "seed": seed,
        "value_mse": value_mse,
        "grad_mse": grad_mse,
        "best_epoch": epoch
    }


def main():
    eps_values = [0.1, 0.25, 0.5, 1.0, 2.0]
    seeds = [42, 123, 456, 789, 1000]
    datasets = {
        "digital_bs": fuzzy_digital_bs,
        "barrier_bs": fuzzy_barrier_bs,
    }

    all_results = {}

    for ds_name, ds_fn in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        ds_results = []
        for eps in eps_values:
            for seed in seeds:
                print(f"  eps_mult={eps}, seed={seed}...", end=" ", flush=True)
                try:
                    result = run_single(ds_fn, eps_mult=eps, seed=seed)
                    result["dataset"] = ds_name
                    ds_results.append(result)
                    print(f"value={result['value_mse']:.4e}, grad={result['grad_mse']:.4e}")
                except Exception as e:
                    print(f"FAILED: {e}")

        all_results[ds_name] = ds_results

    # Summary
    print("\n" + "=" * 80)
    print("FUZZY SENSITIVITY SUMMARY")
    print("=" * 80)

    for ds_name, results in all_results.items():
        print(f"\n{ds_name}:")
        print(f"  {'eps_mult':>10} {'Value MSE (mean±std)':>25} {'Grad MSE (mean±std)':>25}")
        print("  " + "-" * 62)
        for eps in eps_values:
            eps_results = [r for r in results if r["eps_mult"] == eps]
            if eps_results:
                vmses = [r["value_mse"] for r in eps_results]
                gmses = [r["grad_mse"] for r in eps_results]
                print(f"  {eps:>10.2f} {np.mean(vmses):>12.4e} ± {np.std(vmses):.2e}   "
                      f"{np.mean(gmses):>12.4e} ± {np.std(gmses):.2e}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results/fuzzy_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
