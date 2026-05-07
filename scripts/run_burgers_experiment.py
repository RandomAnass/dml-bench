#!/usr/bin/env python3
"""
PDEBench 1D Burgers equation experiment for DML-Bench.
Tests whether derivative supervision helps learn PDE solutions
when spatial gradients du/dx are available as labels.

Dataset: PDEBench 1D Burgers (Nu=0.01), CC-BY 4.0
Source: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import h5py
import torch
from pathlib import Path

from dml_benchmark.model import DmlFeedForward, DmlLoss, VanillaLoss, DataNormalizer, DmlDataset
from dml_benchmark.trainer import DmlTrainer


def load_burgers_data(data_path, n_samples=10000, seed=42):
    """
    Load PDEBench 1D Burgers data and extract pointwise (x, t) -> u samples
    with spatial gradient du/dx computed by central finite differences.

    HDF5 structure:
      - 'tensor': shape (n_simulations, n_timesteps, n_grid_points)
      - 'x-coordinate': shape (n_grid_points,)
      - 't-coordinate': shape (n_timesteps,)
    """
    rng = np.random.RandomState(seed)

    with h5py.File(data_path, 'r') as f:
        # Load solution tensor
        u_all = f['tensor'][:]  # (n_sim, n_t, n_x) or similar
        x_coords = f['x-coordinate'][:]
        t_coords = f['t-coordinate'][:]

    n_sim, n_t, n_x = u_all.shape
    dx = x_coords[1] - x_coords[0]

    print(f"Burgers data: {n_sim} simulations, {n_t} timesteps, {n_x} grid points")
    print(f"x range: [{x_coords[0]:.3f}, {x_coords[-1]:.3f}], dx={dx:.6f}")
    print(f"t range: [{t_coords[0]:.3f}, {t_coords[-1]:.3f}]")

    # Compute du/dx via central finite differences (interior points only)
    dudx_all = np.zeros_like(u_all)
    dudx_all[:, :, 1:-1] = (u_all[:, :, 2:] - u_all[:, :, :-2]) / (2 * dx)
    # Forward/backward difference at boundaries
    dudx_all[:, :, 0] = (u_all[:, :, 1] - u_all[:, :, 0]) / dx
    dudx_all[:, :, -1] = (u_all[:, :, -1] - u_all[:, :, -2]) / dx

    # Sample random (sim, t, x) points
    total_points = n_sim * n_t * n_x
    n_samples = min(n_samples, total_points)

    # Build flat indices
    sim_idx = rng.randint(0, n_sim, n_samples)
    t_idx = rng.randint(0, n_t, n_samples)
    # Avoid boundary points for cleaner gradients
    x_idx = rng.randint(2, n_x - 2, n_samples)

    # Extract features: (x, t)
    X = np.stack([x_coords[x_idx], t_coords[t_idx]], axis=1)  # (n_samples, 2)

    # Extract targets: u value and du/dx
    y = u_all[sim_idx, t_idx, x_idx].reshape(-1, 1)  # (n_samples, 1)
    dydx_x = dudx_all[sim_idx, t_idx, x_idx]  # du/dx

    # We also want du/dt (computed via central FD in time)
    dudt_all = np.zeros_like(u_all)
    dt = t_coords[1] - t_coords[0]
    dudt_all[:, 1:-1, :] = (u_all[:, 2:, :] - u_all[:, :-2, :]) / (2 * dt)
    dudt_all[:, 0, :] = (u_all[:, 1, :] - u_all[:, 0, :]) / dt
    dudt_all[:, -1, :] = (u_all[:, -1, :] - u_all[:, -2, :]) / dt

    dydx_t = dudt_all[sim_idx, t_idx, x_idx]  # du/dt

    dydx = np.stack([dydx_x, dydx_t], axis=1).reshape(-1, 1, 2)  # (n_samples, 1, 2)

    print(f"Extracted {n_samples} pointwise samples")
    print(f"  X shape: {X.shape}, y shape: {y.shape}, dydx shape: {dydx.shape}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  du/dx range: [{dydx_x.min():.3f}, {dydx_x.max():.3f}]")

    return X, y, dydx


def run_experiment(X, y, dydx, method="dml_fixed", seed=42, n_epochs=500,
                   n_train=8000, n_test=2000):
    """Run a single DML experiment on Burgers data.

    J-H4 (2026-04-16): canonical protocol — set_deterministic, AdamW (wd=0),
    val-based early stopping (on val total loss, not training loss),
    explicit shared_layer_name for GradNorm.
    """
    from dml_benchmark.trainer import set_deterministic
    set_deterministic(seed)

    # J-H4: carve val split from train for ES (20% of train)
    idx = np.random.permutation(len(X))
    n_val = max(1, int(n_train * 0.2))
    n_train_actual = n_train - n_val
    X_train = X[idx[:n_train_actual]]; y_train = y[idx[:n_train_actual]]; dydx_train = dydx[idx[:n_train_actual]]
    X_val = X[idx[n_train_actual:n_train]]; y_val = y[idx[n_train_actual:n_train]]; dydx_val = dydx[idx[n_train_actual:n_train]]
    X_test = X[idx[n_train:n_train+n_test]]; y_test = y[idx[n_train:n_train+n_test]]; dydx_test = dydx[idx[n_train:n_train+n_test]]

    # Normalize (train statistics only)
    normalizer = DataNormalizer()
    normalizer.initialize_with_data(X_train, y_train, dydx_train)
    X_n, y_n, dydx_n = normalizer.normalize_all(X_train, y_train, dydx_train)
    X_v, y_v, dydx_v = normalizer.normalize_all(X_val, y_val, dydx_val)

    dim = X_train.shape[1]  # 2
    model = DmlFeedForward(dim, 1, 4, 256, "softplus")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # H-H6 (2026-04-16): dispatch on method — the previous code sent every
    # non-vanilla method to the same DmlLoss(lambda=1), so dml_fixed,
    # dml_gradnorm, and dml_relobralo all produced byte-identical results
    # across seeds (same data, same init, same loss).
    if method == "vanilla":
        loss_fn = VanillaLoss()
    elif method == "dml_fixed":
        loss_fn = DmlLoss(lambda_=1.0, input_dim=dim, lambda_j=normalizer.lambda_j)
    elif method == "dml_gradnorm":
        # J-H4 (2026-04-16): pass explicit shared_layer_name for the flat MLP
        # (matches J4 for the main trainer).
        from dml_benchmark.loss_balancing import GradNormDmlLoss
        loss_fn = GradNormDmlLoss(
            input_dim=dim, shared_layer_name="layers.3.weight",
        ).to(device)
    elif method == "dml_relobralo":
        # K-L5 (2026-04-16): use the FAITHFUL ReLoBRaLoDmlLoss (Bischof &
        # Kraus 2022 Eq.11 with τ=0.1, α=0.999, E[ρ]=0.999) so Burgers
        # is consistent with the current code's dispatch of `dml_relobralo`.
        # Previously used SoftmaxBalanceDmlLoss (simplified variant, τ=1.0).
        from dml_benchmark.loss_balancing import ReLoBRaLoDmlLoss
        loss_fn = ReLoBRaLoDmlLoss(input_dim=dim, seed=seed).to(device)
    else:
        raise ValueError(f"Unknown method: {method}")

    # J-H4: AdamW (wd=0) for cross-arch parity (D015).
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dataset = DmlDataset(X_n, y_n, dydx_n)
    val_dataset = DmlDataset(X_v, y_v, dydx_v)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    ES_PATIENCE = 50

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            db = batch["dydx"].to(device)

            optimizer.zero_grad()
            if method == "vanilla":
                y_pred = model(xb)
                loss_comp = loss_fn(y_pred, yb)
            else:
                y_pred, dydx_pred = model.forward_with_greek(xb)
                loss_comp = loss_fn(y_pred, yb, dydx_pred, db, model)

            loss_comp.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_comp.total.item()

        scheduler.step()
        # J-H4: ES on VALIDATION total loss, not training loss.
        model.eval()
        val_loss_sum = 0.0
        for vb in val_loader:
            xv = vb["x"].to(device); yv = vb["y"].to(device); dv = vb["dydx"].to(device)
            if method == "vanilla":
                yp = model(xv)
                lc = loss_fn(yp, yv)
            else:
                yp, dp = model.forward_with_greek(xv)
                lc = loss_fn(yp, yv, dp, dv, model)
            val_loss_sum += lc.total.item()
        avg_val = val_loss_sum / max(1, len(val_loader))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    # Evaluate
    model.load_state_dict(best_state)
    model.eval()

    X_test_n = normalizer.normalize_x(X_test)
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(device)

    y_pred, dydx_pred = model.forward_with_greek(X_test_t)
    y_pred = normalizer.unscale_y(y_pred.detach().cpu().numpy())
    dydx_pred = normalizer.unscale_dydx(dydx_pred.detach().cpu().numpy())

    value_mse = float(np.mean((y_test - y_pred) ** 2))
    grad_mse = float(np.mean((dydx_test - dydx_pred) ** 2))

    return {
        "method": method,
        "seed": seed,
        "dataset": "burgers_1d_nu0.01",
        "dim": dim,
        "n_train": n_train,
        "n_test": n_test,
        "test_value_mse": value_mse,
        "test_grad_mse": grad_mse,
        "best_epoch": epoch,
    }


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data/pdebench/1D_Burgers_Sols_Nu0.01.hdf5")

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        print("Download with: wget -O data/pdebench/1D_Burgers_Sols_Nu0.01.hdf5 'https://darus.uni-stuttgart.de/api/access/datafile/281363'")
        return

    print("Loading Burgers 1D data...")
    X, y, dydx = load_burgers_data(data_path, n_samples=12000, seed=42)

    methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
    seeds = [42, 123, 456, 789, 1000]
    results = []

    for method in methods:
        for seed in seeds:
            print(f"\nTraining {method} (seed={seed})...")
            result = run_experiment(X, y, dydx, method=method, seed=seed)
            results.append(result)
            print(f"  value_mse={result['test_value_mse']:.6e}, grad_mse={result['test_grad_mse']:.6e}")

    # Summary
    print("\n" + "=" * 70)
    print("BURGERS 1D RESULTS SUMMARY")
    print("=" * 70)

    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        vmses = [r["test_value_mse"] for r in method_results]
        gmses = [r["test_grad_mse"] for r in method_results]
        print(f"{method:<20}: value={np.mean(vmses):.4e} +/- {np.std(vmses):.2e}, "
              f"grad={np.mean(gmses):.4e} +/- {np.std(gmses):.2e}")

    # Compute improvement
    vanilla = [r for r in results if r["method"] == "vanilla"]
    v_vmse = np.mean([r["test_value_mse"] for r in vanilla])
    v_gmse = np.mean([r["test_grad_mse"] for r in vanilla])

    print(f"\nImprovement over vanilla:")
    for method in methods[1:]:
        mr = [r for r in results if r["method"] == method]
        vmse = np.mean([r["test_value_mse"] for r in mr])
        gmse = np.mean([r["test_grad_mse"] for r in mr])
        print(f"  {method:<20}: value {(1-vmse/v_vmse)*100:+.1f}%, "
              f"gradient {v_gmse/gmse:.1f}x")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results/burgers_1d_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
