#!/usr/bin/env python3
"""
SPY robustness analysis: Train DML models and evaluate per moneyness/maturity bucket.
Produces Table for paper: performance breakdown by subgroup.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
from collections import defaultdict

from dml_benchmark.model import DmlFeedForward, DmlLoss, VanillaLoss, DataNormalizer, DmlDataset
from dml_benchmark.trainer import DmlTrainer


def load_spy_data(data_path, temporal_cutoff="2021-07-01", embargo_days=5):
    """Load SPY data with temporal split."""
    data = np.load(data_path, allow_pickle=True)
    X_full = data["X"]
    y = data["y"]
    feature_names = [str(f) for f in data["feature_names"]]
    dates = data["dates"]

    # Select 4 features used in experiments (drop log_volume if present)
    use_cols = [feature_names.index(f) for f in ["moneyness", "T", "r", "iv"] if f in feature_names]
    X = X_full[:, use_cols]
    feature_names = [feature_names[i] for i in use_cols]

    print(f"Loaded SPY data: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")
    print(f"Date range: {sorted(set(dates))[0]} to {sorted(set(dates))[-1]}")

    # Temporal split (dates are strings in YYYY-MM-DD format)
    train_mask = dates < temporal_cutoff
    test_mask = dates >= temporal_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Compute BS Greeks as derivative labels
    moneyness_idx = feature_names.index("moneyness") if "moneyness" in feature_names else 0
    T_idx = feature_names.index("T") if "T" in feature_names else 1
    r_idx = feature_names.index("r") if "r" in feature_names else 2
    iv_idx = feature_names.index("iv") if "iv" in feature_names else 3

    def compute_bs_greeks(X_sub):
        """Compute Black-Scholes Greeks."""
        from scipy.stats import norm
        m = X_sub[:, moneyness_idx]
        T = np.clip(X_sub[:, T_idx], 1e-6, None)
        r = X_sub[:, r_idx]
        iv = np.clip(X_sub[:, iv_idx], 1e-6, None)

        d1 = (np.log(m) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))

        # Delta (∂price/∂moneyness)
        delta = norm.cdf(d1)
        # Theta (∂price/∂T) - simplified
        theta = -(m * norm.pdf(d1) * iv) / (2 * np.sqrt(T))
        # Rho (∂price/∂r)
        rho = T * m * norm.cdf(d1) * np.exp(-r * T) * 0.01
        # Vega (∂price/∂iv)
        vega = m * norm.pdf(d1) * np.sqrt(T) * 0.01

        greeks = np.stack([delta, theta, rho, vega], axis=-1)
        return greeks.reshape(-1, 1, X_sub.shape[1])

    dydx_train = compute_bs_greeks(X_train)
    dydx_test = compute_bs_greeks(X_test)

    return (X_train, y_train.reshape(-1, 1), dydx_train,
            X_test, y_test.reshape(-1, 1), dydx_test,
            feature_names, X_test[:, moneyness_idx], X_test[:, T_idx])


def train_and_predict(X_train, y_train, dydx_train, X_test, method="dml_fixed",
                      seed=42, n_epochs=500, subsample=10000):
    """Train a model and return per-sample predictions."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Subsample training data
    n = min(subsample, len(X_train))
    idx = np.random.choice(len(X_train), n, replace=False)
    X_tr = X_train[idx]
    y_tr = y_train[idx]
    dydx_tr = dydx_train[idx]

    # Normalize
    normalizer = DataNormalizer()
    normalizer.initialize_with_data(X_tr, y_tr, dydx_tr)
    X_n, y_n, dydx_n = normalizer.normalize_all(X_tr, y_tr, dydx_tr)

    dim = X_tr.shape[1]
    model = DmlFeedForward(dim, 1, 4, 256, "softplus")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if method == "vanilla":
        loss_fn = VanillaLoss()
    else:
        loss_fn = DmlLoss(lambda_=1.0, input_dim=dim, lambda_j=normalizer.lambda_j)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dataset = DmlDataset(X_n, y_n, dydx_n)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    best_loss = float("inf")
    best_state = None
    patience = 50
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            x = batch["x"].to(device)
            y_b = batch["y"].to(device)
            dydx_b = batch["dydx"].to(device)

            optimizer.zero_grad()
            if method == "vanilla":
                y_pred = model(x)
                loss_comp = loss_fn(y_pred, y_b)
            else:
                y_pred, dydx_pred = model.forward_with_greek(x)
                loss_comp = loss_fn(y_pred, y_b, dydx_pred, dydx_b, model)

            loss_comp.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_comp.total.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Predict on test set
    X_test_n = normalizer.normalize_x(X_test)
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_n = model(X_test_t).cpu().numpy()

    y_pred, dydx_pred = model.forward_with_greek(X_test_t)
    y_pred = y_pred.detach().cpu().numpy()
    dydx_pred = dydx_pred.detach().cpu().numpy()

    # Unscale
    y_pred_orig = normalizer.unscale_y(y_pred_n)
    dydx_pred_orig = normalizer.unscale_dydx(dydx_pred)

    return y_pred_orig, dydx_pred_orig


def bucket_analysis(y_true, y_pred, dydx_true, dydx_pred, moneyness, maturity, method_name):
    """Compute MSE per moneyness and maturity bucket."""
    results = {}

    # Moneyness buckets
    m_bins = [0.85, 0.93, 0.97, 1.03, 1.07, 1.15]
    m_labels = ["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"]
    results["moneyness"] = {}

    for i in range(len(m_bins) - 1):
        mask = (moneyness >= m_bins[i]) & (moneyness < m_bins[i+1])
        n = mask.sum()
        if n < 10:
            continue
        vmse = np.mean((y_true[mask] - y_pred[mask])**2)
        gmse = np.mean((dydx_true[mask] - dydx_pred[mask])**2) if dydx_pred is not None else None
        results["moneyness"][m_labels[i]] = {
            "n": int(n), "value_mse": float(vmse),
            "grad_mse": float(gmse) if gmse is not None else None
        }

    # Maturity buckets
    t_bins = [0, 0.083, 0.25, 0.5, 1.0, 3.0]
    t_labels = ["<1M", "1-3M", "3-6M", "6-12M", "1Y+"]
    results["maturity"] = {}

    for i in range(len(t_bins) - 1):
        mask = (maturity >= t_bins[i]) & (maturity < t_bins[i+1])
        n = mask.sum()
        if n < 10:
            continue
        vmse = np.mean((y_true[mask] - y_pred[mask])**2)
        gmse = np.mean((dydx_true[mask] - dydx_pred[mask])**2) if dydx_pred is not None else None
        results["maturity"][t_labels[i]] = {
            "n": int(n), "value_mse": float(vmse),
            "grad_mse": float(gmse) if gmse is not None else None
        }

    return results


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data/spy_options/spy_processed.npz")

    print("Loading SPY data...")
    (X_train, y_train, dydx_train,
     X_test, y_test, dydx_test,
     feature_names, test_moneyness, test_maturity) = load_spy_data(data_path)

    methods = ["vanilla", "dml_fixed"]
    seeds = [42, 123, 456]
    all_results = {}

    for method in methods:
        method_results = []
        for seed in seeds:
            print(f"\nTraining {method} (seed={seed})...")
            y_pred, dydx_pred = train_and_predict(
                X_train, y_train, dydx_train, X_test,
                method=method, seed=seed, n_epochs=500, subsample=10000
            )

            buckets = bucket_analysis(
                y_test, y_pred, dydx_test, dydx_pred,
                test_moneyness, test_maturity, method
            )
            method_results.append(buckets)

            # Print summary
            overall_vmse = np.mean((y_test - y_pred)**2)
            overall_gmse = np.mean((dydx_test - dydx_pred)**2)
            print(f"  Overall: value_mse={overall_vmse:.6e}, grad_mse={overall_gmse:.6e}")

        all_results[method] = method_results

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results/spy_robustness_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SPY ROBUSTNESS: MONEYNESS BREAKDOWN")
    print("=" * 80)
    for method in methods:
        print(f"\n{method}:")
        for bucket_type in ["moneyness", "maturity"]:
            print(f"  {bucket_type.upper()}:")
            # Average across seeds
            for label in all_results[method][0][bucket_type]:
                vals = [r[bucket_type][label]["value_mse"] for r in all_results[method]
                        if label in r[bucket_type]]
                if vals:
                    print(f"    {label}: value_mse = {np.mean(vals):.6e} ± {np.std(vals):.6e}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
