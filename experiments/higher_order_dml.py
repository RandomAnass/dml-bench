#!/usr/bin/env python3
"""
Higher-Order Derivative-Enhanced ML — Standalone Experiment.

Tests the effect of adding higher-order derivative information to training:
  Order 0: L = MSE(y)                                        — vanilla
  Order 1: L = MSE(y) + λ·MSE(∇y)/d                          — DML (current)
  Order 2: L = MSE(y) + λ·MSE(∇y)/d + λ·MSE(∇²y)/d²         — DML + Hessian
  Order 3: L = MSE(y) + λ·MSE(∇y)/d + λ·MSE(∇²y)/d² + ...   — DML + Hessian + 3rd

Theoretical motivation (DIC Booklet Ch. 6):
  Each derivative order q should reduce effective dimension by 1:
    d_eff(order=0) = d,  d_eff(order=1) = d-1,  d_eff(order=2) = d-2, ...
  Minimax rate: n^{-2k/(2k + d_eff)}
  So higher-order derivatives should yield strictly better convergence,
  at the cost of O(d^q) additional constraints per sample.

This script is self-contained and does NOT modify the existing codebase.
It reuses the model architecture and data generation but implements its
own training loop with higher-order loss terms.

Usage:
    # Quick test (1 function, 1 dimension, 2 seeds)
    python experiments/higher_order_dml.py --quick

    # Full experiment grid
    python experiments/higher_order_dml.py

    # Specific configuration
    python experiments/higher_order_dml.py --functions poly_trig --dims 2 5 --max-order 3
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from jax import grad, vmap, random

from dml_benchmark.model import DmlFeedForward
from dml_benchmark.trainer import set_deterministic


# ============================================================================
# HIGHER-ORDER DATA GENERATION
# ============================================================================

def _replicate_key_sequence(seed: int):
    """Replicate the exact key splitting pattern of BenchmarkFunctionGenerator.

    BenchmarkFunctionGenerator uses sequential random.split(self.key):
      key = PRNGKey(seed)
      key, subkey1 = random.split(key)  # first _split_key() call
      key, subkey2 = random.split(key)  # second _split_key() call
      ...

    Returns: a generator that yields subkeys in the same order.
    """
    key = random.PRNGKey(seed)
    while True:
        key, subkey = random.split(key)
        yield subkey


def _make_trig_fn(n_dim: int, seed: int = 42):
    """Create the trig scalar function matching BenchmarkFunctionGenerator exactly.

    Key sequence in generate_trigonometric:
      subkey1 → frequencies  (_split_key #1)
      subkey2 → amplitudes   (_split_key #2)
      subkey3 → x data       (_split_key #3 via _generate_random_x)
    """
    keys = _replicate_key_sequence(seed)

    k_freq = next(keys)
    k_amp = next(keys)
    k_x = next(keys)

    frequencies = np.asarray(random.uniform(k_freq, shape=(n_dim,), minval=0.5, maxval=5.0))
    amplitudes = np.asarray(random.uniform(k_amp, shape=(n_dim,), minval=0.5, maxval=2.0))

    freq_jax = jnp.array(frequencies)
    amp_jax = jnp.array(amplitudes)

    def f(x):
        return jnp.sum(amp_jax * jnp.sin(freq_jax * x))

    return f, k_x, frequencies, amplitudes


def _make_poly_trig_fn(n_dim: int, seed: int = 42):
    """Create the poly_trig scalar function matching BenchmarkFunctionGenerator exactly.

    Key sequence in generate_poly_trig:
      subkey1 → coeffs       (_split_key #1)
      subkey2 → x data       (_split_key #2 via _generate_random_x)
    """
    keys = _replicate_key_sequence(seed)

    k_coeffs = next(keys)
    k_x = next(keys)

    poly_degree = 3
    alpha = 0.5
    frequency = 2.0

    coeffs = np.asarray(random.uniform(
        k_coeffs, shape=(n_dim, poly_degree + 1), minval=-1, maxval=1
    )) * np.array([0.9 ** i for i in range(poly_degree + 1)])
    coeffs_jax = jnp.array(coeffs)

    def f(x):
        poly = 0.0
        for j in range(n_dim):
            for k in range(poly_degree + 1):
                poly = poly + coeffs_jax[j, k] * (x[j] ** k)
        trig = alpha * jnp.sum(jnp.sin(frequency * x))
        return poly + trig

    return f, k_x, coeffs


@dataclass
class HigherOrderData:
    """Data container with derivatives up to a given order."""
    x: np.ndarray               # (n, d)
    y: np.ndarray               # (n, 1)
    gradient: np.ndarray        # (n, d)         — order 1
    hessian: Optional[np.ndarray] = None   # (n, d, d)     — order 2
    third: Optional[np.ndarray] = None     # (n, d, d, d)  — order 3
    function_type: str = ""
    n_dim: int = 0


def generate_higher_order_data(
    function_type: str,
    n_dim: int,
    n_samples: int,
    max_order: int = 2,
    seed: int = 42,
) -> HigherOrderData:
    """
    Generate data with derivatives up to `max_order`.

    Uses JAX autodiff for exact computation of all derivative tensors.

    Args:
        function_type: 'trig' or 'poly_trig'
        n_dim: Input dimension
        n_samples: Number of samples
        max_order: Maximum derivative order (1=gradient, 2=+Hessian, 3=+third)
        seed: Random seed

    Returns:
        HigherOrderData with all requested derivative tensors
    """
    # Construct the JAX scalar function (matching BenchmarkFunctionGenerator key sequence)
    if function_type == "trig":
        f, k_x, *_ = _make_trig_fn(n_dim, seed)
        x_low, x_high = -np.pi, np.pi
    elif function_type == "poly_trig":
        f, k_x, *_ = _make_poly_trig_fn(n_dim, seed)
        x_low, x_high = -1.0, 1.0
    else:
        raise ValueError(f"Unsupported function: {function_type}. Use 'trig' or 'poly_trig'.")

    # Generate random x using the exact same key as _generate_random_x
    x = np.asarray(random.uniform(k_x, shape=(n_samples, n_dim), minval=x_low, maxval=x_high))

    # Compute values and derivatives using JAX
    x_jax = jnp.array(x)

    # Order 0: values
    f_vec = vmap(f)
    y = np.asarray(f_vec(x_jax)).reshape(-1, 1)

    # Order 1: gradient
    grad_f = vmap(grad(f))
    gradient = np.asarray(grad_f(x_jax))  # (n, d)

    # Order 2: Hessian
    hessian = None
    if max_order >= 2:
        hessian_f = vmap(jax.hessian(f))
        hessian = np.asarray(hessian_f(x_jax))  # (n, d, d)

    # Order 3: third-order tensor
    third = None
    if max_order >= 3:
        third_f = vmap(jax.jacfwd(jax.hessian(f)))
        third = np.asarray(third_f(x_jax))  # (n, d, d, d)

    return HigherOrderData(
        x=x, y=y, gradient=gradient,
        hessian=hessian, third=third,
        function_type=function_type, n_dim=n_dim
    )


def verify_consistency(data: HigherOrderData, seed: int):
    """Verify our data matches the existing generate_data() output."""
    from dml_benchmark.functions import generate_data

    ref = generate_data(data.function_type, data.n_dim, len(data.x), seed=seed)
    y_match = np.allclose(data.y, ref.y, atol=1e-5)
    grad_match = np.allclose(data.gradient, ref.dydx.reshape(-1, data.n_dim), atol=1e-5)

    if not y_match or not grad_match:
        print(f"  ⚠️  Consistency check FAILED for {data.function_type} d={data.n_dim}")
        print(f"      y match: {y_match}, grad match: {grad_match}")
        if not y_match:
            print(f"      y max diff: {np.max(np.abs(data.y - ref.y)):.2e}")
        if not grad_match:
            ref_grad = ref.dydx.reshape(-1, data.n_dim)
            print(f"      grad max diff: {np.max(np.abs(data.gradient - ref_grad)):.2e}")
        return False
    return True


# ============================================================================
# HIGHER-ORDER FORWARD PASS (NN)
# ============================================================================

def forward_higher_order(
    model: DmlFeedForward,
    x: torch.Tensor,
    max_order: int = 1,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass computing NN output and derivatives up to max_order.

    Args:
        model: The neural network
        x: Input tensor (batch, d)
        max_order: 0=value only, 1=+gradient, 2=+Hessian, 3=+third
        training: If True, build computation graph for backprop through all terms

    Returns:
        Dict with keys 'y', 'gradient', 'hessian', 'third' as available
    """
    x = x.detach().clone().requires_grad_(True)
    y = model(x)  # (batch, 1)
    result = {"y": y}

    if max_order >= 1:
        # Gradient: dy/dx — shape (batch, d)
        dydx = torch.autograd.grad(
            y.sum(), x, create_graph=True, retain_graph=True
        )[0]
        result["gradient"] = dydx

        if max_order >= 2:
            # Hessian: d²y/dxᵢdxⱼ — shape (batch, d, d)
            # create_graph=True is needed whenever we want to backprop through
            # the Hessian loss term (training), or compute higher derivatives.
            batch_size, d = x.shape
            hess_rows = []
            for j in range(d):
                row_j = torch.autograd.grad(
                    dydx[:, j].sum(), x,
                    create_graph=(training or max_order >= 3),
                    retain_graph=True
                )[0]  # (batch, d)
                hess_rows.append(row_j)
            result["hessian"] = torch.stack(hess_rows, dim=1)  # (batch, d, d)

            if max_order >= 3:
                # Third-order: d³y/dxᵢdxⱼdxₖ — shape (batch, d, d, d)
                hessian = result["hessian"]
                third_slices = []
                for j in range(d):
                    third_j = []
                    for k in range(d):
                        grad_jk = torch.autograd.grad(
                            hessian[:, j, k].sum(), x,
                            create_graph=training,
                            retain_graph=True
                        )[0]  # (batch, d)
                        third_j.append(grad_jk)
                    third_slices.append(torch.stack(third_j, dim=1))  # (batch, d, d)
                result["third"] = torch.stack(third_slices, dim=1)  # (batch, d, d, d)

    return result


# ============================================================================
# HIGHER-ORDER LOSS
# ============================================================================

def higher_order_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    order: int,
    lambda_: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the higher-order DML loss.

    L = MSE(y) + λ·[MSE(∇y)/d + MSE(∇²y)/d² + MSE(∇³y)/d³]

    Each derivative term is normalized by the number of components so that
    each individual derivative constraint contributes equally.

    Returns:
        (total_loss, loss_components_dict)
    """
    # Value loss (always present)
    loss_value = torch.nn.functional.mse_loss(predictions["y"], targets["y"])
    components = {"value": loss_value.item()}
    total = loss_value

    d = predictions["gradient"].shape[1] if "gradient" in predictions else 1

    if order >= 1 and "gradient" in predictions:
        loss_grad = torch.nn.functional.mse_loss(
            predictions["gradient"], targets["gradient"]
        ) / d
        total = total + lambda_ * loss_grad
        components["gradient"] = loss_grad.item()

    if order >= 2 and "hessian" in predictions:
        loss_hess = torch.nn.functional.mse_loss(
            predictions["hessian"], targets["hessian"]
        ) / (d * d)
        total = total + lambda_ * loss_hess
        components["hessian"] = loss_hess.item()

    if order >= 3 and "third" in predictions:
        loss_third = torch.nn.functional.mse_loss(
            predictions["third"], targets["third"]
        ) / (d * d * d)
        total = total + lambda_ * loss_third
        components["third"] = loss_third.item()

    components["total"] = total.item()
    return total, components


# ============================================================================
# TRAINING LOOP
# ============================================================================

@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    function_type: str
    n_dim: int
    n_samples: int
    order: int
    seed: int
    test_value_mse: float
    test_grad_mse: float
    test_hessian_mse: float
    best_epoch: int
    train_time_s: float
    loss_history: List[float] = field(default_factory=list)


def train_higher_order(
    function_type: str,
    n_dim: int,
    n_samples: int = 1024,
    order: int = 1,
    seed: int = 42,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 0.005,
    lambda_: float = 1.0,
    n_layers: int = 4,
    hidden_size: int = 256,
    patience: int = 50,
    device: str = "cuda",
    verbose: bool = False,
) -> ExperimentResult:
    """
    Run a single higher-order DML training experiment.

    Args:
        function_type: 'trig' or 'poly_trig'
        n_dim: Input dimension
        n_samples: Total samples (80/20 train/test split)
        order: Derivative order (0=vanilla, 1=DML, 2=+Hessian, 3=+third)
        seed: Random seed
        n_epochs: Maximum training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        lambda_: Derivative loss weight
        n_layers: Network depth
        hidden_size: Hidden layer width
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        ExperimentResult with test metrics
    """
    set_deterministic(seed)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ---- Generate data with higher-order derivatives ----
    max_data_order = min(order, 3)
    data = generate_higher_order_data(function_type, n_dim, n_samples, max_data_order, seed)

    # Train/test split (80/20)
    n_train = int(0.8 * n_samples)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    x_train, x_test = to_tensor(data.x[train_idx]), to_tensor(data.x[test_idx])
    y_train, y_test = to_tensor(data.y[train_idx]), to_tensor(data.y[test_idx])
    grad_train = to_tensor(data.gradient[train_idx])
    grad_test = to_tensor(data.gradient[test_idx])

    hess_train = to_tensor(data.hessian[train_idx]) if data.hessian is not None else None
    hess_test = to_tensor(data.hessian[test_idx]) if data.hessian is not None else None
    third_train = to_tensor(data.third[train_idx]) if data.third is not None else None
    third_test = to_tensor(data.third[test_idx]) if data.third is not None else None

    # ---- Create model ----
    model = DmlFeedForward(n_dim, 1, n_layers, hidden_size, "softplus").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    loss_history = []

    t0 = time.time()

    # Build train dataloader
    train_tensors = [x_train, y_train]
    train_ds = TensorDataset(*train_tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            bx, by = batch
            batch_indices = None  # We track indices separately for higher-order targets

            # For simplicity with batching of higher-order targets:
            # We use the full training set for gradient/hessian matching
            # since batch-level indexing into pre-computed tensors is complex.
            # Instead, use random sub-batches from the training data.
            idx = torch.randint(0, n_train, (min(batch_size, n_train),), device=device)
            bx = x_train[idx]
            by = y_train[idx]

            optimizer.zero_grad()

            preds = forward_higher_order(model, bx, max_order=order, training=True)

            targets = {"y": by, "gradient": grad_train[idx]}
            if hess_train is not None and order >= 2:
                targets["hessian"] = hess_train[idx]
            if third_train is not None and order >= 3:
                targets["third"] = third_train[idx]

            loss, _ = higher_order_loss(preds, targets, order, lambda_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
            val_loss = torch.nn.functional.mse_loss(y_pred, y_test).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if verbose and epoch % 50 == 0:
            print(f"    Epoch {epoch:4d}  train_loss={avg_loss:.6f}  val_mse={val_loss:.6f}")

    train_time = time.time() - t0

    # ---- Evaluate best model ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.enable_grad():
        # Always compute at least order 1 for test evaluation of gradient quality
        eval_order = max(min(order, 2), 1)
        test_preds = forward_higher_order(model, x_test, max_order=eval_order, training=False)

    test_value_mse = torch.nn.functional.mse_loss(
        test_preds["y"].detach(), y_test
    ).item()

    test_grad_mse = torch.nn.functional.mse_loss(
        test_preds["gradient"].detach(), grad_test
    ).item() if "gradient" in test_preds else float("nan")

    test_hessian_mse = 0.0
    if hess_test is not None and "hessian" in test_preds:
        test_hessian_mse = torch.nn.functional.mse_loss(
            test_preds["hessian"].detach(), hess_test
        ).item()

    return ExperimentResult(
        function_type=function_type,
        n_dim=n_dim,
        n_samples=n_samples,
        order=order,
        seed=seed,
        test_value_mse=test_value_mse,
        test_grad_mse=test_grad_mse,
        test_hessian_mse=test_hessian_mse,
        best_epoch=best_epoch,
        train_time_s=train_time,
        loss_history=loss_history,
    )


# ============================================================================
# EXPERIMENT GRID
# ============================================================================

def run_experiment_grid(
    functions: List[str],
    dims: List[int],
    max_order: int,
    n_seeds: int = 5,
    n_samples: int = 1024,
    device: str = "cuda",
    verbose: bool = False,
) -> List[ExperimentResult]:
    """Run the full experiment grid and return results."""
    seeds = [42, 123, 456, 789, 1000][:n_seeds]
    results = []

    # Determine which orders to run for each dimension
    # Order 3 is O(d²) backward passes — skip for d > 5
    def orders_for_dim(d, max_ord):
        if max_ord >= 3 and d > 5:
            return list(range(min(max_ord, 2) + 1))
        return list(range(max_ord + 1))

    total = sum(
        len(orders_for_dim(d, max_order)) * n_seeds
        for _ in functions for d in dims
    )
    count = 0

    for func in functions:
        for d in dims:
            orders = orders_for_dim(d, max_order)
            for order in orders:
                for seed in seeds:
                    count += 1
                    label = f"[{count}/{total}] {func} d={d} order={order} seed={seed}"
                    print(f"  {label} ...", end=" ", flush=True)

                    try:
                        result = train_higher_order(
                            function_type=func, n_dim=d, n_samples=n_samples,
                            order=order, seed=seed, device=device, verbose=verbose,
                        )
                        results.append(result)
                        print(f"val_MSE={result.test_value_mse:.6f} "
                              f"grad_MSE={result.test_grad_mse:.6f} "
                              f"({result.train_time_s:.1f}s, ep={result.best_epoch})")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        import traceback
                        traceback.print_exc()

    return results


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_results(results: List[ExperimentResult]):
    """Analyze and print a comprehensive results table."""
    if not results:
        print("No results to analyze.")
        return

    print("\n" + "=" * 90)
    print("HIGHER-ORDER DML RESULTS")
    print("=" * 90)

    # Group by (function, dim)
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        groups[(r.function_type, r.n_dim)][r.order].append(r)

    # Header
    order_names = {0: "Vanilla", 1: "DML (∇)", 2: "DML+Hess (∇²)", 3: "DML+3rd (∇³)"}

    print(f"\n{'Function':<12} {'d':>3} | ", end="")
    for o in sorted(set(r.order for r in results)):
        print(f"  {order_names.get(o, f'Order {o}'):>16}", end="")
    print(f" |  Best order   Improvement vs vanilla")
    print("-" * 110)

    for (func, dim) in sorted(groups.keys()):
        order_data = groups[(func, dim)]
        orders = sorted(order_data.keys())

        # Compute mean ± std for each order
        means = {}
        stds = {}
        for o in orders:
            vals = [r.test_value_mse for r in order_data[o]]
            means[o] = np.mean(vals)
            stds[o] = np.std(vals)

        # Print row
        print(f"{func:<12} {dim:>3} | ", end="")
        for o in orders:
            print(f"  {means[o]:>10.6f}±{stds[o]:.4f}", end="")

        # Best order and improvement
        best_order = min(orders, key=lambda o: means[o])
        if 0 in means and means[0] > 0:
            improv = (1 - means[best_order] / means[0]) * 100
            print(f" |  Order {best_order} ({order_names.get(best_order, '?'):>12})  {improv:>+6.1f}%")
        else:
            print(f" |  Order {best_order}")

    # Marginal improvement table
    print("\n" + "=" * 90)
    print("MARGINAL IMPROVEMENT (adding each derivative order)")
    print("=" * 90)
    print(f"\n{'Function':<12} {'d':>3} | {'0→1 (add ∇)':>16} {'1→2 (add ∇²)':>16} {'2→3 (add ∇³)':>16}")
    print("-" * 80)

    for (func, dim) in sorted(groups.keys()):
        order_data = groups[(func, dim)]
        orders = sorted(order_data.keys())
        means = {o: np.mean([r.test_value_mse for r in order_data[o]]) for o in orders}

        print(f"{func:<12} {dim:>3} | ", end="")
        for prev, curr in [(0, 1), (1, 2), (2, 3)]:
            if prev in means and curr in means and means[prev] > 0:
                marginal = (1 - means[curr] / means[prev]) * 100
                print(f"  {marginal:>+13.1f}%", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

    # Timing table
    print("\n" + "=" * 90)
    print("TRAINING TIME (seconds, mean across seeds)")
    print("=" * 90)
    print(f"\n{'Function':<12} {'d':>3} | ", end="")
    all_orders = sorted(set(r.order for r in results))
    for o in all_orders:
        print(f"  {order_names.get(o, f'Order {o}'):>16}", end="")
    print(f" |  Order 2 / Order 1 overhead")
    print("-" * 100)

    for (func, dim) in sorted(groups.keys()):
        order_data = groups[(func, dim)]
        orders = sorted(order_data.keys())
        times = {o: np.mean([r.train_time_s for r in order_data[o]]) for o in orders}

        print(f"{func:<12} {dim:>3} | ", end="")
        for o in all_orders:
            if o in times:
                print(f"  {times[o]:>14.1f}s", end="")
            else:
                print(f"  {'N/A':>15}", end="")

        if 1 in times and 2 in times:
            overhead = times[2] / times[1]
            print(f" |  {overhead:.1f}×")
        else:
            print(f" |  N/A")

    # Gradient quality analysis
    print("\n" + "=" * 90)
    print("GRADIENT QUALITY (test gradient MSE)")
    print("=" * 90)
    print(f"\n{'Function':<12} {'d':>3} | ", end="")
    for o in all_orders:
        print(f"  {order_names.get(o, f'Order {o}'):>16}", end="")
    print()
    print("-" * 90)

    for (func, dim) in sorted(groups.keys()):
        order_data = groups[(func, dim)]
        orders = sorted(order_data.keys())
        gmeans = {o: np.mean([r.test_grad_mse for r in order_data[o]]) for o in orders}

        print(f"{func:<12} {dim:>3} | ", end="")
        for o in all_orders:
            if o in gmeans:
                print(f"  {gmeans[o]:>14.6f}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        print()


def save_results(results: List[ExperimentResult], output_dir: Path):
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fname = (f"ho_{r.function_type}_d{r.n_dim}_order{r.order}_"
                 f"seed{r.seed}.json")
        data = asdict(r)
        # Truncate loss history for storage efficiency
        data["loss_history"] = data["loss_history"][-10:]
        with open(output_dir / fname, "w") as f:
            json.dump(data, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Higher-Order DML Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/higher_order_dml.py --quick          # Fast test (~5 min)
  python experiments/higher_order_dml.py                  # Full grid (~2 hours)
  python experiments/higher_order_dml.py --max-order 3    # Include 3rd derivatives
  python experiments/higher_order_dml.py --dims 2 5       # Specific dimensions
        """
    )
    parser.add_argument("--functions", nargs="+", default=["poly_trig", "trig"],
                        choices=["poly_trig", "trig"],
                        help="Functions to test (default: both)")
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10],
                        help="Dimensions to test (default: 2 5 10)")
    parser.add_argument("--max-order", type=int, default=2, choices=[1, 2, 3],
                        help="Maximum derivative order (default: 2)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--n-samples", type=int, default=1024,
                        help="Samples per experiment (default: 1024)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: poly_trig d=2, 2 seeds, orders 0-2")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-epoch training progress")
    args = parser.parse_args()

    # Set GPU
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
    else:
        device = "cpu"

    # Quick mode overrides
    if args.quick:
        args.functions = ["poly_trig"]
        args.dims = [2]
        args.seeds = 2
        args.max_order = 2

    print("=" * 70)
    print("HIGHER-ORDER DML EXPERIMENT")
    print("=" * 70)
    print(f"  Functions:   {args.functions}")
    print(f"  Dimensions:  {args.dims}")
    print(f"  Max order:   {args.max_order}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Samples:     {args.n_samples}")
    print(f"  Device:      {device}")
    print(f"  Note: Order 3 skipped for d > 5 (O(d²) backward passes)")
    print()

    # Verify data consistency first
    print("Verifying data generation consistency...")
    for func in args.functions:
        for d in args.dims:
            data = generate_higher_order_data(func, d, 100, max_order=1, seed=42)
            ok = verify_consistency(data, seed=42)
            status = "✓" if ok else "✗"
            print(f"  {status} {func} d={d}")
    print()

    # Run experiments
    print("Running experiments...")
    results = run_experiment_grid(
        functions=args.functions,
        dims=args.dims,
        max_order=args.max_order,
        n_seeds=args.seeds,
        n_samples=args.n_samples,
        device=device,
        verbose=args.verbose,
    )

    # Analyze
    analyze_results(results)

    # Save
    output_dir = PROJECT_ROOT / "results" / "higher_order_dml"
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
