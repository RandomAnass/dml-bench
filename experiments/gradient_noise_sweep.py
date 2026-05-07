#!/usr/bin/env python3
"""
Gradient Noise Injection Experiment — Controlled σ₁ sweep.

Tests the effect of noisy derivative supervision by injecting Gaussian noise
into the gradient signal at varying magnitudes σ₁, while keeping value noise
at σ₀ = 0. This directly validates the theoretical noise crossover threshold
σ* and reproduces the Heston failure mode in a controlled setting.

Theory (DIC Booklet Ch. 5, Theorem 5.5):
  When derivative noise exceeds σ*, DML hurts more than it helps:
    σ* ≈ σ₀ · n^{1/(2(2k+d))}
  For pure gradient noise (σ₀ = 0), the crossover is at:
    σ₁* ≈ threshold where MSE_DML > MSE_vanilla

This script reuses the data generation and model infrastructure from
higher_order_dml.py but injects controlled noise into the gradient channel.

Usage:
    python experiments/gradient_noise_sweep.py                # Full sweep
    python experiments/gradient_noise_sweep.py --quick        # Fast test
    python experiments/gradient_noise_sweep.py --sigma1 0.0 0.01 0.5 5.0
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dml_benchmark.model import DmlFeedForward
from dml_benchmark.trainer import set_deterministic

# Reuse data generation from higher_order_dml
from experiments.higher_order_dml import (
    generate_higher_order_data,
    verify_consistency,
    forward_higher_order,
)


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class NoiseResult:
    """Result of a single gradient-noise experiment."""
    function_type: str
    n_dim: int
    n_samples: int
    method: str            # 'vanilla' or 'dml'
    sigma1: float          # gradient noise level
    seed: int
    test_value_mse: float
    test_grad_mse: float
    best_epoch: int
    train_time_s: float
    loss_history: List[float] = field(default_factory=list)


# ============================================================================
# TRAINING
# ============================================================================

def train_with_noisy_gradients(
    function_type: str,
    n_dim: int,
    n_samples: int = 4096,
    sigma1: float = 0.0,
    use_derivatives: bool = True,
    seed: int = 42,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 0.005,
    lambda_: float = 1.0,
    n_layers: int = 4,
    hidden_size: int = 256,
    patience: int = 50,
    device: str = "cuda",
) -> NoiseResult:
    """
    Train with optional noisy gradient supervision.

    Generates clean data, then adds N(0, σ₁²) noise to the gradient channel.
    The test set always uses clean gradients for fair evaluation.

    Args:
        function_type: 'poly_trig' or 'trig'
        n_dim: Input dimension
        n_samples: Total samples (80/20 train/test split)
        sigma1: Gradient noise standard deviation (0 = clean)
        use_derivatives: If False, vanilla training (ignore gradients)
        seed: Random seed
        [other args]: Same as higher_order_dml.train_higher_order

    Returns:
        NoiseResult with test metrics
    """
    set_deterministic(seed)
    device = device if torch.cuda.is_available() else "cpu"

    # ---- Generate clean data ----
    data = generate_higher_order_data(function_type, n_dim, n_samples,
                                      max_order=1, seed=seed)

    # ---- Train/test split ----
    n_train = int(0.8 * n_samples)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    x_train, x_test = to_t(data.x[train_idx]), to_t(data.x[test_idx])
    y_train, y_test = to_t(data.y[train_idx]), to_t(data.y[test_idx])
    grad_clean = to_t(data.gradient[train_idx])
    grad_test = to_t(data.gradient[test_idx])

    # ---- Inject noise into training gradients ----
    if sigma1 > 0:
        noise_gen = torch.Generator(device=device).manual_seed(seed + 9999)
        grad_noise = torch.randn(grad_clean.shape, generator=noise_gen,
                                 device=device) * sigma1
        grad_train = grad_clean + grad_noise
    else:
        grad_train = grad_clean

    order = 1 if use_derivatives else 0
    method = "dml" if use_derivatives else "vanilla"

    # ---- Model ----
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

    for epoch in range(n_epochs):
        model.train()

        # Random mini-batch from training set
        idx = torch.randint(0, n_train, (min(batch_size, n_train),), device=device)
        bx, by = x_train[idx], y_train[idx]

        optimizer.zero_grad()

        if use_derivatives:
            preds = forward_higher_order(model, bx, max_order=1, training=True)
            bg = grad_train[idx]
            loss_val = torch.nn.functional.mse_loss(preds["y"], by)
            loss_grad = torch.nn.functional.mse_loss(preds["gradient"], bg) / n_dim
            loss = loss_val + lambda_ * loss_grad
        else:
            y_pred = model(bx)
            loss = torch.nn.functional.mse_loss(y_pred, by)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        loss_history.append(loss.item())

        # ---- Validation (on clean data, value MSE only) ----
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(model(x_test), y_test).item()

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

    train_time = time.time() - t0

    # ---- Evaluate on clean test data ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.enable_grad():
        test_preds = forward_higher_order(model, x_test, max_order=1, training=False)

    test_value_mse = torch.nn.functional.mse_loss(
        test_preds["y"].detach(), y_test).item()
    test_grad_mse = torch.nn.functional.mse_loss(
        test_preds["gradient"].detach(), grad_test).item()

    return NoiseResult(
        function_type=function_type, n_dim=n_dim, n_samples=n_samples,
        method=method, sigma1=sigma1, seed=seed,
        test_value_mse=test_value_mse, test_grad_mse=test_grad_mse,
        best_epoch=best_epoch, train_time_s=train_time,
        loss_history=loss_history[-10:],
    )


# ============================================================================
# EXPERIMENT GRID
# ============================================================================

DEFAULT_SIGMA1 = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
SEEDS = [42, 123, 456, 789, 1000]


def run_noise_sweep(
    functions: List[str],
    dims: List[int],
    sigma1_values: List[float],
    n_seeds: int = 5,
    n_samples: int = 4096,
    device: str = "cuda",
) -> List[NoiseResult]:
    """Run the full noise sweep grid."""
    seeds = SEEDS[:n_seeds]
    results = []

    # Total: funcs × dims × (1 vanilla + len(sigmas) DML) × seeds
    n_configs = len(functions) * len(dims) * (1 + len(sigma1_values)) * n_seeds
    count = 0

    for func in functions:
        for d in dims:
            # ---- Vanilla baseline (no derivatives, run once per seed) ----
            for seed in seeds:
                count += 1
                tag = f"[{count}/{n_configs}] {func} d={d} vanilla seed={seed}"
                print(f"  {tag} ...", end=" ", flush=True)

                r = train_with_noisy_gradients(
                    func, d, n_samples=n_samples, sigma1=0.0,
                    use_derivatives=False, seed=seed, device=device)
                results.append(r)
                print(f"MSE={r.test_value_mse:.6f} ({r.train_time_s:.1f}s, ep={r.best_epoch})")

            # ---- DML with each noise level ----
            for s1 in sigma1_values:
                for seed in seeds:
                    count += 1
                    tag = f"[{count}/{n_configs}] {func} d={d} σ₁={s1:.3f} seed={seed}"
                    print(f"  {tag} ...", end=" ", flush=True)

                    r = train_with_noisy_gradients(
                        func, d, n_samples=n_samples, sigma1=s1,
                        use_derivatives=True, seed=seed, device=device)
                    results.append(r)
                    print(f"MSE={r.test_value_mse:.6f} ({r.train_time_s:.1f}s, ep={r.best_epoch})")

    return results


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_noise_results(results: List[NoiseResult]):
    """Analyze and print noise sweep results with crossover detection."""
    if not results:
        print("No results to analyze.")
        return

    print("\n" + "=" * 100)
    print("GRADIENT NOISE SWEEP RESULTS")
    print("=" * 100)

    # Group by (function, dim)
    groups: Dict[Tuple[str, int], List[NoiseResult]] = defaultdict(list)
    for r in results:
        groups[(r.function_type, r.n_dim)].append(r)

    for (func, dim) in sorted(groups.keys()):
        rlist = groups[(func, dim)]

        # Separate vanilla and DML results
        vanilla = [r for r in rlist if r.method == "vanilla"]
        dml_by_sigma = defaultdict(list)
        for r in rlist:
            if r.method == "dml":
                dml_by_sigma[r.sigma1].append(r)

        van_mean = np.mean([r.test_value_mse for r in vanilla])
        van_std = np.std([r.test_value_mse for r in vanilla])
        van_grad = np.mean([r.test_grad_mse for r in vanilla])

        print(f"\n{'─' * 90}")
        print(f"  {func} d={dim}  |  n={rlist[0].n_samples}  |  "
              f"Vanilla value MSE: {van_mean:.6f} ± {van_std:.6f}")
        print(f"{'─' * 90}")
        print(f"  {'σ₁':>8} | {'DML value MSE':>18} | {'Advantage':>12} | "
              f"{'DML grad MSE':>16} | {'Status':>14}")
        print(f"  {'─'*8}─┼─{'─'*18}─┼─{'─'*12}─┼─{'─'*16}─┼─{'─'*14}")

        sigmas_sorted = sorted(dml_by_sigma.keys())
        crossover_sigma = None
        prev_status = "HELPS"

        for s1 in sigmas_sorted:
            dml_runs = dml_by_sigma[s1]
            dml_mean = np.mean([r.test_value_mse for r in dml_runs])
            dml_std = np.std([r.test_value_mse for r in dml_runs])
            dml_grad = np.mean([r.test_grad_mse for r in dml_runs])

            if van_mean > 0:
                advantage = (1 - dml_mean / van_mean) * 100
            else:
                advantage = 0.0

            if dml_mean < van_mean:
                status = "✅ DML HELPS"
                curr = "HELPS"
            else:
                status = "❌ DML HURTS"
                curr = "HURTS"

            # Detect crossover
            if prev_status == "HELPS" and curr == "HURTS" and crossover_sigma is None:
                crossover_sigma = s1

            prev_status = curr

            print(f"  {s1:>8.3f} | {dml_mean:>10.6f}±{dml_std:.4f} | "
                  f"{advantage:>+10.1f}% | {dml_grad:>14.6f} | {status}")

        # Crossover summary
        print()
        if crossover_sigma is not None:
            # Interpolate crossover between last-good and first-bad
            idx = sigmas_sorted.index(crossover_sigma)
            if idx > 0:
                s_lo = sigmas_sorted[idx - 1]
                s_hi = crossover_sigma
                # Log-space interpolation
                sigma_star = np.exp((np.log(s_lo) + np.log(s_hi)) / 2) if s_lo > 0 else s_hi / 2
                print(f"  ⚡ CROSSOVER DETECTED: σ₁* ∈ ({s_lo:.4f}, {s_hi:.4f}), "
                      f"estimate σ₁* ≈ {sigma_star:.4f}")
            else:
                print(f"  ⚡ DML hurts even at σ₁ = {crossover_sigma:.4f}")
        else:
            if prev_status == "HELPS":
                print(f"  ✅ DML helps at all tested σ₁ levels (no crossover detected)")
            else:
                print(f"  ❌ DML hurts at all tested σ₁ levels")

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("CROSSOVER SUMMARY")
    print("=" * 100)
    print(f"\n  {'Function':<12} {'d':>3} | {'Vanilla MSE':>14} | "
          f"{'σ₁*  (estimated)':>18} | {'Clean DML adv.':>16} | "
          f"{'Max σ₁ DML adv.':>18}")
    print(f"  {'─'*12}─{'─'*3}─┼─{'─'*14}─┼─{'─'*18}─┼─{'─'*16}─┼─{'─'*18}")

    for (func, dim) in sorted(groups.keys()):
        rlist = groups[(func, dim)]
        vanilla = [r for r in rlist if r.method == "vanilla"]
        dml_by_sigma = defaultdict(list)
        for r in rlist:
            if r.method == "dml":
                dml_by_sigma[r.sigma1].append(r)

        van_mean = np.mean([r.test_value_mse for r in vanilla])

        sigmas_sorted = sorted(dml_by_sigma.keys())
        advantages = {}
        crossover = None
        prev_good = True

        for s1 in sigmas_sorted:
            dml_mean = np.mean([r.test_value_mse for r in dml_by_sigma[s1]])
            adv = (1 - dml_mean / van_mean) * 100 if van_mean > 0 else 0
            advantages[s1] = adv
            if dml_mean >= van_mean and prev_good and crossover is None:
                idx = sigmas_sorted.index(s1)
                if idx > 0:
                    s_lo = sigmas_sorted[idx - 1]
                    crossover = np.exp((np.log(max(s_lo, 1e-6)) + np.log(s1)) / 2)
                else:
                    crossover = s1 / 2
            prev_good = dml_mean < van_mean

        clean_adv = advantages.get(0.0, 0)
        max_sigma = sigmas_sorted[-1] if sigmas_sorted else 0
        max_adv = advantages.get(max_sigma, 0)
        xover_str = f"{crossover:.4f}" if crossover else "none"

        print(f"  {func:<12} {dim:>3} | {van_mean:>14.6f} | {xover_str:>18} | "
              f"{clean_adv:>+14.1f}% | {max_adv:>+16.1f}%")

    # ---- Heston comparison ----
    print("\n" + "=" * 100)
    print("HESTON COMPARISON")
    print("=" * 100)
    print("""
  Heston model has σ₁/σ₀ ≈ 50 for implied volatility Greeks.
  This experiment shows what happens at high σ₁ in a controlled setting.

  If DML fails at σ₁ ≈ 1.0–5.0 for poly_trig (which normally gets +78.8%),
  this confirms: the Heston failure is due to derivative noise, not the
  function's smoothness or the model's capacity.

  Key implication: DML + noisy derivatives is worse than no derivatives at all.
  Practitioners should estimate their derivative SNR before applying DML.
""")


def save_results(results: List[NoiseResult], output_dir: Path):
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        s1_tag = f"{r.sigma1:.4f}".replace(".", "p")
        fname = (f"gn_{r.function_type}_d{r.n_dim}_{r.method}_"
                 f"sigma1_{s1_tag}_seed{r.seed}.json")
        with open(output_dir / fname, "w") as f:
            json.dump(asdict(r), f, indent=2)
    print(f"\nSaved {len(results)} results to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gradient Noise Injection Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/gradient_noise_sweep.py                    # Full sweep
  python experiments/gradient_noise_sweep.py --quick            # Fast test
  python experiments/gradient_noise_sweep.py --sigma1 0.0 0.5 5.0  # Custom σ₁
  python experiments/gradient_noise_sweep.py --n-samples 8192   # More data
        """,
    )
    parser.add_argument("--functions", nargs="+", default=["poly_trig"],
                        choices=["poly_trig", "trig"],
                        help="Functions to test (default: poly_trig)")
    parser.add_argument("--dims", nargs="+", type=int, default=[5],
                        help="Dimensions (default: 5)")
    parser.add_argument("--sigma1", nargs="+", type=float, default=None,
                        help=f"σ₁ values (default: {DEFAULT_SIGMA1})")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (default: 5)")
    parser.add_argument("--n-samples", type=int, default=4096,
                        help="Samples per experiment (default: 4096)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: poly_trig d=5, 2 seeds, 3 σ₁ values")
    args = parser.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
    else:
        device = "cpu"

    sigma1_values = args.sigma1 or DEFAULT_SIGMA1

    if args.quick:
        args.functions = ["poly_trig"]
        args.dims = [5]
        args.seeds = 2
        sigma1_values = [0.0, 0.1, 5.0]

    n_configs = len(args.functions) * len(args.dims) * (1 + len(sigma1_values)) * args.seeds

    print("=" * 70)
    print("GRADIENT NOISE INJECTION EXPERIMENT")
    print("=" * 70)
    print(f"  Functions:   {args.functions}")
    print(f"  Dimensions:  {args.dims}")
    print(f"  σ₁ values:   {sigma1_values}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Samples:     {args.n_samples}")
    print(f"  Device:      {device}")
    print(f"  Total runs:  {n_configs}")
    print(f"  Theory:      σ₁* is the noise level where DML breaks even")
    print()

    # Verify data consistency
    print("Verifying data consistency...")
    for func in args.functions:
        for d in args.dims:
            data = generate_higher_order_data(func, d, 100, max_order=1, seed=42)
            ok = verify_consistency(data, seed=42)
            print(f"  {'✓' if ok else '✗'} {func} d={d}")
    print()

    # Run sweep
    print("Running gradient noise sweep...")
    results = run_noise_sweep(
        functions=args.functions,
        dims=args.dims,
        sigma1_values=sigma1_values,
        n_seeds=args.seeds,
        n_samples=args.n_samples,
        device=device,
    )

    # Analyze & save
    analyze_noise_results(results)
    save_results(results, PROJECT_ROOT / "results" / "gradient_noise_sweep")


if __name__ == "__main__":
    main()
