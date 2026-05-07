#!/usr/bin/env python3
"""
Real-World Finance Experiment: U.S. Treasury Yield Curve → Bond Portfolio NPV.

Uses ACTUAL market data from FRED (Federal Reserve Economic Data):
  - Input x ∈ ℝ⁸: daily yield curve (3M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y)
  - Output y: NPV of a fixed-coupon bond portfolio
  - Derivatives ∂y/∂xᵢ: key rate durations (analytically exact)

This is a REAL dataset — yields are actual historical observations from
U.S. Treasury markets (2000–2026, ~6000 daily observations), not sampled
from a parametric model.

Why this works for DML:
  - High-quality derivative labels: key rate durations are exact (closed-form
    from the discounting formula), not estimated via Monte Carlo
  - Practical relevance: duration/convexity hedging is central to fixed income
  - Non-trivial structure: the yield curve is not independent across tenors
    (level/slope/curvature factors), unlike synthetic benchmarks

Usage:
    python run_yieldcurve_experiments.py --gpu 0
    python run_yieldcurve_experiments.py --gpu 0 --resume
"""
import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.trainer import train_single_experiment

# ============================================================================
# CONFIGURATION
# ============================================================================

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS3MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS30"
    "&cosd=2000-01-01&coed=2026-12-31"
)

TENOR_NAMES = ["3M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "30Y"]
TENOR_YEARS = [0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0]

# Bond portfolio: mix of bonds with different maturities
# Each bond: (coupon_rate, maturity_years, face_value)
PORTFOLIO = [
    (0.02, 2.0, 100.0),    # 2% 2Y note
    (0.03, 5.0, 100.0),    # 3% 5Y note
    (0.035, 10.0, 100.0),  # 3.5% 10Y bond
    (0.04, 30.0, 100.0),   # 4% 30Y bond
]

METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

TRAIN_HPARAMS = {
    "n_epochs": 1000,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def download_yield_data(cache_path="data/fred_yields.csv"):
    """Download daily yield curve data from FRED."""
    cache = Path(cache_path)
    
    if cache.exists():
        print(f"  Loading cached data from {cache}")
        import pandas as pd
        df = pd.read_csv(cache, parse_dates=["observation_date"])
        return df
    
    print(f"  Downloading from FRED...")
    import urllib.request
    
    req = urllib.request.Request(FRED_URL, headers={
        "User-Agent": "DML-Benchmark/1.0 (academic research)"
    })
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    
    import pandas as pd
    df = pd.read_csv(StringIO(text), parse_dates=["observation_date"])
    
    # Rename columns
    col_map = {
        "DGS3MO": "3M", "DGS1": "1Y", "DGS2": "2Y", "DGS3": "3Y",
        "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y", "DGS30": "30Y",
    }
    df = df.rename(columns=col_map)
    
    # Replace '.' with NaN (FRED convention for missing data)
    for col in TENOR_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with any missing yields
    df = df.dropna(subset=TENOR_NAMES).reset_index(drop=True)
    
    # Cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} observations to {cache}")
    
    return df


def interpolate_yield(yields_8d, target_tenor, tenor_years=TENOR_YEARS):
    """
    Linear interpolation on the yield curve to get yield at any tenor.
    
    Args:
        yields_8d: (N, 8) yield curve observations (in %)
        target_tenor: float, target tenor in years
        
    Returns:
        (N,) interpolated yields in %
    """
    tenors = np.array(tenor_years)
    
    if target_tenor <= tenors[0]:
        return yields_8d[:, 0]
    if target_tenor >= tenors[-1]:
        return yields_8d[:, -1]
    
    # Find bracketing tenors
    idx = np.searchsorted(tenors, target_tenor) - 1
    t_lo, t_hi = tenors[idx], tenors[idx + 1]
    y_lo, y_hi = yields_8d[:, idx], yields_8d[:, idx + 1]
    
    # Linear interpolation
    w = (target_tenor - t_lo) / (t_hi - t_lo)
    return y_lo + w * (y_hi - y_lo)


def compute_bond_npv(yields_8d, coupon_rate, maturity, face_value,
                     semiannual=True):
    """
    Compute NPV of a bond given yield curve, and key rate durations.
    
    Uses linear interpolation on the 8-point yield curve to get
    discount rates at each cash flow date.
    
    Args:
        yields_8d: (N, 8) yield curve in % (annualized)
        coupon_rate: annual coupon rate
        maturity: years to maturity
        face_value: face value
        semiannual: if True, pay coupon semi-annually
        
    Returns:
        npv: (N,) NPV values
        krd: (N, 8) key rate durations (∂NPV/∂yᵢ)
    """
    # Cash flow schedule
    if semiannual:
        freq = 2
        dt = 0.5
    else:
        freq = 1
        dt = 1.0
    
    n_payments = int(maturity * freq)
    payment_times = np.array([(i + 1) * dt for i in range(n_payments)])
    
    coupon_payment = face_value * coupon_rate / freq
    cash_flows = np.full(n_payments, coupon_payment)
    cash_flows[-1] += face_value  # Principal at maturity
    
    N = yields_8d.shape[0]
    tenors = np.array(TENOR_YEARS)
    
    npv = np.zeros(N)
    krd = np.zeros((N, 8))
    
    for k, t_cf in enumerate(payment_times):
        # Interpolated yield at this tenor (in %)
        y_t = interpolate_yield(yields_8d, t_cf) / 100.0  # Convert to decimal
        
        # Discount factor
        df = np.exp(-y_t * t_cf)  # Continuous compounding
        
        # NPV contribution
        pv_cf = cash_flows[k] * df  # (N,)
        npv += pv_cf
        
        # Key rate duration: ∂NPV/∂yᵢ
        # ∂NPV/∂yᵢ = Σ_k CF_k · ∂DF_k/∂yᵢ
        # DF_k = exp(-y(t_k) · t_k)
        # ∂DF_k/∂yᵢ = DF_k · (-t_k) · ∂y(t_k)/∂yᵢ
        # ∂y(t_k)/∂yᵢ = interpolation weight for node i at tenor t_k
        
        # Compute interpolation weights (∂y(t_cf)/∂yᵢ) for each key rate node
        for i, t_node in enumerate(tenors):
            # Linear interpolation derivative
            if t_cf <= tenors[0]:
                w_i = 1.0 if i == 0 else 0.0
            elif t_cf >= tenors[-1]:
                w_i = 1.0 if i == len(tenors) - 1 else 0.0
            else:
                idx = np.searchsorted(tenors, t_cf) - 1
                if i == idx:
                    w_i = (tenors[idx + 1] - t_cf) / (tenors[idx + 1] - tenors[idx])
                elif i == idx + 1:
                    w_i = (t_cf - tenors[idx]) / (tenors[idx + 1] - tenors[idx])
                else:
                    w_i = 0.0
            
            # ∂NPV/∂yᵢ += CF_k · DF_k · (-t_k) · w_i / 100
            # Division by 100 because yields are in %, so ∂y/∂yᵢ has 1/100 factor
            krd[:, i] += cash_flows[k] * df * (-t_cf) * w_i / 100.0
    
    return npv, krd


def prepare_yield_curve_data(seed=42, train_ratio=0.8):
    """
    Load FRED yields, compute portfolio NPV + key rate durations.
    
    Returns dict compatible with train_single_experiment().
    """
    df = download_yield_data()
    
    # Extract yield matrix
    yields = df[TENOR_NAMES].values.astype(np.float64)  # (N, 8), in %
    N = len(yields)
    
    print(f"  {N} trading days, yields range: "
          f"{yields.min():.2f}% to {yields.max():.2f}%")
    
    # Compute portfolio NPV and derivatives
    total_npv = np.zeros(N)
    total_krd = np.zeros((N, 8))
    
    for coupon, maturity, face in PORTFOLIO:
        npv_bond, krd_bond = compute_bond_npv(yields, coupon, maturity, face)
        total_npv += npv_bond
        total_krd += krd_bond
    
    # Format for DML
    x = yields.astype(np.float32)           # (N, 8)
    y = total_npv.reshape(-1, 1).astype(np.float32)  # (N, 1)
    dydx = total_krd.reshape(N, 1, 8).astype(np.float32)  # (N, 1, 8)
    
    # Train/test split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    split = int(N * train_ratio)
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    metadata = {
        "dataset": "fred_yield_curve",
        "n_observations": N,
        "date_range": f"{df['observation_date'].min()} to {df['observation_date'].max()}",
        "tenors": TENOR_NAMES,
        "portfolio": [
            {"coupon": c, "maturity": m, "face": f}
            for c, m, f in PORTFOLIO
        ],
        "npv_range": f"[{total_npv.min():.2f}, {total_npv.max():.2f}]",
        "npv_mean": float(total_npv.mean()),
        "npv_std": float(total_npv.std()),
        "mean_abs_krd": float(np.abs(total_krd).mean()),
    }
    
    print(f"  NPV range: [{total_npv.min():.2f}, {total_npv.max():.2f}]")
    print(f"  NPV std: {total_npv.std():.2f}")
    print(f"  Mean |KRD|: {np.abs(total_krd).mean():.4f}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "dydx_train": dydx[train_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
        "dydx_test": dydx[test_idx],
        "metadata": metadata,
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_derivatives(x, y, dydx, n_checks=20, eps=1e-4):
    """
    Validate analytically computed KRDs against finite differences.
    
    This is a sanity check that our derivative computation is correct.
    """
    print("\n  Derivative validation (finite differences):")
    max_rel_error = 0
    
    for check in range(min(n_checks, len(x))):
        for dim in range(8):
            x_up = x[check:check+1].copy()
            x_dn = x[check:check+1].copy()
            x_up[0, dim] += eps
            x_dn[0, dim] -= eps
            
            # Recompute NPV
            npv_up = np.zeros(1)
            npv_dn = np.zeros(1)
            for coupon, maturity, face in PORTFOLIO:
                npv_u, _ = compute_bond_npv(x_up, coupon, maturity, face)
                npv_d, _ = compute_bond_npv(x_dn, coupon, maturity, face)
                npv_up += npv_u
                npv_dn += npv_d
            
            fd_deriv = (npv_up[0] - npv_dn[0]) / (2 * eps)
            analytic_deriv = dydx[check, 0, dim]
            
            if abs(analytic_deriv) > 1e-10:
                rel_err = abs(fd_deriv - analytic_deriv) / abs(analytic_deriv)
                max_rel_error = max(max_rel_error, rel_err)
    
    print(f"  Max relative error vs finite differences: {max_rel_error:.2e}")
    if max_rel_error < 1e-3:
        print("  ✓ Derivatives validated (< 0.1% error)")
    else:
        print(f"  ⚠ WARNING: derivative error = {max_rel_error:.2e}")
    
    return max_rel_error


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def make_key(method, seed):
    return f"yieldcurve_d8_s{seed}_{method}"


def main():
    parser = argparse.ArgumentParser(description="Yield Curve DML Experiment")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000",
                        help="Comma-separated seeds")
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    results_dir = Path("results/realworld")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing for resume
    existing = {}
    if args.resume:
        for f in results_dir.glob("yieldcurve_*.json"):
            try:
                with open(f) as fh:
                    d = json.load(fh)
                    existing[d.get("key", f.stem)] = d
            except Exception:
                pass
    
    print("=" * 70)
    print("REAL-WORLD EXPERIMENT: U.S. Treasury Yield Curve")
    print("=" * 70)
    print("Data:       FRED (Federal Reserve Economic Data)")
    print("Input:      8-point yield curve (3M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y)")
    print("Output:     Bond portfolio NPV")
    print("Derivatives: Key Rate Durations (analytically exact)")
    print(f"Methods:    {METHODS}")
    print(f"Seeds:      {seeds}")
    print()
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        
        data = prepare_yield_curve_data(seed=seed)
        
        # Validate derivatives on first seed
        if seed == seeds[0]:
            validate_derivatives(
                data["x_train"][:20],
                data["y_train"][:20],
                data["dydx_train"][:20],
            )
        
        for method in METHODS:
            key = make_key(method, seed)
            
            if args.resume and key in existing:
                print(f"  SKIP (exists): {method}")
                all_results[key] = existing[key]
                continue
            
            print(f"\n  Training: {method}...", end=" ", flush=True)
            t0 = time.time()
            
            try:
                result = train_single_experiment(
                    x_train=data["x_train"],
                    y_train=data["y_train"],
                    dydx_train=data["dydx_train"],
                    x_test=data["x_test"],
                    y_test=data["y_test"],
                    dydx_test=data["dydx_test"],
                    method=method,
                    seed=seed,
                    pbar=False,
                    **TRAIN_HPARAMS,
                )
                elapsed = time.time() - t0
                
                result_dict = {
                    "key": key,
                    "method": method,
                    "dataset": "yieldcurve",
                    "dim": 8,
                    "seed": seed,
                    "lambda": TRAIN_HPARAMS["lambda_"],
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "metadata": data["metadata"],
                    "hparams": TRAIN_HPARAMS,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Save atomically
                path = results_dir / f"{key}.json"
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(result_dict, f, indent=2, default=str)
                tmp.rename(path)
                
                all_results[key] = result_dict
                
                print(f"val_mse={result.test_value_mse:.6e}, "
                      f"grad_mse={result.test_grad_mse:.6e}, "
                      f"epoch={result.best_epoch}, "
                      f"time={elapsed:.1f}s")
                
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
    
    # Analysis
    print("\n" + "=" * 90)
    print("RESULTS: Yield Curve Experiment")
    print("=" * 90)
    
    # Group by method, aggregate across seeds
    from collections import defaultdict
    method_results = defaultdict(list)
    for key, r in all_results.items():
        method_results[r["method"]].append(r)
    
    print(f"\n  {'Method':<25} {'Value MSE (mean±std)':>25} {'Grad MSE (mean±std)':>25}")
    print(f"  {'-'*25} {'-'*25} {'-'*25}")
    
    for method in METHODS:
        if method in method_results:
            vals = [r["test_value_mse"] for r in method_results[method]]
            grads = [r["test_grad_mse"] for r in method_results[method]]
            print(f"  {method:<25} {np.mean(vals):12.4e} ± {np.std(vals):.1e}  "
                  f"{np.mean(grads):12.4e} ± {np.std(grads):.1e}")
    
    # DML improvement
    if "vanilla" in method_results and "dml_fixed" in method_results:
        van_mean = np.mean([r["test_value_mse"] for r in method_results["vanilla"]])
        dml_mean = np.mean([r["test_value_mse"] for r in method_results["dml_fixed"]])
        improvement = van_mean / dml_mean if dml_mean > 0 else float("inf")
        print(f"\n  DML improvement over vanilla: {improvement:.2f}×")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
