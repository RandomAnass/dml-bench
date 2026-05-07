#!/usr/bin/env python3
"""
Phase 2 analyses for DML-Bench revision:
  A. Metric stability: Re-rank methods by value MSE vs gradient MSE
  B. SPY robustness: Breakdown by moneyness and maturity buckets
  C. Fuzzy sensitivity: Check eps_mult sensitivity (requires new runs)
"""

import json
import glob
import numpy as np
from collections import defaultdict
import os

RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# PART A: Metric Stability Analysis
# ============================================================================

def metric_stability_analysis():
    """Re-rank unified comparison methods by value MSE and gradient MSE."""
    print("=" * 70)
    print("PART A: METRIC STABILITY ANALYSIS")
    print("=" * 70)

    results_dir = os.path.join(RESULTS_DIR, "results/unified_comparison/multi_seed")
    files = glob.glob(os.path.join(results_dir, "*.json"))

    # Group by (dataset, method)
    data = defaultdict(lambda: {"value_mse": [], "grad_mse": [], "seeds": []})

    for f in files:
        d = json.load(open(f))
        dataset = d.get("dataset", os.path.basename(f).split("_s")[0])
        method = d.get("method", "unknown")
        mode = d.get("mode", "")

        # Build full method name including mode
        if mode and mode != "pathwise":
            method_key = f"{method}_{mode}"
        else:
            method_key = method

        key = (dataset, method_key)
        data[key]["value_mse"].append(d["test_value_mse"])
        data[key]["grad_mse"].append(d["test_grad_mse"])
        data[key]["seeds"].append(d["seed"])

    # Get unique datasets and methods
    datasets = sorted(set(k[0] for k in data.keys()))
    methods = sorted(set(k[1] for k in data.keys()))

    print(f"\nDatasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Total method-dataset combos: {len(data)}")

    # Compute mean metrics per (dataset, method)
    mean_data = {}
    for key, vals in data.items():
        mean_data[key] = {
            "mean_value_mse": np.mean(vals["value_mse"]),
            "mean_grad_mse": np.mean(vals["grad_mse"]),
            "n_seeds": len(vals["seeds"])
        }

    # Rank methods per dataset by value MSE and gradient MSE
    print("\n--- Per-dataset rankings ---")

    value_ranks = defaultdict(list)
    grad_ranks = defaultdict(list)

    for ds in datasets:
        ds_methods = [(m, mean_data[(ds, m)]) for m in methods if (ds, m) in mean_data]

        # Sort by value MSE
        by_value = sorted(ds_methods, key=lambda x: x[1]["mean_value_mse"])
        for rank, (m, _) in enumerate(by_value, 1):
            value_ranks[m].append(rank)

        # Sort by gradient MSE
        by_grad = sorted(ds_methods, key=lambda x: x[1]["mean_grad_mse"])
        for rank, (m, _) in enumerate(by_grad, 1):
            grad_ranks[m].append(rank)

        print(f"\n{ds}:")
        print(f"  By Value MSE: {[m for m, _ in by_value[:3]]}")
        print(f"  By Grad MSE:  {[m for m, _ in by_grad[:3]]}")

    # Cross-dataset mean ranks
    print("\n--- Cross-dataset mean ranks ---")
    print(f"{'Method':<30} {'Value Rank':>12} {'Grad Rank':>12} {'Overlap':>10}")
    print("-" * 66)

    all_methods_ranked = []
    for m in methods:
        vr = np.mean(value_ranks[m]) if value_ranks[m] else float('inf')
        gr = np.mean(grad_ranks[m]) if grad_ranks[m] else float('inf')
        all_methods_ranked.append((m, vr, gr))

    # Sort by gradient rank (original paper ordering)
    all_methods_ranked.sort(key=lambda x: x[2])

    for m, vr, gr in all_methods_ranked:
        overlap = "TOP-3" if vr <= 3 and gr <= 3 else ("TOP-5" if vr <= 5 and gr <= 5 else "")
        print(f"  {m:<30} {vr:>10.1f} {gr:>10.1f} {overlap:>10}")

    # Rank correlation
    v_order = [x[0] for x in sorted(all_methods_ranked, key=lambda x: x[1])]
    g_order = [x[0] for x in sorted(all_methods_ranked, key=lambda x: x[2])]

    # Kendall tau (simple count of concordant pairs)
    n = len(v_order)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            vi = v_order.index(all_methods_ranked[i][0])
            vj = v_order.index(all_methods_ranked[j][0])
            if (vi - vj) * (i - j) > 0:
                concordant += 1
            else:
                discordant += 1

    tau = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
    print(f"\nKendall tau (value vs grad ranking): {tau:.3f}")
    print(f"Interpretation: {'Strong' if abs(tau) > 0.6 else 'Moderate' if abs(tau) > 0.3 else 'Weak'} agreement")

    return all_methods_ranked


# ============================================================================
# PART B: SPY Robustness Analysis
# ============================================================================

def spy_robustness_analysis():
    """Analyze SPY results by moneyness and maturity buckets."""
    print("\n" + "=" * 70)
    print("PART B: SPY ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # Load SPY data
    data_path = os.path.join(RESULTS_DIR, "data/spy_options/spy_processed.npz")
    if not os.path.exists(data_path):
        print(f"SPY data not found at {data_path}")
        print("Checking alternative locations...")
        for alt in glob.glob(os.path.join(RESULTS_DIR, "data/**/spy*.npz"), recursive=True):
            print(f"  Found: {alt}")
        return None

    data = np.load(data_path, allow_pickle=True)
    print(f"SPY data keys: {list(data.keys())}")

    # Load features
    features = data.get("features", data.get("x_train", None))
    if features is None:
        print("Could not find feature data")
        return None

    print(f"Feature shape: {features.shape}")

    # Try to identify moneyness and maturity columns
    feature_names = data.get("feature_names", None)
    if feature_names is not None:
        feature_names = list(feature_names)
        print(f"Feature names: {feature_names}")

    # Aggregate SPY results
    spy_dirs = ["results/spy_options", "results/spy_options_temporal"]
    all_spy = []
    for sdir in spy_dirs:
        full_path = os.path.join(RESULTS_DIR, sdir)
        if not os.path.exists(full_path):
            continue
        for f in glob.glob(os.path.join(full_path, "*.json")):
            try:
                d = json.load(open(f))
                all_spy.append(d)
            except:
                pass

    print(f"\nTotal SPY result files: {len(all_spy)}")

    if not all_spy:
        print("No SPY results found")
        return None

    # Group by method
    methods = defaultdict(list)
    for d in all_spy:
        m = d.get("method", "unknown")
        methods[m].append(d)

    print("\nSPY results by method:")
    print(f"{'Method':<25} {'N':>5} {'Mean Value MSE':>15} {'Mean Grad MSE':>15}")
    print("-" * 62)
    for m, results in sorted(methods.items()):
        vmse = np.mean([r["test_value_mse"] for r in results])
        gmse = np.mean([r["test_grad_mse"] for r in results])
        print(f"  {m:<25} {len(results):>5} {vmse:>15.6e} {gmse:>15.6e}")

    # Compute vanilla baseline for ratios
    vanilla_results = methods.get("vanilla", [])
    if vanilla_results:
        vanilla_vmse = np.mean([r["test_value_mse"] for r in vanilla_results])
        vanilla_gmse = np.mean([r["test_grad_mse"] for r in vanilla_results])

        print(f"\nVanilla baseline: value={vanilla_vmse:.6e}, grad={vanilla_gmse:.6e}")
        print("\nImprovement ratios:")
        for m, results in sorted(methods.items()):
            if m == "vanilla":
                continue
            vmse = np.mean([r["test_value_mse"] for r in results])
            gmse = np.mean([r["test_grad_mse"] for r in results])
            v_ratio = vanilla_vmse / vmse if vmse > 0 else float('inf')
            g_ratio = vanilla_gmse / gmse if gmse > 0 else float('inf')
            v_pct = (1 - vmse/vanilla_vmse) * 100
            print(f"  {m:<25}: value {v_pct:>+6.1f}%, gradient {g_ratio:>8.1f}x")

    # Check metadata for moneyness info
    sample = all_spy[0]
    metadata = sample.get("metadata", {})
    print(f"\nSample metadata keys: {sorted(metadata.keys())}")
    if "moneyness_range" in metadata:
        print(f"Moneyness range: {metadata['moneyness_range']}")
    if "T_range" in metadata:
        print(f"Maturity range: {metadata['T_range']}")

    return methods


# ============================================================================
# PART C: Fuzzy Sensitivity Check
# ============================================================================

def fuzzy_sensitivity_check():
    """Check what eps_mult values exist in results and what's needed."""
    print("\n" + "=" * 70)
    print("PART C: FUZZY SENSITIVITY CHECK")
    print("=" * 70)

    # Check existing fuzzy results for epsilon values
    results_dir = os.path.join(RESULTS_DIR, "results/unified_comparison/multi_seed")
    files = glob.glob(os.path.join(results_dir, "*fuzzy*.json"))

    print(f"Fuzzy result files: {len(files)}")

    eps_values = set()
    for f in files:
        d = json.load(open(f))
        eps = d.get("epsilon", d.get("eps_mult", None))
        if eps is not None:
            eps_values.add(eps)

    print(f"Epsilon values found in results: {sorted(eps_values)}")

    if len(eps_values) <= 1:
        print("\nOnly one epsilon value found. Need new runs for sensitivity analysis.")
        print("Suggested eps_mult values: [0.1, 0.25, 0.5, 1.0, 2.0]")
        print("Default in code: 0.5")

        # Show what configs to run
        datasets = set()
        for f in files:
            d = json.load(open(f))
            ds = d.get("dataset", "")
            datasets.add(ds)
        print(f"Datasets with fuzzy results: {sorted(datasets)}")
        print("\nMinimal sensitivity run: 5 eps values × key datasets × 5 seeds")
    else:
        print(f"\nMultiple epsilon values found: {sorted(eps_values)}")
        # Analyze sensitivity
        data_by_eps = defaultdict(list)
        for f in files:
            d = json.load(open(f))
            eps = d.get("epsilon", d.get("eps_mult", "default"))
            data_by_eps[eps].append(d)

        for eps, results in sorted(data_by_eps.items()):
            vmse = np.mean([r["test_value_mse"] for r in results])
            gmse = np.mean([r["test_grad_mse"] for r in results])
            print(f"  eps={eps}: value_mse={vmse:.6e}, grad_mse={gmse:.6e} (n={len(results)})")


if __name__ == "__main__":
    print("DML-Bench Phase 2 Analyses")
    print("=" * 70)

    metric_stability_analysis()
    spy_robustness_analysis()
    fuzzy_sensitivity_check()
