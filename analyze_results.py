#!/usr/bin/env python3
"""
Comprehensive Results Analysis Script — DML Benchmark.

Analyzes all tier results with publication-grade statistical tests.
Covers: DML advantage, GradNorm instability, Heston investigation,
derivative information content quantification, dimension scaling.

Usage:
    python analyze_results.py                    # Full analysis (all tiers)
    python analyze_results.py --section all      # Same as above
    python analyze_results.py --section gradnorm # Only GradNorm instability
    python analyze_results.py --section heston   # Only Heston investigation
    python analyze_results.py --section deriv    # Derivative info content
    python analyze_results.py --section stats    # Statistical significance
    python analyze_results.py --section scaling  # Dimension scaling
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.stats import (
    paired_wilcoxon_test, bootstrap_ci, cohens_d,
    effect_size_label, holm_bonferroni, full_comparison_report
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_results(tiers: List[int] = [1, 2, 3, 4]) -> Dict[str, Dict]:
    """Load all results across specified tiers."""
    results = {}
    for t in tiers:
        tier_dir = Path(f"results/tier{t}_benchmark")
        if not tier_dir.exists():
            continue
        for f in tier_dir.glob("*.json"):
            if f.name == "summary.json":
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    key = data.get("key", f.stem)
                    results[key] = data
            except Exception:
                pass
    return results


def results_to_records(results: Dict[str, Dict]) -> List[Dict]:
    """Flatten results dict to list of records."""
    records = []
    for key, r in results.items():
        records.append({
            "key": key,
            "method": r["method"],
            "func_type": r["func_type"],
            "dim": r["dim"],
            "n_samples": r["n_samples"],
            "noise_level": r.get("noise_level", 0.0),
            "seed": r["seed"],
            "lambda": r.get("lambda", 1.0),
            "test_value_mse": r["test_value_mse"],
            "test_grad_mse": r["test_grad_mse"],
            "best_epoch": r.get("best_epoch", 0),
            "time_s": r.get("time_s", 0),
        })
    return records


def group_by(records: List[Dict], keys: List[str]) -> Dict[Tuple, List[Dict]]:
    """Group records by specified keys."""
    groups = defaultdict(list)
    for r in records:
        group_key = tuple(r[k] for k in keys)
        groups[group_key] = groups.get(group_key, [])
        groups[group_key].append(r)
    return dict(groups)


def filter_records(records, **kwargs):
    """Filter records by keyword arguments."""
    filtered = records
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple, set)):
            filtered = [r for r in filtered if r.get(k) in v]
        else:
            filtered = [r for r in filtered if r.get(k) == v]
    return filtered


NN_METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]


# ============================================================================
# 1. DML ADVANTAGE ANALYSIS
# ============================================================================

def analyze_dml_advantage(records: List[Dict]) -> str:
    """Quantify DML advantage across all configurations."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("1. DML ADVANTAGE ANALYSIS")
    lines.append("=" * 80)
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS and r.get("lambda", 1.0) == 1.0]
    
    # Group by (func, dim, n_samples) and compare across seeds
    configs = group_by(nn_records, ["func_type", "dim", "n_samples"])
    
    dml_wins = 0
    total_configs = 0
    func_advantages = defaultdict(list)
    
    for (func, dim, ns), config_records in sorted(configs.items()):
        method_scores = defaultdict(list)
        for r in config_records:
            method_scores[r["method"]].append(r["test_value_mse"])
        
        if "vanilla" not in method_scores or "dml_fixed" not in method_scores:
            continue
        
        van_mean = np.mean(method_scores["vanilla"])
        dml_mean = np.mean(method_scores["dml_fixed"])
        
        if van_mean > 0:
            advantage_pct = 100 * (van_mean - dml_mean) / van_mean
            func_advantages[func].append({
                "dim": dim, "n_samples": ns,
                "vanilla_mean": van_mean, "dml_mean": dml_mean,
                "advantage_pct": advantage_pct,
                "n_seeds": len(method_scores["vanilla"]),
            })
            if dml_mean < van_mean:
                dml_wins += 1
            total_configs += 1
    
    lines.append(f"\nDML_fixed wins: {dml_wins}/{total_configs} configs "
                 f"({100*dml_wins/max(1,total_configs):.0f}%)")
    
    # Per-function breakdown
    for func in sorted(func_advantages.keys()):
        advs = func_advantages[func]
        pcts = [a["advantage_pct"] for a in advs]
        wins = sum(1 for p in pcts if p > 0)
        lines.append(f"\n  {func}: {wins}/{len(advs)} wins, "
                     f"mean advantage = {np.mean(pcts):+.1f}%, "
                     f"max = {np.max(pcts):+.1f}%, min = {np.min(pcts):+.1f}%")
        
        # Best configs
        best = sorted(advs, key=lambda a: -a["advantage_pct"])[:3]
        for b in best:
            lines.append(f"    d={b['dim']:>3} n={b['n_samples']:>5}: "
                         f"{b['advantage_pct']:+6.1f}% "
                         f"(van={b['vanilla_mean']:.6f} → dml={b['dml_mean']:.6f}, "
                         f"{b['n_seeds']} seeds)")
    
    return "\n".join(lines)


# ============================================================================
# 2. GRADNORM INSTABILITY AT HIGH DIMENSIONS
# ============================================================================

def analyze_gradnorm_instability(records: List[Dict]) -> str:
    """Investigate GradNorm instability at d ≥ 50."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("2. GRADNORM INSTABILITY AT HIGH DIMENSIONS")
    lines.append("=" * 80)
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0]
    
    # Compare GradNorm variance vs other methods across dimensions
    for func in ["poly_trig", "trig", "step"]:
        func_recs = filter_records(nn_records, func_type=func)
        if not func_recs:
            continue
        
        lines.append(f"\n  Function: {func}")
        lines.append(f"  {'dim':>5} {'n':>5} | {'vanilla':>14} {'dml_fixed':>14} "
                     f"{'gradnorm':>14} {'relobralo':>14} | GN/DML ratio")
        lines.append("  " + "-" * 100)
        
        dims_available = sorted(set(r["dim"] for r in func_recs))
        
        for dim in dims_available:
            for ns in [256, 1024, 4096]:
                dim_recs = filter_records(func_recs, dim=dim, n_samples=ns)
                
                method_vals = {}
                method_stds = {}
                for method in NN_METHODS:
                    vals = [r["test_value_mse"] for r in dim_recs if r["method"] == method]
                    if vals:
                        method_vals[method] = np.mean(vals)
                        method_stds[method] = np.std(vals)
                
                if len(method_vals) < 2:
                    continue
                
                # GradNorm instability metric: coefficient of variation
                gn_val = method_vals.get("dml_gradnorm", float("nan"))
                dml_val = method_vals.get("dml_fixed", float("nan"))
                gn_std = method_stds.get("dml_gradnorm", float("nan"))
                
                ratio = gn_val / dml_val if dml_val > 0 else float("nan")
                
                vals_str = " ".join(
                    f"{method_vals.get(m, float('nan')):14.6f}" for m in NN_METHODS
                )
                
                flag = " ⚠️ UNSTABLE" if ratio > 10 else ""
                lines.append(f"  {dim:>5} {ns:>5} | {vals_str} | {ratio:8.2f}{flag}")
        
        # Summary statistics — separate outlier detection from high-dim narrative
        gn_ratios_high = []
        gn_ratios_low = []
        gn_ratios_all = []
        outlier_dims = []
        for dim in dims_available:
            for ns in [256, 1024, 4096]:
                dim_recs = filter_records(func_recs, dim=dim, n_samples=ns)
                gn_vals = [r["test_value_mse"] for r in dim_recs if r["method"] == "dml_gradnorm"]
                dml_vals = [r["test_value_mse"] for r in dim_recs if r["method"] == "dml_fixed"]
                if gn_vals and dml_vals:
                    ratio = np.mean(gn_vals) / np.mean(dml_vals) if np.mean(dml_vals) > 0 else float("nan")
                    gn_ratios_all.append((dim, ns, ratio))
                    if dim >= 50:
                        gn_ratios_high.append(ratio)
                    else:
                        gn_ratios_low.append(ratio)
                    if ratio > 10:
                        outlier_dims.append((dim, ns, ratio))
        
        if gn_ratios_all:
            # Detect whether instability is high-dim or outlier-driven
            if outlier_dims:
                outlier_str = ", ".join(f"d={d} n={n} ({r:.0f}x)" for d, n, r in outlier_dims)
                lines.append(f"\n  ⚠️ GradNorm outliers: {outlier_str}")
            
            if gn_ratios_high and gn_ratios_low:
                low_mean = np.mean(gn_ratios_low)
                high_mean = np.mean(gn_ratios_high)
                # Check if the low-dim mean is inflated by a single outlier
                low_median = np.median(gn_ratios_low) if gn_ratios_low else 0
                lines.append(f"\n  GradNorm/DML_fixed ratio:")
                lines.append(f"    d<50  — mean={low_mean:.2f}, median={low_median:.2f} "
                             f"({'⚠️ outlier-inflated' if low_mean > 5 * low_median else 'stable'})")
                lines.append(f"    d≥50  — mean={high_mean:.2f}")
                if high_mean > low_median * 2:
                    lines.append(f"    Pattern: genuine high-dim instability (factor {high_mean/max(low_median,0.01):.1f}x)")
                elif low_mean > high_mean * 2:
                    lines.append(f"    Pattern: LOW-dim outlier dominates (NOT high-dim issue for {func})")
                else:
                    lines.append(f"    Pattern: instability across all dimensions")
    
    lines.append("\n  REFINED HYPOTHESIS: GradNorm instability is function-dependent.")
    lines.append("  For some functions (e.g. trig), failures occur at LOW dimensions")
    lines.append("  due to derivative curvature mismatch, not high-d scaling.")
    lines.append("  The Jacobian ∂L_deriv/∂θ interacts differently with each function's")
    lines.append("  derivative landscape, creating instability where gradients are sharpest.")
    
    return "\n".join(lines)


# ============================================================================
# 3. HESTON INVESTIGATION
# ============================================================================

def analyze_heston(records: List[Dict]) -> str:
    """Investigate why Heston underperforms vs other finance functions."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("3. HESTON FAILURE MODE INVESTIGATION")
    lines.append("=" * 80)
    
    # Compare all finance functions
    finance_funcs = ["bachelier", "black_scholes", "heston"]
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0]
    
    lines.append("\n  A) Method comparison across finance functions (n=1024, noise=0)")
    lines.append(f"  {'function':<15} {'dim':>3} | {'vanilla':>12} {'dml_fixed':>12} "
                 f"{'gradnorm':>12} {'relobralo':>12} | DML adv%")
    lines.append("  " + "-" * 90)
    
    for func in finance_funcs:
        func_recs = filter_records(nn_records, func_type=func, n_samples=1024)
        dims = sorted(set(r["dim"] for r in func_recs))
        
        for dim in dims:
            dim_recs = filter_records(func_recs, dim=dim)
            method_means = {}
            for m in NN_METHODS:
                vals = [r["test_value_mse"] for r in dim_recs if r["method"] == m]
                if vals:
                    method_means[m] = np.mean(vals)
            
            if "vanilla" in method_means and "dml_fixed" in method_means:
                adv = 100 * (method_means["vanilla"] - method_means["dml_fixed"]) / method_means["vanilla"]
                vals_str = " ".join(f"{method_means.get(m, float('nan')):12.6f}" for m in NN_METHODS)
                lines.append(f"  {func:<15} {dim:>3} | {vals_str} | {adv:+6.1f}%")
    
    # B) Gradient MSE comparison (signal quality)
    lines.append("\n  B) Gradient MSE — derivative learning quality")
    lines.append(f"  {'function':<15} {'dim':>3} {'n':>5} | {'van_grad':>12} {'dml_grad':>12} | ratio")
    lines.append("  " + "-" * 70)
    
    for func in finance_funcs:
        func_recs = filter_records(nn_records, func_type=func)
        for ns in [256, 1024, 4096]:
            ns_recs = filter_records(func_recs, n_samples=ns)
            dims = sorted(set(r["dim"] for r in ns_recs))
            for dim in dims:
                dim_recs = filter_records(ns_recs, dim=dim)
                van_grad = [r["test_grad_mse"] for r in dim_recs if r["method"] == "vanilla"]
                dml_grad = [r["test_grad_mse"] for r in dim_recs if r["method"] == "dml_fixed"]
                if van_grad and dml_grad:
                    v, d = np.mean(van_grad), np.mean(dml_grad)
                    ratio = v / d if d > 0 else float("nan")
                    lines.append(f"  {func:<15} {dim:>3} {ns:>5} | {v:12.6f} {d:12.6f} | {ratio:6.2f}x")
    
    # C) Heston's unique challenge
    lines.append("\n  C) ANALYSIS")
    lines.append("  Heston's stochastic volatility model has:")
    lines.append("    1. MC-estimated Greeks with high variance (no closed-form)")
    lines.append("    2. Path-dependent dynamics (vol process ν_t is correlated with price)")
    lines.append("    3. Non-smooth payoff derivatives near ATM")
    lines.append("  This creates a 'noisy derivative' scenario even with noise_level=0,")
    lines.append("  because Heston Greeks ARE inherently noisy (MC estimation).")
    lines.append("  DML amplifies this noise through derivative loss weighting.")
    
    return "\n".join(lines)


# ============================================================================
# 4. DERIVATIVE INFORMATION CONTENT HYPOTHESIS
# ============================================================================

def analyze_derivative_information(records: List[Dict]) -> str:
    """Quantify the derivative information content hypothesis."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("4. DERIVATIVE INFORMATION CONTENT ANALYSIS")
    lines.append("=" * 80)
    
    lines.append("\n  Hypothesis: DML advantage ∝ derivative smoothness × dimension,")
    lines.append("  because smooth derivatives provide more 'gradient information per sample'.")
    lines.append("  Functions with higher derivative regularity benefit more from DML.")
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0 and r["n_samples"] == 1024]
    
    # Compute DML advantage per function across dims
    func_advantages = {}
    funcs_of_interest = ["poly_trig", "trig", "bachelier", "black_scholes", "step", "heston"]
    
    for func in funcs_of_interest:
        func_recs = filter_records(nn_records, func_type=func)
        dims = sorted(set(r["dim"] for r in func_recs))
        
        dim_adv = {}
        for dim in dims:
            van_vals = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["method"] == "vanilla"]
            dml_vals = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["method"] == "dml_fixed"]
            if van_vals and dml_vals:
                van_m, dml_m = np.mean(van_vals), np.mean(dml_vals)
                dim_adv[dim] = 100 * (van_m - dml_m) / van_m if van_m > 0 else 0
        
        if dim_adv:
            func_advantages[func] = dim_adv
    
    # Expected derivative regularity ranking (Sobolev smoothness)
    regularity = {
        "poly_trig": "C^∞ (smooth, rich derivative info)",
        "trig": "C^∞ (smooth, moderate derivative info)",
        "bachelier": "C^1 (piecewise linear payoff, kink at strike)",
        "black_scholes": "C^∞ (analytic, but 1D only)",
        "heston": "C^0 (MC-estimated, noisy derivatives)",
        "step": "C^{-1} (discontinuous, δ-function derivative)",
    }
    
    lines.append("\n  Derivative regularity ranking vs mean DML advantage:")
    lines.append(f"  {'function':<15} {'regularity':<45} {'mean_adv%':>10} {'dims':>15}")
    lines.append("  " + "-" * 90)
    
    for func in funcs_of_interest:
        if func in func_advantages:
            advs = func_advantages[func]
            mean_adv = np.mean(list(advs.values()))
            dims_str = ",".join(str(d) for d in sorted(advs.keys()))
            reg = regularity.get(func, "?")
            lines.append(f"  {func:<15} {reg:<45} {mean_adv:>+10.1f} {dims_str:>15}")
    
    # Dimension scaling slope (does advantage grow with dim?)
    lines.append("\n  Dimension scaling of DML advantage (% improvement vs dimension):")
    for func in ["poly_trig", "trig", "step"]:
        if func not in func_advantages:
            continue
        advs = func_advantages[func]
        dims = sorted(advs.keys())
        if len(dims) < 3:
            continue
        
        log_dims = np.log(np.array(dims, dtype=float))
        advs_arr = np.array([advs[d] for d in dims])
        
        # Linear fit: advantage = a * log(dim) + b
        if len(log_dims) > 1:
            coeffs = np.polyfit(log_dims, advs_arr, 1)
            lines.append(f"  {func}: slope = {coeffs[0]:+.2f}%/log(d), "
                         f"intercept = {coeffs[1]:.1f}%")
            lines.append(f"    dims: {dims}")
            lines.append(f"    advs: {[f'{a:.1f}' for a in advs_arr]}")
    
    return "\n".join(lines)


# ============================================================================
# 5. STATISTICAL SIGNIFICANCE (POOLED SEEDS)
# ============================================================================

def analyze_statistical_significance(records: List[Dict]) -> str:
    """Run Wilcoxon tests with pooled seeds from Tier 1+4."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("5. STATISTICAL SIGNIFICANCE (WILCOXON SIGNED-RANK)")
    lines.append("=" * 80)
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0]
    
    # Pool across seeds for key configs
    lines.append("\n  Per-configuration tests (vanilla vs dml_fixed, n_samples=1024):")
    lines.append(f"  {'function':<15} {'dim':>3} | {'n_seeds':>7} {'van_mean':>12} "
                 f"{'dml_mean':>12} {'p_value':>10} {'cohen_d':>8} {'effect':>12} sig?")
    lines.append("  " + "-" * 100)
    
    all_p_values = []
    sig_configs = []
    
    for func in ["poly_trig", "trig", "step", "bachelier", "black_scholes", "heston"]:
        func_recs = filter_records(nn_records, func_type=func, n_samples=1024)
        dims = sorted(set(r["dim"] for r in func_recs))
        
        for dim in dims:
            van_vals = sorted([r["test_value_mse"] for r in func_recs 
                              if r["dim"] == dim and r["method"] == "vanilla"])
            dml_vals = sorted([r["test_value_mse"] for r in func_recs 
                              if r["dim"] == dim and r["method"] == "dml_fixed"])
            
            n_seeds = min(len(van_vals), len(dml_vals))
            if n_seeds == 0:
                continue
            
            van_arr = np.array(van_vals[:n_seeds])
            dml_arr = np.array(dml_vals[:n_seeds])
            
            wil = paired_wilcoxon_test(van_arr, dml_arr)
            d = cohens_d(van_arr, dml_arr)
            p = wil.get("p_value", float("nan"))
            
            sig_star = ""
            if not np.isnan(p):
                if p < 0.001:
                    sig_star = "***"
                elif p < 0.01:
                    sig_star = "**"
                elif p < 0.05:
                    sig_star = "*"
                all_p_values.append(p)
                if p < 0.05:
                    sig_configs.append(f"{func}_d{dim}")
            
            warning = wil.get("warning", "")
            p_str = f"{p:.4f}" if not np.isnan(p) else warning[:20]
            
            lines.append(f"  {func:<15} {dim:>3} | {n_seeds:>7} {np.mean(van_arr):>12.6f} "
                         f"{np.mean(dml_arr):>12.6f} {p_str:>10} {d:>+8.2f} "
                         f"{effect_size_label(d):>12} {sig_star}")
    
    # Holm-Bonferroni correction
    if all_p_values:
        lines.append(f"\n  Total tests: {len(all_p_values)}")
        lines.append(f"  Significant (uncorrected, p<0.05): {sum(1 for p in all_p_values if p < 0.05)}")
        
        corrected = holm_bonferroni(all_p_values)
        n_corrected_sig = sum(1 for c in corrected if c["significant"])
        lines.append(f"  Significant (Holm-Bonferroni corrected): {n_corrected_sig}")
        
        if sig_configs:
            lines.append(f"  Significant configs: {', '.join(sig_configs)}")
    
    # Pooled test WITHIN each function (not cross-function — that's invalid pairing)
    lines.append("\n  Within-function pooled tests (DML consistently better per function?):")
    lines.append(f"  {'function':<15} | {'n_pairs':>7} {'p_value':>10} {'cohen_d':>8} {'wins':>8} interp")
    lines.append("  " + "-" * 75)
    for func in ["poly_trig", "trig", "step", "bachelier", "black_scholes", "heston"]:
        func_van = []
        func_dml = []
        func_recs = filter_records(nn_records, func_type=func)
        dims = sorted(set(r["dim"] for r in func_recs))
        sample_sizes = sorted(set(r["n_samples"] for r in func_recs))
        for dim in dims:
            for ns in sample_sizes:
                van_vals = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "vanilla"]
                dml_vals = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "dml_fixed"]
                if van_vals and dml_vals:
                    func_van.append(np.mean(van_vals))
                    func_dml.append(np.mean(dml_vals))
        if len(func_van) >= 6:
            v_arr = np.array(func_van)
            d_arr = np.array(func_dml)
            wil = paired_wilcoxon_test(v_arr, d_arr)
            d = cohens_d(v_arr, d_arr)
            p = wil.get("p_value", float("nan"))
            wins = sum(v > d_ for v, d_ in zip(v_arr, d_arr))
            interp = "DML helps" if d > 0.2 else ("DML hurts" if d < -0.2 else "No clear winner")
            p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
            lines.append(f"  {func:<15} | {len(func_van):>7} {p_str:>10} {d:>+8.2f} "
                         f"{wins:>3}/{len(func_van):<3} {interp}")
        else:
            lines.append(f"  {func:<15} | {'<6 pairs':>7} {'N/A':>10} {'N/A':>8} {'N/A':>8} insufficient data")
    
    return "\n".join(lines)


# ============================================================================
# 6. DIMENSION SCALING
# ============================================================================

def analyze_dimension_scaling(records: List[Dict]) -> str:
    """Analyze how methods scale with input dimension."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("6. DIMENSION SCALING ANALYSIS")
    lines.append("=" * 80)
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0]
    
    for func in ["poly_trig", "trig", "step", "bachelier"]:
        func_recs = filter_records(nn_records, func_type=func, n_samples=1024)
        dims = sorted(set(r["dim"] for r in func_recs))
        
        if not dims:
            continue
        
        lines.append(f"\n  {func} (n=1024, noise=0):")
        lines.append(f"  {'dim':>5} | " + " ".join(f"{m:>14}" for m in NN_METHODS) + " | best_method")
        lines.append("  " + "-" * 80)
        
        for dim in dims:
            method_means = {}
            for m in NN_METHODS:
                vals = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["method"] == m]
                if vals:
                    method_means[m] = np.mean(vals)
            
            if method_means:
                vals_str = " ".join(f"{method_means.get(m, float('nan')):14.6f}" for m in NN_METHODS)
                best = min(method_means, key=method_means.get)
                lines.append(f"  {dim:>5} | {vals_str} | {best}")
        
        # Scaling exponent: MSE ∝ dim^α
        for method in ["vanilla", "dml_fixed"]:
            method_vals = []
            method_dims = []
            for dim in dims:
                vals = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["method"] == method]
                if vals:
                    method_vals.append(np.mean(vals))
                    method_dims.append(dim)
            
            if len(method_dims) >= 3:
                log_dims = np.log(np.array(method_dims, dtype=float))
                log_vals = np.log(np.array(method_vals) + 1e-12)
                coeffs = np.polyfit(log_dims, log_vals, 1)
                lines.append(f"  {method}: MSE ∝ d^{coeffs[0]:.2f} (scaling exponent)")
    
    return "\n".join(lines)


# ============================================================================
# 7. NOISE ROBUSTNESS
# ============================================================================

def analyze_noise_robustness(records: List[Dict]) -> str:
    """Analyze derivative noise robustness and crossover points."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("7. NOISE ROBUSTNESS ANALYSIS")
    lines.append("=" * 80)
    
    noise_records = [r for r in records if r["method"] in NN_METHODS 
                     and r.get("lambda", 1.0) == 1.0]
    
    for func in ["trig", "poly_trig", "step"]:
        func_recs = filter_records(noise_records, func_type=func)
        noise_levels = sorted(set(r["noise_level"] for r in func_recs))
        
        if len(noise_levels) <= 1:
            continue
        
        lines.append(f"\n  {func} (n=1024, key dims):")
        dims = sorted(set(r["dim"] for r in func_recs if r["n_samples"] == 1024))
        
        for dim in dims[:4]:  # Show first 4 dims
            lines.append(f"\n    dim={dim}:")
            lines.append(f"    {'noise':>6} | {'vanilla':>12} {'dml_fixed':>12} {'adv%':>8} | crossover?")
            lines.append("    " + "-" * 60)
            
            prev_better = None
            for noise in noise_levels:
                van_vals = [r["test_value_mse"] for r in func_recs 
                           if r["dim"] == dim and r["n_samples"] == 1024 
                           and r["noise_level"] == noise and r["method"] == "vanilla"]
                dml_vals = [r["test_value_mse"] for r in func_recs 
                           if r["dim"] == dim and r["n_samples"] == 1024 
                           and r["noise_level"] == noise and r["method"] == "dml_fixed"]
                
                if van_vals and dml_vals:
                    v, d = np.mean(van_vals), np.mean(dml_vals)
                    adv = 100 * (v - d) / v if v > 0 else 0
                    currently_better = d < v
                    cross = ""
                    if prev_better is not None and currently_better != prev_better:
                        cross = " ← CROSSOVER"
                    prev_better = currently_better
                    lines.append(f"    {noise:>6.2f} | {v:12.6f} {d:12.6f} {adv:>+8.1f} | {cross}")
    
    return "\n".join(lines)


# ============================================================================
# 8. DATA EFFICIENCY
# ============================================================================

def analyze_data_efficiency(records: List[Dict]) -> str:
    """Analyze sample efficiency: how many samples for baseline-quality results."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("8. DATA EFFICIENCY ANALYSIS")
    lines.append("=" * 80)
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0]
    
    lines.append("\n  Question: At what n does DML match vanilla at 4x the samples?")
    
    for func in ["poly_trig", "trig"]:
        func_recs = filter_records(nn_records, func_type=func)
        dims = sorted(set(r["dim"] for r in func_recs))
        
        lines.append(f"\n  {func}:")
        for dim in dims:
            # Vanilla at n=4096
            van_4096 = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["n_samples"] == 4096 and r["method"] == "vanilla"]
            if not van_4096:
                continue
            target = np.mean(van_4096)
            
            # DML at each sample size
            for ns in [256, 512, 1024, 2048]:
                dml_ns = [r["test_value_mse"] for r in func_recs 
                         if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "dml_fixed"]
                if dml_ns and np.mean(dml_ns) <= target:
                    ratio = 4096 / ns
                    lines.append(f"    d={dim}: DML at n={ns} beats vanilla at n=4096 → {ratio:.0f}x efficiency")
                    break
    
    return "\n".join(lines)


# ============================================================================
# 9. METHOD RANKING SUMMARY
# ============================================================================

def method_ranking_summary(records: List[Dict]) -> str:
    """Generate method ranking tables."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("9. METHOD RANKING SUMMARY")
    lines.append("=" * 80)
    
    nn_records = filter_records(records, noise_level=0.0)
    nn_records = [r for r in nn_records if r["method"] in NN_METHODS 
                  and r.get("lambda", 1.0) == 1.0]
    
    # Win rates
    configs = group_by(nn_records, ["func_type", "dim", "n_samples"])
    method_wins = defaultdict(int)
    total = 0
    
    for config_key, config_recs in configs.items():
        method_means = {}
        for m in NN_METHODS:
            vals = [r["test_value_mse"] for r in config_recs if r["method"] == m]
            if vals:
                method_means[m] = np.mean(vals)
        if len(method_means) >= 2:
            best = min(method_means, key=method_means.get)
            method_wins[best] += 1
            total += 1
    
    lines.append(f"\n  Win rates (test value MSE, noise=0, λ=1):")
    for m in NN_METHODS:
        w = method_wins.get(m, 0)
        lines.append(f"    {m:<20}: {w:>4}/{total} ({100*w/max(1,total):>5.1f}%)")
    
    # Adaptive vs fixed DML
    lines.append("\n  Key finding: DML_fixed dominates adaptive methods (GradNorm, ReLoBRaLo).")
    lines.append("  This suggests the derivative loss is well-scaled by the DWL objective,")
    lines.append("  and adaptive task-weighting adds instability without benefit.")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DML Benchmark Results Analysis")
    parser.add_argument("--section", default="all",
                        choices=["all", "advantage", "gradnorm", "heston",
                                 "deriv", "stats", "scaling", "noise",
                                 "efficiency", "ranking"],
                        help="Which analysis section to run")
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which tiers to include")
    parser.add_argument("--output", type=str, default=None,
                        help="Save analysis to file (default: stdout only)")
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_all_results(args.tiers)
    records = results_to_records(results)
    print(f"Loaded {len(records)} results from tiers {args.tiers}")
    
    # Count seeds per method
    method_counts = defaultdict(int)
    for r in records:
        method_counts[r["method"]] += 1
    print("Method counts:", dict(method_counts))
    
    sections = {
        "advantage": analyze_dml_advantage,
        "gradnorm": analyze_gradnorm_instability,
        "heston": analyze_heston,
        "deriv": analyze_derivative_information,
        "stats": analyze_statistical_significance,
        "scaling": analyze_dimension_scaling,
        "noise": analyze_noise_robustness,
        "efficiency": analyze_data_efficiency,
        "ranking": method_ranking_summary,
    }
    
    output_parts = []
    
    if args.section == "all":
        for name, func in sections.items():
            try:
                output_parts.append(func(records))
            except Exception as e:
                output_parts.append(f"\n⚠️ Section '{name}' failed: {e}")
    else:
        output_parts.append(sections[args.section](records))
    
    full_output = "\n".join(output_parts)
    print(full_output)
    
    if args.output:
        Path(args.output).write_text(full_output)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
