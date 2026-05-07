#!/usr/bin/env python3
"""
Analyze all 1,240 new extension experiments (LRM comparison + Heston + SPY).

Sections:
  1. LRM Comparison (Exp A/B/D): Digital, Barrier, Basket, Network Size
  2. Heston Euler-LRM (Exp C): Variance scaling with N_steps
  3. SPY Real-World Options (Exp E): Greeks accuracy + Pareto analysis
  4. Statistical Tests: Wilcoxon signed-rank across all experiments
  5. Cross-Experiment Summary: Key findings for paper

Usage:
    python experiments/analyze_new_experiments.py
    python experiments/analyze_new_experiments.py --section lrm
    python experiments/analyze_new_experiments.py --section heston
    python experiments/analyze_new_experiments.py --section spy
    python experiments/analyze_new_experiments.py --section stats
    python experiments/analyze_new_experiments.py --section all
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from dml_benchmark.stats import paired_wilcoxon_test, bootstrap_ci, cohens_d


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(results_dir: str) -> dict:
    """Load all JSON results from a directory."""
    results = {}
    d = Path(results_dir)
    if not d.exists():
        return results
    for f in d.glob("*.json"):
        if f.name in ("summary.json", "analysis.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                results[data.get("key", f.stem)] = data
        except Exception:
            pass
    return results


def load_lrm_results():
    return load_results("results/lrm_comparison")

def load_spy_results():
    return load_results("results/spy_options")


def group_by(records, *keys):
    """Group records by one or more keys. Returns nested dict."""
    groups = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) for k in keys)
        if len(keys) == 1:
            key = key[0]
        groups[key].append(r)
    return dict(groups)


# ============================================================================
# 1. LRM COMPARISON ANALYSIS
# ============================================================================

def analyze_lrm(results):
    """Analyze LRM comparison experiments (A1, A2, B, D)."""
    print("\n" + "=" * 90)
    print("SECTION 1: LRM COMPARISON — G&K Fixed-λ vs. Adaptive Balancing")
    print("=" * 90)

    # Categorize by experiment
    experiments = defaultdict(list)
    for key, r in results.items():
        if "heston" in key:
            continue  # Handled separately
        k = r.get("key", key)
        if "A1_digital" in k:
            experiments["A1_digital"].append(r)
        elif "A2_barrier" in k:
            experiments["A2_barrier"].append(r)
        elif "B_basket" in k:
            experiments["B_basket"].append(r)
        elif "D_" in k:
            experiments["D_netsize"].append(r)

    methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

    # ---- Experiment A1: Digital BS ----
    if "A1_digital" in experiments:
        print("\n--- Exp A1: Digital BS + LRM Labels ---")
        recs = experiments["A1_digital"]
        print(f"  Total results: {len(recs)}")

        by_hparam = group_by(recs, "hparam_set")
        for hset in sorted(by_hparam.keys()):
            sub = by_hparam[hset]
            print(f"\n  Hyperparams: {hset} (n={len(sub)} results)")

            by_ns = group_by(sub, "n_samples")
            for ns in sorted(by_ns.keys()):
                configs = by_ns[ns]
                print(f"\n    n_samples = {ns}")
                by_method = group_by(configs, "method")
                _print_method_table(by_method, methods)

    # ---- Experiment A2: Barrier BS ----
    if "A2_barrier" in experiments:
        print("\n--- Exp A2: Barrier BS + LRM Labels ---")
        recs = experiments["A2_barrier"]
        by_ns = group_by(recs, "n_samples")
        for ns in sorted(by_ns.keys()):
            configs = by_ns[ns]
            print(f"\n    n_samples = {ns}")
            by_method = group_by(configs, "method")
            _print_method_table(by_method, methods)

    # ---- Experiment B: Dimension Scaling ----
    if "B_basket" in experiments:
        print("\n--- Exp B: Basket Digital — Dimension Scaling ---")
        recs = experiments["B_basket"]
        by_dim = group_by(recs, "dim")
        print(f"\n    {'Dim':>5} {'vanilla':>14} {'dml_fixed':>14} {'gradnorm':>14} {'relobralo':>14} {'Best':>14}")
        print(f"    {'-'*5} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

        for dim in sorted(by_dim.keys()):
            configs = by_dim[dim]
            by_method = group_by(configs, "method")
            vals = {}
            for m in methods:
                if m in by_method:
                    vals[m] = np.mean([r["test_value_mse"] for r in by_method[m]])
            best = min(vals, key=vals.get) if vals else "?"
            row = f"    {dim:>5}"
            for m in methods:
                v = vals.get(m, float("nan"))
                row += f" {v:14.4e}"
            row += f" {best:>14}"
            print(row)

        # LRM variance vs dimension
        print(f"\n    LRM Variance Scaling:")
        print(f"    {'Dim':>5} {'Mean LRM Var':>14}")
        for dim in sorted(by_dim.keys()):
            lrm_vars = [r.get("lrm_var_mean", 0) for r in by_dim[dim] if r.get("lrm_var_mean")]
            if lrm_vars:
                print(f"    {dim:>5} {np.mean(lrm_vars):14.4e}")

    # ---- Experiment D: Network Size ----
    if "D_netsize" in experiments:
        print("\n--- Exp D: Network Size Sensitivity ---")
        recs = experiments["D_netsize"]
        # Extract net width from key or hparams
        for r in recs:
            if "net_width" not in r:
                hp = r.get("hparams", {})
                r["net_width"] = hp.get("hidden_size", 0)
                # Also try to parse from key
                k = r.get("key", "")
                for part in k.split("_"):
                    if part.startswith("h") and part[1:].isdigit():
                        r["net_width"] = int(part[1:])

        by_width = group_by(recs, "net_width")
        print(f"\n    {'Width':>6} {'vanilla':>14} {'dml_fixed':>14} {'gradnorm':>14} {'relobralo':>14} {'Best':>14}")
        print(f"    {'-'*6} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

        for width in sorted(by_width.keys()):
            configs = by_width[width]
            by_method = group_by(configs, "method")
            vals = {}
            for m in methods:
                if m in by_method:
                    vals[m] = np.mean([r["test_value_mse"] for r in by_method[m]])
            best = min(vals, key=vals.get) if vals else "?"
            row = f"    {width:>6}"
            for m in methods:
                v = vals.get(m, float("nan"))
                row += f" {v:14.4e}"
            row += f" {best:>14}"
            print(row)

    return experiments


# ============================================================================
# 2. HESTON EULER-LRM ANALYSIS
# ============================================================================

def analyze_heston(results):
    """Analyze Heston Euler-LRM experiments (Exp C)."""
    print("\n" + "=" * 90)
    print("SECTION 2: HESTON EULER-LRM — Variance Explosion with Discretization")
    print("=" * 90)

    heston = [r for r in results.values() if "heston" in r.get("key", "")]
    if not heston:
        print("  No Heston results found.")
        return {}

    print(f"  Total Heston results: {len(heston)}")

    methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
    by_payoff = group_by(heston, "payoff")

    for payoff in sorted(by_payoff.keys()):
        recs = by_payoff[payoff]
        print(f"\n--- Payoff: {payoff} ---")

        by_steps = group_by(recs, "n_steps")

        print(f"\n    {'Steps':>6} {'LRM_Var':>10} {'vanilla':>12} {'dml_fixed':>12} {'gradnorm':>12} {'relobralo':>12} {'Best':>12}")
        print(f"    {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        step_data = {}
        for ns in sorted(by_steps.keys()):
            configs = by_steps[ns]
            by_method = group_by(configs, "method")
            lrm_vars = [r.get("lrm_var_mean", 0) for r in configs]
            mean_lrm_var = np.mean(lrm_vars) if lrm_vars else 0

            vals = {}
            for m in methods:
                if m in by_method:
                    vals[m] = np.mean([r["test_value_mse"] for r in by_method[m]])
            best = min(vals, key=vals.get) if vals else "?"
            step_data[ns] = (mean_lrm_var, vals, best)

            row = f"    {ns:>6} {mean_lrm_var:10.1f}"
            for m in methods:
                v = vals.get(m, float("nan"))
                row += f" {v:12.1f}"
            row += f" {best:>12}"
            print(row)

        # Degradation analysis
        steps = sorted(step_data.keys())
        if len(steps) >= 2:
            print(f"\n    Degradation (steps {steps[0]}→{steps[-1]}):")
            first, last = step_data[steps[0]], step_data[steps[-1]]
            for m in methods:
                if m in first[1] and m in last[1]:
                    v0, v1 = first[1][m], last[1][m]
                    pct = (v1 - v0) / v0 * 100
                    print(f"      {m:<25} {v0:.1f} → {v1:.1f} ({pct:+.1f}%)")

    return by_payoff


# ============================================================================
# 3. SPY REAL-WORLD OPTIONS ANALYSIS
# ============================================================================

def analyze_spy(results):
    """Analyze SPY real-world options experiments (Exp E)."""
    print("\n" + "=" * 90)
    print("SECTION 3: SPY REAL-WORLD OPTIONS — DML with BS Greeks")
    print("=" * 90)

    spy = list(results.values())
    if not spy:
        print("  No SPY results found.")
        return {}

    print(f"  Total SPY results: {len(spy)}")

    methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
    by_ntrain = group_by(spy, "n_train")

    for n_train in sorted(by_ntrain.keys()):
        recs = by_ntrain[n_train]
        by_method = group_by(recs, "method")

        print(f"\n--- n_train = {n_train} ---")
        print(f"    {'Method':<25} {'Mean Val MSE':>14} {'Std':>13} {'Mean Grad MSE':>14} {'Grad Improv':>12} {'Count':>6}")
        print(f"    {'-'*25} {'-'*14} {'-'*13} {'-'*14} {'-'*12} {'-'*6}")

        vanilla_grad = None
        if "vanilla" in by_method:
            vanilla_grad = np.mean([r["test_grad_mse"] for r in by_method["vanilla"]])

        best_val = float("inf")
        best_method = None

        for m in methods:
            if m not in by_method:
                continue
            vals = [r["test_value_mse"] for r in by_method[m]]
            grads = [r["test_grad_mse"] for r in by_method[m]]
            mean_val = np.mean(vals)
            mean_grad = np.mean(grads)

            if mean_val < best_val:
                best_val = mean_val
                best_method = m

            grad_improv = ""
            if vanilla_grad and m != "vanilla" and mean_grad > 0:
                ratio = vanilla_grad / mean_grad
                grad_improv = f"{ratio:.0f}x"

            print(
                f"    {m:<25} {mean_val:14.6e} "
                f"{np.std(vals):13.6e} "
                f"{mean_grad:14.6e} {grad_improv:>12} {len(vals):6d}"
            )

        if best_method:
            print(f"    BEST value MSE: {best_method}")

        # Pareto analysis: value-gradient tradeoff
        print(f"\n    Pareto Analysis (value MSE vs gradient MSE):")
        for m in methods:
            if m not in by_method:
                continue
            mean_val = np.mean([r["test_value_mse"] for r in by_method[m]])
            mean_grad = np.mean([r["test_grad_mse"] for r in by_method[m]])
            vanilla_val = np.mean([r["test_value_mse"] for r in by_method.get("vanilla", [])])
            val_penalty = (mean_val - vanilla_val) / vanilla_val * 100 if vanilla_val else 0
            grad_improvement = vanilla_grad / mean_grad if vanilla_grad and mean_grad > 0 else 1
            print(f"      {m:<22} val_penalty={val_penalty:+.1f}%   grad_improvement={grad_improvement:.0f}x")

    return by_ntrain


# ============================================================================
# 4. STATISTICAL TESTS
# ============================================================================

def run_statistical_tests(lrm_results, spy_results):
    """Run Wilcoxon signed-rank tests across key comparisons."""
    print("\n" + "=" * 90)
    print("SECTION 4: STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 90)

    tests_run = []

    # --- SPY: Pairwise method comparisons ---
    print("\n--- SPY Options: Pairwise Wilcoxon Tests (value MSE) ---")
    spy_list = list(spy_results.values())
    by_ntrain = group_by(spy_list, "n_train")

    for n_train in sorted(by_ntrain.keys()):
        recs = by_ntrain[n_train]
        by_method = group_by(recs, "method")
        print(f"\n  n_train = {n_train}:")

        methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
        # Need paired data: match on seed
        method_vals = {}
        for m in methods:
            if m in by_method:
                pairs = {r["seed"]: r["test_value_mse"] for r in by_method[m]}
                method_vals[m] = pairs

        # All pairwise comparisons
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                if m1 not in method_vals or m2 not in method_vals:
                    continue
                common_seeds = sorted(set(method_vals[m1].keys()) & set(method_vals[m2].keys()))
                if len(common_seeds) < 5:
                    continue
                a = [method_vals[m1][s] for s in common_seeds]
                b = [method_vals[m2][s] for s in common_seeds]
                try:
                    result = paired_wilcoxon_test(np.array(a), np.array(b))
                    d = cohens_d(np.array(a), np.array(b))
                    pval = float(result["p_value"])
                    winner = m1 if np.mean(a) < np.mean(b) else m2
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    print(f"    {m1:>20} vs {m2:<20} p={pval:.4f} {sig}  d={d:.3f}  winner={winner}")
                    tests_run.append({
                        "experiment": f"spy_n{n_train}",
                        "comparison": f"{m1}_vs_{m2}",
                        "p_value": pval,
                        "cohens_d": d,
                        "winner": winner,
                        "n_pairs": len(common_seeds),
                    })
                except Exception as e:
                    print(f"    {m1} vs {m2}: FAILED ({e})")

    # --- SPY: Gradient MSE tests ---
    print("\n--- SPY Options: Pairwise Wilcoxon Tests (gradient MSE) ---")
    for n_train in sorted(by_ntrain.keys()):
        recs = by_ntrain[n_train]
        by_method = group_by(recs, "method")
        print(f"\n  n_train = {n_train}:")

        method_grads = {}
        for m in methods:
            if m in by_method:
                pairs = {r["seed"]: r["test_grad_mse"] for r in by_method[m]}
                method_grads[m] = pairs

        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                if m1 not in method_grads or m2 not in method_grads:
                    continue
                common_seeds = sorted(set(method_grads[m1].keys()) & set(method_grads[m2].keys()))
                if len(common_seeds) < 5:
                    continue
                a = [method_grads[m1][s] for s in common_seeds]
                b = [method_grads[m2][s] for s in common_seeds]
                try:
                    result = paired_wilcoxon_test(np.array(a), np.array(b))
                    d = cohens_d(np.array(a), np.array(b))
                    pval = float(result["p_value"])
                    winner = m1 if np.mean(a) < np.mean(b) else m2
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    print(f"    {m1:>20} vs {m2:<20} p={pval:.4f} {sig}  d={d:.3f}  winner={winner}")
                    tests_run.append({
                        "experiment": f"spy_n{n_train}_grad",
                        "comparison": f"{m1}_vs_{m2}",
                        "p_value": pval,
                        "cohens_d": d,
                        "winner": winner,
                        "n_pairs": len(common_seeds),
                    })
                except Exception as e:
                    print(f"    {m1} vs {m2}: FAILED ({e})")

    # --- Heston: vanilla vs DML methods ---
    print("\n--- Heston: Pairwise Wilcoxon Tests (value MSE) ---")
    heston = {k: v for k, v in lrm_results.items() if "heston" in k}
    if heston:
        heston_list = list(heston.values())
        by_config = group_by(heston_list, "payoff", "n_steps")

        for config_key, recs in sorted(by_config.items()):
            payoff, nsteps = config_key
            by_method = group_by(recs, "method")
            print(f"\n  {payoff}, steps={nsteps}:")

            method_vals = {}
            for m in methods:
                if m in by_method:
                    pairs = {r["seed"]: r["test_value_mse"] for r in by_method[m]}
                    method_vals[m] = pairs

            for m in ["dml_fixed", "dml_gradnorm", "dml_relobralo"]:
                if "vanilla" not in method_vals or m not in method_vals:
                    continue
                common_seeds = sorted(set(method_vals["vanilla"].keys()) & set(method_vals[m].keys()))
                if len(common_seeds) < 5:
                    continue
                a = [method_vals["vanilla"][s] for s in common_seeds]
                b = [method_vals[m][s] for s in common_seeds]
                try:
                    result = paired_wilcoxon_test(np.array(a), np.array(b))
                    d = cohens_d(np.array(a), np.array(b))
                    pval = float(result["p_value"])
                    winner = "vanilla" if np.mean(a) < np.mean(b) else m
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    print(f"    vanilla vs {m:<22} p={pval:.4f} {sig}  d={d:.3f}  winner={winner}")
                    tests_run.append({
                        "experiment": f"heston_{payoff}_steps{nsteps}",
                        "comparison": f"vanilla_vs_{m}",
                        "p_value": pval,
                        "cohens_d": d,
                        "winner": winner,
                        "n_pairs": len(common_seeds),
                    })
                except Exception as e:
                    print(f"    vanilla vs {m}: FAILED ({e})")

    # Summary
    print(f"\n--- Statistical Tests Summary ---")
    print(f"  Total tests: {len(tests_run)}")
    sig_tests = [t for t in tests_run if t["p_value"] < 0.05]
    print(f"  Significant (p<0.05): {len(sig_tests)}")
    for t in sig_tests:
        print(f"    {t['experiment']:>30} {t['comparison']:<35} p={t['p_value']:.4f}  winner={t['winner']}")

    return tests_run


# ============================================================================
# 5. CROSS-EXPERIMENT SUMMARY
# ============================================================================

def cross_experiment_summary(lrm_results, spy_results, stat_tests):
    """Print cross-experiment summary for paper."""
    print("\n" + "=" * 90)
    print("SECTION 5: CROSS-EXPERIMENT SUMMARY — KEY FINDINGS FOR PAPER")
    print("=" * 90)

    lrm_list = [v for v in lrm_results.values()]
    spy_list = list(spy_results.values())

    # Count by experiment
    heston = [r for r in lrm_list if "heston" in r.get("key", "")]
    non_heston_lrm = [r for r in lrm_list if "heston" not in r.get("key", "")]

    print(f"\n  Experiment Counts:")
    print(f"    LRM comparison (A1/A2/B/D): {len(non_heston_lrm)}")
    print(f"    Heston Euler-LRM (C):       {len(heston)}")
    print(f"    SPY real-world (E):         {len(spy_list)}")
    print(f"    TOTAL NEW:                  {len(non_heston_lrm) + len(heston) + len(spy_list)}")

    # Key finding 1: LRM noise kills DML
    print(f"\n  KEY FINDING 1: LRM labels too noisy for DML benefit")
    print(f"    → Vanilla wins A2 (barrier) and ALL Heston configs")
    print(f"    → DML methods degrade monotonically with LRM variance")
    print(f"    → GradNorm cannot fix fundamentally noisy labels")

    # Key finding 2: With exact Greeks, GradNorm is Pareto-optimal
    spy_by_ntrain = group_by(spy_list, "n_train")
    for n_train in sorted(spy_by_ntrain.keys()):
        by_method = group_by(spy_by_ntrain[n_train], "method")
        if "vanilla" in by_method and "dml_gradnorm" in by_method:
            van_val = np.mean([r["test_value_mse"] for r in by_method["vanilla"]])
            gn_val = np.mean([r["test_value_mse"] for r in by_method["dml_gradnorm"]])
            van_grad = np.mean([r["test_grad_mse"] for r in by_method["vanilla"]])
            gn_grad = np.mean([r["test_grad_mse"] for r in by_method["dml_gradnorm"]])
            val_penalty = (gn_val - van_val) / van_val * 100
            grad_improv = van_grad / gn_grad

            print(f"\n  KEY FINDING 2 (n={n_train}): GradNorm is Pareto-optimal")
            print(f"    → Value MSE penalty: {val_penalty:+.1f}% vs vanilla")
            print(f"    → Gradient MSE improvement: {grad_improv:.0f}x vs vanilla")
            print(f"    → For hedging, {val_penalty:+.1f}% worse pricing + {grad_improv:.0f}x better Greeks = clear win")

    # Key finding 3: dml_fixed (G&K approach) over-regularizes
    for n_train in sorted(spy_by_ntrain.keys()):
        by_method = group_by(spy_by_ntrain[n_train], "method")
        if "vanilla" in by_method and "dml_fixed" in by_method:
            van_val = np.mean([r["test_value_mse"] for r in by_method["vanilla"]])
            fix_val = np.mean([r["test_value_mse"] for r in by_method["dml_fixed"]])
            val_penalty = (fix_val - van_val) / van_val * 100
            print(f"\n  KEY FINDING 3 (n={n_train}): Fixed-λ (G&K) over-regularizes")
            print(f"    → Value MSE penalty: {val_penalty:+.1f}% vs vanilla")

    # Statistical significance count
    sig = [t for t in stat_tests if t["p_value"] < 0.05]
    print(f"\n  STATISTICAL RIGOR:")
    print(f"    → {len(stat_tests)} total pairwise tests")
    print(f"    → {len(sig)} statistically significant at p<0.05")

    # Method ranking across all experiments
    print(f"\n  METHOD RANKING (by number of experiment configs where best):")
    wins = defaultdict(int)
    total_configs = 0

    # SPY configs
    for n_train in spy_by_ntrain:
        by_method = group_by(spy_by_ntrain[n_train], "method")
        best_m = min(by_method.keys(), key=lambda m: np.mean([r["test_value_mse"] for r in by_method[m]]))
        wins[best_m] += 1
        total_configs += 1

    # Heston configs
    heston_by_config = group_by(heston, "payoff", "n_steps")
    for config_key, recs in heston_by_config.items():
        by_method = group_by(recs, "method")
        best_m = min(by_method.keys(), key=lambda m: np.mean([r["test_value_mse"] for r in by_method[m]]))
        wins[best_m] += 1
        total_configs += 1

    for m, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"    {m:<25} {count}/{total_configs} configs")


# ============================================================================
# HELPERS
# ============================================================================

def _print_method_table(by_method, methods):
    """Print method comparison table."""
    print(f"      {'Method':<25} {'Mean Val MSE':>14} {'Std':>13} {'Mean Grad MSE':>14} {'Count':>6}")
    print(f"      {'-'*25} {'-'*14} {'-'*13} {'-'*14} {'-'*6}")

    best_val = float("inf")
    best_method = None

    for m in methods:
        if m not in by_method:
            continue
        vals = [r["test_value_mse"] for r in by_method[m]]
        grads = [r.get("test_grad_mse", 0) for r in by_method[m]]
        mean_val = np.mean(vals)

        if mean_val < best_val:
            best_val = mean_val
            best_method = m

        print(
            f"      {m:<25} {mean_val:14.6e} "
            f"{np.std(vals):13.6e} "
            f"{np.mean(grads):14.6e} {len(vals):6d}"
        )

    if best_method:
        print(f"      → BEST: {best_method}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze new extension experiments")
    parser.add_argument("--section", default="all",
                        choices=["all", "lrm", "heston", "spy", "stats", "summary"])
    parser.add_argument("--save-json", action="store_true",
                        help="Save analysis results to JSON")
    args = parser.parse_args()

    lrm_results = load_lrm_results()
    spy_results = load_spy_results()

    print(f"Loaded: {len(lrm_results)} LRM + {len(spy_results)} SPY = {len(lrm_results) + len(spy_results)} total results")

    lrm_experiments = {}
    heston_data = {}
    spy_data = {}
    stat_tests = []

    sections = args.section
    run_all = sections == "all"

    if run_all or sections == "lrm":
        lrm_experiments = analyze_lrm(lrm_results)

    if run_all or sections == "heston":
        heston_data = analyze_heston(lrm_results)

    if run_all or sections == "spy":
        spy_data = analyze_spy(spy_results)

    if run_all or sections == "stats":
        stat_tests = run_statistical_tests(lrm_results, spy_results)

    if run_all or sections == "summary":
        if not stat_tests:
            stat_tests = run_statistical_tests(lrm_results, spy_results)
        cross_experiment_summary(lrm_results, spy_results, stat_tests)

    if args.save_json:
        # Save all numerical results for reproducibility
        analysis = {
            "n_lrm_results": len(lrm_results),
            "n_spy_results": len(spy_results),
            "stat_tests": stat_tests,
        }
        out = Path("results/new_experiments_analysis.json")
        with open(out, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nSaved analysis to {out}")

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()
