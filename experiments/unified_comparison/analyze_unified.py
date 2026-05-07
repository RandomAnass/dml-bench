#!/usr/bin/env python3
"""
Publication-Grade Analysis of Unified Discontinuous-Payoff Comparison.

Analyzes 550 experiments: 11 methods × 5 datasets × 10 seeds.

Sections:
  1. Per-Dataset Summary Tables (mean ± std, sorted by grad MSE)
  2. Relative Performance vs Vanilla (value penalty %, grad improvement ×)
  3. Pairwise Wilcoxon Tests + Holm-Bonferroni Correction
  4. Cohen's d Effect Sizes (practical significance)
  5. Bootstrap 95% CIs per method per dataset
  6. Cross-Dataset Method Ranking (mean rank, Friedman-style)
  7. Pareto Analysis (value-gradient tradeoff, ≤10% value penalty)
  8. Label-Type Comparison (pathwise vs LRM vs fuzzy)
  9. Auto-generated Markdown Report

Methods (11):
  Pathwise labels: vanilla, dml_fixed, dml_gradnorm, dml_relobralo, dml_warmup
  LRM labels:      dml_lrm, dml_gradnorm_lrm, dml_warmup_lrm
  Fuzzy labels:    dml_fuzzy, dml_gradnorm_fuzzy, dml_warmup_fuzzy

Datasets (5):
  digital_bs (1D, eval=analytical), barrier_bs (1D, eval=analytical),
  heston_digital (1D, eval=COS semi-analytical), basket_d1 (1D, eval=analytical),
  basket_d7 (7D, eval=high-k MC 100K)

Usage:
  python experiments/unified_comparison/analyze_unified.py
  python experiments/unified_comparison/analyze_unified.py --mode single_seed
  python experiments/unified_comparison/analyze_unified.py --section all
  python experiments/unified_comparison/analyze_unified.py --section summary
  python experiments/unified_comparison/analyze_unified.py --section wilcoxon
  python experiments/unified_comparison/analyze_unified.py --section pareto
  python experiments/unified_comparison/analyze_unified.py --latex
  python experiments/unified_comparison/analyze_unified.py --save-report
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.stats import (
    paired_wilcoxon_test, bootstrap_ci, cohens_d, cohens_d_ci,
    effect_size_label, holm_bonferroni,
)


# ============================================================================
# CONSTANTS
# ============================================================================

RESULTS_DIR = Path("results/unified_comparison")

ALL_METHODS = [
    "vanilla",
    "dml_fixed",
    "dml_gradnorm",
    "dml_relobralo",
    "dml_warmup",
    "dml_lrm",
    "dml_gradnorm_lrm",
    "dml_warmup_lrm",
    "dml_fuzzy",
    "dml_gradnorm_fuzzy",
    "dml_warmup_fuzzy",
]

DATASET_ORDER = ["digital_bs", "barrier_bs", "heston_digital", "basket_d1", "basket_d7"]

METHOD_LABELS = {
    "vanilla":            "Vanilla (no deriv.)",
    "dml_fixed":          "DML fixed λ (PW)",
    "dml_gradnorm":       "GradNorm (PW)",
    "dml_relobralo":      "ReLoBRaLo (PW)",
    "dml_warmup":         "Warmup (PW)",
    "dml_lrm":            "DML fixed λ (LRM)",
    "dml_gradnorm_lrm":   "GradNorm (LRM)",
    "dml_warmup_lrm":     "Warmup (LRM)",
    "dml_fuzzy":           "DML fixed λ (Fuzzy)",
    "dml_gradnorm_fuzzy":  "GradNorm (Fuzzy)",
    "dml_warmup_fuzzy":    "Warmup (Fuzzy)",
}

LABEL_FAMILIES = {
    "pathwise": ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"],
    "lrm":     ["dml_lrm", "dml_gradnorm_lrm", "dml_warmup_lrm"],
    "fuzzy":   ["dml_fuzzy", "dml_gradnorm_fuzzy", "dml_warmup_fuzzy"],
}

DATASET_LABELS = {
    "digital_bs":      "Digital BS (1D, analytical)",
    "barrier_bs":      "Barrier BS (1D, analytical)",
    "heston_digital":  "Heston Digital (1D, COS method)",
    "basket_d1":       "Basket Digital d=1 (analytical)",
    "basket_d7":       "Basket Digital d=7 (high-k MC 100K)",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(mode: str = "multi_seed") -> Dict[str, Dict]:
    """Load all JSON results from the specified mode directory."""
    results_dir = RESULTS_DIR / mode
    results = {}
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist.")
        return results
    for f in results_dir.glob("*.json"):
        if f.name.startswith("summary") or f.name.startswith("analysis"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                key = data.get("key", f.stem)
                results[key] = data
        except Exception:
            pass
    return results


def group_by_dataset(results: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Group results by dataset name."""
    groups = defaultdict(list)
    for r in results.values():
        groups[r["dataset"]].append(r)
    return dict(groups)


def group_by_method(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by method name."""
    groups = defaultdict(list)
    for r in records:
        groups[r["method"]].append(r)
    return dict(groups)


def get_values(records: List[Dict], key: str) -> np.ndarray:
    """Extract a metric array from a list of records, sorted by seed for pairing."""
    sorted_recs = sorted(records, key=lambda r: r["seed"])
    return np.array([r[key] for r in sorted_recs])


# ============================================================================
# 1. PER-DATASET SUMMARY TABLES
# ============================================================================

def analyze_summary(by_dataset: Dict[str, List[Dict]], latex: bool = False) -> str:
    """Generate per-dataset summary tables (mean ± std for value & grad MSE)."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 1: PER-DATASET SUMMARY TABLES")
    lines.append("=" * 100)

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)

        eval_src = recs[0].get("eval_source", "unknown")
        dim = recs[0].get("dim", "?")
        n_train = recs[0].get("n_train", "?")
        n_test = recs[0].get("n_test", "?")
        eps = recs[0].get("epsilon")
        lrm_var = recs[0].get("lrm_var_mean")

        lines.append(f"\n{'─' * 100}")
        lines.append(f"Dataset: {DATASET_LABELS.get(dataset, dataset)}")
        lines.append(f"  dim={dim}, n_train={n_train}, n_test={n_test}, eval={eval_src}"
                      + (f", ε={eps:.2f}" if eps else "")
                      + (f", LRM_var={lrm_var:.4f}" if lrm_var else ""))
        lines.append(f"{'─' * 100}")

        # Compute stats per method
        method_stats = {}
        for method in ALL_METHODS:
            if method not in by_method:
                continue
            vals = get_values(by_method[method], "test_value_mse")
            grads = get_values(by_method[method], "test_grad_mse")
            method_stats[method] = {
                "val_mean": np.mean(vals),
                "val_std": np.std(vals),
                "grad_mean": np.mean(grads),
                "grad_std": np.std(grads),
                "n": len(vals),
                "val_arr": vals,
                "grad_arr": grads,
            }

        # Sort by gradient MSE (ascending)
        sorted_methods = sorted(method_stats.keys(), key=lambda m: method_stats[m]["grad_mean"])

        if latex:
            lines.append(_latex_summary_table(dataset, method_stats, sorted_methods))
        else:
            lines.append(f"\n  {'Method':<25} {'Val MSE (mean±std)':>28} {'Grad MSE (mean±std)':>28} {'N':>4}")
            lines.append(f"  {'-'*25} {'-'*28} {'-'*28} {'-'*4}")
            for method in sorted_methods:
                s = method_stats[method]
                val_str = f"{s['val_mean']:.4e} ± {s['val_std']:.4e}"
                grad_str = f"{s['grad_mean']:.4e} ± {s['grad_std']:.4e}"
                lines.append(f"  {method:<25} {val_str:>28} {grad_str:>28} {s['n']:>4}")

            # Best markers
            best_val = min(sorted_methods, key=lambda m: method_stats[m]["val_mean"])
            best_grad = sorted_methods[0]  # already sorted by grad
            lines.append(f"\n  Best value MSE:    {best_val} ({method_stats[best_val]['val_mean']:.4e})")
            lines.append(f"  Best gradient MSE: {best_grad} ({method_stats[best_grad]['grad_mean']:.4e})")

    return "\n".join(lines)


def _latex_summary_table(dataset: str, stats: dict, sorted_methods: list) -> str:
    """Generate LaTeX table for one dataset."""
    lines = [
        f"% {dataset}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Val MSE & $\pm$ Std & Grad MSE & $\pm$ Std \\",
        r"\midrule",
    ]
    best_val = min(sorted_methods, key=lambda m: stats[m]["val_mean"])
    best_grad = sorted_methods[0]
    for method in sorted_methods:
        s = stats[method]
        label = METHOD_LABELS.get(method, method)
        vm = f"\\mathbf{{{s['val_mean']:.2e}}}" if method == best_val else f"{s['val_mean']:.2e}"
        gm = f"\\mathbf{{{s['grad_mean']:.2e}}}" if method == best_grad else f"{s['grad_mean']:.2e}"
        lines.append(f"  {label} & ${vm}$ & ${s['val_std']:.2e}$ & ${gm}$ & ${s['grad_std']:.2e}$ \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


# ============================================================================
# 2. RELATIVE PERFORMANCE VS VANILLA
# ============================================================================

def analyze_relative(by_dataset: Dict[str, List[Dict]]) -> str:
    """Compute value penalty and gradient improvement relative to vanilla."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 2: RELATIVE PERFORMANCE VS VANILLA")
    lines.append("=" * 100)

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)
        if "vanilla" not in by_method:
            continue

        van_val = np.mean(get_values(by_method["vanilla"], "test_value_mse"))
        van_grad = np.mean(get_values(by_method["vanilla"], "test_grad_mse"))

        lines.append(f"\n{'─' * 90}")
        lines.append(f"Dataset: {dataset}  |  Vanilla: val={van_val:.4e}, grad={van_grad:.4e}")
        lines.append(f"{'─' * 90}")
        lines.append(f"  {'Method':<25} {'Val Penalty':>14} {'Grad Improvement':>18} {'Grad MSE':>14}")
        lines.append(f"  {'-'*25} {'-'*14} {'-'*18} {'-'*14}")

        for method in ALL_METHODS:
            if method == "vanilla" or method not in by_method:
                continue
            mv = np.mean(get_values(by_method[method], "test_value_mse"))
            mg = np.mean(get_values(by_method[method], "test_grad_mse"))

            val_pen = (mv - van_val) / van_val * 100 if van_val > 0 else 0
            if van_grad > 0 and mg > 0:
                grad_imp = van_grad / mg
                grad_str = f"{grad_imp:17.1f}x"
            elif mg == 0:
                grad_str = "              ∞"
            else:
                grad_str = "            N/A"
            lines.append(f"  {method:<25} {val_pen:+13.1f}% {grad_str} {mg:14.4e}")

    return "\n".join(lines)


# ============================================================================
# 3. PAIRWISE WILCOXON TESTS + HOLM-BONFERRONI
# ============================================================================

def analyze_wilcoxon(by_dataset: Dict[str, List[Dict]], metric: str = "both") -> str:
    """Run pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 3: PAIRWISE WILCOXON SIGNED-RANK TESTS (HOLM-BONFERRONI CORRECTED)")
    lines.append("=" * 100)

    metrics_to_test = []
    if metric in ("both", "value"):
        metrics_to_test.append(("test_value_mse", "Value MSE"))
    if metric in ("both", "gradient"):
        metrics_to_test.append(("test_grad_mse", "Gradient MSE"))

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)

        for metric_key, metric_label in metrics_to_test:
            lines.append(f"\n{'─' * 90}")
            lines.append(f"Dataset: {dataset}  |  Metric: {metric_label}")
            lines.append(f"{'─' * 90}")

            # Build results dict for pairwise tests
            results_dict = {}
            for method in ALL_METHODS:
                if method not in by_method:
                    continue
                results_dict[method] = get_values(by_method[method], metric_key)

            methods_list = sorted(results_dict.keys())
            n_methods = len(methods_list)
            if n_methods < 2:
                lines.append("  Insufficient methods for comparison.")
                continue

            # All pairwise Wilcoxon + Cohen's d (no bootstrap on diffs for speed)
            pairwise_results = []
            raw_p_values = []
            for i in range(n_methods):
                for j in range(i + 1, n_methods):
                    a = results_dict[methods_list[i]]
                    b = results_dict[methods_list[j]]
                    test_res = paired_wilcoxon_test(a, b)
                    d = cohens_d(a, b)
                    pairwise_results.append({
                        "comparison": f"{methods_list[i]} vs {methods_list[j]}",
                        "method_a": methods_list[i],
                        "method_b": methods_list[j],
                        "p_value": test_res.get("p_value", 1.0),
                        "effect_size": d,
                        "effect_label": effect_size_label(d),
                    })
                    raw_p_values.append(test_res.get("p_value", 1.0))

            # Holm-Bonferroni correction
            corrected = holm_bonferroni(raw_p_values) if raw_p_values else []
            for idx, pr in enumerate(pairwise_results):
                pr["adjusted_p"] = corrected[idx]["adjusted_p"] if idx < len(corrected) else 1.0
                pr["significant"] = corrected[idx]["significant"] if idx < len(corrected) else False

            sig_tests = [t for t in pairwise_results if t["significant"]]

            lines.append(f"  Total pairwise tests: {len(pairwise_results)}")
            lines.append(f"  Significant after Holm-Bonferroni (α=0.05): {len(sig_tests)}")

            if sig_tests:
                lines.append(f"\n  {'Comparison':<50} {'p_raw':>10} {'p_adj':>10} {'Cohen d':>10} {'Effect':>12}")
                lines.append(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
                for t in sorted(sig_tests, key=lambda x: x["adjusted_p"]):
                    lines.append(
                        f"  {t['comparison']:<50} {t['p_value']:>10.4f} "
                        f"{t['adjusted_p']:>10.4f} "
                        f"{t['effect_size']:>10.3f} {t['effect_label']:>12}"
                    )

            # Key comparisons to highlight
            key_pairs = [
                ("vanilla", "dml_fuzzy"),
                ("vanilla", "dml_lrm"),
                ("dml_fuzzy", "dml_lrm"),
                ("dml_warmup_fuzzy", "dml_warmup_lrm"),
                ("dml_gradnorm_fuzzy", "dml_gradnorm_lrm"),
            ]
            lines.append(f"\n  Key Comparisons:")
            lines.append(f"  {'Comparison':<50} {'p_raw':>10} {'p_adj':>10} {'d':>8} {'Effect':>10} {'Winner':>15}")
            lines.append(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*15}")

            # Build a lookup for corrected p-values
            corr_lookup = {t["comparison"]: t for t in pairwise_results}

            for ma, mb in key_pairs:
                if ma not in results_dict or mb not in results_dict:
                    continue
                comp_str = f"{ma} vs {mb}"
                # Try both orderings
                t = corr_lookup.get(comp_str) or corr_lookup.get(f"{mb} vs {ma}")
                if t is None:
                    continue

                winner = ma if np.mean(results_dict[ma]) < np.mean(results_dict[mb]) else mb
                sig_marker = " *" if t["adjusted_p"] < 0.05 else "  " if t["adjusted_p"] < 0.1 else ""
                lines.append(
                    f"  {comp_str:<50} {t['p_value']:>10.4f} "
                    f"{t['adjusted_p']:>10.4f} {t['effect_size']:>8.3f} {t['effect_label']:>10} "
                    f"{winner:>15}{sig_marker}"
                )

    return "\n".join(lines)


# ============================================================================
# 4. COHEN'S D EFFECT SIZES
# ============================================================================

def analyze_effect_sizes(by_dataset: Dict[str, List[Dict]]) -> str:
    """Compute Cohen's d for all DML methods vs vanilla, per dataset."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 4: COHEN'S D EFFECT SIZES (VS VANILLA)")
    lines.append("=" * 100)
    lines.append("  Interpretation: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)
        if "vanilla" not in by_method:
            continue

        van_vals = get_values(by_method["vanilla"], "test_value_mse")
        van_grads = get_values(by_method["vanilla"], "test_grad_mse")

        lines.append(f"\n{'─' * 90}")
        lines.append(f"Dataset: {dataset}")
        lines.append(f"{'─' * 90}")
        lines.append(f"  {'Method':<25} {'d (value)':>12} {'95% CI (v)':>18} {'Effect(v)':>12} {'d (gradient)':>14} {'95% CI (g)':>18} {'Effect(g)':>12}")
        lines.append(f"  {'-'*25} {'-'*12} {'-'*18} {'-'*12} {'-'*14} {'-'*18} {'-'*12}")

        for method in ALL_METHODS:
            if method == "vanilla" or method not in by_method:
                continue
            m_vals = get_values(by_method[method], "test_value_mse")
            m_grads = get_values(by_method[method], "test_grad_mse")

            dv = cohens_d_ci(van_vals, m_vals, n_bootstrap=9999)
            dg = cohens_d_ci(van_grads, m_grads, n_bootstrap=9999)

            lines.append(
                f"  {method:<25} {dv['d']:>12.3f} {'['+f'{dv["ci_lower"]:.2f}, {dv["ci_upper"]:.2f}'+']':>18} {dv['label']:>12} "
                f"{dg['d']:>14.3f} {'['+f'{dg["ci_lower"]:.2f}, {dg["ci_upper"]:.2f}'+']':>18} {dg['label']:>12}"
            )

    return "\n".join(lines)


# ============================================================================
# 5. BOOTSTRAP 95% CIs
# ============================================================================

def analyze_bootstrap_ci(by_dataset: Dict[str, List[Dict]]) -> str:
    """Compute bootstrap 95% CIs for each method per dataset."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 5: BOOTSTRAP 95% CONFIDENCE INTERVALS")
    lines.append("=" * 100)

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)

        lines.append(f"\n{'─' * 100}")
        lines.append(f"Dataset: {dataset}")
        lines.append(f"{'─' * 100}")

        for metric_key, metric_label in [("test_value_mse", "Value MSE"), ("test_grad_mse", "Gradient MSE")]:
            lines.append(f"\n  {metric_label}:")
            lines.append(f"  {'Method':<25} {'Mean':>14} {'95% CI':>30}")
            lines.append(f"  {'-'*25} {'-'*14} {'-'*30}")

            for method in ALL_METHODS:
                if method not in by_method:
                    continue
                arr = get_values(by_method[method], metric_key)
                ci = bootstrap_ci(arr, n_bootstrap=2000, alpha=0.05)
                ci_str = f"[{ci['ci_lower']:.4e}, {ci['ci_upper']:.4e}]"
                lines.append(f"  {method:<25} {ci['mean']:>14.4e} {ci_str:>30}")

    return "\n".join(lines)


# ============================================================================
# 6. CROSS-DATASET METHOD RANKING
# ============================================================================

def analyze_ranking(by_dataset: Dict[str, List[Dict]]) -> str:
    """Rank methods across datasets (mean rank, Friedman-style)."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 6: CROSS-DATASET METHOD RANKING (MEAN RANK)")
    lines.append("=" * 100)

    for metric_key, metric_label in [("test_value_mse", "Value MSE"), ("test_grad_mse", "Gradient MSE")]:
        lines.append(f"\n  {metric_label}:")

        all_ranks = defaultdict(list)

        for dataset in DATASET_ORDER:
            if dataset not in by_dataset:
                continue
            recs = by_dataset[dataset]
            by_method = group_by_method(recs)

            # Mean metric per method for this dataset
            means = {}
            for method in ALL_METHODS:
                if method not in by_method:
                    continue
                means[method] = np.mean(get_values(by_method[method], metric_key))

            # Rank (1 = best = lowest MSE)
            sorted_methods = sorted(means.keys(), key=means.get)
            for rank, method in enumerate(sorted_methods, 1):
                all_ranks[method].append(rank)

        # Compute mean rank
        mean_ranks = {m: np.mean(ranks) for m, ranks in all_ranks.items()}
        sorted_by_rank = sorted(mean_ranks.keys(), key=mean_ranks.get)

        lines.append(f"  {'Rank':>4} {'Method':<25} {'Mean Rank':>10} {'Ranks per Dataset':>40}")
        lines.append(f"  {'-'*4} {'-'*25} {'-'*10} {'-'*40}")

        for overall_rank, method in enumerate(sorted_by_rank, 1):
            ranks_str = ", ".join(f"{r}" for r in all_ranks[method])
            lines.append(
                f"  {overall_rank:>4} {method:<25} {mean_ranks[method]:>10.1f} "
                f"  [{ranks_str}]"
            )

    return "\n".join(lines)


# ============================================================================
# 7. PARETO ANALYSIS
# ============================================================================

def analyze_pareto(by_dataset: Dict[str, List[Dict]]) -> str:
    """Identify Pareto-optimal methods (value vs gradient MSE tradeoff)."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 7: PARETO ANALYSIS (VALUE-GRADIENT TRADEOFF)")
    lines.append("=" * 100)
    lines.append("  Pareto-optimal: no other method is better on BOTH value AND gradient MSE.")
    lines.append("  Constrained Pareto (≤10% val penalty): best gradient MSE with ≤10% worse value than vanilla.")

    global_pareto_counts = defaultdict(int)

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)

        lines.append(f"\n{'─' * 90}")
        lines.append(f"Dataset: {dataset}")
        lines.append(f"{'─' * 90}")

        # Mean metrics per method
        means = {}
        for method in ALL_METHODS:
            if method not in by_method:
                continue
            mv = np.mean(get_values(by_method[method], "test_value_mse"))
            mg = np.mean(get_values(by_method[method], "test_grad_mse"))
            means[method] = (mv, mg)

        # Find Pareto frontier
        pareto = []
        for m, (mv, mg) in means.items():
            dominated = False
            for m2, (mv2, mg2) in means.items():
                if m2 != m and mv2 <= mv and mg2 <= mg and (mv2 < mv or mg2 < mg):
                    dominated = True
                    break
            if not dominated:
                pareto.append(m)
                global_pareto_counts[m] += 1

        lines.append(f"  Pareto-optimal methods: {', '.join(pareto)}")

        # Constrained Pareto (≤10% value penalty)
        van_val = means.get("vanilla", (None, None))[0]
        if van_val is not None:
            eligible = {m: (mv, mg) for m, (mv, mg) in means.items()
                        if (mv - van_val) / van_val <= 0.10}
            if eligible:
                best_constrained = min(eligible.keys(), key=lambda m: eligible[m][1])
                mv, mg = eligible[best_constrained]
                val_pen = (mv - van_val) / van_val * 100
                grad_imp = van_val / mg if mg > 0 else float("inf")
                lines.append(f"  Best (≤10% val penalty): {best_constrained} "
                             f"(val_pen={val_pen:+.1f}%, grad={mg:.4e})")
            else:
                lines.append(f"  No method within ≤10% value penalty.")

        # Table
        lines.append(f"\n  {'Method':<25} {'Val MSE':>14} {'Grad MSE':>14} {'Pareto?':>8}")
        lines.append(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*8}")
        for method in ALL_METHODS:
            if method not in means:
                continue
            mv, mg = means[method]
            is_pareto = "  ★" if method in pareto else ""
            lines.append(f"  {method:<25} {mv:14.4e} {mg:14.4e} {is_pareto}")

    # Global summary
    lines.append(f"\n{'=' * 60}")
    lines.append(f"GLOBAL PARETO FREQUENCY (across all {len(DATASET_ORDER)} datasets):")
    lines.append(f"{'=' * 60}")
    for method, count in sorted(global_pareto_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {method:<25} Pareto-optimal in {count}/{len(DATASET_ORDER)} datasets")

    return "\n".join(lines)


# ============================================================================
# 8. LABEL-TYPE COMPARISON
# ============================================================================

def analyze_label_types(by_dataset: Dict[str, List[Dict]]) -> str:
    """Compare label families: pathwise (excl. vanilla) vs LRM vs fuzzy."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("SECTION 8: LABEL-TYPE COMPARISON (PATHWISE vs LRM vs FUZZY)")
    lines.append("=" * 100)
    lines.append("  Compares the three label families, aggregated across balancing strategies.")
    lines.append("  Pathwise methods exclude vanilla (which uses no derivative labels).")

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)

        lines.append(f"\n{'─' * 90}")
        lines.append(f"Dataset: {dataset}")
        lines.append(f"{'─' * 90}")

        # Compare matching balancing strategies across label types
        # fixed: dml_fixed vs dml_lrm vs dml_fuzzy
        # gradnorm: dml_gradnorm vs dml_gradnorm_lrm vs dml_gradnorm_fuzzy
        # warmup: dml_warmup vs dml_warmup_lrm vs dml_warmup_fuzzy

        strategy_groups = [
            ("Fixed λ=1", "dml_fixed", "dml_lrm", "dml_fuzzy"),
            ("GradNorm", "dml_gradnorm", "dml_gradnorm_lrm", "dml_gradnorm_fuzzy"),
            ("Warmup", "dml_warmup", "dml_warmup_lrm", "dml_warmup_fuzzy"),
        ]

        for strategy_name, pw_method, lrm_method, fuzzy_method in strategy_groups:
            if pw_method not in by_method or lrm_method not in by_method or fuzzy_method not in by_method:
                continue

            lines.append(f"\n  Strategy: {strategy_name}")
            lines.append(f"  {'Label Type':<12} {'Method':<25} {'Val MSE':>14} {'Grad MSE':>14}")
            lines.append(f"  {'-'*12} {'-'*25} {'-'*14} {'-'*14}")

            for label_type, method_name in [("Pathwise", pw_method), ("LRM", lrm_method), ("Fuzzy", fuzzy_method)]:
                mv = np.mean(get_values(by_method[method_name], "test_value_mse"))
                mg = np.mean(get_values(by_method[method_name], "test_grad_mse"))
                lines.append(f"  {label_type:<12} {method_name:<25} {mv:14.4e} {mg:14.4e}")

            # Wilcoxon: fuzzy vs LRM for this strategy
            fuzzy_vals = get_values(by_method[fuzzy_method], "test_value_mse")
            lrm_vals = get_values(by_method[lrm_method], "test_value_mse")
            fuzzy_grads = get_values(by_method[fuzzy_method], "test_grad_mse")
            lrm_grads = get_values(by_method[lrm_method], "test_grad_mse")

            test_val = paired_wilcoxon_test(fuzzy_vals, lrm_vals)
            test_grad = paired_wilcoxon_test(fuzzy_grads, lrm_grads)

            val_winner = "Fuzzy" if np.mean(fuzzy_vals) < np.mean(lrm_vals) else "LRM"
            grad_winner = "Fuzzy" if np.mean(fuzzy_grads) < np.mean(lrm_grads) else "LRM"

            lines.append(f"    Fuzzy vs LRM (value):    p={test_val.get('p_value', 1.0):.4f}  → {val_winner}"
                         + (" *" if test_val.get("significant_005", False) else ""))
            lines.append(f"    Fuzzy vs LRM (gradient): p={test_grad.get('p_value', 1.0):.4f}  → {grad_winner}"
                         + (" *" if test_grad.get("significant_005", False) else ""))

    return "\n".join(lines)


# ============================================================================
# 9. MARKDOWN REPORT
# ============================================================================

def generate_markdown_report(
    by_dataset: Dict[str, List[Dict]],
    mode: str,
    n_results: int,
) -> str:
    """Generate a publication-ready markdown report."""
    lines = []
    lines.append("# Unified Discontinuous-Payoff Comparison — Multi-Seed Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Mode:** {mode}")
    lines.append(f"**Total experiments:** {n_results}")
    lines.append(f"**Methods:** {len(ALL_METHODS)} (5 pathwise + 3 LRM + 3 fuzzy)")
    lines.append(f"**Datasets:** {len(DATASET_ORDER)}")
    lines.append(f"**Seeds:** 10 ([42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999])")
    lines.append(f"**Epochs:** 500, **Architecture:** 4×256 softplus, **Optimizer:** Adam (lr=0.005)")
    lines.append("")

    # Summary table per dataset
    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)
        eval_src = recs[0].get("eval_source", "?")

        lines.append(f"## {DATASET_LABELS.get(dataset, dataset)}")
        lines.append("")

        van_val = np.mean(get_values(by_method.get("vanilla", [{}]), "test_value_mse")) if "vanilla" in by_method else None
        van_grad = np.mean(get_values(by_method.get("vanilla", [{}]), "test_grad_mse")) if "vanilla" in by_method else None

        # Main table
        lines.append("| Method | Val MSE (mean±std) | Grad MSE (mean±std) | Val Penalty | Grad Improv. |")
        lines.append("|---|---|---|---|---|")

        for method in ALL_METHODS:
            if method not in by_method:
                continue
            vals = get_values(by_method[method], "test_value_mse")
            grads = get_values(by_method[method], "test_grad_mse")
            mv, sv = np.mean(vals), np.std(vals)
            mg, sg = np.mean(grads), np.std(grads)

            if method == "vanilla":
                val_pen_str = "—"
                grad_imp_str = "baseline"
            elif van_val is not None and van_grad is not None:
                val_pen = (mv - van_val) / van_val * 100
                val_pen_str = f"{val_pen:+.1f}%"
                if van_grad > 0 and mg > 0:
                    grad_imp = van_grad / mg
                    grad_imp_str = f"{grad_imp:.1f}×"
                else:
                    grad_imp_str = "N/A"
            else:
                val_pen_str = "N/A"
                grad_imp_str = "N/A"

            label = METHOD_LABELS.get(method, method)
            lines.append(f"| {label} | {mv:.4e} ± {sv:.4e} | {mg:.4e} ± {sg:.4e} | {val_pen_str} | {grad_imp_str} |")

        lines.append("")

        # Finding per dataset
        if van_val is not None and van_grad is not None:
            # Best method within 10% val penalty
            eligible = {}
            for method in ALL_METHODS:
                if method not in by_method:
                    continue
                mv = np.mean(get_values(by_method[method], "test_value_mse"))
                mg = np.mean(get_values(by_method[method], "test_grad_mse"))
                if (mv - van_val) / van_val <= 0.10:
                    eligible[method] = mg
            if eligible:
                best = min(eligible, key=eligible.get)
                mg = eligible[best]
                val_pen = (np.mean(get_values(by_method[best], "test_value_mse")) - van_val) / van_val * 100
                grad_imp = van_grad / mg if mg > 0 else float("inf")
                lines.append(f"**Best (≤10% val penalty):** {METHOD_LABELS.get(best, best)} — "
                             f"val penalty {val_pen:+.1f}%, gradient improvement {grad_imp:.1f}×")
                lines.append("")

    # Cross-dataset summary
    lines.append("## Cross-Dataset Method Ranking (by Gradient MSE)")
    lines.append("")

    all_ranks = defaultdict(list)
    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        recs = by_dataset[dataset]
        by_method = group_by_method(recs)
        means = {m: np.mean(get_values(by_method[m], "test_grad_mse"))
                 for m in ALL_METHODS if m in by_method}
        sorted_m = sorted(means, key=means.get)
        for rank, m in enumerate(sorted_m, 1):
            all_ranks[m].append(rank)

    mean_ranks = {m: np.mean(r) for m, r in all_ranks.items()}
    sorted_by_rank = sorted(mean_ranks, key=mean_ranks.get)

    lines.append("| Rank | Method | Mean Rank |")
    lines.append("|---|---|---|")
    for i, m in enumerate(sorted_by_rank, 1):
        label = METHOD_LABELS.get(m, m)
        lines.append(f"| {i} | {label} | {mean_ranks[m]:.1f} |")
    lines.append("")

    # (narrative removed — tables only)
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze unified comparison results")
    parser.add_argument("--mode", default="multi_seed",
                        choices=["smoke_test", "single_seed", "multi_seed"])
    parser.add_argument("--section", default="all",
                        choices=["all", "summary", "relative", "wilcoxon",
                                 "effect", "bootstrap", "ranking", "pareto",
                                 "labels", "report"])
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    parser.add_argument("--save-report", action="store_true",
                        help="Save markdown report to file")
    parser.add_argument("--output-dir", default="results/unified_comparison",
                        help="Directory for output files")
    args = parser.parse_args()

    # Load results
    results = load_results(args.mode)
    if not results:
        print(f"No results found for mode '{args.mode}'")
        sys.exit(1)

    print(f"\nLoaded {len(results)} results (mode: {args.mode})")

    # Verify completeness
    by_dataset = group_by_dataset(results)
    for dataset in DATASET_ORDER:
        if dataset in by_dataset:
            recs = by_dataset[dataset]
            methods = set(r["method"] for r in recs)
            seeds = set(r["seed"] for r in recs)
            print(f"  {dataset}: {len(recs)} results, {len(methods)} methods, {len(seeds)} seeds")
        else:
            print(f"  {dataset}: MISSING")

    # Run sections
    output_parts = []

    sections = {
        "summary":   lambda: analyze_summary(by_dataset, latex=args.latex),
        "relative":  lambda: analyze_relative(by_dataset),
        "wilcoxon":  lambda: analyze_wilcoxon(by_dataset),
        "effect":    lambda: analyze_effect_sizes(by_dataset),
        "bootstrap": lambda: analyze_bootstrap_ci(by_dataset),
        "ranking":   lambda: analyze_ranking(by_dataset),
        "pareto":    lambda: analyze_pareto(by_dataset),
        "labels":    lambda: analyze_label_types(by_dataset),
    }

    if args.section == "all":
        for name, func in sections.items():
            print(f"\n  Running section: {name}...")
            output = func()
            print(output)
            output_parts.append(output)
    elif args.section == "report":
        report = generate_markdown_report(by_dataset, args.mode, len(results))
        print(report)
        output_parts.append(report)
    else:
        output = sections[args.section]()
        print(output)
        output_parts.append(output)

    # Save report if requested
    if args.save_report or args.section == "report":
        report = generate_markdown_report(by_dataset, args.mode, len(results))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"UNIFIED_RESULTS_{args.mode.upper()}.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Report saved to: {report_path}")

        # Also save full analysis as text
        if args.section == "all":
            analysis_path = output_dir / f"analysis_{args.mode}.txt"
            with open(analysis_path, "w") as f:
                f.write("\n\n".join(output_parts))
            print(f"  Full analysis saved to: {analysis_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
