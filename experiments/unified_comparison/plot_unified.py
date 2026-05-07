#!/usr/bin/env python3
"""
Publication-Quality Figures for Unified Discontinuous-Payoff Comparison.

Generates Nature-style (Wong 2011 colorblind-safe, Arial 8pt, 300dpi) plots
from 550 multi-seed experiments (11 methods × 5 datasets × 10 seeds).

Figures:
  1. gradient_improvement   — Per-dataset bar chart of gradient improvement over vanilla
  2. pareto                 — Value penalty vs gradient improvement scatter (Pareto frontier)
  3. ranking_heatmap        — Cross-dataset rank heatmap (methods × datasets)
  4. label_comparison       — Grouped bars: label type (PW / LRM / Fuzzy) per strategy
  5. bootstrap_ci           — Forest plot of bootstrap 95% CIs per dataset
  6. effect_heatmap         — Cohen's d effect sizes (methods × datasets) heatmap

Usage:
    python experiments/unified_comparison/plot_unified.py                     # all figures
    python experiments/unified_comparison/plot_unified.py --figure pareto     # single figure
    python experiments/unified_comparison/plot_unified.py --format png        # PNG output
    python experiments/unified_comparison/plot_unified.py --outdir figures/unified  # custom dir
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dml_benchmark.stats import bootstrap_ci


# ============================================================================
# CONSTANTS (from analyze_unified.py — keep in sync)
# ============================================================================

RESULTS_DIR = Path("results/unified_comparison")

ALL_METHODS = [
    "vanilla",
    "dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup",
    "dml_lrm", "dml_gradnorm_lrm", "dml_warmup_lrm",
    "dml_fuzzy", "dml_gradnorm_fuzzy", "dml_warmup_fuzzy",
]

DATASET_ORDER = ["digital_bs", "barrier_bs", "heston_digital", "basket_d1", "basket_d7"]

DATASET_SHORT = {
    "digital_bs":     "Digital BS",
    "barrier_bs":     "Barrier BS",
    "heston_digital": "Heston Dig.",
    "basket_d1":      "Basket d=1",
    "basket_d7":      "Basket d=7",
}

METHOD_SHORT = {
    "vanilla":            "Vanilla",
    "dml_fixed":          "Fixed (PW)",
    "dml_gradnorm":       "GradNorm (PW)",
    "dml_relobralo":      "ReLoBRaLo (PW)",
    "dml_warmup":         "Warmup (PW)",
    "dml_lrm":            "Fixed (LRM)",
    "dml_gradnorm_lrm":   "GradNorm (LRM)",
    "dml_warmup_lrm":     "Warmup (LRM)",
    "dml_fuzzy":           "Fixed (Fuzzy)",
    "dml_gradnorm_fuzzy":  "GradNorm (Fuzzy)",
    "dml_warmup_fuzzy":    "Warmup (Fuzzy)",
}

# Wong 2011 colorblind-safe palette — extended for 11 methods
# Grouped by label family: pathwise blues, LRM oranges, fuzzy greens
COLORS = {
    "vanilla":            "#56B4E9",  # Sky blue
    "dml_fixed":          "#0072B2",  # Blue
    "dml_gradnorm":       "#332288",  # Indigo
    "dml_relobralo":      "#CC79A7",  # Pink
    "dml_warmup":         "#88CCEE",  # Light cyan
    "dml_lrm":            "#D55E00",  # Vermillion
    "dml_gradnorm_lrm":   "#E69F00",  # Orange
    "dml_warmup_lrm":     "#F0E442",  # Yellow
    "dml_fuzzy":           "#009E73",  # Green
    "dml_gradnorm_fuzzy":  "#44AA99",  # Teal
    "dml_warmup_fuzzy":    "#117733",  # Dark green
}

# For label-type comparison (3 families)
FAMILY_COLORS = {
    "pathwise": "#0072B2",
    "lrm":     "#D55E00",
    "fuzzy":   "#009E73",
}

FAMILY_LABELS = {
    "pathwise": "Pathwise",
    "lrm":     "LRM",
    "fuzzy":   "Fuzzy",
}


# ============================================================================
# STYLE
# ============================================================================

def setup_style():
    """Nature-quality figure style: Arial 8pt, 300dpi, colorblind-safe."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(mode: str = "multi_seed") -> Dict[str, Dict]:
    """Load all JSON results."""
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
    groups = defaultdict(list)
    for r in results.values():
        groups[r["dataset"]].append(r)
    return dict(groups)


def group_by_method(records: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for r in records:
        groups[r["method"]].append(r)
    return dict(groups)


def get_values(records: List[Dict], key: str) -> np.ndarray:
    sorted_recs = sorted(records, key=lambda r: r["seed"])
    return np.array([r[key] for r in sorted_recs])


def compute_means(by_method: Dict[str, List[Dict]]) -> Dict[str, Tuple[float, float]]:
    """Return {method: (mean_val_mse, mean_grad_mse)}."""
    out = {}
    for m, recs in by_method.items():
        mv = np.mean(get_values(recs, "test_value_mse"))
        mg = np.mean(get_values(recs, "test_grad_mse"))
        out[m] = (mv, mg)
    return out


# ============================================================================
# FIGURE 1: GRADIENT IMPROVEMENT BAR CHART
# ============================================================================

def fig_gradient_improvement(by_dataset, out_dir, fmt="pdf"):
    """
    Per-dataset grouped bar chart: gradient improvement over vanilla (log scale).
    Excludes vanilla (baseline = 1.0×). Methods colored by label family.
    Error bars show 95% bootstrap CI on per-seed improvement ratios.
    """
    setup_style()

    n_datasets = len([d for d in DATASET_ORDER if d in by_dataset])
    fig, axes = plt.subplots(1, n_datasets, figsize=(3.0 * n_datasets, 3.5), squeeze=False)

    methods_no_van = [m for m in ALL_METHODS if m != "vanilla"]

    for col, dataset in enumerate(DATASET_ORDER):
        if dataset not in by_dataset:
            continue
        ax = axes[0, col]
        by_method = group_by_method(by_dataset[dataset])

        van_grad_arr = get_values(by_method["vanilla"], "test_grad_mse")

        improvements = []
        ci_lower = []
        ci_upper = []
        colors = []
        labels = []
        for m in methods_no_van:
            if m not in by_method:
                continue
            method_grad_arr = get_values(by_method[m], "test_grad_mse")

            # Per-seed improvement ratios (paired by seed)
            per_seed_imp = van_grad_arr / np.clip(method_grad_arr, 1e-30, None)
            mean_imp = float(np.mean(per_seed_imp))

            # Bootstrap CI on per-seed ratios
            ci = bootstrap_ci(per_seed_imp, n_bootstrap=5000, alpha=0.05)
            improvements.append(mean_imp)
            ci_lower.append(mean_imp - ci["ci_lower"])  # distance below mean
            ci_upper.append(ci["ci_upper"] - mean_imp)  # distance above mean
            colors.append(COLORS.get(m, "#999"))
            labels.append(METHOD_SHORT.get(m, m))

        x = np.arange(len(improvements))
        yerr = [ci_lower, ci_upper]
        bars = ax.bar(x, improvements, color=colors, edgecolor="black",
                      linewidth=0.4, width=0.75,
                      yerr=yerr, capsize=2, error_kw=dict(
                          linewidth=0.8, color="black", capthick=0.8))

        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
        ax.set_title(DATASET_SHORT.get(dataset, dataset), fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Gradient improvement (×)", fontsize=8)

        # Add value annotations on top of tall bars
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            if imp >= 10:
                y_top = imp + ci_upper[i]
                ax.text(bar.get_x() + bar.get_width() / 2, y_top * 1.15,
                        f"{imp:.0f}×", ha="center", va="bottom", fontsize=5.5,
                        fontweight="bold")

    plt.tight_layout()
    path = out_dir / f"gradient_improvement.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 2: PARETO SCATTER (VALUE PENALTY vs GRADIENT IMPROVEMENT)
# ============================================================================

def fig_pareto(by_dataset, out_dir, fmt="pdf"):
    """
    Per-dataset scatter: x = value penalty (%), y = gradient improvement (×).
    Pareto-optimal points highlighted with marker. 10% value penalty zone shaded.
    Error crosshairs show ±1σ across 10 seeds in both dimensions.
    """
    setup_style()

    n_datasets = len([d for d in DATASET_ORDER if d in by_dataset])
    fig, axes = plt.subplots(1, n_datasets, figsize=(3.0 * n_datasets, 3.5), squeeze=False)

    for col, dataset in enumerate(DATASET_ORDER):
        if dataset not in by_dataset:
            continue
        ax = axes[0, col]
        by_method = group_by_method(by_dataset[dataset])

        if "vanilla" not in by_method:
            continue
        van_val_arr = get_values(by_method["vanilla"], "test_value_mse")
        van_grad_arr = get_values(by_method["vanilla"], "test_grad_mse")

        # Compute per-seed statistics for all methods
        points = {}
        point_errs = {}
        for m, recs in by_method.items():
            method_val_arr = get_values(recs, "test_value_mse")
            method_grad_arr = get_values(recs, "test_grad_mse")

            # Per-seed paired metrics
            per_seed_vp = (method_val_arr - van_val_arr) / van_val_arr * 100
            per_seed_gi = van_grad_arr / np.clip(method_grad_arr, 1e-30, None)

            mean_vp = float(np.mean(per_seed_vp))
            mean_gi = float(np.mean(per_seed_gi))
            std_vp = float(np.std(per_seed_vp, ddof=1))
            std_gi = float(np.std(per_seed_gi, ddof=1))

            points[m] = (mean_vp, mean_gi)
            point_errs[m] = (std_vp, std_gi)

        # Identify Pareto-optimal (lower val_pen + higher grad_imp is better)
        pareto_set = set()
        for m, (vp, gi) in points.items():
            dominated = False
            for m2, (vp2, gi2) in points.items():
                if m2 != m and vp2 <= vp and gi2 >= gi and (vp2 < vp or gi2 > gi):
                    dominated = True
                    break
            if not dominated:
                pareto_set.add(m)

        # 10% penalty zone
        ax.axvspan(-100, 10, alpha=0.06, color="green", zorder=0)
        ax.axvline(x=10, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)

        for m, (vp, gi) in points.items():
            color = COLORS.get(m, "#999")
            vp_err, gi_err = point_errs[m]

            if m in pareto_set:
                ax.errorbar(vp, gi, xerr=vp_err, yerr=gi_err,
                            fmt="*", color=color, markersize=8,
                            markeredgecolor="black", markeredgewidth=0.5,
                            ecolor=color, elinewidth=0.6, capsize=2, capthick=0.5,
                            alpha=0.85, zorder=10,
                            label=METHOD_SHORT.get(m, m))
            else:
                ax.errorbar(vp, gi, xerr=vp_err, yerr=gi_err,
                            fmt="o", color=color, markersize=5,
                            ecolor=color, elinewidth=0.6, capsize=2, capthick=0.5,
                            alpha=0.6, zorder=5,
                            label=METHOD_SHORT.get(m, m))

        ax.set_yscale("log")
        ax.set_title(DATASET_SHORT.get(dataset, dataset), fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Gradient improvement (×)", fontsize=8)
        ax.set_xlabel("Value penalty (%)", fontsize=7)

        # Compact legend only on first panel
        if col == 0:
            ax.legend(fontsize=5, loc="upper left", framealpha=0.7,
                      ncol=1, handletextpad=0.3, columnspacing=0.5)

    plt.tight_layout()
    path = out_dir / f"pareto_val_vs_grad.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 3: CROSS-DATASET RANKING HEATMAP
# ============================================================================

def fig_ranking_heatmap(by_dataset, out_dir, fmt="pdf"):
    """
    Heatmap: rows = methods (sorted by mean rank), columns = datasets.
    Cell color = rank (1=best=green, 11=worst=red). Annotated with rank number.
    Two panels: value MSE and gradient MSE.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    for panel_idx, (metric_key, metric_title) in enumerate([
        ("test_value_mse", "Value MSE Rank"),
        ("test_grad_mse", "Gradient MSE Rank"),
    ]):
        ax = axes[panel_idx]
        all_ranks = defaultdict(list)

        for dataset in DATASET_ORDER:
            if dataset not in by_dataset:
                continue
            by_method = group_by_method(by_dataset[dataset])
            means = {m: np.mean(get_values(by_method[m], metric_key))
                     for m in ALL_METHODS if m in by_method}
            sorted_m = sorted(means, key=means.get)
            for rank, m in enumerate(sorted_m, 1):
                all_ranks[m].append(rank)

        # Sort methods by mean rank
        mean_ranks = {m: np.mean(r) for m, r in all_ranks.items()}
        sorted_methods = sorted(mean_ranks, key=mean_ranks.get)

        # Build matrix
        n_methods = len(sorted_methods)
        n_datasets = len(DATASET_ORDER)
        rank_matrix = np.full((n_methods, n_datasets), np.nan)

        for i, m in enumerate(sorted_methods):
            for j, d in enumerate(DATASET_ORDER):
                if j < len(all_ranks[m]):
                    rank_matrix[i, j] = all_ranks[m][j]

        # Plot
        cmap = plt.cm.RdYlGn_r  # 1=green(best), 11=red(worst)
        im = ax.imshow(rank_matrix, aspect="auto", cmap=cmap, vmin=1, vmax=11)

        # Annotate cells
        for i in range(n_methods):
            for j in range(n_datasets):
                val = rank_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val <= 3 or val >= 9 else "black"
                    ax.text(j, i, f"{int(val)}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

        # Labels
        ax.set_xticks(range(n_datasets))
        ax.set_xticklabels([DATASET_SHORT.get(d, d) for d in DATASET_ORDER],
                           rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(n_methods))
        method_labels = [f"{METHOD_SHORT.get(m, m)}  ({mean_ranks[m]:.1f})"
                         for m in sorted_methods]
        ax.set_yticklabels(method_labels, fontsize=7)
        ax.set_title(metric_title, fontsize=9, fontweight="bold")

    # Shared colorbar
    fig.subplots_adjust(left=0.18, right=0.86, wspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Rank (1 = best)", fontsize=8)
    path = out_dir / f"ranking_heatmap.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 4: LABEL-TYPE COMPARISON (GROUPED BARS)
# ============================================================================

def fig_label_comparison(by_dataset, out_dir, fmt="pdf"):
    """
    For each dataset: 3 balancing strategies (Fixed, GradNorm, Warmup) on x-axis,
    3 bars per group (Pathwise, LRM, Fuzzy), y = gradient MSE (log scale).
    Two rows: value MSE (top) and gradient MSE (bottom).
    """
    setup_style()

    strategy_groups = [
        ("Fixed",    "dml_fixed",    "dml_lrm",          "dml_fuzzy"),
        ("GradNorm", "dml_gradnorm", "dml_gradnorm_lrm", "dml_gradnorm_fuzzy"),
        ("Warmup",   "dml_warmup",   "dml_warmup_lrm",   "dml_warmup_fuzzy"),
    ]

    n_datasets = len([d for d in DATASET_ORDER if d in by_dataset])
    fig, axes = plt.subplots(2, n_datasets, figsize=(2.8 * n_datasets, 5.5), squeeze=False)

    bar_width = 0.25
    for col, dataset in enumerate(DATASET_ORDER):
        if dataset not in by_dataset:
            continue
        by_method = group_by_method(by_dataset[dataset])

        for row, (metric_key, metric_label) in enumerate([
            ("test_value_mse", "Value MSE"),
            ("test_grad_mse", "Gradient MSE"),
        ]):
            ax = axes[row, col]
            x_positions = np.arange(len(strategy_groups))

            for fam_idx, (family, fam_color) in enumerate([
                ("pathwise", FAMILY_COLORS["pathwise"]),
                ("lrm",      FAMILY_COLORS["lrm"]),
                ("fuzzy",    FAMILY_COLORS["fuzzy"]),
            ]):
                vals = []
                errs = []
                for _, pw_m, lrm_m, fuzzy_m in strategy_groups:
                    method_map = {"pathwise": pw_m, "lrm": lrm_m, "fuzzy": fuzzy_m}
                    m = method_map[family]
                    if m in by_method:
                        arr = get_values(by_method[m], metric_key)
                        vals.append(np.mean(arr))
                        errs.append(np.std(arr))
                    else:
                        vals.append(np.nan)
                        errs.append(0)

                offset = (fam_idx - 1) * bar_width
                ax.bar(x_positions + offset, vals, bar_width, yerr=errs,
                       color=fam_color, edgecolor="black", linewidth=0.3,
                       label=FAMILY_LABELS[family] if col == 0 and row == 0 else "",
                       capsize=2, alpha=0.85)

            ax.set_yscale("log")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([s[0] for s in strategy_groups], fontsize=7)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=8)
            if row == 0:
                ax.set_title(DATASET_SHORT.get(dataset, dataset), fontsize=9,
                             fontweight="bold")

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = out_dir / f"label_comparison.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 5: BOOTSTRAP CI FOREST PLOT
# ============================================================================

def fig_bootstrap_ci(by_dataset, out_dir, fmt="pdf"):
    """
    Forest plot: each method as a row, horizontal line = 95% bootstrap CI.
    One panel per dataset, two sub-columns: value MSE and gradient MSE.
    """
    setup_style()

    n_datasets = len([d for d in DATASET_ORDER if d in by_dataset])
    fig, axes = plt.subplots(n_datasets, 2, figsize=(8, 2.0 * n_datasets), squeeze=False)

    display_methods = ALL_METHODS[::-1]  # bottom-to-top = best-to-worst visually

    for row, dataset in enumerate(DATASET_ORDER):
        if dataset not in by_dataset:
            continue
        by_method = group_by_method(by_dataset[dataset])

        for col, (metric_key, metric_label) in enumerate([
            ("test_value_mse", "Value MSE"),
            ("test_grad_mse", "Gradient MSE"),
        ]):
            ax = axes[row, col]
            y_positions = []
            y_labels = []
            y_idx = 0

            for m in display_methods:
                if m not in by_method:
                    continue
                arr = get_values(by_method[m], metric_key)
                ci = bootstrap_ci(arr, n_bootstrap=2000, alpha=0.05)

                color = COLORS.get(m, "#999")
                ax.plot([ci["ci_lower"], ci["ci_upper"]], [y_idx, y_idx],
                        color=color, linewidth=2, solid_capstyle="round")
                ax.plot(ci["mean"], y_idx, "o", color=color, markersize=4,
                        markeredgecolor="black", markeredgewidth=0.3)

                y_positions.append(y_idx)
                y_labels.append(METHOD_SHORT.get(m, m))
                y_idx += 1

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=6)
            ax.set_xscale("log")
            ax.set_xlabel(metric_label, fontsize=7)

            if col == 0:
                ax.set_ylabel(DATASET_SHORT.get(dataset, dataset), fontsize=8,
                              fontweight="bold")
            if row == 0:
                ax.set_title(metric_label, fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = out_dir / f"bootstrap_ci_forest.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 6: COHEN'S D EFFECT SIZE HEATMAP
# ============================================================================

def fig_effect_heatmap(by_dataset, out_dir, fmt="pdf"):
    """
    Heatmap: rows = methods (excl. vanilla), columns = datasets.
    Cell color = Cohen's d (vs vanilla) for gradient MSE.
    Positive d = method better than vanilla on gradients.
    Second panel: same for value MSE (negative d = method worse than vanilla on value).
    """
    setup_style()
    from dml_benchmark.stats import cohens_d

    methods_no_van = [m for m in ALL_METHODS if m != "vanilla"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for panel_idx, (metric_key, metric_title, diverging_center) in enumerate([
        ("test_value_mse", "Cohen's d — Value MSE (vs Vanilla)", 0),
        ("test_grad_mse",  "Cohen's d — Gradient MSE (vs Vanilla)", 0),
    ]):
        ax = axes[panel_idx]
        n_methods = len(methods_no_van)
        n_datasets = len(DATASET_ORDER)
        d_matrix = np.full((n_methods, n_datasets), np.nan)

        for j, dataset in enumerate(DATASET_ORDER):
            if dataset not in by_dataset:
                continue
            by_method = group_by_method(by_dataset[dataset])
            if "vanilla" not in by_method:
                continue
            van_arr = get_values(by_method["vanilla"], metric_key)

            for i, m in enumerate(methods_no_van):
                if m not in by_method:
                    continue
                method_arr = get_values(by_method[m], metric_key)
                # d > 0 means vanilla > method (method is better = lower MSE)
                d_val = cohens_d(van_arr, method_arr)
                d_matrix[i, j] = d_val

        # Diverging colormap centered at 0
        vmax = min(np.nanmax(np.abs(d_matrix)), 20)  # cap at 20
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = "RdYlGn" if panel_idx == 1 else "RdYlGn_r"
        # For value MSE: positive d means method better (green), negative means worse (red)
        # For gradient MSE: positive d means method better (green)

        im = ax.imshow(d_matrix, aspect="auto", cmap=cmap, norm=norm)

        # Annotate
        for i in range(n_methods):
            for j in range(n_datasets):
                val = d_matrix[i, j]
                if not np.isnan(val):
                    txt = f"{val:.1f}" if abs(val) < 10 else f"{val:.0f}"
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=5.5, color=color)

        ax.set_xticks(range(n_datasets))
        ax.set_xticklabels([DATASET_SHORT.get(d, d) for d in DATASET_ORDER],
                           rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels([METHOD_SHORT.get(m, m) for m in methods_no_van], fontsize=6.5)
        ax.set_title(metric_title, fontsize=8, fontweight="bold")

    fig.subplots_adjust(left=0.14, right=0.86, wspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Cohen's d", fontsize=8)
    path = out_dir / f"effect_size_heatmap.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality plots for unified comparison"
    )
    parser.add_argument("--mode", default="multi_seed",
                        choices=["smoke_test", "single_seed", "multi_seed"])
    parser.add_argument("--figure", default="all",
                        choices=["all", "gradient_improvement", "pareto",
                                 "ranking_heatmap", "label_comparison",
                                 "bootstrap_ci", "effect_heatmap"])
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--outdir", default="figures/unified",
                        help="Output directory for figures")
    args = parser.parse_args()

    # Load data
    results = load_results(args.mode)
    if not results:
        print(f"No results found for mode '{args.mode}'")
        sys.exit(1)

    print(f"Loaded {len(results)} results (mode: {args.mode})")
    by_dataset = group_by_dataset(results)
    for d in DATASET_ORDER:
        if d in by_dataset:
            n = len(by_dataset[d])
            methods = len(set(r["method"] for r in by_dataset[d]))
            seeds = len(set(r["seed"] for r in by_dataset[d]))
            print(f"  {d}: {n} results, {methods} methods, {seeds} seeds")

    # Output directory
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure dispatch
    figures = {
        "gradient_improvement": fig_gradient_improvement,
        "pareto":              fig_pareto,
        "ranking_heatmap":     fig_ranking_heatmap,
        "label_comparison":    fig_label_comparison,
        "bootstrap_ci":        fig_bootstrap_ci,
        "effect_heatmap":      fig_effect_heatmap,
    }

    if args.figure == "all":
        for name, func in figures.items():
            print(f"\n  Generating: {name}...")
            func(by_dataset, out_dir, args.format)
    else:
        figures[args.figure](by_dataset, out_dir, args.format)

    print(f"\nDone. Figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
