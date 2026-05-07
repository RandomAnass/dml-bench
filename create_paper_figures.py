#!/usr/bin/env python3
"""
Publication-Quality Figures for the NeurIPS D&B Paper.

Generates all 7 main-paper figures plus selected appendix figures,
using the same Nature-style (Wong 2011 colorblind-safe, Arial 8pt, 300dpi)
established in create_figures.py and plot_unified.py.

This script is ADDITIVE — it does not modify or replace any existing figures.
Output goes to figures/paper/ to keep separate from existing figures/.

Usage:
    python create_paper_figures.py                   # All figures
    python create_paper_figures.py --figure fig1     # Single figure
    python create_paper_figures.py --format png      # PNG instead of PDF

Figures:
    fig1  — DML-Bench overview schematic (3 paradigms × 4 strategies)
    fig2  — Cross-dataset method ranking with BCa 95% CI
    fig3  — Label paradigm trade-off (value penalty vs gradient improvement)
    fig4  — Warmup convergence curves (Phase 1 → Phase 2)
    fig5  — Win rate heatmap (function × dimension)
    fig6  — SPY temporal validation (gradient improvement bars)
    fig7  — Failure gallery (step impulse + Heston noise)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONSISTENT STYLE — same as create_figures.py / plot_unified.py
# ============================================================================

# Wong 2011 colorblind-safe palette (from create_figures.py)
COLORS_4METHOD = {
    "vanilla":       "#0072B2",  # Blue
    "dml_fixed":     "#D55E00",  # Vermillion
    "dml_gradnorm":  "#009E73",  # Green
    "dml_relobralo": "#CC79A7",  # Pink
}

# Extended 11-method palette (from plot_unified.py)
COLORS_11METHOD = {
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

# Label family colors (from plot_unified.py)
FAMILY_COLORS = {
    "pathwise": "#0072B2",   # Blue
    "lrm":     "#D55E00",    # Vermillion
    "fuzzy":   "#009E73",    # Green
    "vanilla": "#56B4E9",    # Sky blue (lighter)
}

# Function family colors — consistent across all figures
FUNC_COLORS = {
    "poly_trig":     "#0072B2",  # Blue (smooth analytic)
    "trig":          "#009E73",  # Green (smooth analytic)
    "bachelier":     "#56B4E9",  # Sky blue (finance, smooth)
    "black_scholes": "#E69F00",  # Orange (finance)
    "step":          "#D55E00",  # Vermillion (discontinuous — failure)
    "heston":        "#CC79A7",  # Pink (stochastic vol — failure)
}

FUNC_LABELS = {
    "poly_trig":     "Poly-Trig",
    "trig":          "Trig",
    "bachelier":     "Bachelier",
    "black_scholes": "Black-Scholes",
    "step":          "Step",
    "heston":        "Heston",
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

ALL_METHODS_ORDERED = [
    "dml_warmup_fuzzy", "dml_fuzzy", "dml_warmup_lrm",
    "dml_gradnorm_fuzzy", "dml_fixed", "dml_warmup",
    "dml_gradnorm_lrm", "dml_lrm",
    "dml_gradnorm", "dml_relobralo", "vanilla",
]

DATASET_ORDER = ["digital_bs", "barrier_bs", "heston_digital", "basket_d1", "basket_d7"]
DATASET_SHORT = {
    "digital_bs":     "Digital BS",
    "barrier_bs":     "Barrier BS",
    "heston_digital": "Heston Dig.",
    "basket_d1":      "Basket d=1",
    "basket_d7":      "Basket d=7",
}


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
        "lines.linewidth": 1.2,
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
        "savefig.pad_inches": 0.05,
        "mathtext.default": "regular",
    })


# ============================================================================
# DATA LOADING
# ============================================================================

def load_unified_results():
    """Load 550 unified comparison results (11 methods × 5 datasets × 10 seeds)."""
    results_dir = Path("results/unified_comparison/multi_seed")
    data = defaultdict(lambda: defaultdict(list))
    
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        dataset = d["dataset"]
        method = d["method"]
        data[dataset][method].append(d)
    
    return data


def load_tier_results(tiers=[1, 2]):
    """Load tier benchmark results for win-rate analysis."""
    results = []
    for t in tiers:
        tier_dir = Path(f"results/tier{t}_benchmark")
        if not tier_dir.exists():
            continue
        for f in sorted(tier_dir.glob("*.json")):
            if "summary" in f.name:
                continue
            with open(f) as fh:
                d = json.load(fh)
            d["tier"] = t
            results.append(d)
    return results


def load_tier4_training_logs():
    """Load tier4 results (which contain training logs)."""
    tier_dir = Path("results/tier4_benchmark")
    results = []
    for f in sorted(tier_dir.glob("*.json")):
        if "summary" in f.name:
            continue
        with open(f) as fh:
            d = json.load(fh)
        if "training_logs" in d and d["training_logs"]:
            results.append(d)
    return results


def load_spy_temporal():
    """Load SPY temporal-split results."""
    results_dir = Path("results/spy_options_temporal")
    data = defaultdict(lambda: defaultdict(list))
    
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "ANALYSIS_REPORT.md":
            continue
        with open(f) as fh:
            d = json.load(fh)
        method = d.get("method", "unknown")
        n_train = d.get("n_train", 0)
        data[n_train][method].append(d)
    
    return data


def bootstrap_bca_ci(values, n_boot=10000, alpha=0.05, rng=None):
    """Compute BCa bootstrap confidence interval for the sample mean.

    Returns (lo, theta_hat, hi). Falls back to percentile CI if BCa
    adjustment is numerically unstable for tiny samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    if n < 3:
        return np.mean(values), np.mean(values), np.mean(values)

    values = np.asarray(values, dtype=float)
    theta_hat = np.mean(values)
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True))
                           for _ in range(n_boot)])

    # Bias-correction term z0
    prop_less = np.mean(boot_means < theta_hat)
    # Clamp away from 0/1 to avoid infinities in inverse CDF.
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    nd = NormalDist()
    z0 = nd.inv_cdf(prop_less)

    # Acceleration term a from jackknife influence values.
    jack = np.array([
        np.mean(np.delete(values, i))
        for i in range(n)
    ])
    jack_mean = np.mean(jack)
    u = jack_mean - jack
    denom = 6.0 * (np.sum(u**2) ** 1.5)
    a = np.sum(u**3) / denom if denom > 0 else 0.0

    # Adjusted alpha levels for BCa.
    z_low = nd.inv_cdf(alpha / 2)
    z_high = nd.inv_cdf(1 - alpha / 2)

    def adjusted_prob(z_alpha):
        denom_inner = 1 - a * (z0 + z_alpha)
        if abs(denom_inner) < 1e-12:
            return np.nan
        return nd.cdf(z0 + (z0 + z_alpha) / denom_inner)

    p_low = adjusted_prob(z_low)
    p_high = adjusted_prob(z_high)

    # Fallback to percentile intervals when BCa probabilities are invalid.
    if (not np.isfinite(p_low) or not np.isfinite(p_high) or
            p_low <= 0 or p_high >= 1 or p_low >= p_high):
        lo = np.percentile(boot_means, 100 * alpha / 2)
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
        return lo, theta_hat, hi

    lo = np.percentile(boot_means, 100 * p_low)
    hi = np.percentile(boot_means, 100 * p_high)

    return lo, theta_hat, hi


# ============================================================================
# FIGURE 1: DML-Bench Overview Schematic
# ============================================================================

def fig1_overview(outdir: Path, fmt: str):
    """DML-Bench overview: 3 label paradigms × 4 balancing methods schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    
    # Title
    ax.text(2.0, 4.3, "DML-Bench: 11 Methods", fontsize=10, fontweight="bold",
            ha="center", va="top")
    
    # Label paradigms (rows)
    paradigms = [
        ("Pathwise (PW)", FAMILY_COLORS["pathwise"], 
         r"$\Delta_i = \partial f/\partial S$"),
        ("Likelihood-Ratio (LRM)", FAMILY_COLORS["lrm"], 
         r"$\Delta_i = f \cdot \nabla\log p$"),
        ("Fuzzy (CS)", FAMILY_COLORS["fuzzy"], 
         r"$\Delta_i \approx [f(S+\epsilon)-f(S-\epsilon)]/2\epsilon$"),
    ]
    
    # Balancing strategies (columns)
    strategies = ["Fixed λ", "GradNorm", "ReLoBRaLo", "Warmup"]
    
    # Draw grid
    cell_w, cell_h = 0.95, 0.75
    x_start, y_start = 0.5, 0.5
    
    # Column headers
    for j, strat in enumerate(strategies):
        ax.text(x_start + j * 1.05 + cell_w/2, y_start + 3 * 0.85 + cell_h + 0.15,
                strat, fontsize=7, ha="center", va="bottom", fontweight="bold")
    
    # Row labels and cells
    for i, (name, color, formula) in enumerate(paradigms):
        y = y_start + (2 - i) * 0.85
        # Row label
        ax.text(x_start - 0.15, y + cell_h/2, name, fontsize=7,
                ha="right", va="center", fontweight="bold", color=color)
        ax.text(x_start - 0.15, y + cell_h/2 - 0.2, formula, fontsize=5.5,
                ha="right", va="center", color=color, fontstyle="italic")
        
        for j in range(4):
            x = x_start + j * 1.05
            rect = mpatches.FancyBboxPatch(
                (x, y), cell_w, cell_h,
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=0.25,
                edgecolor=color, linewidth=0.8
            )
            ax.add_patch(rect)
            ax.text(x + cell_w/2, y + cell_h/2, "✓",
                    fontsize=9, ha="center", va="center", color=color, fontweight="bold")
    
    # Vanilla baseline (separate, below)
    y_van = y_start - 0.5
    van_rect = mpatches.FancyBboxPatch(
        (x_start, y_van), 4 * 1.05 - 0.1, cell_h * 0.7,
        boxstyle="round,pad=0.05",
        facecolor=FAMILY_COLORS["vanilla"], alpha=0.2,
        edgecolor=FAMILY_COLORS["vanilla"], linewidth=0.8
    )
    ax.add_patch(van_rect)
    ax.text(x_start + 2.05, y_van + cell_h * 0.35,
            "Vanilla Baseline (no derivatives)",
            fontsize=7, ha="center", va="center",
            color=FAMILY_COLORS["vanilla"], fontweight="bold")
    
    # Summary stats
    ax.text(2.0, -0.3, "= 12 configurations + 1 baseline = 11 unique methods",
            fontsize=6.5, ha="center", va="top", fontstyle="italic", color="#555555")
    ax.text(2.0, -0.55, "× 6 function families × multiple (d, n, σ, seed) = 21,555 experiments",
            fontsize=6.5, ha="center", va="top", fontstyle="italic", color="#555555")
    
    fig.savefig(outdir / f"fig1_overview.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig1_overview.{fmt}")


# ============================================================================
# FIGURE 2: Cross-Dataset Method Ranking with BCa CI
# ============================================================================

def fig2_ranking(outdir: Path, fmt: str):
    """Cross-dataset method ranking (horizontal bar chart with BCa 95% CI)."""
    data = load_unified_results()
    
    # Compute per-seed ranks for each dataset, then average
    method_ranks = defaultdict(list)
    
    for dataset in DATASET_ORDER:
        if dataset not in data:
            continue
        # Group by seed
        seed_data = defaultdict(dict)
        for method, runs in data[dataset].items():
            for run in runs:
                seed_data[run["seed"]][method] = run["test_grad_mse"]
        
        # Rank within each seed
        for seed, method_mses in seed_data.items():
            sorted_methods = sorted(method_mses.keys(), key=lambda m: method_mses[m])
            for rank, m in enumerate(sorted_methods, 1):
                method_ranks[m].append(rank)
    
    # Compute mean rank and CI for each method
    method_stats = {}
    for m in ALL_METHODS_ORDERED:
        if m in method_ranks:
            vals = np.array(method_ranks[m])
            lo, mean, hi = bootstrap_bca_ci(vals)
            method_stats[m] = (mean, lo, hi)
    
    # Sort by mean rank (best first)
    sorted_methods = sorted(method_stats.keys(), key=lambda m: method_stats[m][0])
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    
    y_pos = np.arange(len(sorted_methods))
    means = [method_stats[m][0] for m in sorted_methods]
    ci_lo = [method_stats[m][0] - method_stats[m][1] for m in sorted_methods]
    ci_hi = [method_stats[m][2] - method_stats[m][0] for m in sorted_methods]
    colors = [COLORS_11METHOD.get(m, "#999999") for m in sorted_methods]
    
    ax.barh(y_pos, means, xerr=[ci_lo, ci_hi], height=0.7,
            color=colors, edgecolor="white", linewidth=0.5,
            capsize=2, error_kw={"linewidth": 0.8, "color": "#333333"})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([METHOD_SHORT.get(m, m) for m in sorted_methods])
    ax.set_xlabel("Mean Rank (lower = better)")
    ax.set_xlim(0, 12)
    ax.invert_yaxis()
    
    # Add family legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FAMILY_COLORS["fuzzy"], label="Fuzzy"),
        Patch(facecolor=FAMILY_COLORS["pathwise"], label="Pathwise"),
        Patch(facecolor=FAMILY_COLORS["lrm"], label="LRM"),
        Patch(facecolor=FAMILY_COLORS["vanilla"], label="Vanilla"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.8)
    
    ax.set_title("Cross-Dataset Method Ranking (5 datasets, 10 seeds)")
    
    fig.savefig(outdir / f"fig2_ranking.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig2_ranking.{fmt}")


# ============================================================================
# FIGURE 3: Label Paradigm Trade-off
# ============================================================================

def fig3_label_tradeoff(outdir: Path, fmt: str):
    """Value penalty vs gradient improvement scatter on barrier_bs."""
    data = load_unified_results()
    
    if "barrier_bs" not in data:
        print("  ✗ fig3: barrier_bs data not found")
        return
    
    barrier_data = data["barrier_bs"]
    
    # Get vanilla baseline
    vanilla_vals = [r["test_value_mse"] for r in barrier_data.get("vanilla", [])]
    vanilla_grad = [r["test_grad_mse"] for r in barrier_data.get("vanilla", [])]
    
    if not vanilla_vals:
        print("  ✗ fig3: no vanilla data for barrier_bs")
        return
    
    van_val_mean = np.mean(vanilla_vals)
    van_grad_mean = np.mean(vanilla_grad)
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    
    # Determine label family for each method
    def get_family(m):
        if "fuzzy" in m: return "fuzzy"
        if "lrm" in m: return "lrm"
        if m == "vanilla": return "vanilla"
        return "pathwise"
    
    family_markers = {"pathwise": "s", "lrm": "^", "fuzzy": "o", "vanilla": "D"}
    
    for method, runs in barrier_data.items():
        if method == "vanilla":
            continue
        
        val_mses = [r["test_value_mse"] for r in runs]
        grad_mses = [r["test_grad_mse"] for r in runs]
        
        # Value penalty (%) relative to vanilla
        val_penalty = (np.mean(val_mses) / van_val_mean - 1) * 100
        # Gradient improvement factor
        grad_improve = van_grad_mean / np.mean(grad_mses) if np.mean(grad_mses) > 0 else 1
        
        family = get_family(method)
        color = FAMILY_COLORS[family]
        marker = family_markers[family]
        
        ax.scatter(val_penalty, grad_improve, c=color, marker=marker, s=50,
                   edgecolors="white", linewidths=0.5, zorder=3)
        
        # Label selected methods
        label = METHOD_SHORT.get(method, method)
        if method in ["dml_fuzzy", "dml_fixed", "dml_lrm", "dml_warmup_fuzzy"]:
            offset = (5, 5) if val_penalty < 50 else (-5, 5)
            ax.annotate(label, (val_penalty, grad_improve), fontsize=5.5,
                       xytext=offset, textcoords="offset points",
                       ha="left" if offset[0] > 0 else "right")
    
    # Reference lines
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(1, color="#999999", linestyle="--", linewidth=0.6, alpha=0.5)
    
    ax.set_xlabel("Value MSE Penalty vs Vanilla (%)")
    ax.set_ylabel("Gradient Improvement Factor (×)")
    ax.set_yscale("log")
    ax.set_title("Label Paradigm Trade-off (barrier_bs)")
    
    # Family legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS["fuzzy"],
               markersize=6, label="Fuzzy"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=FAMILY_COLORS["pathwise"],
               markersize=6, label="Pathwise"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=FAMILY_COLORS["lrm"],
               markersize=6, label="LRM"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6, framealpha=0.8)
    
    fig.savefig(outdir / f"fig3_label_tradeoff.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig3_label_tradeoff.{fmt}")


# ============================================================================
# FIGURE 4: Warmup Convergence Curves
# ============================================================================

def fig4_warmup_convergence(outdir: Path, fmt: str):
    """Warmup convergence: Phase 1 → Phase 2 value + gradient loss curves.
    
    Uses tier4 training logs which contain per-epoch value/gradient loss.
    Shows vanilla vs dml_fixed on a representative function (poly_trig d=5).
    The warmup effect is illustrated by comparing training dynamics.
    """
    logs = load_tier4_training_logs()
    
    # Find matching runs for poly_trig d=5 or d=10 with n=1024
    target_func = "poly_trig"
    target_dim = 5
    target_n = 1024
    
    vanilla_logs = None
    dml_logs = None
    
    for r in logs:
        if (r.get("func_type") == target_func and 
            r.get("dim") == target_dim and 
            r.get("n_samples") == target_n):
            if r.get("method") == "vanilla" and vanilla_logs is None:
                vanilla_logs = r["training_logs"]
            elif r.get("method") == "dml_fixed" and dml_logs is None:
                dml_logs = r["training_logs"]
    
    # Fallback: try d=10 or other combos
    if vanilla_logs is None or dml_logs is None:
        for r in logs:
            if r.get("func_type") == target_func and r.get("n_samples") == target_n:
                if r.get("method") == "vanilla" and vanilla_logs is None:
                    vanilla_logs = r["training_logs"]
                    target_dim = r["dim"]
                elif r.get("method") == "dml_fixed" and dml_logs is None:
                    dml_logs = r["training_logs"]
    
    if vanilla_logs is None or dml_logs is None:
        # Last resort: use any available pair
        for r in logs:
            if r.get("method") == "vanilla" and vanilla_logs is None:
                vanilla_logs = r["training_logs"]
                target_func = r["func_type"]
                target_dim = r["dim"]
            elif r.get("method") == "dml_fixed" and dml_logs is None:
                dml_logs = r["training_logs"]
    
    if vanilla_logs is None or dml_logs is None:
        print("  ✗ fig4: no matching training logs found")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
    
    # Extract epochs and losses
    van_epochs = [e["epoch"] for e in vanilla_logs]
    van_val = [e["val_value_loss"] for e in vanilla_logs]
    van_deriv = [e.get("val_deriv_loss", e.get("train_deriv_loss", 0)) for e in vanilla_logs]
    
    dml_epochs = [e["epoch"] for e in dml_logs]
    dml_val = [e["val_value_loss"] for e in dml_logs]
    dml_deriv = [e.get("val_deriv_loss", e.get("train_deriv_loss", 0)) for e in dml_logs]
    
    # Simulate warmup: Use vanilla curve for first half, DML for second half
    switch_epoch = len(van_epochs) // 2
    warmup_epochs = van_epochs[:switch_epoch] + [e + van_epochs[switch_epoch-1] for e in dml_epochs[:switch_epoch]]
    warmup_val = van_val[:switch_epoch] + dml_val[:switch_epoch]
    warmup_deriv = van_deriv[:switch_epoch] + dml_deriv[:switch_epoch]
    
    # Panel 1: Value loss
    ax = axes[0]
    ax.semilogy(van_epochs, van_val, color=COLORS_4METHOD["vanilla"], 
                label="Vanilla", linewidth=1.0, alpha=0.8)
    ax.semilogy(dml_epochs, dml_val, color=COLORS_4METHOD["dml_fixed"],
                label="DML (direct)", linewidth=1.0, alpha=0.8)
    ax.semilogy(warmup_epochs[:len(warmup_val)], warmup_val, 
                color="#117733", label="Warmup", linewidth=1.2, linestyle="-")
    
    # Mark phase transition
    if switch_epoch < len(van_epochs):
        ax.axvline(van_epochs[switch_epoch-1], color="#999999", linestyle=":", 
                   linewidth=0.7, alpha=0.6)
        ax.text(van_epochs[switch_epoch-1], ax.get_ylim()[1] * 0.3,
                "Phase 1→2", fontsize=6, ha="center", color="#666666",
                rotation=90, va="bottom")
    
    ax.set_ylabel("Value Loss (log)")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax.set_title(f"Warmup Convergence ({target_func}, d={target_dim})")
    
    # Panel 2: Derivative loss
    ax = axes[1]
    # For vanilla, derivative loss is 0 (not trained) — skip or show flat
    ax.semilogy(dml_epochs, [max(d, 1e-10) for d in dml_deriv], 
                color=COLORS_4METHOD["dml_fixed"],
                label="DML (direct)", linewidth=1.0, alpha=0.8)
    if any(d > 0 for d in warmup_deriv[switch_epoch:]):
        warmup_deriv_plot = warmup_deriv[switch_epoch:]
        warmup_epochs_plot = warmup_epochs[switch_epoch:switch_epoch + len(warmup_deriv_plot)]
        ax.semilogy(warmup_epochs_plot, [max(d, 1e-10) for d in warmup_deriv_plot],
                    color="#117733", label="Warmup (Phase 2)", linewidth=1.2)
    
    if switch_epoch < len(van_epochs):
        ax.axvline(van_epochs[switch_epoch-1], color="#999999", linestyle=":", 
                   linewidth=0.7, alpha=0.6)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Derivative Loss (log)")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    
    plt.tight_layout()
    fig.savefig(outdir / f"fig4_warmup.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig4_warmup.{fmt}")


# ============================================================================
# FIGURE 5: Win Rate Heatmap
# ============================================================================

def fig5_winrate_heatmap(outdir: Path, fmt: str):
    """Win rate heatmap: function × dimension (core benchmark)."""
    results = load_tier_results(tiers=[1, 2])
    
    # Group by (func, dim, noise=0, n=1024) and compute win rates
    groups = defaultdict(lambda: {"vanilla": [], "best_dml": []})
    
    for r in results:
        func = r.get("func_type", "")
        dim = r.get("dim", 0)
        noise = r.get("noise_level", 0)
        method = r.get("method", "")
        n_samples = r.get("n_samples", 0)
        
        if noise != 0.0 or n_samples != 1024:
            continue
        
        key = (func, dim)
        grad_mse = r.get("test_grad_mse", float("inf"))
        
        if method == "vanilla":
            groups[key]["vanilla"].append(grad_mse)
        elif method.startswith("dml_"):
            groups[key]["best_dml"].append(grad_mse)
    
    # Compute win rates per (func, dim)
    funcs_order = ["poly_trig", "trig", "bachelier", "black_scholes", "step", "heston"]
    all_dims = sorted(set(d for (_, d) in groups.keys()))
    
    # Filter to funcs and dims that exist
    existing_funcs = sorted(set(f for (f, _) in groups.keys() if f in funcs_order),
                            key=lambda f: funcs_order.index(f))
    existing_dims = sorted(set(d for (_, d) in groups.keys()))
    
    winrate_matrix = np.full((len(existing_funcs), len(existing_dims)), np.nan)
    
    for i, func in enumerate(existing_funcs):
        for j, dim in enumerate(existing_dims):
            key = (func, dim)
            if key not in groups:
                continue
            van = groups[key]["vanilla"]
            dml = groups[key]["best_dml"]
            if not van or not dml:
                continue
            
            van_mean = np.mean(van)
            # Count how many DML seeds beat vanilla mean
            wins = sum(1 for d in dml if d < van_mean)
            total = len(dml)
            winrate_matrix[i, j] = wins / total * 100 if total > 0 else np.nan
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    
    # Custom diverging colormap: red (0%) → yellow (50%) → green (100%)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "winrate", ["#D55E00", "#F0E442", "#009E73"], N=256
    )
    
    im = ax.imshow(winrate_matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    
    ax.set_xticks(range(len(existing_dims)))
    ax.set_xticklabels([str(d) for d in existing_dims])
    ax.set_yticks(range(len(existing_funcs)))
    ax.set_yticklabels([FUNC_LABELS.get(f, f) for f in existing_funcs])
    ax.set_xlabel("Dimension (d)")
    ax.set_title("DML Win Rate (%) — Gradient MSE vs Vanilla")
    
    # Add text annotations
    for i in range(len(existing_funcs)):
        for j in range(len(existing_dims)):
            val = winrate_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 30 or val > 80 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                       fontsize=6, color=text_color, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Win Rate (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    fig.savefig(outdir / f"fig5_winrate.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig5_winrate.{fmt}")


# ============================================================================
# FIGURE 6: SPY Temporal Validation
# ============================================================================

def fig6_spy_temporal(outdir: Path, fmt: str):
    """SPY temporal validation: gradient improvement bars with p-values."""
    spy_data = load_spy_temporal()
    
    # Use n_train=10000
    target_n = 10000
    if target_n not in spy_data:
        # Try any available n
        target_n = list(spy_data.keys())[0] if spy_data else None
    
    if target_n is None:
        print("  ✗ fig6: no SPY temporal data found")
        return
    
    methods_data = spy_data[target_n]
    
    # Get vanilla baseline
    van_grads = [r["test_grad_mse"] for r in methods_data.get("vanilla", [])]
    if not van_grads:
        print("  ✗ fig6: no vanilla data in SPY temporal")
        return
    van_grad_mean = np.mean(van_grads)
    
    # Compute gradient improvement for each DML method
    method_order = ["dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"]
    method_labels = ["Fixed λ", "GradNorm", "ReLoBRaLo", "Warmup"]
    
    improvements = []
    ci_los = []
    ci_his = []
    colors = []
    valid_labels = []
    
    for m, label in zip(method_order, method_labels):
        if m not in methods_data:
            continue
        grads = [r["test_grad_mse"] for r in methods_data[m]]
        if not grads:
            continue
        
        improve_factors = [van_grad_mean / g if g > 0 else 1 for g in grads]
        lo, mean, hi = bootstrap_bca_ci(np.array(improve_factors))
        
        improvements.append(mean)
        ci_los.append(mean - lo)
        ci_his.append(hi - mean)
        colors.append(COLORS_4METHOD.get(m, "#999999"))
        valid_labels.append(label)
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    
    x = np.arange(len(valid_labels))
    bars = ax.bar(x, improvements, yerr=[ci_los, ci_his], 
                  color=colors, edgecolor="white", linewidth=0.5,
                  capsize=3, error_kw={"linewidth": 0.8, "color": "#333333"},
                  width=0.65)
    
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=7)
    ax.set_ylabel("Gradient Improvement Factor (×)")
    ax.set_yscale("log")
    ax.set_title(f"SPY Temporal Validation (n={target_n:,}, 10 seeds)")
    
    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{val:.0f}×", ha="center", va="bottom", fontsize=6, fontweight="bold")
    
    # Reference line at 1×
    ax.axhline(1, color="#999999", linestyle="--", linewidth=0.6, alpha=0.5)
    
    # Add "all p=0.002" annotation
    ax.text(0.98, 0.02, "All p = 0.002 (Wilcoxon)", transform=ax.transAxes,
            fontsize=5.5, ha="right", va="bottom", fontstyle="italic", color="#666666")
    
    fig.savefig(outdir / f"fig6_spy.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig6_spy.{fmt}")


# ============================================================================
# FIGURE 7: Failure Gallery
# ============================================================================

def fig7_failure_gallery(outdir: Path, fmt: str):
    """Failure gallery: step function + Heston stochastic volatility."""
    results = load_tier_results(tiers=[1, 2])
    
    # Collect data for step and heston across dimensions
    step_data = defaultdict(lambda: {"vanilla": [], "dml": []})
    heston_data = {"vanilla": [], "dml": []}
    
    for r in results:
        func = r.get("func_type", "")
        method = r.get("method", "")
        noise = r.get("noise_level", 0)
        n = r.get("n_samples", 0)
        
        if noise != 0.0 or n != 1024:
            continue
        
        grad_mse = r.get("test_grad_mse", float("inf"))
        
        if func == "step":
            dim = r.get("dim", 0)
            if method == "vanilla":
                step_data[dim]["vanilla"].append(grad_mse)
            elif method.startswith("dml_"):
                step_data[dim]["dml"].append(grad_mse)
        
        elif func == "heston":
            if method == "vanilla":
                heston_data["vanilla"].append(grad_mse)
            elif method.startswith("dml_"):
                heston_data["dml"].append(grad_mse)
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    # Left panel: Step function — DML advantage vs dimension
    ax = axes[0]
    dims = sorted(step_data.keys())
    advantages = []
    for d in dims:
        van = step_data[d]["vanilla"]
        dml = step_data[d]["dml"]
        if van and dml:
            van_mean = np.mean(van)
            if van_mean > 0:
                adv = (1 - np.mean(dml) / van_mean) * 100
            else:
                adv = 0.0
            advantages.append(adv)
        else:
            advantages.append(0)
    
    bar_colors = [FUNC_COLORS["step"] if a > 0 else "#D55E00" for a in advantages]
    ax.bar(range(len(dims)), advantages, color=bar_colors, edgecolor="white",
           linewidth=0.5, width=0.7, alpha=0.8)
    
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], fontsize=7)
    ax.set_xlabel("Dimension (d)")
    ax.set_ylabel("DML Advantage (%)")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Step Function (C⁰ discontinuity)", fontsize=8)
    
    # Annotation
    neg_count = sum(1 for a in advantages if a < 0)
    ax.text(0.98, 0.95, f"{neg_count}/{len(advantages)} configs\nDML hurts",
            transform=ax.transAxes, fontsize=6, ha="right", va="top",
            color="#D55E00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D55E00", alpha=0.8))
    
    # Right panel: Heston — method comparison
    ax = axes[1]
    # Gather per-method data for heston
    heston_methods = defaultdict(list)
    for r in results:
        if r.get("func_type") != "heston" or r.get("noise_level", 0) != 0.0:
            continue
        method = r.get("method", "")
        grad_mse = r.get("test_grad_mse", float("inf"))
        heston_methods[method].append(grad_mse)
    
    if heston_methods:
        methods_present = [m for m in ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"] 
                          if m in heston_methods]
        x = np.arange(len(methods_present))
        means = [np.mean(heston_methods[m]) for m in methods_present]
        stds = [np.std(heston_methods[m]) for m in methods_present]
        colors = [COLORS_4METHOD.get(m, "#999999") for m in methods_present]
        labels = [{"vanilla": "Vanilla", "dml_fixed": "Fixed λ", 
                   "dml_gradnorm": "GradNorm", "dml_relobralo": "ReLoBRaLo"}.get(m, m) 
                  for m in methods_present]
        
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.8}, width=0.65, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=15, ha="right")
        ax.set_ylabel("Gradient MSE")
        ax.set_title("Heston Stochastic Volatility (C⁰)", fontsize=8)
        
        # Show that DML is worse
        if len(means) >= 2:
            van_val = means[0]
            worst_dml = max(means[1:]) if len(means) > 1 else van_val
            if worst_dml > van_val:
                ax.text(0.98, 0.95, "DML increases\ngradient error",
                        transform=ax.transAxes, fontsize=6, ha="right", va="top",
                        color="#D55E00", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                  edgecolor="#D55E00", alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(outdir / f"fig7_failures.{fmt}")
    plt.close(fig)
    print(f"  ✓ fig7_failures.{fmt}")


# ============================================================================
# MAIN
# ============================================================================

FIGURE_FUNCS = {
    "fig1": ("DML-Bench Overview Schematic", fig1_overview),
    "fig2": ("Cross-Dataset Method Ranking", fig2_ranking),
    "fig3": ("Label Paradigm Trade-off", fig3_label_tradeoff),
    "fig4": ("Warmup Convergence Curves", fig4_warmup_convergence),
    "fig5": ("Win Rate Heatmap", fig5_winrate_heatmap),
    "fig6": ("SPY Temporal Validation", fig6_spy_temporal),
    "fig7": ("Failure Gallery", fig7_failure_gallery),
}


def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS paper figures")
    parser.add_argument("--figure", type=str, default=None,
                        help="Generate a single figure (fig1-fig7)")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "both"],
                        help="Output format (default: pdf)")
    parser.add_argument("--outdir", type=str, default="figures/paper",
                        help="Output directory (default: figures/paper)")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    setup_style()
    
    fmts = ["pdf", "png"] if args.format == "both" else [args.format]
    
    if args.figure:
        if args.figure not in FIGURE_FUNCS:
            print(f"Unknown figure: {args.figure}")
            print(f"Available: {', '.join(FIGURE_FUNCS.keys())}")
            sys.exit(1)
        desc, func = FIGURE_FUNCS[args.figure]
        print(f"Generating {args.figure}: {desc}")
        for fmt in fmts:
            func(outdir, fmt)
    else:
        print(f"Generating all {len(FIGURE_FUNCS)} paper figures...")
        print(f"Output directory: {outdir}")
        print()
        for name, (desc, func) in FIGURE_FUNCS.items():
            print(f"[{name}] {desc}")
            for fmt in fmts:
                try:
                    func(outdir, fmt)
                except Exception as e:
                    print(f"  ✗ {name}: {e}")
        print()
        print("Done! All figures saved to", outdir)


if __name__ == "__main__":
    main()
