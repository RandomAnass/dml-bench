#!/usr/bin/env python3
"""
Enhanced Figure Generator v2 — DML Benchmark.

Adds publication-critical figures missing from v1:
  - Phase transition diagram (dim × noise, colored by DML advantage)
  - Sample efficiency curves (MSE vs n_samples)
  - All-function heatmaps (not just 2)
  - Gradient MSE comparison
  - Noise crossover visualization
  - Computational Pareto (MSE vs time)
  - Per-seed stability box plots

Usage:
    python create_figures_v2.py --figure all
    python create_figures_v2.py --figure phase_transition
    python create_figures_v2.py --figure sample_efficiency
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from create_figures import (
    setup_style, load_all_results, to_records, COLORS, METHOD_LABELS, NN_METHODS,
    fig_heatmap, fig_scaling, fig_noise, fig_convergence,
    fig_gradnorm, fig_method_comparison, fig_finance,
)

EXTENDED_LABELS = {
    **METHOD_LABELS,
    "baseline_gp": "GP (RBF)",
    "baseline_krr": "KRR (RBF)",
    "baseline_rf": "Random Forest",
}

FUNCS = ["poly_trig", "trig", "step", "bachelier", "black_scholes", "heston"]
FUNC_LABELS = {
    "poly_trig": "Poly-Trig",
    "trig": "Trigonometric",
    "step": "Step",
    "bachelier": "Bachelier",
    "black_scholes": "Black-Scholes",
    "heston": "Heston",
}

OUT_DIR = Path("figures")


def filter_recs(records, **kwargs):
    out = records
    for k, v in kwargs.items():
        out = [r for r in out if r.get(k) == v]
    return out


# ============================================================================
# FIG A: PHASE TRANSITION DIAGRAM
# ============================================================================

def fig_phase_transition(records, fmt="pdf"):
    """2D heatmap: (dim, noise) colored by DML advantage percentage."""
    clean_records = [r for r in records if r["method"] in ["vanilla", "dml_fixed"]
                     and r.get("lambda", 1.0) == 1.0]

    for func in ["poly_trig", "trig", "step"]:
        func_recs = [r for r in clean_records if r["func_type"] == func]
        noise_levels = sorted(set(r["noise_level"] for r in func_recs))
        dims = sorted(set(r["dim"] for r in func_recs))

        if len(noise_levels) < 2 or len(dims) < 2:
            continue

        # Build advantage matrix
        adv_matrix = np.full((len(dims), len(noise_levels)), np.nan)
        for i, dim in enumerate(dims):
            for j, noise in enumerate(noise_levels):
                for ns in [1024]:
                    van = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["noise_level"] == noise
                           and r["n_samples"] == ns and r["method"] == "vanilla"]
                    dml = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["noise_level"] == noise
                           and r["n_samples"] == ns and r["method"] == "dml_fixed"]
                    if van and dml:
                        v_m, d_m = np.mean(van), np.mean(dml)
                        adv_matrix[i, j] = 100 * (v_m - d_m) / v_m if v_m > 0 else 0

        fig, ax = plt.subplots(figsize=(4.5, 3.5))

        # Diverging colormap: green=DML helps, red=DML hurts
        vmax = max(abs(np.nanmax(adv_matrix)), abs(np.nanmin(adv_matrix)))
        vmax = min(vmax, 100)  # cap at 100%
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(adv_matrix, aspect="auto", origin="lower",
                       cmap="RdYlGn", norm=norm,
                       extent=[-0.5, len(noise_levels)-0.5, -0.5, len(dims)-0.5])

        # Add text annotations
        for i in range(len(dims)):
            for j in range(len(noise_levels)):
                val = adv_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.7 else "black"
                    ax.text(j, i, f"{val:+.0f}%", ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold")

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{n:.2f}" for n in noise_levels], rotation=45)
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels([str(d) for d in dims])
        ax.set_xlabel("Noise level (σ)")
        ax.set_ylabel("Dimension (d)")
        ax.set_title(f"DML Advantage — {FUNC_LABELS.get(func, func)}")

        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("DML advantage (%)")

        # Draw the zero contour if possible
        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(np.nan_to_num(adv_matrix), sigma=0.5)
            ax.contour(range(len(noise_levels)), range(len(dims)),
                       smoothed, levels=[0], colors="black",
                       linewidths=1.5, linestyles="--")
        except ImportError:
            pass

        plt.tight_layout()
        fpath = OUT_DIR / f"phase_transition_{func}.{fmt}"
        fig.savefig(fpath)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ============================================================================
# FIG B: SAMPLE EFFICIENCY CURVES
# ============================================================================

def fig_sample_efficiency(records, fmt="pdf"):
    """MSE vs n_samples for each method, per function × dim."""
    clean_records = [r for r in records if r["method"] in NN_METHODS
                     and r.get("lambda", 1.0) == 1.0 and r["noise_level"] == 0.0]

    for func in ["poly_trig", "trig", "bachelier"]:
        func_recs = [r for r in clean_records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))

        if not dims:
            continue

        n_plots = min(len(dims), 6)
        plot_dims = dims[:n_plots]
        fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3), squeeze=False)

        for idx, dim in enumerate(plot_dims):
            ax = axes[0, idx]
            for method in NN_METHODS:
                ns_vals = sorted(set(r["n_samples"] for r in func_recs
                                    if r["dim"] == dim and r["method"] == method))
                means, stds, ns_plotted = [], [], []
                for ns in ns_vals:
                    vals = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns
                           and r["method"] == method]
                    if vals:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
                        ns_plotted.append(ns)

                if means:
                    ax.errorbar(ns_plotted, means, yerr=stds,
                               label=METHOD_LABELS.get(method, method),
                               color=COLORS.get(method, "#333"),
                               marker="o", capsize=2, linewidth=1.0, markersize=3)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("n samples")
            ax.set_title(f"d = {dim}")
            if idx == 0:
                ax.set_ylabel("Value MSE")
            if idx == n_plots - 1:
                ax.legend(fontsize=6, framealpha=0.7)

        fig.suptitle(f"Sample Efficiency — {FUNC_LABELS.get(func, func)}", fontsize=10)
        plt.tight_layout()
        fpath = OUT_DIR / f"sample_efficiency_{func}.{fmt}"
        fig.savefig(fpath)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ============================================================================
# FIG C: ALL-FUNCTION HEATMAP
# ============================================================================

def fig_all_heatmaps(records, fmt="pdf"):
    """DML advantage heatmap for ALL 6 functions (not just 2)."""
    clean_records = [r for r in records if r["method"] in ["vanilla", "dml_fixed"]
                     and r.get("lambda", 1.0) == 1.0 and r["noise_level"] == 0.0]

    funcs_with_data = [f for f in FUNCS
                       if any(r["func_type"] == f for r in clean_records)]

    if len(funcs_with_data) < 2:
        print("  Not enough functions for all-function heatmap")
        return

    n_funcs = len(funcs_with_data)
    fig, axes = plt.subplots(1, n_funcs, figsize=(2.8 * n_funcs, 3.5), squeeze=False)

    for fi, func in enumerate(funcs_with_data):
        ax = axes[0, fi]
        func_recs = [r for r in clean_records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))
        sample_sizes = sorted(set(r["n_samples"] for r in func_recs))

        if not dims or not sample_sizes:
            ax.set_title(FUNC_LABELS.get(func, func))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        adv_matrix = np.full((len(dims), len(sample_sizes)), np.nan)
        for i, dim in enumerate(dims):
            for j, ns in enumerate(sample_sizes):
                van = [r["test_value_mse"] for r in func_recs
                       if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "vanilla"]
                dml = [r["test_value_mse"] for r in func_recs
                       if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "dml_fixed"]
                if van and dml:
                    v_m, d_m = np.mean(van), np.mean(dml)
                    adv_matrix[i, j] = 100 * (v_m - d_m) / v_m if v_m > 0 else 0

        vmax = max(abs(np.nanmax(adv_matrix)), abs(np.nanmin(adv_matrix)), 1)
        vmax = min(vmax, 100)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(adv_matrix, aspect="auto", origin="lower",
                       cmap="RdYlGn", norm=norm)

        for i in range(len(dims)):
            for j in range(len(sample_sizes)):
                val = adv_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                            fontsize=5.5, color=color)

        ax.set_xticks(range(len(sample_sizes)))
        ax.set_xticklabels([str(s) for s in sample_sizes], rotation=45, fontsize=6)
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels([str(d) for d in dims], fontsize=6)
        ax.set_title(FUNC_LABELS.get(func, func), fontsize=8, fontweight="bold")
        if fi == 0:
            ax.set_ylabel("Dimension")
        ax.set_xlabel("n samples")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("DML advantage (%)", fontsize=7)

    fig.suptitle("DML Advantage Across All Functions", fontsize=10, y=1.02)
    fpath = OUT_DIR / f"all_function_heatmap.{fmt}"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fpath}")


# ============================================================================
# FIG D: GRADIENT MSE COMPARISON
# ============================================================================

def fig_gradient_mse(records, fmt="pdf"):
    """Side-by-side: value MSE improvement vs gradient MSE improvement."""
    clean_records = [r for r in records if r["method"] in ["vanilla", "dml_fixed"]
                     and r.get("lambda", 1.0) == 1.0 and r["noise_level"] == 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Scatter: value improvement vs gradient improvement
    val_imps, grad_imps, labels = [], [], []
    for func in FUNCS:
        func_recs = [r for r in clean_records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))
        for dim in dims:
            for ns in [1024]:
                van_val = [r["test_value_mse"] for r in func_recs
                          if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "vanilla"]
                dml_val = [r["test_value_mse"] for r in func_recs
                          if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "dml_fixed"]
                van_grad = [r["test_grad_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "vanilla"]
                dml_grad = [r["test_grad_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns and r["method"] == "dml_fixed"]

                if van_val and dml_val and van_grad and dml_grad:
                    vv, dv = np.mean(van_val), np.mean(dml_val)
                    vg, dg = np.mean(van_grad), np.mean(dml_grad)
                    vi = 100 * (vv - dv) / vv if vv > 0 else 0
                    gi = 100 * (vg - dg) / vg if vg > 0 else 0
                    val_imps.append(vi)
                    grad_imps.append(gi)
                    labels.append(func)

    if val_imps:
        ax = axes[0]
        func_colors = {f: plt.cm.Set2(i) for i, f in enumerate(FUNCS)}
        for f in set(labels):
            mask = [l == f for l in labels]
            ax.scatter([v for v, m in zip(val_imps, mask) if m],
                       [g for g, m in zip(grad_imps, mask) if m],
                       label=FUNC_LABELS.get(f, f), color=func_colors.get(f),
                       s=30, alpha=0.8)

        # Diagonal line
        lims = [min(min(val_imps), min(grad_imps)) - 5,
                max(max(val_imps), max(grad_imps)) + 5]
        ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Value MSE improvement (%)")
        ax.set_ylabel("Gradient MSE improvement (%)")
        ax.set_title("Value vs Gradient Improvement")
        ax.legend(fontsize=6, framealpha=0.7)
        ax.text(0.05, 0.95, "Above diagonal:\ngrad improves more",
                transform=ax.transAxes, fontsize=6, va="top", style="italic", alpha=0.6)

    # Bar: mean improvement by function
    ax = axes[1]
    func_val_means = {}
    func_grad_means = {}
    for f in FUNCS:
        fv = [v for v, l in zip(val_imps, labels) if l == f]
        fg = [g for g, l in zip(grad_imps, labels) if l == f]
        if fv:
            func_val_means[f] = np.mean(fv)
            func_grad_means[f] = np.mean(fg)

    if func_val_means:
        x = np.arange(len(func_val_means))
        width = 0.35
        funcs_ordered = list(func_val_means.keys())
        ax.bar(x - width/2, [func_val_means[f] for f in funcs_ordered],
               width, label="Value MSE", color=COLORS["vanilla"], alpha=0.8)
        ax.bar(x + width/2, [func_grad_means[f] for f in funcs_ordered],
               width, label="Gradient MSE", color=COLORS["dml_fixed"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([FUNC_LABELS.get(f, f) for f in funcs_ordered],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("DML improvement (%)")
        ax.set_title("Mean Improvement by Function")
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fpath = OUT_DIR / f"gradient_mse_analysis.{fmt}"
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  Saved {fpath}")


# ============================================================================
# FIG E: NOISE CROSSOVER VISUALIZATION
# ============================================================================

def fig_noise_crossover(records, fmt="pdf"):
    """Visualize where DML advantage crosses zero as noise increases."""
    clean_records = [r for r in records if r["method"] in ["vanilla", "dml_fixed"]
                     and r.get("lambda", 1.0) == 1.0]

    for func in ["poly_trig", "trig", "step"]:
        func_recs = [r for r in clean_records if r["func_type"] == func]
        noise_levels = sorted(set(r["noise_level"] for r in func_recs))
        dims = sorted(set(r["dim"] for r in func_recs))

        if len(noise_levels) < 2:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))

        dim_colors = plt.cm.viridis(np.linspace(0, 0.9, len(dims)))

        for di, dim in enumerate(dims):
            advantages = []
            used_noise = []
            for noise in noise_levels:
                for ns in [1024]:
                    van = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["noise_level"] == noise
                           and r["n_samples"] == ns and r["method"] == "vanilla"]
                    dml = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["noise_level"] == noise
                           and r["n_samples"] == ns and r["method"] == "dml_fixed"]
                    if van and dml:
                        v_m, d_m = np.mean(van), np.mean(dml)
                        adv = 100 * (v_m - d_m) / v_m if v_m > 0 else 0
                        advantages.append(adv)
                        used_noise.append(noise)

            if advantages:
                ax.plot(used_noise, advantages, "-o", color=dim_colors[di],
                        label=f"d={dim}", markersize=3, linewidth=1.0)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between([min(noise_levels), max(noise_levels)], 0, -300,
                        alpha=0.05, color="red")
        ax.fill_between([min(noise_levels), max(noise_levels)], 0, 300,
                        alpha=0.05, color="green")
        ax.set_xlabel("Noise level (σ)")
        ax.set_ylabel("DML advantage (%)")
        ax.set_title(f"Noise Crossover — {FUNC_LABELS.get(func, func)}")
        ax.legend(fontsize=6, ncol=2, framealpha=0.7)
        ax.text(0.02, 0.02, "DML hurts ↓", transform=ax.transAxes,
                fontsize=7, color="red", alpha=0.6)
        ax.text(0.02, 0.98, "DML helps ↑", transform=ax.transAxes,
                fontsize=7, color="green", alpha=0.6, va="top")

        plt.tight_layout()
        fpath = OUT_DIR / f"noise_crossover_{func}.{fmt}"
        fig.savefig(fpath)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ============================================================================
# FIG F: STABILITY BOX PLOTS
# ============================================================================

def fig_stability(records, fmt="pdf"):
    """Box plots showing per-seed variance."""
    clean_records = [r for r in records if r["method"] in NN_METHODS
                     and r.get("lambda", 1.0) == 1.0 and r["noise_level"] == 0.0]

    for func in ["poly_trig", "trig"]:
        func_recs = [r for r in clean_records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))

        if not dims:
            continue

        n_plots = min(len(dims), 6)
        plot_dims = dims[:n_plots]
        fig, axes = plt.subplots(1, n_plots, figsize=(3.2 * n_plots, 3), squeeze=False)

        for idx, dim in enumerate(plot_dims):
            ax = axes[0, idx]
            data = []
            labels_list = []
            colors_list = []

            for method in NN_METHODS:
                vals = [r["test_value_mse"] for r in func_recs
                       if r["dim"] == dim and r["n_samples"] == 1024
                       and r["method"] == method]
                if vals:
                    data.append(vals)
                    labels_list.append(METHOD_LABELS.get(method, method))
                    colors_list.append(COLORS.get(method, "#333"))

            if data:
                bp = ax.boxplot(data, patch_artist=True, widths=0.6)
                for patch, color in zip(bp["boxes"], colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=6)

            ax.set_title(f"d = {dim}", fontsize=8)
            if idx == 0:
                ax.set_ylabel("Value MSE")

        fig.suptitle(f"Per-Seed Variance — {FUNC_LABELS.get(func, func)}", fontsize=10)
        plt.tight_layout()
        fpath = OUT_DIR / f"stability_{func}.{fmt}"
        fig.savefig(fpath)
        plt.close(fig)
        print(f"  Saved {fpath}")


# ============================================================================
# FIG G: COMPUTATIONAL PARETO
# ============================================================================

def fig_pareto(records, fmt="pdf"):
    """Pareto frontier: MSE vs training time."""
    clean_records = [r for r in records if r["method"] in NN_METHODS
                     and r.get("lambda", 1.0) == 1.0 and r["noise_level"] == 0.0
                     and r.get("time_s", 0) > 0]

    if not clean_records:
        print("  No timing data for Pareto plot")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    for method in NN_METHODS:
        method_recs = [r for r in clean_records if r["method"] == method]
        if not method_recs:
            continue

        # Group by config, take means
        configs = defaultdict(list)
        for r in method_recs:
            key = (r["func_type"], r["dim"], r["n_samples"])
            configs[key].append(r)

        times, mses = [], []
        for key, recs in configs.items():
            times.append(np.mean([r["time_s"] for r in recs]))
            mses.append(np.mean([r["test_value_mse"] for r in recs]))

        ax.scatter(times, mses,
                   label=METHOD_LABELS.get(method, method),
                   color=COLORS.get(method, "#333"),
                   s=15, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Value MSE")
    ax.set_title("Cost-Accuracy Pareto")
    ax.legend(fontsize=7, framealpha=0.7)

    plt.tight_layout()
    fpath = OUT_DIR / f"pareto_cost_accuracy.{fmt}"
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  Saved {fpath}")


# ============================================================================
# FIG H: EXTENDED BASELINES (d=50,100) — DML vs Classical Methods
# ============================================================================

def _load_tier5(subdir):
    """Load results from a tier5 subdirectory."""
    tier_dir = Path(f"results/{subdir}")
    recs = []
    if not tier_dir.exists():
        return recs
    for f in tier_dir.glob("*.json"):
        try:
            with open(f) as fh:
                recs.append(json.load(fh))
        except Exception:
            pass
    return recs


def fig_extended_baselines(records=None, fmt="pdf"):
    """Bar chart: DML vs baselines (KRR, RF) at d=50,100 for poly_trig, trig."""
    raw = _load_tier5("tier5_extended_baselines")
    if not raw:
        print("  No extended baseline data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    func_order = ["poly_trig", "trig"]
    dims = [50, 100]
    methods = ["vanilla", "dml_fixed", "baseline_krr", "baseline_rf"]
    method_labels = ["Vanilla NN", "DML (λ=1)", "KRR (RBF)", "Random Forest"]
    method_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for fi, func in enumerate(func_order):
        ax = axes[fi]
        func_recs = [r for r in raw if r["func_type"] == func]

        x = np.arange(len(dims))
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

        for mi, method in enumerate(methods):
            means, stds = [], []
            for d in dims:
                vals = [r["test_value_mse"] for r in func_recs
                        if r["method"] == method and r["dim"] == d]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)

            bars = ax.bar(x + offsets[mi], means, width, yerr=stds,
                          label=method_labels[mi], color=method_colors[mi],
                          alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in dims])
        ax.set_ylabel("Value MSE")
        ax.set_title(FUNC_LABELS.get(func, func))
        ax.legend(fontsize=7, framealpha=0.7)
        ax.set_yscale("log")

    fig.suptitle("Extended Baselines: DML vs Classical Methods at High Dimensions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = OUT_DIR / f"extended_baselines_comparison.{fmt}"
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  Saved {fpath}")


# ============================================================================
# FIG I: ARCHITECTURE ABLATION — DML Robustness Across Architectures
# ============================================================================

def fig_architecture_ablation(records=None, fmt="pdf"):
    """Grouped bar: vanilla vs DML across small/default/large architectures."""
    raw = _load_tier5("tier5_arch_ablation")
    if not raw:
        print("  No architecture ablation data found")
        return

    archs = ["small", "default", "large"]
    arch_labels = ["Small\n(2L×128H)", "Default\n(4L×256H)", "Large\n(6L×512H)"]
    methods = ["vanilla", "dml_fixed"]
    method_labels = ["Vanilla NN", "DML (λ=1)"]
    method_colors = ["#2196F3", "#FF5722"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: MSE comparison
    ax = axes[0]
    x = np.arange(len(archs))
    width = 0.3

    for mi, method in enumerate(methods):
        means, stds = [], []
        for arch in archs:
            vals = [r["test_value_mse"] for r in raw
                    if r["method"] == method and r.get("arch_name") == arch]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.bar(x + (mi - 0.5) * width, means, width, yerr=stds,
               label=method_labels[mi], color=method_colors[mi],
               alpha=0.85, capsize=4, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, fontsize=9)
    ax.set_ylabel("Value MSE")
    ax.set_title("MSE by Architecture")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Panel B: Improvement percentage
    ax2 = axes[1]
    improvements = []
    for arch in archs:
        van_vals = [r["test_value_mse"] for r in raw
                    if r["method"] == "vanilla" and r.get("arch_name") == arch]
        dml_vals = [r["test_value_mse"] for r in raw
                    if r["method"] == "dml_fixed" and r.get("arch_name") == arch]
        if van_vals and dml_vals:
            imp = (1 - np.mean(dml_vals) / np.mean(van_vals)) * 100
        else:
            imp = 0
        improvements.append(imp)

    bars = ax2.bar(x, improvements, 0.5, color="#FF5722", alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{improvements[i]:.1f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels, fontsize=9)
    ax2.set_ylabel("DML Improvement (%)")
    ax2.set_title("DML Improvement by Architecture")
    ax2.set_ylim(0, 105)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Architecture Ablation: poly_trig, d=10, n=1024",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = OUT_DIR / f"architecture_ablation.{fmt}"
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  Saved {fpath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DML Enhanced Figures v2")
    parser.add_argument("--figure", default="all",
                        choices=["all", "phase_transition", "sample_efficiency",
                                 "all_heatmaps", "gradient_mse", "noise_crossover",
                                 "stability", "pareto",
                                 "extended_baselines", "architecture_ablation",
                                 # v1 figures
                                 "heatmap", "scaling", "noise",
                                 "convergence", "gradnorm",
                                 "method_comparison", "finance"])
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    setup_style()
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_all_results(args.tiers)
    records = to_records(results)
    print(f"Loaded {len(records)} results")

    v2_figures = {
        "phase_transition": fig_phase_transition,
        "sample_efficiency": fig_sample_efficiency,
        "all_heatmaps": fig_all_heatmaps,
        "gradient_mse": fig_gradient_mse,
        "noise_crossover": fig_noise_crossover,
        "stability": fig_stability,
        "pareto": fig_pareto,
        "extended_baselines": fig_extended_baselines,
        "architecture_ablation": fig_architecture_ablation,
    }

    v1_figures = {
        "heatmap": lambda recs, fmt: fig_heatmap(recs, OUT_DIR, fmt),
        "scaling": lambda recs, fmt: fig_scaling(recs, OUT_DIR, fmt),
        "noise": lambda recs, fmt: fig_noise(recs, OUT_DIR, fmt),
        "convergence": lambda recs, fmt: fig_convergence(recs, OUT_DIR, fmt),
        "gradnorm": lambda recs, fmt: fig_gradnorm(recs, OUT_DIR, fmt),
        "method_comparison": lambda recs, fmt: fig_method_comparison(recs, OUT_DIR, fmt),
        "finance": lambda recs, fmt: fig_finance(recs, OUT_DIR, fmt),
    }

    if args.figure == "all":
        # Run all v2 figures
        for name, func in v2_figures.items():
            print(f"\nGenerating {name}...")
            try:
                func(records, args.format)
            except Exception as e:
                import traceback
                print(f"  ⚠️ {name} failed: {e}")
                traceback.print_exc()

        # Also run all v1 figures
        print("\n--- V1 figures ---")
        for name, func in v1_figures.items():
            print(f"\nGenerating {name}...")
            try:
                func(records, args.format)
            except Exception as e:
                print(f"  ⚠️ {name} failed: {e}")
    elif args.figure in v2_figures:
        v2_figures[args.figure](records, args.format)
    elif args.figure in v1_figures:
        v1_figures[args.figure](records, args.format)
    else:
        print(f"Unknown figure: {args.figure}")

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
