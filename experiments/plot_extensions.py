#!/usr/bin/env python3
"""
Publication-Quality Figures for Extension Experiments (LRM, SPY, Warmup).

Covers:
  1. spy_pareto        — SPY Pareto scatter: value penalty vs gradient improvement
  2. spy_method_bars   — SPY grouped bar chart (value MSE + gradient MSE)
  3. heston_degradation — Heston value MSE vs N_steps line plot (LRM variance growth)
  4. warmup_spy        — Warmup vs vanilla vs GradNorm on SPY
  5. warmup_heston     — Warmup vs vanilla vs GradNorm on Heston configs
  6. lrm_variance      — LRM variance scaling with dimension (basket)

Usage:
    python experiments/plot_extensions.py                        # all figures
    python experiments/plot_extensions.py --figure spy_pareto    # single figure
    python experiments/plot_extensions.py --format png           # PNG output
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dml_benchmark.stats import bootstrap_ci


# ============================================================================
# CONSTANTS
# ============================================================================

METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
WARMUP_METHOD = "dml_warmup"

METHOD_LABELS = {
    "vanilla":       "Vanilla",
    "dml_fixed":     "Fixed λ (G&K)",
    "dml_gradnorm":  "GradNorm",
    "dml_relobralo": "ReLoBRaLo",
    "dml_warmup":    "Warmup",
}

COLORS = {
    "vanilla":       "#56B4E9",
    "dml_fixed":     "#0072B2",
    "dml_gradnorm":  "#D55E00",
    "dml_relobralo": "#CC79A7",
    "dml_warmup":    "#009E73",
}

MARKERS = {
    "vanilla":       "o",
    "dml_fixed":     "s",
    "dml_gradnorm":  "D",
    "dml_relobralo": "^",
    "dml_warmup":    "P",
}


# ============================================================================
# STYLE
# ============================================================================

def setup_style():
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
        "lines.markersize": 5,
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

def load_results(directory):
    results = {}
    d = Path(directory)
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


def load_all():
    lrm = load_results("results/lrm_comparison")
    spy = load_results("results/spy_options")
    warmup = load_results("results/warmup_experiments")
    return lrm, spy, warmup


def group_by(records, *keys):
    groups = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) for k in keys)
        if len(keys) == 1:
            key = key[0]
        groups[key].append(r)
    return dict(groups)


# ============================================================================
# FIGURE 1: SPY PARETO SCATTER
# ============================================================================

def fig_spy_pareto(spy, out_dir, fmt="pdf"):
    """Value penalty (%) vs gradient improvement (×) on SPY data."""
    setup_style()
    spy_list = list(spy.values())
    by_ntrain = group_by(spy_list, "n_train")

    fig, axes = plt.subplots(1, len(by_ntrain), figsize=(4 * len(by_ntrain), 3.5),
                             squeeze=False)

    for col, n_train in enumerate(sorted(by_ntrain.keys())):
        ax = axes[0, col]
        recs = by_ntrain[n_train]
        by_method = group_by(recs, "method")

        van_val = np.mean([r["test_value_mse"] for r in by_method.get("vanilla", [])])
        van_grad = np.mean([r["test_grad_mse"] for r in by_method.get("vanilla", [])])

        # 10% value penalty zone
        ax.axvspan(-20, 10, alpha=0.06, color="green", zorder=0)
        ax.axvline(x=10, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

        for m in METHODS:
            if m not in by_method:
                continue
            mv = np.mean([r["test_value_mse"] for r in by_method[m]])
            mg = np.mean([r["test_grad_mse"] for r in by_method[m]])
            vp = (mv - van_val) / van_val * 100
            gi = van_grad / mg if mg > 0 else 1

            ax.plot(vp, gi, marker=MARKERS.get(m, "o"), color=COLORS.get(m, "#999"),
                    markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                    label=METHOD_LABELS.get(m, m), zorder=10)

            # Annotate
            ax.annotate(METHOD_LABELS.get(m, m), (vp, gi),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=6, color=COLORS.get(m, "#999"))

        ax.set_yscale("log")
        ax.set_xlabel("Value penalty (%)", fontsize=8)
        if col == 0:
            ax.set_ylabel("Gradient improvement (×)", fontsize=8)
        ax.set_title(f"SPY Options (n={n_train:,})", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    path = out_dir / f"spy_pareto.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 2: SPY METHOD BARS
# ============================================================================

def fig_spy_method_bars(spy, out_dir, fmt="pdf"):
    """Grouped bar chart: value MSE and gradient MSE on SPY, per n_train."""
    setup_style()
    spy_list = list(spy.values())
    by_ntrain = group_by(spy_list, "n_train")

    fig, axes = plt.subplots(2, len(by_ntrain), figsize=(4 * len(by_ntrain), 5),
                             squeeze=False)

    for col, n_train in enumerate(sorted(by_ntrain.keys())):
        recs = by_ntrain[n_train]
        by_method = group_by(recs, "method")

        for row, (metric, label) in enumerate([
            ("test_value_mse", "Value MSE"),
            ("test_grad_mse", "Gradient MSE"),
        ]):
            ax = axes[row, col]
            x = np.arange(len(METHODS))
            means = []
            stds = []
            colors = []
            for m in METHODS:
                if m in by_method:
                    arr = np.array([r[metric] for r in by_method[m]])
                    means.append(np.mean(arr))
                    stds.append(np.std(arr))
                else:
                    means.append(np.nan)
                    stds.append(0)
                colors.append(COLORS.get(m, "#999"))

            ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
                   linewidth=0.4, capsize=3, width=0.65)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in METHODS],
                               rotation=30, ha="right", fontsize=6.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == 0:
                ax.set_title(f"SPY (n={n_train:,})", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = out_dir / f"spy_method_comparison.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 3: HESTON DEGRADATION
# ============================================================================

def fig_heston_degradation(lrm, out_dir, fmt="pdf"):
    """Line plot: value MSE vs N_steps for each method, both payoff types."""
    setup_style()
    heston = [r for r in lrm.values() if "heston" in r.get("key", "")]
    if not heston:
        print("  No Heston results found — skipping.")
        return

    by_payoff = group_by(heston, "payoff")
    fig, axes = plt.subplots(1, len(by_payoff), figsize=(4.5 * len(by_payoff), 3.5),
                             squeeze=False)

    for col, payoff in enumerate(sorted(by_payoff.keys())):
        ax = axes[0, col]
        recs = by_payoff[payoff]
        by_steps = group_by(recs, "n_steps")
        steps = sorted(by_steps.keys())

        for m in METHODS:
            y_vals = []
            y_errs = []
            x_vals = []
            for ns in steps:
                by_method = group_by(by_steps[ns], "method")
                if m in by_method:
                    arr = np.array([r["test_value_mse"] for r in by_method[m]])
                    y_vals.append(np.mean(arr))
                    y_errs.append(np.std(arr))
                    x_vals.append(ns)

            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                            marker=MARKERS.get(m, "o"), color=COLORS.get(m, "#999"),
                            label=METHOD_LABELS.get(m, m), capsize=3,
                            markeredgecolor="black", markeredgewidth=0.3)

        # Add LRM variance on secondary axis
        ax2 = ax.twinx()
        lrm_vars = []
        for ns in steps:
            vars_at_ns = [r.get("lrm_var_mean", 0) for r in by_steps[ns]]
            lrm_vars.append(np.mean(vars_at_ns))
        ax2.plot(steps, lrm_vars, "--", color="gray", alpha=0.5, linewidth=1)
        ax2.set_ylabel("LRM Variance", fontsize=7, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=6)

        ax.set_xlabel("N steps (Euler)", fontsize=8)
        if col == 0:
            ax.set_ylabel("Value MSE", fontsize=8)
        ax.set_title(f"Heston — {payoff.title()}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

    plt.tight_layout()
    path = out_dir / f"heston_degradation.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 4: WARMUP vs VANILLA vs GRADNORM ON SPY
# ============================================================================

def fig_warmup_spy(spy, warmup, out_dir, fmt="pdf"):
    """Bar chart: warmup vs vanilla vs GradNorm on SPY (value + gradient MSE)."""
    setup_style()

    # Combine SPY results + warmup SPY results
    spy_list = list(spy.values())
    warmup_spy = [r for r in warmup.values() if r.get("dataset") == "spy_options"]

    by_ntrain_spy = group_by(spy_list, "n_train")
    warmup_by_ntrain = group_by(warmup_spy, "n_train")

    # Merge n_train keys (filter out None)
    all_ntrains = sorted(k for k in set(list(by_ntrain_spy.keys()) + list(warmup_by_ntrain.keys()))
                         if k is not None)

    if not all_ntrains:
        print("  No SPY or warmup-SPY results found — skipping.")
        return

    fig, axes = plt.subplots(2, len(all_ntrains), figsize=(5 * len(all_ntrains), 5.5),
                             squeeze=False)

    warmup_fractions = sorted(set(r["warmup_fraction"] for r in warmup_spy)) if warmup_spy else []
    compare_methods = ["vanilla", "dml_gradnorm"]
    warmup_labels = [f"Warmup w{int(wf*100)}" for wf in warmup_fractions]
    all_labels = [METHOD_LABELS.get(m, m) for m in compare_methods] + warmup_labels
    all_colors = [COLORS.get(m, "#999") for m in compare_methods] + \
                 [plt.cm.Greens(0.4 + 0.2 * i) for i in range(len(warmup_fractions))]

    for col, n_train in enumerate(all_ntrains):
        spy_recs = by_ntrain_spy.get(n_train, [])
        warmup_recs = warmup_by_ntrain.get(n_train, [])
        by_method_spy = group_by(spy_recs, "method")
        by_wf = group_by(warmup_recs, "warmup_fraction")

        for row, (metric, label) in enumerate([
            ("test_value_mse", "Value MSE"),
            ("test_grad_mse", "Gradient MSE"),
        ]):
            ax = axes[row, col]

            means = []
            stds = []
            colors = []
            labels = []

            for m in compare_methods:
                if m in by_method_spy:
                    arr = np.array([r[metric] for r in by_method_spy[m]])
                    means.append(np.mean(arr))
                    stds.append(np.std(arr))
                else:
                    means.append(np.nan)
                    stds.append(0)
                colors.append(COLORS.get(m, "#999"))
                labels.append(METHOD_LABELS.get(m, m))

            for i, wf in enumerate(warmup_fractions):
                if wf in by_wf:
                    arr = np.array([r[metric] for r in by_wf[wf]])
                    means.append(np.mean(arr))
                    stds.append(np.std(arr))
                else:
                    means.append(np.nan)
                    stds.append(0)
                colors.append(plt.cm.Greens(0.4 + 0.2 * i))
                labels.append(f"Warmup w{int(wf*100)}")

            x = np.arange(len(means))
            ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
                   linewidth=0.4, capsize=3, width=0.65)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == 0:
                ax.set_title(f"SPY (n={n_train:,})", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = out_dir / f"warmup_spy_comparison.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 5: WARMUP vs VANILLA vs GRADNORM ON HESTON
# ============================================================================

def fig_warmup_heston(lrm, warmup, out_dir, fmt="pdf"):
    """Grouped bar chart: warmup vs vanilla vs GradNorm across Heston configs."""
    setup_style()

    heston_lrm = [r for r in lrm.values() if "heston" in r.get("key", "")]
    heston_warmup = [r for r in warmup.values() if r.get("model") == "heston_euler_lrm"]

    if not heston_lrm and not heston_warmup:
        print("  No Heston results found — skipping.")
        return

    # Combine all heston results
    all_heston = heston_lrm + heston_warmup
    by_payoff = group_by(all_heston, "payoff")

    fig, axes = plt.subplots(1, len(by_payoff), figsize=(5 * len(by_payoff), 4),
                             squeeze=False)

    compare_methods = ["vanilla", "dml_gradnorm", "dml_warmup"]
    bar_width = 0.25

    for col, payoff in enumerate(sorted(by_payoff.keys())):
        ax = axes[0, col]
        recs = by_payoff[payoff]
        by_steps = group_by(recs, "n_steps")
        steps = sorted(by_steps.keys())
        x = np.arange(len(steps))

        for i, m in enumerate(compare_methods):
            means = []
            stds = []
            for ns in steps:
                by_method = group_by(by_steps[ns], "method")
                if m in by_method:
                    arr = np.array([r["test_value_mse"] for r in by_method[m]])
                    means.append(np.mean(arr))
                    stds.append(np.std(arr))
                else:
                    means.append(np.nan)
                    stds.append(0)

            offset = (i - 1) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds,
                   color=COLORS.get(m, "#999"), edgecolor="black", linewidth=0.3,
                   label=METHOD_LABELS.get(m, m), capsize=2, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={ns}" for ns in steps], fontsize=7)
        ax.set_xlabel("Euler Steps", fontsize=8)
        if col == 0:
            ax.set_ylabel("Value MSE", fontsize=8)
        ax.set_title(f"Heston — {payoff.title()}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

    plt.tight_layout()
    path = out_dir / f"warmup_heston_comparison.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 6: LRM VARIANCE SCALING WITH DIMENSION
# ============================================================================

def fig_lrm_variance_scaling(lrm, out_dir, fmt="pdf"):
    """LRM variance vs dimension for basket + method performance overlay."""
    setup_style()

    basket = [r for r in lrm.values()
              if "B_basket" in r.get("key", "") and r.get("lrm_var_mean")]
    if not basket:
        print("  No basket dimension scaling results found — skipping.")
        return

    by_dim = group_by(basket, "dim")
    dims = sorted(by_dim.keys())

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))

    # LRM variance bars
    lrm_vars = [np.mean([r["lrm_var_mean"] for r in by_dim[d]]) for d in dims]
    ax1.bar(range(len(dims)), lrm_vars, color="#E69F00", alpha=0.5, edgecolor="black",
            linewidth=0.3, label="LRM Variance")
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels([str(d) for d in dims], fontsize=7)
    ax1.set_xlabel("Dimension", fontsize=8)
    ax1.set_ylabel("Mean LRM Variance", fontsize=8, color="#E69F00")
    ax1.tick_params(axis="y", labelcolor="#E69F00")

    # Overlay: DML advantage (vanilla / dml_fixed value MSE ratio)
    ax2 = ax1.twinx()
    advantages = []
    for d in dims:
        by_method = group_by(by_dim[d], "method")
        van = np.mean([r["test_value_mse"] for r in by_method.get("vanilla", [])]) \
            if "vanilla" in by_method else np.nan
        fix = np.mean([r["test_value_mse"] for r in by_method.get("dml_fixed", [])]) \
            if "dml_fixed" in by_method else np.nan
        advantages.append(van / fix if fix > 0 else np.nan)

    ax2.plot(range(len(dims)), advantages, "o-", color="#0072B2",
             markersize=5, markeredgecolor="black", markeredgewidth=0.3,
             label="DML Advantage (vanilla/fixed)", linewidth=1.2)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.set_ylabel("DML Advantage (ratio)", fontsize=8, color="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#0072B2")

    ax1.set_title("LRM Variance Scaling & DML Advantage vs Dimension",
                  fontsize=9, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")

    plt.tight_layout()
    path = out_dir / f"lrm_variance_scaling.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality figures for extension experiments"
    )
    parser.add_argument("--figure", default="all",
                        choices=["all", "spy_pareto", "spy_method_bars",
                                 "heston_degradation", "warmup_spy",
                                 "warmup_heston", "lrm_variance"])
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--outdir", default="figures/extensions",
                        help="Output directory for figures")
    args = parser.parse_args()

    lrm, spy, warmup = load_all()
    print(f"Loaded: {len(lrm)} LRM, {len(spy)} SPY, {len(warmup)} Warmup results")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "spy_pareto":         lambda: fig_spy_pareto(spy, out_dir, args.format),
        "spy_method_bars":    lambda: fig_spy_method_bars(spy, out_dir, args.format),
        "heston_degradation": lambda: fig_heston_degradation(lrm, out_dir, args.format),
        "warmup_spy":         lambda: fig_warmup_spy(spy, warmup, out_dir, args.format),
        "warmup_heston":      lambda: fig_warmup_heston(lrm, warmup, out_dir, args.format),
        "lrm_variance":       lambda: fig_lrm_variance_scaling(lrm, out_dir, args.format),
    }

    if args.figure == "all":
        for name, func in figures.items():
            print(f"\n  Generating: {name}...")
            func()
    else:
        figures[args.figure]()

    print(f"\nDone. Figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
