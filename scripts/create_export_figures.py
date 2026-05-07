#!/usr/bin/env python3
"""
Export-ready figures for the DML-Bench paper revision.

Generates updated plots from the fresh re-run results (post autodiff fix).
Uses Nature-style formatting with Wong 2011 colorblind-safe palette.

Output: figures/export/

Figures generated:
  1. autodiff_impact     — Old (zeros) vs New (autodiff) vanilla gradient MSE
  2. honest_improvement  — Corrected DML gradient improvement ratios
  3. tier2_method_comparison — Value & gradient MSE across methods × functions
  4. spy_combined        — SPY temporal + purged CV results
  5. winrate_heatmap     — Updated win-rate from fresh tier1+tier2
  6. unified_ranking     — Study 1 unified comparison ranking
  7. noise_crossover     — Gradient MSE vs noise level
  8. dimension_scaling   — DML benefit vs dimension
  9. sample_efficiency   — DML benefit vs sample size
  10. lambda_ablation    — Lambda_j sensitivity
"""

import json
import glob
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Style: Nature-style, Wong 2011 colorblind-safe
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Wong 2011 colorblind-safe palette
COLORS = {
    "vanilla":       "#0072B2",
    "dml_fixed":     "#D55E00",
    "dml_gradnorm":  "#009E73",
    "dml_relobralo": "#CC79A7",
    "dml_warmup":    "#E69F00",
}

COLORS_11 = {
    "vanilla":            "#56B4E9",
    "dml_fixed":          "#0072B2",
    "dml_gradnorm":       "#332288",
    "dml_relobralo":      "#CC79A7",
    "dml_warmup":         "#88CCEE",
    "dml_lrm":            "#D55E00",
    "dml_gradnorm_lrm":   "#E69F00",
    "dml_warmup_lrm":     "#F0E442",
    "dml_fuzzy":          "#009E73",
    "dml_gradnorm_fuzzy": "#44AA99",
    "dml_warmup_fuzzy":   "#117733",
}

FUNC_COLORS = {
    "poly_trig":     "#0072B2",
    "trig":          "#009E73",
    "bachelier":     "#56B4E9",
    "black_scholes": "#E69F00",
    "step":          "#D55E00",
    "heston":        "#CC79A7",
}

FUNC_LABELS = {
    "poly_trig": "Poly-Trig", "trig": "Trig",
    "bachelier": "Bachelier", "black_scholes": "Black-Scholes",
    "step": "Step", "heston": "Heston",
}

METHOD_LABELS = {
    "vanilla": "Vanilla", "dml_fixed": "Fixed \u03bb",
    "dml_gradnorm": "GradNorm", "dml_relobralo": "ReLoBRaLo",
    "dml_warmup": "Warmup",
}

RESULTS = Path(__file__).parent.parent / "results"
OUTDIR = Path(__file__).parent.parent / "figures" / "export"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_tier_results(tier, methods=None, functions=None):
    """Load all results from a tier, grouped by (function, method)."""
    data = defaultdict(lambda: {"val": [], "grad": [], "configs": []})
    pattern = str(RESULTS / f"tier{tier}_benchmark" / "*.json")
    for f in glob.glob(pattern):
        d = json.load(open(f))
        base = os.path.basename(f).replace(".json", "")

        # Extract method
        method = None
        for m in ["dml_warmup", "dml_gradnorm", "dml_relobralo", "dml_fixed",
                   "vanilla", "baseline_rf", "baseline_gp", "baseline_krr"]:
            if base.endswith(f"_{m}"):
                method = m
                break
        if method is None:
            continue
        if methods and method not in methods:
            continue

        # Extract function name
        func_part = base[:base.rfind(f"_{method}")]
        # Remove seed suffix
        func_part = func_part[:func_part.rfind("_s")]
        # Extract function name (before _dN_nN...)
        parts = func_part.split("_d")
        func = parts[0]
        if functions and func not in functions:
            continue

        v = d.get("test_value_mse") or d.get("val_mse", 0)
        g = d.get("test_grad_mse") or d.get("grad_mse", 0)
        dim = d.get("dim") or d.get("d")
        n = d.get("n_train")
        noise = d.get("noise_level", 0.0)

        data[(func, method)]["val"].append(v)
        data[(func, method)]["grad"].append(g)
        data[(func, method)]["configs"].append({"d": dim, "n": n, "noise": noise})
    return data


def load_old_tier_results(tier, methods=None, functions=None):
    """Load pre-autodiff backup results."""
    data = defaultdict(lambda: {"val": [], "grad": []})
    pattern = str(RESULTS / f"_pre_autodiff_full_backup_20260406/tier{tier}_benchmark" / "*.json")
    for f in glob.glob(pattern):
        d = json.load(open(f))
        base = os.path.basename(f).replace(".json", "")
        method = None
        for m in ["dml_warmup", "dml_gradnorm", "dml_relobralo", "dml_fixed",
                   "vanilla", "baseline_rf", "baseline_gp", "baseline_krr"]:
            if base.endswith(f"_{m}"):
                method = m
                break
        if method is None:
            continue
        if methods and method not in methods:
            continue
        func_part = base[:base.rfind(f"_{method}")]
        func_part = func_part[:func_part.rfind("_s")]
        parts = func_part.split("_d")
        func = parts[0]
        if functions and func not in functions:
            continue
        v = d.get("test_value_mse") or d.get("val_mse", 0)
        g = d.get("test_grad_mse") or d.get("grad_mse", 0)
        data[(func, method)]["val"].append(v)
        data[(func, method)]["grad"].append(g)
    return data


# =========================================================================
# Figure 1: Autodiff Impact — Old vs New vanilla gradient MSE
# =========================================================================
def fig_autodiff_impact():
    print("  [1/10] autodiff_impact")
    funcs = ["bachelier", "black_scholes", "poly_trig", "trig", "step", "heston"]
    old = load_old_tier_results(2, methods=["vanilla"], functions=funcs)
    new = load_tier_results(2, methods=["vanilla"], functions=funcs)

    old_means, new_means, labels = [], [], []
    for func in funcs:
        og = old.get((func, "vanilla"), {}).get("grad", [])
        ng = new.get((func, "vanilla"), {}).get("grad", [])
        if og and ng:
            old_means.append(np.mean(og))
            new_means.append(np.mean(ng))
            labels.append(FUNC_LABELS.get(func, func))

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, old_means, w, label="Before fix (zeros baseline)",
                   color="#999999", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, new_means, w, label="After fix (autodiff)",
                   color="#0072B2", edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Vanilla Gradient MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="upper left", fontsize=6)

    # Add ratio annotations
    for i, (o, n) in enumerate(zip(old_means, new_means)):
        if n > 0 and o > 0:
            ratio = o / n
            if ratio > 1.5:
                ax.annotate(f"{ratio:.0f}\u00d7", xy=(i + w/2, n),
                           xytext=(0, -10), textcoords="offset points",
                           ha="center", fontsize=5.5, color="#0072B2", fontweight="bold")

    fig.savefig(OUTDIR / "fig_autodiff_impact.pdf")
    fig.savefig(OUTDIR / "fig_autodiff_impact.png")
    plt.close(fig)


# =========================================================================
# Figure 2: Honest improvement ratios (DML vs vanilla autodiff)
# =========================================================================
def fig_honest_improvement():
    print("  [2/10] honest_improvement")
    funcs = ["bachelier", "black_scholes", "poly_trig", "trig", "step", "heston"]
    nn_methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
    new = load_tier_results(2, methods=nn_methods, functions=funcs)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: Value MSE
    ax = axes[0]
    x = np.arange(len(funcs))
    w = 0.18
    for i, method in enumerate(nn_methods):
        means = []
        for func in funcs:
            vals = new.get((func, method), {}).get("val", [])
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + i*w - 1.5*w, means, w, label=METHOD_LABELS[method],
               color=COLORS[method], edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylabel("Value MSE \u2193")
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_LABELS[f] for f in funcs], rotation=25, ha="right")
    ax.set_title("a) Value MSE by method", fontsize=8, loc="left")

    # Right: Gradient improvement ratio over vanilla
    ax = axes[1]
    for i, method in enumerate(["dml_fixed", "dml_gradnorm", "dml_relobralo"]):
        ratios = []
        for func in funcs:
            van_g = new.get((func, "vanilla"), {}).get("grad", [])
            dml_g = new.get((func, method), {}).get("grad", [])
            if van_g and dml_g:
                ratio = np.mean(van_g) / np.mean(dml_g)
                ratios.append(ratio)
            else:
                ratios.append(0)
        ax.bar(x + (i-1)*w, ratios, w, label=METHOD_LABELS[method],
               color=COLORS[method], edgecolor="white", linewidth=0.3)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Gradient improvement \u00d7 (vs vanilla autodiff) \u2191")
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_LABELS[f] for f in funcs], rotation=25, ha="right")
    ax.set_title("b) Gradient improvement ratio", fontsize=8, loc="left")
    ax.legend(fontsize=5.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_honest_improvement.pdf")
    fig.savefig(OUTDIR / "fig_honest_improvement.png")
    plt.close(fig)


# =========================================================================
# Figure 3: Tier 2 method comparison heatmap
# =========================================================================
def fig_tier2_heatmap():
    print("  [3/10] tier2_heatmap")
    funcs = ["bachelier", "black_scholes", "poly_trig", "trig", "step", "heston"]
    methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
    data = load_tier_results(2, methods=methods, functions=funcs)

    # Build gradient improvement matrix
    matrix = np.ones((len(funcs), len(methods) - 1))
    for i, func in enumerate(funcs):
        van_g = data.get((func, "vanilla"), {}).get("grad", [])
        if not van_g:
            continue
        van_mean = np.mean(van_g)
        for j, method in enumerate(["dml_fixed", "dml_gradnorm", "dml_relobralo"]):
            dml_g = data.get((func, method), {}).get("grad", [])
            if dml_g and van_mean > 0:
                matrix[i, j] = np.mean(dml_g) / van_mean  # <1 means DML better

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(np.log10(matrix), cmap="RdYlGn_r", aspect="auto",
                   vmin=-2, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Fixed \u03bb", "GradNorm", "ReLoBRaLo"], fontsize=7)
    ax.set_yticks(range(len(funcs)))
    ax.set_yticklabels([FUNC_LABELS[f] for f in funcs], fontsize=7)

    # Annotate
    for i in range(len(funcs)):
        for j in range(3):
            val = matrix[i, j]
            if val < 1:
                txt = f"{1/val:.0f}\u00d7"
                color = "white" if val < 0.1 else "black"
            else:
                txt = f"{val:.1f}\u00d7\u2191"
                color = "white" if val > 5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log\u2081\u2080(DML / Vanilla grad MSE)", fontsize=7)
    ax.set_title("Gradient improvement over vanilla (autodiff)", fontsize=8)
    fig.savefig(OUTDIR / "fig_tier2_heatmap.pdf")
    fig.savefig(OUTDIR / "fig_tier2_heatmap.png")
    plt.close(fig)


# =========================================================================
# Figure 4: SPY combined (temporal + purged CV)
# =========================================================================
def fig_spy_combined():
    print("  [4/10] spy_combined")
    methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"]

    def load_spy(subdir):
        out = defaultdict(lambda: {"val": [], "grad": []})
        for f in glob.glob(str(RESULTS / subdir / "*.json")):
            d = json.load(open(f))
            m = d.get("method", "")
            if m in methods:
                v = d.get("test_value_mse", 0)
                g = d.get("test_grad_mse", 0)
                out[m]["val"].append(v)
                out[m]["grad"].append(g)
        return out

    temporal = load_spy("spy_options_temporal")
    cv = load_spy("spy_options_purged_cv")

    fig, axes = plt.subplots(2, 2, figsize=(6, 4.5))

    for col, (split_data, split_name) in enumerate([(temporal, "Temporal"), (cv, "Purged CV")]):
        for row, (metric, ylabel) in enumerate([("val", "Value MSE \u2193"), ("grad", "Gradient MSE \u2193")]):
            ax = axes[row, col]
            means, stds, colors, labels = [], [], [], []
            for m in methods:
                vals = split_data[m][metric]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    colors.append(COLORS.get(m, "#999999"))
                    labels.append(METHOD_LABELS.get(m, m))
            x = np.arange(len(means))
            ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                       capsize=2, linewidth=0.5)
            if row == 0:
                ax.set_title(f"{split_name} split", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_spy_combined.pdf")
    fig.savefig(OUTDIR / "fig_spy_combined.png")
    plt.close(fig)


# =========================================================================
# Figure 5: Win-rate heatmap (function × dimension) — fresh data
# =========================================================================
def fig_winrate():
    print("  [5/10] winrate_heatmap")
    funcs = ["bachelier", "black_scholes", "poly_trig", "trig", "step", "heston"]
    dims_set = set()
    nn_methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

    # Load tier1 + tier2 data, compute win rates per (func, dim)
    wins = defaultdict(lambda: {"win": 0, "total": 0})

    for tier in [1, 2]:
        data = load_tier_results(tier, methods=nn_methods, functions=funcs)
        # Group by (func, dim, noise, n_train, seed)
        configs = defaultdict(lambda: defaultdict(list))
        for f in glob.glob(str(RESULTS / f"tier{tier}_benchmark" / "*.json")):
            d = json.load(open(f))
            base = os.path.basename(f).replace(".json", "")
            method = None
            for m in nn_methods:
                if base.endswith(f"_{m}"):
                    method = m
                    break
            if method is None or method.startswith("baseline"):
                continue

            func_part = base[:base.rfind(f"_{method}")]
            seed_part = func_part[func_part.rfind("_s")+2:]
            func_part2 = func_part[:func_part.rfind("_s")]
            parts = func_part2.split("_d")
            func = parts[0]
            if func not in funcs:
                continue

            dim = d.get("dim") or d.get("d", 1)
            dims_set.add(dim)
            config_key = func_part2 + f"_s{seed_part}"
            v = d.get("test_value_mse") or d.get("val_mse", 0)
            configs[(func, dim)][config_key].append((method, v))

        # Count wins
        for (func, dim), config_results in configs.items():
            for config_key, method_vals in config_results.items():
                if len(method_vals) < 2:
                    continue
                vanilla_val = None
                best_dml_val = float("inf")
                for m, v in method_vals:
                    if m == "vanilla":
                        vanilla_val = v
                    else:
                        best_dml_val = min(best_dml_val, v)
                if vanilla_val is not None:
                    wins[(func, dim)]["total"] += 1
                    if best_dml_val < vanilla_val:
                        wins[(func, dim)]["win"] += 1

    dims = sorted(dims_set)
    if not dims:
        print("    Skipping — no data")
        return

    matrix = np.full((len(funcs), len(dims)), np.nan)
    for i, func in enumerate(funcs):
        for j, dim in enumerate(dims):
            entry = wins.get((func, dim))
            if entry and entry["total"] > 0:
                matrix[i, j] = entry["win"] / entry["total"] * 100

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], fontsize=6)
    ax.set_xlabel("Dimension")
    ax.set_yticks(range(len(funcs)))
    ax.set_yticklabels([FUNC_LABELS[f] for f in funcs], fontsize=7)

    for i in range(len(funcs)):
        for j in range(len(dims)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if (v < 30 or v > 80) else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                       fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DML win rate (%)", fontsize=7)
    ax.set_title("DML win rate: value MSE (any DML < vanilla)", fontsize=8)
    fig.savefig(OUTDIR / "fig_winrate_heatmap.pdf")
    fig.savefig(OUTDIR / "fig_winrate_heatmap.png")
    plt.close(fig)


# =========================================================================
# Figure 6: Unified ranking (Study 1)
# =========================================================================
def fig_unified_ranking():
    print("  [6/10] unified_ranking")
    uni_dir = RESULTS / "unified_comparison" / "multi_seed"
    if not uni_dir.exists():
        print("    Skipping — no unified results")
        return

    methods_11 = list(COLORS_11.keys())
    datasets = ["digital_bs", "barrier_bs", "heston_digital", "basket_digital"]

    data = defaultdict(lambda: defaultdict(lambda: {"val": [], "grad": []}))
    for f in glob.glob(str(uni_dir / "*.json")):
        d = json.load(open(f))
        method = d.get("method", "")
        dataset = d.get("dataset", "")
        # Normalize dataset name
        for ds in datasets:
            if ds in os.path.basename(f):
                dataset = ds
                break
        v = d.get("test_value_mse") or d.get("val_mse", 0)
        g = d.get("test_grad_mse") or d.get("grad_mse", 0)
        data[dataset][method]["val"].append(v)
        data[dataset][method]["grad"].append(g)

    if not data:
        print("    Skipping — no data parsed")
        return

    # Compute mean ranks per method across datasets
    method_ranks = defaultdict(list)
    for dataset in data:
        grad_means = {}
        for method in data[dataset]:
            grads = data[dataset][method]["grad"]
            if grads:
                grad_means[method] = np.mean(grads)
        if grad_means:
            sorted_methods = sorted(grad_means, key=grad_means.get)
            for rank, m in enumerate(sorted_methods, 1):
                method_ranks[m].append(rank)

    if not method_ranks:
        print("    Skipping — no rankings computed")
        return

    # Plot
    methods_ranked = sorted(method_ranks, key=lambda m: np.mean(method_ranks[m]))
    means = [np.mean(method_ranks[m]) for m in methods_ranked]
    stds = [np.std(method_ranks[m]) for m in methods_ranked]

    fig, ax = plt.subplots(figsize=(4, 3))
    y = np.arange(len(methods_ranked))
    colors = [COLORS_11.get(m, "#999999") for m in methods_ranked]
    labels = []
    for m in methods_ranked:
        short = m.replace("dml_", "").replace("_", " ").title()
        labels.append(short)

    ax.barh(y, means, xerr=stds, color=colors, edgecolor="white",
            linewidth=0.5, capsize=2, error_kw={"linewidth": 0.5})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Mean rank (lower = better)")
    ax.set_title("Method ranking by gradient MSE\n(Study 1: discontinuous payoffs)", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_unified_ranking.pdf")
    fig.savefig(OUTDIR / "fig_unified_ranking.png")
    plt.close(fig)


# =========================================================================
# Figure 7: Noise crossover
# =========================================================================
def fig_noise_crossover():
    print("  [7/10] noise_crossover")
    # Only include functions that have multiple noise levels in tier2
    nn_methods = ["vanilla", "dml_fixed"]

    noise_data = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(RESULTS / "tier2_benchmark" / "*.json")):
        d = json.load(open(f))
        base = os.path.basename(f).replace(".json", "")
        method = None
        for m in nn_methods:
            if base.endswith(f"_{m}"):
                method = m
                break
        if method is None:
            continue
        func_part = base[:base.rfind(f"_{method}")]
        func_part = func_part[:func_part.rfind("_s")]
        func = func_part.split("_d")[0]
        noise = d.get("noise_level", 0.0)
        g = d.get("test_grad_mse") or d.get("grad_mse", 0)
        noise_data[(func, method, noise)]["grad"].append(g)

    if not noise_data:
        print("    Skipping — no data")
        return

    # Filter to functions with >1 noise level
    func_noises = defaultdict(set)
    for (func, method, noise) in noise_data:
        func_noises[func].add(noise)
    funcs = [f for f in sorted(func_noises) if len(func_noises[f]) > 1]

    if not funcs:
        print("    Skipping — no functions with multiple noise levels")
        return

    ncols = min(len(funcs), 4)
    fig, axes = plt.subplots(1, ncols, figsize=(1.8 * ncols, 2.2), sharey=False)
    if ncols == 1:
        axes = [axes]

    for idx, func in enumerate(funcs[:ncols]):
        ax = axes[idx]
        for method in nn_methods:
            noise_vals = sorted(set(n for (f, m, n) in noise_data if f == func and m == method))
            if not noise_vals or len(noise_vals) < 2:
                continue
            means = [np.mean(noise_data[(func, method, n)]["grad"]) for n in noise_vals]
            ax.plot(noise_vals, means, "o-", color=COLORS[method],
                   label=METHOD_LABELS[method], markersize=3, linewidth=1)
        ax.set_title(FUNC_LABELS.get(func, func), fontsize=7)
        ax.set_xlabel("Noise \u03c3")
        if idx == 0:
            ax.set_ylabel("Gradient MSE")
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=5)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_noise_crossover.pdf")
    fig.savefig(OUTDIR / "fig_noise_crossover.png")
    plt.close(fig)


# =========================================================================
# Figure 8: Dimension scaling
# =========================================================================
def fig_dimension_scaling():
    print("  [8/10] dimension_scaling")
    funcs = ["poly_trig", "trig"]
    nn_methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

    dim_data = defaultdict(lambda: defaultdict(list))
    for tier in [1, 2]:
        for f in glob.glob(str(RESULTS / f"tier{tier}_benchmark" / "*.json")):
            d = json.load(open(f))
            base = os.path.basename(f).replace(".json", "")
            method = None
            for m in nn_methods:
                if base.endswith(f"_{m}"):
                    method = m
                    break
            if method is None:
                continue
            func_part = base[:base.rfind(f"_{method}")]
            func_part = func_part[:func_part.rfind("_s")]
            func = func_part.split("_d")[0]
            if func not in funcs:
                continue
            dim = d.get("dim") or d.get("d", 1)
            noise = d.get("noise_level", 0.0)
            if noise > 0:
                continue  # clean only
            v = d.get("test_value_mse") or d.get("val_mse", 0)
            dim_data[(func, method, dim)]["val"].append(v)

    fig, axes = plt.subplots(1, len(funcs), figsize=(5.5, 2.4))
    if len(funcs) == 1:
        axes = [axes]

    for idx, func in enumerate(funcs):
        ax = axes[idx]
        for method in nn_methods:
            dims = sorted(set(d for (f, m, d) in dim_data if f == func and m == method))
            if not dims:
                continue
            means = [np.mean(dim_data[(func, method, d)]["val"]) for d in dims]
            ax.plot(dims, means, "o-", color=COLORS[method],
                   label=METHOD_LABELS[method], markersize=3, linewidth=1)
        ax.set_title(FUNC_LABELS[func], fontsize=8)
        ax.set_xlabel("Dimension")
        if idx == 0:
            ax.set_ylabel("Value MSE \u2193")
        ax.set_yscale("log")
        ax.set_xscale("log")
        if idx == len(funcs) - 1:
            ax.legend(fontsize=5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_dimension_scaling.pdf")
    fig.savefig(OUTDIR / "fig_dimension_scaling.png")
    plt.close(fig)


# =========================================================================
# Figure 9: Sample efficiency
# =========================================================================
def fig_sample_efficiency():
    print("  [9/10] sample_efficiency")
    # Use functions/dims that have multiple sample sizes
    configs = [("poly_trig", 2), ("trig", 5), ("bachelier", 1)]
    nn_methods = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]

    n_data = defaultdict(lambda: defaultdict(list))
    for tier in [1, 2]:
        for f in glob.glob(str(RESULTS / f"tier{tier}_benchmark" / "*.json")):
            d = json.load(open(f))
            base = os.path.basename(f).replace(".json", "")
            method = None
            for m in nn_methods:
                if base.endswith(f"_{m}"):
                    method = m
                    break
            if method is None:
                continue
            func_part = base[:base.rfind(f"_{method}")]
            func_part = func_part[:func_part.rfind("_s")]
            func = func_part.split("_d")[0]
            dim = d.get("dim") or d.get("d", 1)
            if (func, dim) not in configs:
                continue
            noise = d.get("noise_level", 0.0)
            if noise > 0:
                continue
            n = d.get("n_samples") or d.get("n_train", 0)
            v = d.get("test_value_mse") or d.get("val_mse", 0)
            g = d.get("test_grad_mse") or d.get("grad_mse", 0)
            n_data[(func, dim, method, n)]["val"].append(v)
            n_data[(func, dim, method, n)]["grad"].append(g)

    fig, axes = plt.subplots(1, len(configs), figsize=(7, 2.4))
    if len(configs) == 1:
        axes = [axes]
    for idx, (func, dim) in enumerate(configs):
        ax = axes[idx]
        for method in nn_methods:
            ns = sorted(set(n for (f, d, m, n) in n_data if f == func and d == dim and m == method))
            if not ns:
                continue
            val_means = [np.mean(n_data[(func, dim, method, n)]["val"]) for n in ns]
            ax.plot(ns, val_means, "o-", color=COLORS[method],
                   label=METHOD_LABELS[method], markersize=3, linewidth=1)
        ax.set_title(f"{FUNC_LABELS[func]} (d={dim}, \u03c3=0)", fontsize=8)
        ax.set_xlabel("Training samples")
        if idx == 0:
            ax.set_ylabel("Value MSE \u2193")
        ax.set_yscale("log")
        ax.set_xscale("log")
        if idx == len(configs) - 1:
            ax.legend(fontsize=5, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_sample_efficiency.pdf")
    fig.savefig(OUTDIR / "fig_sample_efficiency.png")
    plt.close(fig)


# =========================================================================
# Figure 10: Lambda ablation
# =========================================================================
def fig_lambda_ablation():
    print("  [10/10] lambda_ablation")
    lam_dir = RESULTS / "lambda_j_ablation"
    files = glob.glob(str(lam_dir / "*.json"))
    if not files:
        print("    Skipping — no data")
        return

    # Group by (dataset, lambda_j_source)
    lam_data = defaultdict(lambda: {"val": [], "grad": []})
    for f in files:
        d = json.load(open(f))
        dataset = d.get("dataset", "unknown")
        source = d.get("lambda_j_source", "unknown")
        v = d.get("test_value_mse", 0)
        g = d.get("test_grad_mse", 0)
        lam_data[(dataset, source)]["val"].append(v)
        lam_data[(dataset, source)]["grad"].append(g)

    if not lam_data:
        print("    Skipping — no data parsed")
        return

    datasets = sorted(set(ds for ds, _ in lam_data if ds != "unknown"))
    sources = sorted(set(src for _, src in lam_data if src != "unknown"))

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = np.arange(len(datasets))
    w = 0.35
    src_colors = {"hs_lambda_j": "#0072B2", "our_lambda_j": "#D55E00"}
    src_labels = {"hs_lambda_j": "H&S \u03bb_j", "our_lambda_j": "Our \u03bb_j"}

    for metric_idx, (metric, ylabel) in enumerate([("val", "Value MSE \u2193"), ("grad", "Gradient MSE \u2193")]):
        ax = axes[metric_idx]
        for i, source in enumerate(sources):
            means = []
            stds = []
            for ds in datasets:
                vals = lam_data[(ds, source)][metric]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            offset = (i - 0.5) * w
            ax.bar(x + offset, means, w, yerr=stds,
                   label=src_labels.get(source, source),
                   color=src_colors.get(source, "#999999"),
                   edgecolor="white", linewidth=0.5, capsize=2,
                   error_kw={"linewidth": 0.5})
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ds_short = {"digital_bs": "Digital", "barrier_bs": "Barrier",
                     "heston_digital": "Heston", "basket_d1": "Basket d=1",
                     "basket_d7": "Basket d=7"}
        ax.set_xticklabels([ds_short.get(d, d) for d in datasets], rotation=25, ha="right", fontsize=6)
        if metric_idx == 0:
            ax.legend(fontsize=6)
        ax.set_title("a) Value" if metric_idx == 0 else "b) Gradient", fontsize=8, loc="left")

    fig.suptitle("\u03bb_j formula comparison (H&S vs Ours)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_lambda_ablation.pdf")
    fig.savefig(OUTDIR / "fig_lambda_ablation.png")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print(f"Generating export figures to {OUTDIR}/")
    os.chdir(Path(__file__).parent.parent)

    fig_autodiff_impact()
    fig_honest_improvement()
    fig_tier2_heatmap()
    fig_spy_combined()
    fig_winrate()
    fig_unified_ranking()
    fig_noise_crossover()
    fig_dimension_scaling()
    fig_sample_efficiency()
    fig_lambda_ablation()

    print(f"\nDone! {len(list(OUTDIR.glob('*.pdf')))} PDFs + {len(list(OUTDIR.glob('*.png')))} PNGs saved to {OUTDIR}/")
