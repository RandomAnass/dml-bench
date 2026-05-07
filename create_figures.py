#!/usr/bin/env python3
"""
Publication-Quality Figure Generator — DML Benchmark.

Produces Nature-style figures (PDF/PNG) from benchmark results.
All figures use consistent styling: Arial 8pt, 300dpi, colorblind-safe palette.

Usage:
    python create_figures.py                          # All figures
    python create_figures.py --figure heatmap         # Just heatmap
    python create_figures.py --figure scaling          # Dimension scaling
    python create_figures.py --figure noise            # Noise robustness
    python create_figures.py --figure convergence      # Learning curves
    python create_figures.py --figure gradnorm         # GradNorm instability
    python create_figures.py --figure method_comparison # Bar charts
    python create_figures.py --format png              # PNG instead of PDF
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ============================================================================
# NATURE STYLE CONFIG
# ============================================================================

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "vanilla":       "#0072B2",  # Blue
    "dml_fixed":     "#D55E00",  # Vermillion
    "dml_gradnorm":  "#009E73",  # Green
    "dml_relobralo": "#CC79A7",  # Pink
    "baseline_gp":   "#56B4E9",  # Sky blue
    "baseline_krr":  "#E69F00",  # Orange
    "baseline_rf":   "#F0E442",  # Yellow
}

METHOD_LABELS = {
    "vanilla": "Vanilla",
    "dml_fixed": "DML (fixed λ)",
    "dml_gradnorm": "DML + GradNorm",
    "dml_relobralo": "DML + ReLoBRaLo",
}

NN_METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]


def setup_style():
    """Configure matplotlib for Nature-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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
# DATA LOADING (reuse from analyze_results)
# ============================================================================

def load_all_results(tiers=[1, 2, 3, 4]):
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
                    results[data.get("key", f.stem)] = data
            except Exception:
                pass
    return results


def to_records(results):
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
            "training_logs": r.get("training_logs", None),
        })
    return records


def filt(records, **kw):
    out = records
    for k, v in kw.items():
        if isinstance(v, (list, tuple, set)):
            out = [r for r in out if r.get(k) in v]
        else:
            out = [r for r in out if r.get(k) == v]
    return out


# ============================================================================
# FIGURE 1: DML ADVANTAGE HEATMAP
# ============================================================================

def fig_heatmap(records, out_dir, fmt="pdf"):
    """DML advantage heatmap: dim × n_samples for poly_trig and trig."""
    setup_style()
    
    for func in ["poly_trig", "trig"]:
        func_recs = filt(records, func_type=func, noise_level=0.0)
        func_recs = [r for r in func_recs if r["method"] in ["vanilla", "dml_fixed"]
                     and r.get("lambda", 1.0) == 1.0]
        
        dims = sorted(set(r["dim"] for r in func_recs))
        samples = sorted(set(r["n_samples"] for r in func_recs))
        
        if not dims or not samples:
            continue
        
        advantage = np.full((len(dims), len(samples)), np.nan)
        
        for i, d in enumerate(dims):
            for j, n in enumerate(samples):
                van = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == d and r["n_samples"] == n and r["method"] == "vanilla"]
                dml = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == d and r["n_samples"] == n and r["method"] == "dml_fixed"]
                if van and dml:
                    v, dm = np.mean(van), np.mean(dml)
                    advantage[i, j] = 100 * (v - dm) / v if v > 0 else 0
        
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        
        vmax = np.nanmax(np.abs(advantage))
        vmax = max(vmax, 1)  # Avoid degenerate colorbar
        
        im = ax.imshow(advantage.T, cmap="RdYlGn", aspect="auto",
                       vmin=-vmax, vmax=vmax, origin="lower")
        
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims)
        ax.set_yticks(range(len(samples)))
        ax.set_yticklabels(samples)
        ax.set_xlabel("Input dimension $d$")
        ax.set_ylabel("Sample size $n$")
        ax.set_title(f"DML advantage (%) — {func}")
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("% improvement over vanilla")
        
        # Annotate cells
        for i in range(len(dims)):
            for j in range(len(samples)):
                val = advantage[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(i, j, f"{val:.1f}", ha="center", va="center",
                            color=color, fontsize=7)
        
        plt.tight_layout()
        path = out_dir / f"heatmap_{func}.{fmt}"
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")


# ============================================================================
# FIGURE 2: DIMENSION SCALING
# ============================================================================

def fig_scaling(records, out_dir, fmt="pdf"):
    """Dimension scaling: MSE vs dimension for each method."""
    setup_style()
    
    for func in ["poly_trig", "trig", "step", "bachelier"]:
        func_recs = filt(records, func_type=func, noise_level=0.0, n_samples=1024)
        func_recs = [r for r in func_recs if r["method"] in NN_METHODS
                     and r.get("lambda", 1.0) == 1.0]
        
        dims = sorted(set(r["dim"] for r in func_recs))
        if len(dims) < 2:
            continue
        
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        
        for method in NN_METHODS:
            means, stds, ds = [], [], []
            for dim in dims:
                vals = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["method"] == method]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    ds.append(dim)
            
            if ds:
                ax.errorbar(ds, means, yerr=stds, label=METHOD_LABELS.get(method, method),
                           color=COLORS.get(method, "gray"), capsize=2, marker="o",
                           markersize=4, linewidth=1.2)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Input dimension $d$")
        ax.set_ylabel("Test value MSE")
        ax.set_title(f"Dimension scaling — {func}")
        ax.legend(loc="best", framealpha=0.9)
        
        plt.tight_layout()
        path = out_dir / f"scaling_{func}.{fmt}"
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")


# ============================================================================
# FIGURE 3: NOISE ROBUSTNESS
# ============================================================================

def fig_noise(records, out_dir, fmt="pdf"):
    """Noise robustness: MSE vs noise level with crossover detection."""
    setup_style()
    
    for func in ["poly_trig", "trig"]:
        func_recs = [r for r in records if r["func_type"] == func
                     and r["method"] in NN_METHODS and r.get("lambda", 1.0) == 1.0]
        
        noise_levels = sorted(set(r["noise_level"] for r in func_recs))
        if len(noise_levels) <= 1:
            continue
        
        dims_available = sorted(set(r["dim"] for r in func_recs if r["n_samples"] == 1024))
        dims_to_plot = [d for d in [2, 10, 50] if d in dims_available]
        
        if not dims_to_plot:
            continue
        
        fig, axes = plt.subplots(1, len(dims_to_plot), figsize=(3.5 * len(dims_to_plot), 2.8),
                                 squeeze=False)
        
        for idx, dim in enumerate(dims_to_plot):
            ax = axes[0, idx]
            
            for method in ["vanilla", "dml_fixed"]:
                means = []
                for noise in noise_levels:
                    vals = [r["test_value_mse"] for r in func_recs 
                           if r["dim"] == dim and r["n_samples"] == 1024
                           and r["noise_level"] == noise and r["method"] == method]
                    means.append(np.mean(vals) if vals else np.nan)
                
                ax.plot(noise_levels, means, "o-", label=METHOD_LABELS.get(method, method),
                       color=COLORS.get(method), markersize=4, linewidth=1.2)
            
            ax.set_xlabel("Derivative noise $\\sigma$")
            if idx == 0:
                ax.set_ylabel("Test value MSE")
            ax.set_title(f"$d = {dim}$")
            ax.legend(loc="best", fontsize=6)
            ax.set_yscale("log")
        
        fig.suptitle(f"Noise robustness — {func}", fontsize=10)
        plt.tight_layout()
        path = out_dir / f"noise_{func}.{fmt}"
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")


# ============================================================================
# FIGURE 4: LEARNING CURVES (requires --save-logs data)
# ============================================================================

def fig_convergence(records, out_dir, fmt="pdf"):
    """Learning curves from training logs (if available)."""
    setup_style()
    
    # Find records with training logs
    log_records = [r for r in records if r.get("training_logs")]
    
    if not log_records:
        print("  ⚠️ No training logs found. Run Tier 4 with --save-logs to generate.")
        return
    
    # Group by (func, dim, n_samples) — pick representative configs
    configs = defaultdict(list)
    for r in log_records:
        key = (r["func_type"], r["dim"], r["n_samples"])
        configs[key].append(r)
    
    for (func, dim, ns), cfg_records in sorted(configs.items()):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        
        for r in cfg_records:
            method = r["method"]
            logs = r["training_logs"]
            
            epochs = [l["epoch"] for l in logs]
            val_loss = [l["val_loss"] for l in logs]
            
            color = COLORS.get(method, "gray")
            label = METHOD_LABELS.get(method, method)
            
            # Left: total val loss
            axes[0].plot(epochs, val_loss, label=label, color=color, linewidth=1.0)
            
            # Right: value vs deriv components (if available)
            if "val_value_loss" in logs[0]:
                val_value = [l["val_value_loss"] for l in logs]
                val_deriv = [l["val_deriv_loss"] for l in logs]
                axes[1].plot(epochs, val_value, color=color, linewidth=0.8, linestyle="-")
                axes[1].plot(epochs, val_deriv, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation loss")
        axes[0].set_yscale("log")
        axes[0].legend(loc="best", fontsize=6)
        axes[0].set_title("Total validation loss")
        
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss component")
        axes[1].set_yscale("log")
        axes[1].set_title("Value (—) vs derivative (--) loss")
        
        fig.suptitle(f"{func} d={dim} n={ns}", fontsize=10)
        plt.tight_layout()
        path = out_dir / f"convergence_{func}_d{dim}_n{ns}.{fmt}"
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")


# ============================================================================
# FIGURE 5: GRADNORM INSTABILITY
# ============================================================================

def fig_gradnorm(records, out_dir, fmt="pdf"):
    """GradNorm instability visualization across dimensions."""
    setup_style()
    
    for func in ["poly_trig", "trig"]:
        func_recs = filt(records, func_type=func, noise_level=0.0, n_samples=1024)
        func_recs = [r for r in func_recs if r["method"] in NN_METHODS
                     and r.get("lambda", 1.0) == 1.0]
        
        dims = sorted(set(r["dim"] for r in func_recs))
        if len(dims) < 3:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        
        # Left: Absolute MSE vs dim per method
        for method in NN_METHODS:
            means, ds = [], []
            for dim in dims:
                vals = [r["test_value_mse"] for r in func_recs 
                       if r["dim"] == dim and r["method"] == method]
                if vals:
                    means.append(np.mean(vals))
                    ds.append(dim)
            if ds:
                axes[0].plot(ds, means, "o-", label=METHOD_LABELS.get(method, method),
                           color=COLORS.get(method), linewidth=1.2, markersize=4)
        
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Dimension $d$")
        axes[0].set_ylabel("Test value MSE")
        axes[0].set_title(f"All methods — {func}")
        axes[0].legend(fontsize=6)
        
        # Right: GradNorm/DML_fixed ratio vs dim
        ratios = []
        ratio_dims = []
        for dim in dims:
            gn = [r["test_value_mse"] for r in func_recs 
                 if r["dim"] == dim and r["method"] == "dml_gradnorm"]
            dml = [r["test_value_mse"] for r in func_recs 
                  if r["dim"] == dim and r["method"] == "dml_fixed"]
            if gn and dml:
                ratios.append(np.mean(gn) / np.mean(dml))
                ratio_dims.append(dim)
        
        if ratio_dims:
            axes[1].bar(range(len(ratio_dims)), ratios,
                       color=[COLORS["dml_gradnorm"] if r > 2 else "#999999" for r in ratios],
                       edgecolor="black", linewidth=0.5)
            axes[1].axhline(y=1, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
            axes[1].set_xticks(range(len(ratio_dims)))
            axes[1].set_xticklabels(ratio_dims)
            axes[1].set_xlabel("Dimension $d$")
            axes[1].set_ylabel("GradNorm / DML_fixed MSE ratio")
            axes[1].set_title("GradNorm instability")
            axes[1].set_yscale("log")
        
        plt.tight_layout()
        path = out_dir / f"gradnorm_instability_{func}.{fmt}"
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")


# ============================================================================
# FIGURE 6: METHOD COMPARISON BAR CHARTS
# ============================================================================

def fig_method_comparison(records, out_dir, fmt="pdf"):
    """Method comparison bar charts with CIs for key configurations."""
    setup_style()
    from dml_benchmark.stats import bootstrap_ci
    
    key_configs = [
        ("poly_trig", 10, 1024),
        ("poly_trig", 50, 1024),
        ("trig", 10, 1024),
        ("bachelier", 5, 1024),
    ]
    
    available = []
    for func, dim, ns in key_configs:
        recs = filt(records, func_type=func, dim=dim, n_samples=ns, noise_level=0.0)
        recs = [r for r in recs if r["method"] in NN_METHODS and r.get("lambda", 1.0) == 1.0]
        if recs:
            available.append((func, dim, ns, recs))
    
    if not available:
        print("  ⚠️ No data for method comparison plots")
        return
    
    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(2.5 * n_plots, 3.0), squeeze=False)
    
    for idx, (func, dim, ns, cfg_recs) in enumerate(available):
        ax = axes[0, idx]
        
        means, cis, colors, labels = [], [], [], []
        for method in NN_METHODS:
            vals = [r["test_value_mse"] for r in cfg_recs if r["method"] == method]
            if vals:
                ci = bootstrap_ci(np.array(vals))
                means.append(ci["mean"])
                cis.append(ci["ci_upper"] - ci["mean"])
                colors.append(COLORS.get(method, "gray"))
                labels.append(METHOD_LABELS.get(method, method))
        
        x = np.arange(len(means))
        ax.bar(x, means, yerr=cis, color=colors, edgecolor="black", linewidth=0.5,
               capsize=3, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Test MSE" if idx == 0 else "")
        ax.set_title(f"{func}\n$d={dim}, n={ns}$", fontsize=8)
        ax.set_yscale("log")
    
    plt.tight_layout()
    path = out_dir / f"method_comparison.{fmt}"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 7: FINANCE COMPARISON
# ============================================================================

def fig_finance(records, out_dir, fmt="pdf"):
    """Comparison of all finance functions."""
    setup_style()
    
    finance_funcs = ["bachelier", "black_scholes", "heston"]
    nn_recs = filt(records, noise_level=0.0, n_samples=1024)
    nn_recs = [r for r in nn_recs if r["method"] in ["vanilla", "dml_fixed"]
               and r.get("lambda", 1.0) == 1.0]
    
    available_funcs = []
    for func in finance_funcs:
        if any(r["func_type"] == func for r in nn_recs):
            available_funcs.append(func)
    
    if not available_funcs:
        print("  ⚠️ No finance data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    
    # Left: Value MSE across functions/dims
    for method in ["vanilla", "dml_fixed"]:
        x_labels, y_vals = [], []
        for func in available_funcs:
            func_recs = filt(nn_recs, func_type=func, method=method)
            dims = sorted(set(r["dim"] for r in func_recs))
            for dim in dims:
                vals = [r["test_value_mse"] for r in func_recs if r["dim"] == dim]
                if vals:
                    x_labels.append(f"{func}\nd={dim}")
                    y_vals.append(np.mean(vals))
        
        if x_labels:
            x = np.arange(len(x_labels))
            offset = -0.15 if method == "vanilla" else 0.15
            axes[0].bar(x + offset, y_vals, width=0.3,
                       color=COLORS.get(method), label=METHOD_LABELS.get(method),
                       edgecolor="black", linewidth=0.5)
    
    if x_labels:
        axes[0].set_xticks(np.arange(len(x_labels)))
        axes[0].set_xticklabels(x_labels, fontsize=6)
    axes[0].set_ylabel("Test value MSE")
    axes[0].set_title("Value MSE — Finance functions")
    axes[0].legend(fontsize=7)
    axes[0].set_yscale("log")
    
    # Right: Gradient MSE
    for method in ["vanilla", "dml_fixed"]:
        x_labels, y_vals = [], []
        for func in available_funcs:
            func_recs = filt(nn_recs, func_type=func, method=method)
            dims = sorted(set(r["dim"] for r in func_recs))
            for dim in dims:
                vals = [r["test_grad_mse"] for r in func_recs if r["dim"] == dim]
                if vals:
                    x_labels.append(f"{func}\nd={dim}")
                    y_vals.append(np.mean(vals))
        
        if x_labels:
            x = np.arange(len(x_labels))
            offset = -0.15 if method == "vanilla" else 0.15
            axes[1].bar(x + offset, y_vals, width=0.3,
                       color=COLORS.get(method), label=METHOD_LABELS.get(method),
                       edgecolor="black", linewidth=0.5)
    
    if x_labels:
        axes[1].set_xticks(np.arange(len(x_labels)))
        axes[1].set_xticklabels(x_labels, fontsize=6)
    axes[1].set_ylabel("Test gradient MSE")
    axes[1].set_title("Gradient MSE — Finance functions")
    axes[1].legend(fontsize=7)
    axes[1].set_yscale("log")
    
    plt.tight_layout()
    path = out_dir / f"finance_comparison.{fmt}"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DML Benchmark Figure Generator")
    parser.add_argument("--figure", default="all",
                        choices=["all", "heatmap", "scaling", "noise", "convergence",
                                 "gradnorm", "method_comparison", "finance"],
                        help="Which figure to generate")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--outdir", default="figures", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(args.tiers)
    records = to_records(results)
    print(f"Loaded {len(records)} results")
    
    figure_fns = {
        "heatmap": fig_heatmap,
        "scaling": fig_scaling,
        "noise": fig_noise,
        "convergence": fig_convergence,
        "gradnorm": fig_gradnorm,
        "method_comparison": fig_method_comparison,
        "finance": fig_finance,
    }
    
    if args.figure == "all":
        for name, fn in figure_fns.items():
            print(f"\nGenerating: {name}")
            try:
                fn(records, out_dir, args.format)
            except Exception as e:
                print(f"  ⚠️ {name} failed: {e}")
    else:
        figure_fns[args.figure](records, out_dir, args.format)
    
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
