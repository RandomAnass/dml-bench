#!/usr/bin/env python3
"""
Figure Generator for Gradient Noise Sweep Experiment.

Creates publication-quality figures showing:
  1. DML advantage vs gradient noise σ₁ (main result)
  2. Test MSE vs σ₁ for DML and vanilla (with crossover)
  3. Gradient MSE degradation under noise

Usage:
    python experiments/plot_noise_sweep.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Style ----
COLORS = {
    "vanilla": "#636363",
    "dml": "#2166ac",
    "crossover": "#d62728",
}
FUNC_MARKERS = {"poly_trig": "o", "trig": "s"}
DIM_STYLES = {2: "-", 5: "--"}


def setup_style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load_results(results_dir: Path):
    """Load all gradient noise sweep results from JSON."""
    records = []
    for f in sorted(results_dir.glob("gn_*.json")):
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def group_results(records):
    """Group results by (function, dim) → {sigma1: [records], 'vanilla': [records]}."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["function_type"], r["n_dim"])
        if r["method"] == "vanilla":
            groups[key]["vanilla"].append(r)
        else:
            groups[key][r["sigma1"]].append(r)
    return groups


def fig_advantage_curve(records, out_dir: Path):
    """
    Figure 1: DML advantage (%) vs gradient noise σ₁.
    One line per (function, dim). Shows crossover point.
    """
    setup_style()
    groups = group_results(records)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.5)

    for (func, dim) in sorted(groups.keys()):
        g = groups[(func, dim)]
        vanilla_mse = np.mean([r["test_value_mse"] for r in g["vanilla"]])

        sigmas = sorted(s for s in g if s != "vanilla")
        advantages = []
        for s in sigmas:
            dml_mse = np.mean([r["test_value_mse"] for r in g[s]])
            adv = (1 - dml_mse / vanilla_mse) * 100 if vanilla_mse > 0 else 0
            advantages.append(adv)

        label = f"{func} d={dim}"
        marker = FUNC_MARKERS.get(func, "o")
        ls = DIM_STYLES.get(dim, "-")

        # Use log scale for x but handle σ₁=0
        x = [max(s, 5e-4) for s in sigmas]
        ax.semilogx(x, advantages, marker=marker, ls=ls, lw=1.8,
                     markersize=5, label=label)

        # Mark crossover
        for i in range(len(advantages) - 1):
            if advantages[i] > 0 >= advantages[i + 1]:
                # Log-space interpolation
                if sigmas[i] > 0 and sigmas[i + 1] > 0:
                    sigma_star = np.exp(np.interp(
                        0, [advantages[i + 1], advantages[i]],
                        [np.log(sigmas[i + 1]), np.log(sigmas[i])]))
                else:
                    sigma_star = sigmas[i + 1] / 2
                ax.axvline(sigma_star, color=COLORS["crossover"], lw=0.8,
                           ls="--", alpha=0.6)
                ax.plot(sigma_star, 0, "*", color=COLORS["crossover"],
                        markersize=10, zorder=5)
                ax.annotate(f"σ₁*≈{sigma_star:.2f}",
                            (sigma_star, 0), fontsize=7,
                            textcoords="offset points", xytext=(5, 8))
                break

    ax.set_xlabel("Gradient noise σ₁")
    ax.set_ylabel("DML advantage over vanilla (%)")
    ax.set_title("Effect of Gradient Noise on DML Performance")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=min(-50, ax.get_ylim()[0]))

    path = out_dir / "noise_sweep_advantage.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig_mse_curves(records, out_dir: Path):
    """
    Figure 2: Test MSE vs σ₁ for both methods, one subplot per (function, dim).
    """
    setup_style()
    groups = group_results(records)
    configs = sorted(groups.keys())
    n = len(configs)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), squeeze=False)

    for idx, (func, dim) in enumerate(configs):
        ax = axes[0, idx]
        g = groups[(func, dim)]

        # Vanilla (constant line)
        van_mses = [r["test_value_mse"] for r in g["vanilla"]]
        van_mean, van_std = np.mean(van_mses), np.std(van_mses)

        sigmas = sorted(s for s in g if s != "vanilla")
        dml_means, dml_stds = [], []
        for s in sigmas:
            mses = [r["test_value_mse"] for r in g[s]]
            dml_means.append(np.mean(mses))
            dml_stds.append(np.std(mses))

        dml_means = np.array(dml_means)
        dml_stds = np.array(dml_stds)
        x = np.array([max(s, 5e-4) for s in sigmas])

        # Vanilla band
        ax.axhspan(van_mean - van_std, van_mean + van_std,
                    color=COLORS["vanilla"], alpha=0.15)
        ax.axhline(van_mean, color=COLORS["vanilla"], lw=1.5, ls="--",
                    label="Vanilla")

        # DML curve with error band
        ax.semilogx(x, dml_means, "o-", color=COLORS["dml"], lw=1.8,
                     markersize=4, label="DML (order 1)")
        ax.fill_between(x, dml_means - dml_stds, dml_means + dml_stds,
                         color=COLORS["dml"], alpha=0.15)

        ax.set_xlabel("Gradient noise σ₁")
        ax.set_ylabel("Test value MSE")
        ax.set_title(f"{func} d={dim}")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_yscale("log")

    fig.suptitle("Test MSE vs Gradient Noise Level", y=1.02)
    fig.tight_layout()
    path = out_dir / "noise_sweep_mse.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig_grad_mse(records, out_dir: Path):
    """
    Figure 3: Gradient prediction MSE vs injected noise — shows how noise
    propagates through the model.
    """
    setup_style()
    groups = group_results(records)

    fig, ax = plt.subplots(figsize=(6, 4))

    for (func, dim) in sorted(groups.keys()):
        g = groups[(func, dim)]
        sigmas = sorted(s for s in g if s != "vanilla")
        grad_mses = []
        for s in sigmas:
            grad_mses.append(np.mean([r["test_grad_mse"] for r in g[s]]))

        x = [max(s, 5e-4) for s in sigmas]
        marker = FUNC_MARKERS.get(func, "o")
        ls = DIM_STYLES.get(dim, "-")
        ax.semilogx(x, grad_mses, marker=marker, ls=ls, lw=1.8,
                     markersize=5, label=f"{func} d={dim}")

    ax.set_xlabel("Injected gradient noise σ₁")
    ax.set_ylabel("Test gradient MSE (clean targets)")
    ax.set_title("Gradient Quality Degradation Under Noise")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    path = out_dir / "noise_sweep_grad_mse.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    results_dir = PROJECT_ROOT / "results" / "gradient_noise_sweep"
    out_dir = PROJECT_ROOT / "figures" / "gradient_noise_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(results_dir)
    if not records:
        print(f"No results found in {results_dir}")
        sys.exit(1)

    print(f"Loaded {len(records)} results from {results_dir}")
    print(f"Generating figures in {out_dir}/\n")

    fig_advantage_curve(records, out_dir)
    fig_mse_curves(records, out_dir)
    fig_grad_mse(records, out_dir)

    print(f"\nDone — {len(list(out_dir.glob('*.png')))} figures saved")


if __name__ == "__main__":
    main()
