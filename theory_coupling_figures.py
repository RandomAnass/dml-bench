#!/usr/bin/env python3
"""
Theory–Experiment Coupling Figures — DML Benchmark.

Generates publication-quality plots that stand alone empirically
but overlay theoretical predictions from the DIC framework.

Figures:
  1. Empirical sample-scaling exponent β vs dimension
     (with theoretical reference bands for different k)
  2. DML advantage vs dimension (empirical) with DIC prediction curves
  3. Noise crossover σ* vs dimension (empirical + theoretical formula)
  4. σ* vs sample size n (empirical, with theoretical growth rate)
  5. Empirical advantage ratio vs DIC-predicted advantage (scatter)
  6. Gradient vs Value improvement ratio (theory predicts grad > val)

Usage:
    python theory_coupling_figures.py --tiers 1 2 4
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

# ============================================================================
# STYLE
# ============================================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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

COLORS = {
    "vanilla": "#0072B2",
    "dml_fixed": "#D55E00",
    "dml_gradnorm": "#009E73",
    "dml_relobralo": "#CC79A7",
    "theory": "#999999",
}

FUNC_COLORS = {
    "poly_trig": "#0072B2",
    "trig": "#D55E00",
    "step": "#009E73",
    "bachelier": "#CC79A7",
    "black_scholes": "#E69F00",
    "heston": "#56B4E9",
}

FUNC_LABELS = {
    "poly_trig": "Poly-Trig",
    "trig": "Trigonometric",
    "step": "Step",
    "bachelier": "Bachelier",
    "black_scholes": "Black-Scholes",
    "heston": "Heston",
}

FUNC_MARKERS = {
    "poly_trig": "o",
    "trig": "s",
    "step": "^",
    "bachelier": "D",
    "black_scholes": "v",
    "heston": "X",
}

NN_METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo"]
OUT_DIR = Path("figures")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_records(tiers):
    records = []
    for t in tiers:
        d = Path(f"results/tier{t}_benchmark")
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            if f.name == "summary.json":
                continue
            try:
                r = json.load(open(f))
                records.append({
                    "method": r["method"],
                    "func_type": r["func_type"],
                    "dim": int(r["dim"]),
                    "n_samples": int(r["n_samples"]),
                    "noise_level": float(r.get("noise_level", 0.0)),
                    "seed": r["seed"],
                    "lambda": float(r.get("lambda", 1.0)),
                    "test_value_mse": float(r["test_value_mse"]),
                    "test_grad_mse": float(r["test_grad_mse"]),
                    "time_s": float(r.get("time_s", 0)),
                })
            except Exception:
                pass
    return records


def filt(records, **kw):
    out = records
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


# ============================================================================
# THEORETICAL PREDICTIONS
# ============================================================================

def theory_beta(k, d, method="vanilla"):
    """Theoretical MSE ∝ n^{-β}. Returns β."""
    if method == "vanilla":
        return 2 * k / (2 * k + d)
    else:  # DML
        return 2 * k / (2 * k + d - 1)


def theory_dic(k, d):
    """DIC = exponent gain from gradient observations."""
    return 2 * k / ((2 * k + d - 1) * (2 * k + d))


def theory_advantage_ratio(k, d, n):
    """Predicted R_n^{(0)} / R_n^{(p)} ≈ n^{DIC}."""
    dic = theory_dic(k, d)
    return n ** dic


def theory_sigma_star(sigma0, n, k, d):
    """Approximate σ* ≈ σ₀ · n^{1/(2(2k+d))}."""
    return sigma0 * n ** (1.0 / (2 * (2 * k + d)))


# ============================================================================
# EMPIRICAL COMPUTATIONS
# ============================================================================

def compute_beta(records, func, dim, method):
    """Fit MSE ∝ n^{-β} from data. Returns (β, r², n_points)."""
    clean = [r for r in records if r["func_type"] == func
             and r["dim"] == dim and r["method"] == method
             and r["noise_level"] == 0.0
             and r.get("lambda", 1.0) == 1.0]

    ns_list = sorted(set(r["n_samples"] for r in clean))
    if len(ns_list) < 3:
        return None, None, 0

    means, ns_valid = [], []
    for ns in ns_list:
        vals = [r["test_value_mse"] for r in clean if r["n_samples"] == ns]
        if vals and np.mean(vals) > 1e-15:
            means.append(np.mean(vals))
            ns_valid.append(ns)

    if len(ns_valid) < 3:
        return None, None, 0

    log_n = np.log(np.array(ns_valid, dtype=float))
    log_mse = np.log(np.array(means))
    coeffs = np.polyfit(log_n, log_mse, 1)
    ss_res = np.sum((log_mse - np.polyval(coeffs, log_n)) ** 2)
    ss_tot = np.sum((log_mse - np.mean(log_mse)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return -coeffs[0], r2, len(ns_valid)


def compute_advantage(records, func, dim, n_samples=1024):
    """Compute % advantage of DML over vanilla."""
    clean = [r for r in records if r["func_type"] == func
             and r["dim"] == dim and r["n_samples"] == n_samples
             and r["noise_level"] == 0.0
             and r.get("lambda", 1.0) == 1.0]

    van = [r["test_value_mse"] for r in clean if r["method"] == "vanilla"]
    dml = [r["test_value_mse"] for r in clean if r["method"] == "dml_fixed"]

    if not van or not dml:
        return None
    v, d = np.mean(van), np.mean(dml)
    return (v - d) / v if v > 1e-15 else 0


def compute_sigma_star(records, func, dim, n_samples=1024):
    """Find σ* via linear interpolation where DML advantage crosses zero."""
    clean = [r for r in records if r["func_type"] == func
             and r["dim"] == dim and r["n_samples"] == n_samples
             and r.get("lambda", 1.0) == 1.0]

    noise_levels = sorted(set(r["noise_level"] for r in clean))
    if len(noise_levels) < 2:
        return None, {}

    advantages = {}
    for noise in noise_levels:
        van = [r["test_value_mse"] for r in clean
               if r["noise_level"] == noise and r["method"] == "vanilla"]
        dml = [r["test_value_mse"] for r in clean
               if r["noise_level"] == noise and r["method"] == "dml_fixed"]
        if van and dml:
            v, d = np.mean(van), np.mean(dml)
            advantages[noise] = (v - d) / v if v > 0 else 0

    sorted_noise = sorted(advantages.keys())
    sorted_adv = [advantages[n] for n in sorted_noise]

    # Find crossover
    sigma_star = None
    for i in range(len(sorted_adv) - 1):
        if sorted_adv[i] > 0 and sorted_adv[i + 1] <= 0:
            a1, a2 = sorted_adv[i], sorted_adv[i + 1]
            n1, n2 = sorted_noise[i], sorted_noise[i + 1]
            sigma_star = n1 + (n2 - n1) * a1 / (a1 - a2)
            break

    if sigma_star is None and all(a > 0 for a in sorted_adv):
        sigma_star = float("inf")  # DML always helps
    elif sigma_star is None and sorted_adv[0] <= 0:
        sigma_star = 0.0  # DML never helps

    return sigma_star, advantages


# ============================================================================
# FIGURE 1: EMPIRICAL β VS DIMENSION (+ THEORY BANDS)
# ============================================================================

def fig_beta_vs_dim(records, fmt="pdf"):
    """
    Empirical sample-scaling exponent β by dimension.
    Overlays theoretical prediction bands for k=2,3,5.
    STANDS ALONE: x-axis is dim, y-axis is empirical β.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for mi, method in enumerate(["vanilla", "dml_fixed"]):
        ax = axes[mi]
        method_label = "Vanilla" if method == "vanilla" else "DML (fixed λ)"

        # Empirical points per function
        for func in ["poly_trig", "trig", "step", "bachelier"]:
            dims_used, betas, r2s = [], [], []
            all_dims = sorted(set(r["dim"] for r in records
                                  if r["func_type"] == func))
            for dim in all_dims:
                beta, r2, npts = compute_beta(records, func, dim, method)
                if beta is not None and r2 is not None and r2 > 0.5:
                    dims_used.append(dim)
                    betas.append(beta)
                    r2s.append(r2)

            if dims_used:
                ax.scatter(dims_used, betas,
                           label=FUNC_LABELS.get(func, func),
                           color=FUNC_COLORS[func],
                           marker=FUNC_MARKERS[func],
                           s=40, zorder=5, edgecolors="white", linewidth=0.5)
                # Connect with thin line
                ax.plot(dims_used, betas, color=FUNC_COLORS[func],
                        alpha=0.4, linewidth=0.8, zorder=4)

        # Theory bands for different k values
        d_range = np.linspace(1, 105, 200)
        for k, ls, alpha in [(2, "--", 0.4), (3, "-.", 0.35), (5, ":", 0.3)]:
            beta_theory = [theory_beta(k, d, method) for d in d_range]
            ax.plot(d_range, beta_theory, ls, color=COLORS["theory"],
                    alpha=alpha, linewidth=1.5, label=f"Theory k={k}")

        ax.set_xlabel("Input dimension d")
        ax.set_ylabel("Sample scaling exponent β\n(MSE ∝ n$^{-β}$)")
        ax.set_title(method_label)
        ax.set_xlim(0, 110)
        ax.set_ylim(-0.5, 3.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=6, ncol=2, framealpha=0.7, loc="upper right")

    fig.suptitle("Sample Complexity Exponent: Empirical vs Theory", fontsize=11)
    plt.tight_layout()
    path = OUT_DIR / f"theory_beta_vs_dim.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 2: DML ADVANTAGE VS DIMENSION (+ DIC PREDICTIONS)
# ============================================================================

def fig_advantage_vs_dim(records, fmt="pdf"):
    """
    Empirical DML advantage (%) by dimension.
    Overlays DIC-predicted advantage for reference.
    STANDS ALONE: bars are purely empirical.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    funcs = ["poly_trig", "trig", "step", "bachelier"]
    all_dims = sorted(set(r["dim"] for r in records
                          if r["func_type"] in funcs and r["dim"] <= 100))

    bar_width = 0.18
    x = np.arange(len(all_dims))

    for fi, func in enumerate(funcs):
        advantages = []
        for dim in all_dims:
            adv = compute_advantage(records, func, dim, 1024)
            advantages.append(100 * adv if adv is not None else 0)

        offset = (fi - len(funcs) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, advantages, bar_width,
                      label=FUNC_LABELS[func], color=FUNC_COLORS[func],
                      edgecolor="white", linewidth=0.3, alpha=0.85)

    # DIC reference curve (k=3 and k=5 as envelope)
    for k, ls, label in [(3, "--", "DIC k=3"), (5, ":", "DIC k=5")]:
        dic_advantages = []
        for dim in all_dims:
            ratio = theory_advantage_ratio(k, dim, 1024)
            dic_advantages.append(100 * (1 - 1 / ratio))  # Convert ratio to % improvement
        ax.plot(x, dic_advantages, ls, color=COLORS["theory"],
                linewidth=2.0, alpha=0.6, label=label, zorder=10)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in all_dims])
    ax.set_xlabel("Input dimension d")
    ax.set_ylabel("DML advantage (%)\n(positive = DML better)")
    ax.set_title("DML Advantage vs Dimension (n=1024, σ=0)")
    ax.legend(fontsize=7, ncol=3, framealpha=0.7, loc="upper right")

    # Annotation: theory gap
    ax.annotate("Theory predicts\nmodest gains",
                xy=(3, 8), fontsize=7, color=COLORS["theory"],
                style="italic", alpha=0.7)

    plt.tight_layout()
    path = OUT_DIR / f"theory_advantage_vs_dim.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 3: NOISE CROSSOVER σ* VS DIMENSION
# ============================================================================

def fig_sigma_star_vs_dim(records, fmt="pdf"):
    """
    Empirical σ* (noise tolerance) vs dimension.
    Overlays theoretical scaling for reference.
    STANDS ALONE: the empirical σ* points are the main content.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: σ* vs dim
    ax = axes[0]
    for func in ["poly_trig", "trig", "step"]:
        func_recs = [r for r in records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))
        sigma_stars, used_dims = [], []

        for dim in dims:
            ss, advs = compute_sigma_star(records, func, dim, 1024)
            if ss is not None and ss > 0 and ss != float("inf"):
                sigma_stars.append(ss)
                used_dims.append(dim)
            elif ss == float("inf"):
                # DML always helps → σ* > 0.5 (our max noise)
                sigma_stars.append(0.55)
                used_dims.append(dim)

        if used_dims:
            # Distinguish finite σ* from "always helps"
            finite_mask = [s < 0.55 for s in sigma_stars]
            inf_mask = [s >= 0.55 for s in sigma_stars]

            dims_f = [d for d, m in zip(used_dims, finite_mask) if m]
            ss_f = [s for s, m in zip(sigma_stars, finite_mask) if m]
            dims_i = [d for d, m in zip(used_dims, inf_mask) if m]
            ss_i = [s for s, m in zip(sigma_stars, inf_mask) if m]

            if dims_f:
                ax.scatter(dims_f, ss_f, color=FUNC_COLORS[func],
                           marker=FUNC_MARKERS[func], s=60, zorder=5,
                           label=FUNC_LABELS[func], edgecolors="white", linewidth=0.5)
            if dims_i:
                ax.scatter(dims_i, ss_i, color=FUNC_COLORS[func],
                           marker=FUNC_MARKERS[func], s=60, zorder=5,
                           facecolors="none", edgecolors=FUNC_COLORS[func],
                           linewidth=1.5)

    # Add "DML never helps" zone for step
    step_dims = sorted(set(r["dim"] for r in records if r["func_type"] == "step"))
    for dim in step_dims:
        ss, _ = compute_sigma_star(records, "step", dim, 1024)
        if ss is not None and ss == 0:
            ax.scatter([dim], [0.0], color=FUNC_COLORS["step"],
                       marker="x", s=50, zorder=5, linewidth=1.5)

    # Shading
    ax.axhspan(0.0, 0.0, color="red", alpha=0.05)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.text(80, 0.52, "max tested σ", fontsize=6, color="gray", alpha=0.7)

    ax.set_xlabel("Input dimension d")
    ax.set_ylabel("Noise tolerance σ*\n(max σ where DML still helps)")
    ax.set_title("Derivative Tolerance Threshold σ*")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.set_ylim(-0.05, 0.65)
    ax.set_xlim(0, 110)

    # Right: advantage curves at different noise levels
    ax = axes[1]
    for func in ["poly_trig", "trig"]:
        func_recs = [r for r in records if r["func_type"] == func]
        noise_levels = sorted(set(r["noise_level"] for r in func_recs))
        dims = sorted(set(r["dim"] for r in func_recs))

        for dim in [2, 5, 10]:
            if dim not in dims:
                continue
            _, advantages = compute_sigma_star(records, func, dim, 1024)
            if advantages:
                noise_sorted = sorted(advantages.keys())
                adv_sorted = [100 * advantages[n] for n in noise_sorted]
                label = f"{FUNC_LABELS[func]} d={dim}"
                ls = "-" if func == "poly_trig" else "--"
                ax.plot(noise_sorted, adv_sorted, ls,
                        color=FUNC_COLORS[func], alpha=min(0.9, 0.5 + 0.05 * dim),
                        marker="o", markersize=3, label=label)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between([0, 0.5], 0, -1000, alpha=0.03, color="red")
    ax.fill_between([0, 0.5], 0, 1000, alpha=0.03, color="green")
    ax.set_xlabel("Derivative noise level σ")
    ax.set_ylabel("DML advantage (%)")
    ax.set_title("DML Advantage vs Noise Level")
    ax.legend(fontsize=6, ncol=2, framealpha=0.7)
    ax.set_ylim(-200, 110)
    ax.set_xlim(-0.02, 0.55)

    plt.tight_layout()
    path = OUT_DIR / f"theory_sigma_star.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 4: σ* VS SAMPLE SIZE n
# ============================================================================

def fig_sigma_star_vs_n(records, fmt="pdf"):
    """
    σ* at different sample sizes n, for fixed (func, dim).
    Theory predicts σ* ∝ n^{1/(2(2k+d))} — very slow growth.
    STANDS ALONE: empirical points with optional trend line.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    sample_sizes = sorted(set(r["n_samples"] for r in records))

    configs = [
        ("poly_trig", 2), ("poly_trig", 5),
        ("trig", 2), ("trig", 5),
    ]

    for func, dim in configs:
        sigma_stars, valid_ns = [], []
        for ns in sample_sizes:
            ss, advs = compute_sigma_star(records, func, dim, ns)
            if ss is not None and ss > 0 and ss != float("inf"):
                sigma_stars.append(ss)
                valid_ns.append(ns)

        if len(valid_ns) >= 2:
            ax.scatter(valid_ns, sigma_stars, color=FUNC_COLORS[func],
                       marker=FUNC_MARKERS[func], s=50, zorder=5,
                       edgecolors="white", linewidth=0.5,
                       label=f"{FUNC_LABELS[func]} d={dim}")

            # Fit empirical trend
            if len(valid_ns) >= 3:
                log_n = np.log(np.array(valid_ns, dtype=float))
                log_ss = np.log(np.array(sigma_stars))
                slope, intercept = np.polyfit(log_n, log_ss, 1)
                n_fit = np.linspace(min(valid_ns), max(valid_ns), 50)
                ss_fit = np.exp(intercept) * n_fit ** slope
                ax.plot(n_fit, ss_fit, "--", color=FUNC_COLORS[func],
                        alpha=0.4, linewidth=1.0)
                # Annotate slope
                ax.annotate(f"∝ n$^{{{slope:.2f}}}$",
                            xy=(valid_ns[-1], sigma_stars[-1]),
                            xytext=(10, 5), textcoords="offset points",
                            fontsize=6, color=FUNC_COLORS[func], alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Noise tolerance σ*")
    ax.set_title("How σ* Scales with Data Size")
    ax.legend(fontsize=7, framealpha=0.7)

    plt.tight_layout()
    path = OUT_DIR / f"theory_sigma_star_vs_n.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 5: EMPIRICAL VS DIC-PREDICTED ADVANTAGE (SCATTER)
# ============================================================================

def fig_dic_scatter(records, fmt="pdf"):
    """
    Scatter: x = DIC-predicted advantage ratio, y = empirical advantage ratio.
    Each point is one (func, dim, n) config.
    The gap between diagonal and data reveals NN structural exploitation.
    STANDS ALONE: interpretable as "does theory predict practice?"
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    data_points = []

    for func in ["poly_trig", "trig", "step", "bachelier"]:
        func_recs = [r for r in records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))

        for dim in dims:
            for ns in [256, 1024, 4096]:
                adv = compute_advantage(records, func, dim, ns)
                if adv is None:
                    continue

                # Empirical advantage ratio = 1/(1-adv) if adv < 1, else use raw
                if adv > 0 and adv < 1:
                    emp_ratio = 1.0 / (1.0 - adv)
                elif adv >= 1:
                    emp_ratio = 100.0  # Cap for display
                else:
                    emp_ratio = 1.0 / (1.0 - adv)  # < 1 when DML hurts

                # Try different k values and pick the one that best represents
                # Use k=3 as default
                for k in [3]:
                    dic_ratio = theory_advantage_ratio(k, dim, ns)

                data_points.append({
                    "func": func, "dim": dim, "n": ns,
                    "dic_ratio": dic_ratio, "emp_ratio": emp_ratio,
                })

    # Plot per function
    for func in ["poly_trig", "trig", "step", "bachelier"]:
        pts = [p for p in data_points if p["func"] == func]
        if pts:
            x = [p["dic_ratio"] for p in pts]
            y = [p["emp_ratio"] for p in pts]
            ax.scatter(x, y, color=FUNC_COLORS[func],
                       marker=FUNC_MARKERS[func], s=30, alpha=0.7,
                       label=FUNC_LABELS[func], edgecolors="white",
                       linewidth=0.3, zorder=5)

    # Diagonal (theory = practice)
    lims = [0.95, max(p["emp_ratio"] for p in data_points) * 1.1]
    ax.plot([0.95, lims[1]], [0.95, lims[1]], "-", color="gray",
            linewidth=1.0, alpha=0.5, label="Theory = Practice")

    # Below diagonal = theory overpredicts
    ax.fill_between([0.95, lims[1]], [0.95, lims[1]], 0.5,
                    alpha=0.03, color="red")
    ax.fill_between([0.95, lims[1]], [0.95, lims[1]], 200,
                    alpha=0.03, color="green")
    ax.text(1.05, 5, "Empirical > Theory\n(NN exploits structure)",
            fontsize=7, color="green", alpha=0.5, style="italic")
    ax.text(1.3, 0.7, "Theory > Empirical\n(assumption violated?)",
            fontsize=7, color="red", alpha=0.5, style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DIC-predicted advantage ratio (k=3)")
    ax.set_ylabel("Empirical advantage ratio\n(vanilla MSE / DML MSE)")
    ax.set_title("Theory vs Practice: DIC Predictions")
    ax.legend(fontsize=6, framealpha=0.7, loc="upper left")
    ax.set_xlim(0.98, lims[1])
    ax.set_ylim(0.3, max(200, lims[1]))

    plt.tight_layout()
    path = OUT_DIR / f"theory_dic_scatter.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 6: GRADIENT VS VALUE IMPROVEMENT
# ============================================================================

def fig_grad_vs_val_improvement(records, fmt="pdf"):
    """
    Scatter: x = value MSE improvement (%), y = gradient MSE improvement (%).
    Theory predicts points above the diagonal (grad improves more).
    STANDS ALONE: directly shows where derivatives help most.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    clean = [r for r in records if r["method"] in ["vanilla", "dml_fixed"]
             and r["noise_level"] == 0.0
             and r.get("lambda", 1.0) == 1.0]

    func_points = defaultdict(list)

    for func in ["poly_trig", "trig", "step", "bachelier", "black_scholes"]:
        func_recs = [r for r in clean if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))
        for dim in dims:
            for ns in [256, 1024, 4096]:
                van_val = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns
                           and r["method"] == "vanilla"]
                dml_val = [r["test_value_mse"] for r in func_recs
                           if r["dim"] == dim and r["n_samples"] == ns
                           and r["method"] == "dml_fixed"]
                van_grad = [r["test_grad_mse"] for r in func_recs
                            if r["dim"] == dim and r["n_samples"] == ns
                            and r["method"] == "vanilla"]
                dml_grad = [r["test_grad_mse"] for r in func_recs
                            if r["dim"] == dim and r["n_samples"] == ns
                            and r["method"] == "dml_fixed"]

                if van_val and dml_val and van_grad and dml_grad:
                    vv, dv = np.mean(van_val), np.mean(dml_val)
                    vg, dg = np.mean(van_grad), np.mean(dml_grad)
                    val_imp = 100 * (vv - dv) / vv if vv > 1e-15 else 0
                    grad_imp = 100 * (vg - dg) / vg if vg > 1e-15 else 0
                    func_points[func].append((val_imp, grad_imp, dim, ns))

    for func, points in func_points.items():
        if not points:
            continue
        val_imps = [p[0] for p in points]
        grad_imps = [p[1] for p in points]
        ax.scatter(val_imps, grad_imps, color=FUNC_COLORS[func],
                   marker=FUNC_MARKERS[func], s=30, alpha=0.7,
                   label=FUNC_LABELS[func], edgecolors="white",
                   linewidth=0.3, zorder=5)

    # Diagonal
    all_vals = [p[0] for pts in func_points.values() for p in pts]
    all_grads = [p[1] for pts in func_points.values() for p in pts]
    if all_vals and all_grads:
        lo = min(min(all_vals), min(all_grads), -10)
        hi = max(max(all_vals), max(all_grads), 10)
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.5)

    # Cross-hairs at origin
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    # Upper-left triangle annotation
    ax.text(0.05, 0.92, "Gradient improves more\nthan value (expected by FTC)",
            transform=ax.transAxes, fontsize=7, color="#009E73",
            style="italic", alpha=0.6, va="top")
    ax.text(0.92, 0.05, "Value improves\nmore than gradient",
            transform=ax.transAxes, fontsize=7, color="#D55E00",
            style="italic", alpha=0.6, ha="right")

    # Upper-left triangle shading
    if all_vals:
        ax.fill_between([lo, hi], [lo, hi], hi,
                        alpha=0.03, color="green")
        ax.fill_between([lo, hi], lo, [lo, hi],
                        alpha=0.03, color="orange")

    # Count
    above = sum(1 for pts in func_points.values()
                for p in pts if p[1] > p[0])
    total = sum(len(pts) for pts in func_points.values())
    ax.text(0.05, 0.82, f"Grad > Val: {above}/{total} configs",
            transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax.set_xlabel("Value MSE improvement (%)")
    ax.set_ylabel("Gradient MSE improvement (%)")
    ax.set_title("Gradient vs Value: Where Does DML Help Most?")
    ax.legend(fontsize=6, framealpha=0.7, loc="lower right")

    plt.tight_layout()
    path = OUT_DIR / f"theory_grad_vs_val.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 7: SMOOTHNESS ASSUMPTION VALIDATION
# ============================================================================

def fig_smoothness_matters(records, fmt="pdf"):
    """
    Compare DML advantage across smooth vs non-smooth functions.
    Theory: C^k satisfied → DML helps; violated → DML fails.
    STANDS ALONE: grouped bar chart, no theory needed to interpret.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    funcs_smooth = ["poly_trig", "bachelier", "trig"]
    funcs_nonsmooth = ["step"]
    all_funcs = funcs_smooth + funcs_nonsmooth

    bar_data = {}
    for func in all_funcs:
        advantages = []
        func_recs = [r for r in records if r["func_type"] == func]
        dims = sorted(set(r["dim"] for r in func_recs))
        for dim in dims:
            for ns in [256, 1024, 4096]:
                adv = compute_advantage(records, func, dim, ns)
                if adv is not None:
                    advantages.append(100 * adv)
        if advantages:
            bar_data[func] = {
                "mean": np.mean(advantages),
                "std": np.std(advantages),
                "wins": sum(1 for a in advantages if a > 0),
                "total": len(advantages),
            }

    x = np.arange(len(bar_data))
    funcs_ordered = list(bar_data.keys())
    means = [bar_data[f]["mean"] for f in funcs_ordered]
    stds = [bar_data[f]["std"] for f in funcs_ordered]
    wins = [bar_data[f]["wins"] for f in funcs_ordered]
    totals = [bar_data[f]["total"] for f in funcs_ordered]

    colors = [FUNC_COLORS[f] for f in funcs_ordered]
    edge_colors = ["black" if f in funcs_nonsmooth else "white"
                   for f in funcs_ordered]
    hatches = ["//" if f in funcs_nonsmooth else None
               for f in funcs_ordered]

    bars = ax.bar(x, means, yerr=stds, capsize=3,
                  color=colors, edgecolor=edge_colors, linewidth=1.0,
                  alpha=0.85)

    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    # Win rate labels
    for i, (w, t) in enumerate(zip(wins, totals)):
        y_pos = means[i] + stds[i] + 3
        ax.text(i, y_pos, f"{w}/{t} wins", ha="center", fontsize=7,
                fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_LABELS[f] for f in funcs_ordered])
    ax.set_ylabel("Mean DML advantage (%)")
    ax.set_title("Smoothness Determines DML Success")

    # Theory annotation
    ax.annotate("Smooth ($C^k$ satisfied) →\nDML helps",
                xy=(0.3, 0.85), xycoords="axes fraction",
                fontsize=8, color="#009E73", style="italic")
    ax.annotate("Non-smooth ($C^k$ violated) →\nDML hurts",
                xy=(0.7, 0.15), xycoords="axes fraction",
                fontsize=8, color="#D55E00", style="italic",
                ha="center")

    plt.tight_layout()
    path = OUT_DIR / f"theory_smoothness_matters.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Theory-Experiment Coupling Figures")
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--figure", default="all",
                        choices=["all", "beta", "advantage", "sigma_star",
                                 "sigma_star_n", "dic_scatter",
                                 "grad_val", "smoothness"])
    args = parser.parse_args()

    setup_style()
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading results...")
    records = load_records(args.tiers)
    print(f"Loaded {len(records)} results from tiers {args.tiers}")

    figures = {
        "beta": ("Fig 1: Sample scaling β vs dimension", fig_beta_vs_dim),
        "advantage": ("Fig 2: DML advantage vs dimension + DIC", fig_advantage_vs_dim),
        "sigma_star": ("Fig 3: Noise threshold σ* vs dimension", fig_sigma_star_vs_dim),
        "sigma_star_n": ("Fig 4: σ* vs sample size", fig_sigma_star_vs_n),
        "dic_scatter": ("Fig 5: DIC prediction vs empirical", fig_dic_scatter),
        "grad_val": ("Fig 6: Gradient vs value improvement", fig_grad_vs_val_improvement),
        "smoothness": ("Fig 7: Smoothness assumption validation", fig_smoothness_matters),
    }

    if args.figure == "all":
        for name, (desc, func) in figures.items():
            print(f"\n{desc}...")
            try:
                func(records, args.format)
            except Exception as e:
                import traceback
                print(f"  ⚠️ {name} failed: {e}")
                traceback.print_exc()
    else:
        desc, func = figures[args.figure]
        print(f"\n{desc}...")
        func(records, args.format)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
