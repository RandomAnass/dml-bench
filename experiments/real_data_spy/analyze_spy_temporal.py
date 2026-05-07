#!/usr/bin/env python3
"""
Analysis of SPY Real-World Options Experiments — Temporal Split.

Analyzes 100 experiments: 5 methods × 2 train sizes × 10 seeds.
Produces publication-grade figures and markdown report.

Sections:
  1. Summary Table  (mean ± std per method per train size)
  2. DML Improvement (relative to vanilla baseline)
  3. Statistical Tests (paired Wilcoxon + Holm-Bonferroni)
  4. Bootstrap CIs (95% confidence intervals)
  5. Figures:
     - Bar chart: Value & Gradient MSE per method (with CI whiskers)
     - Improvement heatmap: grad-MSE improvement over vanilla
     - Training time comparison
  6. Markdown Report (auto-generated)

Usage:
  python experiments/real_data_spy/analyze_spy_temporal.py
  python experiments/real_data_spy/analyze_spy_temporal.py --results-dir results/spy_options_temporal
  python experiments/real_data_spy/analyze_spy_temporal.py --save-report
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dml_benchmark.stats import (
        paired_wilcoxon_test, bootstrap_ci, cohens_d,
        effect_size_label, holm_bonferroni,
    )
    HAS_STATS = True
except ImportError:
    HAS_STATS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_RESULTS_DIR = Path("results/spy_options_temporal")
FIGURE_DIR = Path("figures/spy_temporal")

METHODS = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"]

METHOD_LABELS = {
    "vanilla":       "Vanilla",
    "dml_fixed":     "DML fixed λ",
    "dml_gradnorm":  "DML GradNorm",
    "dml_relobralo": "DML ReLoBRaLo",
    "dml_warmup":    "DML Warmup",
}

METHOD_COLORS = {
    "vanilla":       "#4C72B0",
    "dml_fixed":     "#55A868",
    "dml_gradnorm":  "#C44E52",
    "dml_relobralo": "#8172B2",
    "dml_warmup":    "#CCB974",
}


# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_results(results_dir: Path):
    """Load all JSON results."""
    results = {}
    for f in results_dir.glob("*.json"):
        if f.name in ("summary.json", "analysis.json", "report.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                key = data.get("key", f.stem)
                results[key] = data
        except Exception:
            pass
    return results


def group_results(results):
    """Group by (n_train, method) -> list of result dicts."""
    groups = defaultdict(list)
    for key, res in results.items():
        n_train = res.get("n_train", 0)
        method = res.get("method", "?")
        groups[(n_train, method)].append(res)
    return dict(groups)


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary(groups):
    """Print per-method summary with mean ± std."""
    train_sizes = sorted(set(k[0] for k in groups))

    for n_train in train_sizes:
        print(f"\n{'='*80}")
        print(f"n_train = {n_train:,}")
        print(f"{'='*80}")
        print(f"  {'Method':<20} {'Value MSE':>16} {'Grad MSE':>16} {'Time (s)':>10} {'N':>4}")
        print(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*10} {'-'*4}")

        for method in METHODS:
            key = (n_train, method)
            if key not in groups:
                continue
            vals = [r["test_value_mse"] for r in groups[key]]
            grads = [r["test_grad_mse"] for r in groups[key]]
            times = [r.get("time_s", 0) for r in groups[key]]

            mean_v, std_v = np.mean(vals), np.std(vals)
            mean_g, std_g = np.mean(grads), np.std(grads)
            mean_t = np.mean(times)
            n = len(vals)

            print(
                f"  {METHOD_LABELS.get(method, method):<20} "
                f"{mean_v:.4e}±{std_v:.2e} "
                f"{mean_g:.4e}±{std_g:.2e} "
                f"{mean_t:10.1f} {n:4d}"
            )


# ============================================================================
# DML IMPROVEMENT ANALYSIS
# ============================================================================

def compute_improvements(groups):
    """Compute improvement of each DML method vs vanilla."""
    train_sizes = sorted(set(k[0] for k in groups))
    improvements = {}

    for n_train in train_sizes:
        vanilla_key = (n_train, "vanilla")
        if vanilla_key not in groups:
            continue
        vanilla_val = np.mean([r["test_value_mse"] for r in groups[vanilla_key]])
        vanilla_grad = np.mean([r["test_grad_mse"] for r in groups[vanilla_key]])

        for method in METHODS:
            if method == "vanilla":
                continue
            key = (n_train, method)
            if key not in groups:
                continue
            method_val = np.mean([r["test_value_mse"] for r in groups[key]])
            method_grad = np.mean([r["test_grad_mse"] for r in groups[key]])

            val_penalty_pct = (method_val - vanilla_val) / vanilla_val * 100
            grad_improvement_x = vanilla_grad / method_grad if method_grad > 0 else float("inf")

            improvements[(n_train, method)] = {
                "val_penalty_pct": val_penalty_pct,
                "grad_improvement_x": grad_improvement_x,
                "method_val_mse": method_val,
                "method_grad_mse": method_grad,
                "vanilla_val_mse": vanilla_val,
                "vanilla_grad_mse": vanilla_grad,
            }

    return improvements


def print_improvements(improvements):
    """Print DML improvement table."""
    train_sizes = sorted(set(k[0] for k in improvements))
    print(f"\n{'='*80}")
    print("DML IMPROVEMENT vs VANILLA")
    print(f"{'='*80}")

    for n_train in train_sizes:
        print(f"\n  n_train = {n_train:,}")
        print(f"  {'Method':<20} {'Val Penalty %':>14} {'Grad Improve ×':>15}")
        print(f"  {'-'*20} {'-'*14} {'-'*15}")

        for method in METHODS:
            if method == "vanilla":
                continue
            key = (n_train, method)
            if key not in improvements:
                continue
            imp = improvements[key]
            sign = "+" if imp["val_penalty_pct"] > 0 else ""
            print(
                f"  {METHOD_LABELS.get(method, method):<20} "
                f"{sign}{imp['val_penalty_pct']:13.1f}% "
                f"{imp['grad_improvement_x']:14.1f}×"
            )


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_statistical_tests(groups):
    """Run paired Wilcoxon tests between each DML method and vanilla."""
    if not HAS_STATS:
        print("  [stats module not available — skipping]")
        return {}

    train_sizes = sorted(set(k[0] for k in groups))
    all_tests = {}

    for n_train in train_sizes:
        vanilla_key = (n_train, "vanilla")
        if vanilla_key not in groups:
            continue

        # Sort by seed to ensure pairing
        vanilla_grads = sorted(groups[vanilla_key], key=lambda r: r.get("seed", 0))
        vanilla_g = [r["test_grad_mse"] for r in vanilla_grads]
        vanilla_v = [r["test_value_mse"] for r in vanilla_grads]

        print(f"\n  n_train = {n_train:,}: Paired Wilcoxon (Grad MSE)")
        print(f"  {'Method':<20} {'p-value':>10} {'Effect (d)':>12} {'Signif':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8}")

        p_values = []
        for method in METHODS:
            if method == "vanilla":
                continue
            key = (n_train, method)
            if key not in groups:
                continue

            method_grads = sorted(groups[key], key=lambda r: r.get("seed", 0))
            method_g = [r["test_grad_mse"] for r in method_grads]

            if len(vanilla_g) != len(method_g) or len(vanilla_g) < 5:
                continue

            result = paired_wilcoxon_test(
                np.array(vanilla_g), np.array(method_g)
            )
            stat = result["statistic"]
            p = result["p_value"]
            d = cohens_d(np.array(vanilla_g), np.array(method_g))
            es = effect_size_label(abs(d))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            p_values.append((method, p))
            all_tests[(n_train, method)] = {
                "statistic": float(stat), "p_value": float(p),
                "cohens_d": float(d), "effect_label": es,
            }

            print(
                f"  {METHOD_LABELS.get(method, method):<20} "
                f"{p:10.4f} {d:12.2f} ({es}) {sig:>8}"
            )

        # Holm-Bonferroni correction
        if p_values:
            names = [p[0] for p in p_values]
            pvals = [p[1] for p in p_values]
            adjusted = holm_bonferroni(pvals)
            print(f"\n  Holm-Bonferroni adjusted p-values:")
            for name, adj_result in zip(names, adjusted):
                adj_p = adj_result["adjusted_p"]
                sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
                print(f"    {METHOD_LABELS.get(name, name):<20} p_adj={adj_p:.4f} {sig}")

    return all_tests


# ============================================================================
# BOOTSTRAP CIs
# ============================================================================

def compute_bootstrap_cis(groups, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap 95% CIs for each (n_train, method)."""
    if not HAS_STATS:
        return {}

    cis = {}
    for (n_train, method), res_list in groups.items():
        vals = np.array([r["test_value_mse"] for r in res_list])
        grads = np.array([r["test_grad_mse"] for r in res_list])

        if len(vals) < 3:
            # Not enough for meaningful bootstrap
            cis[(n_train, method)] = {
                "value_mse": {"mean": float(np.mean(vals)), "ci_low": float(np.mean(vals) - np.std(vals)), "ci_high": float(np.mean(vals) + np.std(vals))},
                "grad_mse": {"mean": float(np.mean(grads)), "ci_low": float(np.mean(grads) - np.std(grads)), "ci_high": float(np.mean(grads) + np.std(grads))},
            }
            continue

        val_ci = bootstrap_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha)
        grad_ci = bootstrap_ci(grads, n_bootstrap=n_bootstrap, alpha=alpha)

        cis[(n_train, method)] = {
            "value_mse": {"mean": float(np.mean(vals)), "ci_low": float(val_ci["ci_lower"]), "ci_high": float(val_ci["ci_upper"])},
            "grad_mse": {"mean": float(np.mean(grads)), "ci_low": float(grad_ci["ci_lower"]), "ci_high": float(grad_ci["ci_upper"])},
        }
    return cis


# ============================================================================
# FIGURES
# ============================================================================

def plot_bar_chart(groups, cis, figure_dir):
    """Bar chart: Value & Gradient MSE per method, grouped by train size."""
    if not HAS_MPL:
        print("  [matplotlib not available — skipping figures]")
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    train_sizes = sorted(set(k[0] for k in groups))

    for n_train in train_sizes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        methods_present = [m for m in METHODS if (n_train, m) in groups]
        x = np.arange(len(methods_present))
        labels = [METHOD_LABELS.get(m, m) for m in methods_present]
        colors = [METHOD_COLORS.get(m, "#999999") for m in methods_present]

        for ax_idx, (metric, title) in enumerate([
            ("test_value_mse", "Value MSE"),
            ("test_grad_mse", "Gradient MSE"),
        ]):
            ax = axes[ax_idx]
            means = []
            ci_lows = []
            ci_highs = []

            for method in methods_present:
                vals = [r[metric] for r in groups[(n_train, method)]]
                m = np.mean(vals)
                means.append(m)
                if cis and (n_train, method) in cis:
                    ci_key = "value_mse" if metric == "test_value_mse" else "grad_mse"
                    ci_lows.append(m - cis[(n_train, method)][ci_key]["ci_low"])
                    ci_highs.append(cis[(n_train, method)][ci_key]["ci_high"] - m)
                else:
                    std = np.std(vals)
                    ci_lows.append(std)
                    ci_highs.append(std)

            bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
            ax.errorbar(x, means, yerr=[ci_lows, ci_highs], fmt="none",
                        ecolor="black", capsize=4, capthick=1)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(title)
            ax.set_title(f"{title} (n_train={n_train:,})")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"SPY Options — Temporal Split (n_train={n_train:,})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        fname = figure_dir / f"spy_bar_n{n_train}.pdf"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_grad_improvement(improvements, figure_dir):
    """Heatmap: gradient improvement factor for each method × train size."""
    if not HAS_MPL:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    train_sizes = sorted(set(k[0] for k in improvements))
    dml_methods = [m for m in METHODS if m != "vanilla"]

    data = np.zeros((len(dml_methods), len(train_sizes)))
    for i, method in enumerate(dml_methods):
        for j, n_train in enumerate(train_sizes):
            key = (n_train, method)
            if key in improvements:
                data[i, j] = improvements[key]["grad_improvement_x"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(train_sizes)))
    ax.set_xticklabels([f"{n:,}" for n in train_sizes], fontsize=10)
    ax.set_yticks(range(len(dml_methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in dml_methods], fontsize=10)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("DML Method")
    ax.set_title("Gradient MSE Improvement over Vanilla (×)")

    for i in range(len(dml_methods)):
        for j in range(len(train_sizes)):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.0f}×", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Improvement ×")
    plt.tight_layout()

    fname = figure_dir / "spy_grad_improvement.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_bs_comparison(groups, figure_dir):
    """Compare model predictions vs Black-Scholes benchmark."""
    if not HAS_MPL:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    train_sizes = sorted(set(k[0] for k in groups))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METHODS))
    width = 0.35

    for i, n_train in enumerate(train_sizes):
        means = []
        stds = []
        for method in METHODS:
            if (n_train, method) in groups:
                vals = [np.sqrt(r["test_value_mse"]) for r in groups[(n_train, method)]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, label=f"n_train={n_train:,}",
                      yerr=stds, capsize=3, edgecolor="black", linewidth=0.5)

    # Add BS benchmark line
    bs_rmses = []
    for res_list in groups.values():
        for r in res_list:
            if r.get("bs_vs_mid_rmse"):
                bs_rmses.append(r["bs_vs_mid_rmse"])
    if bs_rmses:
        bs_mean = np.mean(bs_rmses)
        ax.axhline(y=bs_mean, color="red", linestyle="--", linewidth=2,
                    label=f"BS analytical (RMSE={bs_mean:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in METHODS],
                        rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Value RMSE")
    ax.set_title("Model Value RMSE vs Black-Scholes Benchmark")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fname = figure_dir / "spy_vs_bs.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ============================================================================
# MARKDOWN REPORT
# ============================================================================

def generate_report(groups, improvements, tests, cis, results_dir):
    """Generate markdown report."""
    lines = []
    lines.append("# SPY Real-World Options — Temporal Split Analysis")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nResults directory: `{results_dir}`\n")

    train_sizes = sorted(set(k[0] for k in groups))
    total = sum(len(v) for v in groups.values())
    lines.append(f"**Total experiments:** {total}")
    lines.append(f"**Train sizes:** {train_sizes}")
    lines.append(f"**Methods:** {METHODS}")

    # Get metadata from first result
    first_res = next(iter(next(iter(groups.values()))))
    meta = first_res.get("metadata", {})
    if meta:
        lines.append(f"\n## Data & Split")
        lines.append(f"- Split mode: **{meta.get('split_mode', 'temporal')}**")
        lines.append(f"- Train period: ≤ {meta.get('train_end_date', '?')}")
        lines.append(f"- Test period: ≥ {meta.get('test_start_date', '?')}")
        lines.append(f"- Embargo: {meta.get('embargo_days', '?')} trading days "
                      f"({meta.get('n_embargo_excluded', '?')} samples excluded)")
        lines.append(f"- Train pool: {meta.get('train_pool_size', '?'):,} samples")
        lines.append(f"- Test pool: {meta.get('test_pool_size', '?'):,} samples")

    # Summary tables
    for n_train in train_sizes:
        lines.append(f"\n## Results: n_train = {n_train:,}")
        lines.append(f"\n| Method | Value MSE (mean±std) | Grad MSE (mean±std) | Time (s) | N |")
        lines.append(f"|--------|---------------------|--------------------:|--------:|--:|")

        for method in METHODS:
            key = (n_train, method)
            if key not in groups:
                continue
            vals = [r["test_value_mse"] for r in groups[key]]
            grads = [r["test_grad_mse"] for r in groups[key]]
            times = [r.get("time_s", 0) for r in groups[key]]

            lines.append(
                f"| {METHOD_LABELS.get(method, method)} "
                f"| {np.mean(vals):.4e} ± {np.std(vals):.2e} "
                f"| {np.mean(grads):.4e} ± {np.std(grads):.2e} "
                f"| {np.mean(times):.0f} | {len(vals)} |"
            )

    # Improvements
    if improvements:
        lines.append(f"\n## DML Improvement over Vanilla")
        for n_train in train_sizes:
            lines.append(f"\n### n_train = {n_train:,}")
            lines.append(f"\n| Method | Value Penalty | Grad Improvement |")
            lines.append(f"|--------|-------------:|----------------:|")

            for method in METHODS:
                if method == "vanilla":
                    continue
                key = (n_train, method)
                if key not in improvements:
                    continue
                imp = improvements[key]
                sign = "+" if imp["val_penalty_pct"] > 0 else ""
                lines.append(
                    f"| {METHOD_LABELS.get(method, method)} "
                    f"| {sign}{imp['val_penalty_pct']:.1f}% "
                    f"| {imp['grad_improvement_x']:.0f}× |"
                )

    # Statistical tests
    if tests:
        lines.append(f"\n## Statistical Significance (Paired Wilcoxon)")
        for n_train in train_sizes:
            lines.append(f"\n### n_train = {n_train:,}")
            lines.append(f"\n| Method | p-value | Cohen's d | Effect |")
            lines.append(f"|--------|--------:|----------:|--------|")

            for method in METHODS:
                if method == "vanilla":
                    continue
                key = (n_train, method)
                if key not in tests:
                    continue
                t = tests[key]
                sig = "***" if t["p_value"] < 0.001 else "**" if t["p_value"] < 0.01 else "*" if t["p_value"] < 0.05 else "ns"
                lines.append(
                    f"| {METHOD_LABELS.get(method, method)} "
                    f"| {t['p_value']:.4f} {sig} "
                    f"| {t['cohens_d']:.2f} "
                    f"| {t['effect_label']} |"
                )

    report_text = "\n".join(lines)
    report_path = results_dir / "ANALYSIS_REPORT.md"
    report_path.write_text(report_text)
    print(f"\n  Report saved: {report_path}")

    return report_text


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze SPY temporal experiments")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--save-report", action="store_true", default=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} experiments")

    if not results:
        print("No results found. Run experiments first.")
        return

    groups = group_results(results)

    # 1. Summary
    print_summary(groups)

    # 2. Improvements
    improvements = compute_improvements(groups)
    print_improvements(improvements)

    # 3. Statistical tests
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")
    tests = run_statistical_tests(groups)

    # 4. Bootstrap CIs
    cis = compute_bootstrap_cis(groups) if HAS_STATS else {}

    # 5. Figures
    if not args.no_figures:
        print(f"\n{'='*80}")
        print("FIGURES")
        print(f"{'='*80}")
        plot_bar_chart(groups, cis, FIGURE_DIR)
        plot_grad_improvement(improvements, FIGURE_DIR)
        plot_bs_comparison(groups, FIGURE_DIR)

    # 6. Report
    if args.save_report:
        generate_report(groups, improvements, tests, cis, results_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
