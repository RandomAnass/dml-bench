#!/usr/bin/env python3
"""
Figure (HIGH-ROI defense §8.4): four-panel feature-distribution histogram
for SPY options, train (dates < 2021-07-01) vs test (dates >= 2021-07-01).
Demonstrates that the temporal split produces a real distribution shift
(rate-hike regime change) — pre-empts the "test set isn't really new"
reviewer concern.

Reads the SPY options npz directly (data/spy_options/spy_processed.npz) and
splits by the same DEFAULT_TEMPORAL_CUTOFF used in the SPY runner
(2021-07-01). Draws histograms for the four model-input features
(moneyness, T, r, iv).

Output: latex/figures/fig_spy_distribution_shift.pdf
        + .summary.txt with mean ± std per feature per split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/spy_options/spy_processed.npz"
OUT = ROOT / "papers/neurips_DB/latex/figures/fig_spy_distribution_shift.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

CUTOFF = "2021-07-01"
TRAIN_COLOR = "#0072B2"  # darkblue
TEST_COLOR = "#D55E00"   # vermillion

FEATURE_NAMES = ["moneyness", "T (yrs)", "r", "iv"]
FEATURE_LIMITS = [
    (0.7, 1.4),     # moneyness
    (0.0, 1.0),     # T
    (-0.005, 0.06), # r
    (0.05, 0.6),    # iv
]


def main():
    if not DATA.exists():
        print(f"missing data file: {DATA}", file=sys.stderr)
        sys.exit(1)
    d = np.load(DATA)
    X = d["X"]                       # (n, 5): moneyness, T, r, iv, log_volume
    dates = d["dates"]                # (n,) string YYYY-MM-DD
    train_mask = dates < CUTOFF
    test_mask = dates >= CUTOFF
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0))
    for i, (ax, name, (lo, hi)) in enumerate(zip(axes, FEATURE_NAMES, FEATURE_LIMITS)):
        train_vals = X[train_mask, i]
        test_vals = X[test_mask, i]
        # Trim to limits for visualization (does not affect counts shown)
        train_vis = train_vals[(train_vals >= lo) & (train_vals <= hi)]
        test_vis = test_vals[(test_vals >= lo) & (test_vals <= hi)]
        bins = np.linspace(lo, hi, 41)
        ax.hist(train_vis, bins=bins, alpha=0.55, color=TRAIN_COLOR,
                density=True, label=f"train (<{CUTOFF}, n={n_train:,})")
        ax.hist(test_vis, bins=bins, alpha=0.55, color=TEST_COLOR,
                density=True, label=f"test (>={CUTOFF}, n={n_test:,})")
        ax.set_title(name)
        ax.set_xlabel(name)
        if i == 0:
            ax.set_ylabel("density")
        ax.grid(alpha=0.20)

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"SPY options feature distributions: train (pre-rate-hike) vs test (rate-hike regime)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)

    # Also dump a quick summary table to txt
    summary_path = OUT.with_suffix(".summary.txt")
    lines = [
        "SPY temporal split — feature distribution summary",
        f"  cutoff:   {CUTOFF}",
        f"  n_train:  {n_train:,}",
        f"  n_test:   {n_test:,}",
        "",
        "  Feature        train mean ± std       test mean ± std",
    ]
    for i, name in enumerate(FEATURE_NAMES):
        tr = X[train_mask, i]; te = X[test_mask, i]
        lines.append(
            f"  {name:<13s}  {tr.mean():+.4f} ± {tr.std():.4f}    "
            f"{te.mean():+.4f} ± {te.std():.4f}"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"figure_id=F_spy_dist n_train={n_train} n_test={n_test} output={OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
