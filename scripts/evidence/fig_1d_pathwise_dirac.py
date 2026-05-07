#!/usr/bin/env python3
"""
Figure (HIGH-ROI #2 from inspiration §10.2): three-panel illustrative plot of
the digital payoff and three derivative-label paradigms.

  Panel A: digital payoff f(x) = 1[x > K]
  Panel B: pathwise gradient (Dirac spike at K — drawn as an arrow + zero
           function elsewhere)
  Panel C: fuzzy call-spread approximation 1/eps · 1[|x-K|<eps/2]
           (the smooth bump used in DML-Bench's fuzzy method)

Uses the Wong-2011 colorblind palette inline (no project dep).

Output: latex/figures/fig_1d_pathwise_dirac.pdf
Run from anywhere; uses absolute paths.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[3]   # repo root
OUT = ROOT / "papers/neurips_DB/latex/figures/fig_1d_pathwise_dirac.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Wong 2011 palette
WONG = {
    "black": "#000000", "orange": "#E69F00", "blue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "darkblue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}

# Domain
K = 1.0           # strike
EPS = 0.30        # fuzzy bandwidth (visible at this plot resolution)
x = np.linspace(0.0, 2.0, 1001)


def digital(x, K=K):
    return (x > K).astype(float)


def fuzzy_kernel(x, K=K, eps=EPS):
    """Indicator-of-interval / eps. Integrates to 1 (Dirac approx)."""
    return ((np.abs(x - K) < eps / 2.0).astype(float)) / eps


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharex=True)

# ---- Panel A: digital payoff ----
ax = axes[0]
ax.plot(x, digital(x), color=WONG["black"], linewidth=2.0)
ax.axvline(K, color=WONG["vermillion"], linestyle=":", linewidth=1.0, alpha=0.7)
ax.set_title(r"(A) Payoff: $f(x) = \mathbf{1}[x > K]$")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_ylim(-0.20, 1.30)
ax.set_yticks([0.0, 1.0])
ax.text(K + 0.04, 0.50, "$K$", color=WONG["vermillion"], fontsize=11)
ax.grid(alpha=0.25)

# ---- Panel B: pathwise gradient (Dirac spike) ----
ax = axes[1]
ax.plot(x, np.zeros_like(x), color=WONG["black"], linewidth=2.0)
# Stylised Dirac arrow at x = K
arrow = FancyArrowPatch(
    (K, 0.0), (K, 1.20),
    arrowstyle="-|>", mutation_scale=22,
    color=WONG["darkblue"], linewidth=2.5, alpha=0.95,
)
ax.add_patch(arrow)
ax.text(K + 0.05, 0.95, r"$\delta(x{-}K)$",
        color=WONG["darkblue"], fontsize=12)
ax.axvline(K, color=WONG["vermillion"], linestyle=":", linewidth=1.0, alpha=0.7)
ax.set_title(r"(B) Pathwise gradient: $f'(x) = \delta(x{-}K)$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$f'(x)$")
ax.set_ylim(-0.20, 1.30)
ax.set_yticks([0.0, 1.0])
ax.grid(alpha=0.25)

# ---- Panel C: fuzzy call-spread derivative ----
ax = axes[2]
ax.plot(x, fuzzy_kernel(x), color=WONG["green"], linewidth=2.5)
ax.axvline(K, color=WONG["vermillion"], linestyle=":", linewidth=1.0, alpha=0.7)
# Shade the bandwidth
ax.fill_between(
    x, 0, fuzzy_kernel(x),
    where=(np.abs(x - K) < EPS / 2.0),
    color=WONG["green"], alpha=0.15,
)
ax.set_title(r"(C) Fuzzy approx.: $\widehat{f}'_\varepsilon(x) = \frac{1}{\varepsilon}\,\mathbf{1}[|x{-}K|<\frac{\varepsilon}{2}]$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\widehat{f}'_\varepsilon(x)$")
ax.set_ylim(-0.20 / EPS, 1.30 / EPS)
ax.text(K + EPS / 2 + 0.05, 1.0 / EPS * 0.6, fr"$\varepsilon = {EPS}$",
        color=WONG["green"], fontsize=11)
ax.grid(alpha=0.25)

fig.suptitle(
    "Three derivative-label paradigms on a digital payoff",
    fontsize=12, y=1.02,
)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
print(f"figure_id=F_1D_dirac panels=3 K={K} eps={EPS} output={OUT}")
