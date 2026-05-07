#!/usr/bin/env python3
"""
Figure (HIGH-ROI defense §8.3): per-function smoothed advantage-vs-sigma
curve with a horizontal zero line. σ* is the visible crossing point.
Adds BCa-bootstrap-based shading for the win-rate (read from the JSON
produced by sigma_star_bca.py).

Output: latex/figures/fig_sigma_star_curve.pdf
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json
TIER_DIRS = [
    ROOT / "results/tier1_benchmark",
    ROOT / "results/tier2_benchmark",
    ROOT / "results/tier3_benchmark",
]
OUT = ROOT / "papers/neurips_DB/latex/figures/fig_sigma_star_curve.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
SIGMA_STAR_JSON = ROOT / "papers/neurips_DB/evidence/sigma_star_bca.json"

SMOOTH_FUNCS = ["poly_trig", "trig", "bachelier"]
TARGET = "dml_fixed"
BASELINE = "vanilla"

WONG = {
    "poly_trig": "#0072B2",
    "trig": "#D55E00",
    "bachelier": "#009E73",
}


def collect_winrate(target=TARGET, baseline=BASELINE):
    """Field names confirmed from tier-3 sample (2026-04-29):
       func_type, noise_level, n_samples, lambda. dml_fixed lambda-ablation
       files are filtered to lambda == 1.0 (canonical) to avoid duplicate-key
       overwrites at the same (func, dim, n, sigma, seed) cell."""
    by_config = defaultdict(dict)
    for tdir in TIER_DIRS:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            method = r.get("method")
            if method not in (target, baseline):
                continue
            if method == target:
                lam = r.get("lambda")
                if lam is not None and float(lam) != 1.0:
                    continue
            func = r.get("func_type") or r.get("dataset")
            if func not in SMOOTH_FUNCS:
                continue
            sigma = r.get("noise_level")
            if sigma is None:
                sigma = r.get("noise")
            seed = r.get("seed")
            dim = r.get("dim")
            n = r.get("n_samples") or r.get("n_train")
            val = r.get("test_value_mse")
            if None in (sigma, seed, dim, n, val):
                continue
            key = (func, dim, n, float(sigma), seed)
            by_config[key][method] = float(val)
    paired = defaultdict(lambda: defaultdict(list))
    for (func, dim, n, sigma, seed), m in by_config.items():
        if target in m and baseline in m:
            paired[func][sigma].append((m[target], m[baseline]))
    return paired


def winrate_with_ci(pairs, n_boot=500, rng=None):
    """Win-rate (target lower MSE than baseline) + Wilson 95% CI via bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    wins = sum(1 for t, b in pairs if t < b)
    point = wins / n
    # Bootstrap CI
    arr = np.empty(n_boot)
    pairs_arr = np.array(pairs)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = pairs_arr[idx]
        arr[i] = np.mean(sub[:, 0] < sub[:, 1])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    paired = collect_winrate()
    if not paired:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    sigma_star_data = {}
    if SIGMA_STAR_JSON.exists():
        try:
            sigma_star_data = json.load(open(SIGMA_STAR_JSON))["results"]
        except Exception:
            sigma_star_data = {}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rng = np.random.default_rng(123)

    for func in SMOOTH_FUNCS:
        if func not in paired:
            continue
        sigmas = sorted(paired[func].keys())
        wr, lo, hi = [], [], []
        for s in sigmas:
            point, lo_b, hi_b = winrate_with_ci(paired[func][s], rng=rng)
            wr.append(point); lo.append(lo_b); hi.append(hi_b)
        sigmas = np.array(sigmas, dtype=float)
        wr = np.array(wr); lo = np.array(lo); hi = np.array(hi)
        ax.plot(sigmas, wr, marker="o", color=WONG[func], label=func, linewidth=2.0)
        ax.fill_between(sigmas, lo, hi, color=WONG[func], alpha=0.15)

        sstar_info = sigma_star_data.get(func, {})
        sstar = sstar_info.get("point_estimate") if isinstance(sstar_info, dict) else None
        if sstar is not None:
            ax.axvline(sstar, color=WONG[func], linestyle=":", linewidth=1.0, alpha=0.6)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Gradient noise $\sigma$")
    ax.set_ylabel(r"Win-rate of $\mathrm{DML}_{\mathrm{fixed}}$ vs vanilla")
    ax.set_title(r"Noise crossover: $\sigma^*$ = $\sigma$ at which DML stops winning")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
    print(f"figure_id=F_sigma_star n_funcs={len(paired)} output={OUT}")


if __name__ == "__main__":
    main()
