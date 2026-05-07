#!/usr/bin/env python3
"""
S4: Dolan & Moré (2002) performance profiles for the synthetic
DML-vs-vanilla comparison, stratified by function family.

Per-cell statistic: R1 (paired log-ratio, canonical per
`research_notes/ratio_definitions.md`). For each (func, dim, n, sigma) cell,
match DML and vanilla results by seed, compute the per-seed paired ratio
log10(MSE_DML / MSE_vanilla), and aggregate as the cell-level median in
log space. The cell statistic is exp10(median) — a paired-log-ratio
median, not a between-pool ratio of medians.

Plot rho(tau) = fraction of cells with r <= tau, for tau in [10^-4, 10^4]
on log axis. One subplot per function family; one curve per sigma. The
ECDF still operates on the per-cell statistic; only the per-cell
aggregation has changed (R2 -> R1).

Output:
  papers/neurips_DB/latex/figures/fig_perf_profile_value.pdf
  papers/neurips_DB/latex/figures/fig_perf_profile_grad.pdf
  papers/neurips_DB/evidence/perf_profile_data.json
"""
from __future__ import annotations

import json
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
TIER_DIRS = [ROOT / f"results/tier{i}_benchmark" for i in (1, 2, 3, 4)]
OUT_FIG_VAL = ROOT / "papers/neurips_DB/latex/figures/fig_perf_profile_value.pdf"
OUT_FIG_GRAD = ROOT / "papers/neurips_DB/latex/figures/fig_perf_profile_grad.pdf"
OUT_DATA = ROOT / "papers/neurips_DB/evidence/perf_profile_data.json"

DML = "dml_fixed"
VANILLA = "vanilla"
EPS = 1e-12

FAMILIES = ["bachelier", "black_scholes", "poly_trig", "trig", "step", "heston"]
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.5]
WONG = {0.0: "#0072B2", 0.05: "#E69F00", 0.1: "#009E73",
        0.2: "#CC79A7", 0.5: "#D55E00"}


def load_pairs():
    """Per (func, dim, n, sigma): R1 paired-log-ratio statistic (DML / vanilla).

    For each (cell, seed) where both DML and vanilla results exist, compute
    log10(MSE_DML / MSE_van). Aggregate across seeds within the cell as the
    median (R1; canonical per `research_notes/ratio_definitions.md`). The
    cell-level statistic returned is exp10(median) so the ECDF below sees
    a ratio in the same display space as before. Cells without a paired
    seed are dropped.
    """
    # First pass: collect per-(cell, seed) MSEs per method.
    by_cell_seed = defaultdict(lambda: defaultdict(dict))
    for tdir in TIER_DIRS:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            method = r.get("method")
            if method not in (DML, VANILLA):
                continue
            if method == DML:
                lam = r.get("lambda")
                if lam is not None and float(lam) != 1.0:
                    continue
            func = r.get("func_type") or r.get("dataset")
            dim = r.get("dim"); n = r.get("n_samples") or r.get("n_train")
            sigma = r.get("noise_level"); sigma = sigma if sigma is not None else r.get("noise")
            seed = r.get("seed")
            v = r.get("test_value_mse"); g = r.get("test_grad_mse")
            if None in (func, dim, n, sigma, seed, v, g):
                continue
            cfg = (func, int(dim), int(n), float(sigma))
            by_cell_seed[cfg][int(seed)][method] = (
                max(EPS, float(v)), max(EPS, float(g))
            )

    # Second pass: median of paired per-seed log-ratios within each cell.
    out = {}
    for cfg, seeds in by_cell_seed.items():
        log_v = []; log_g = []
        for seed, by_method in seeds.items():
            if DML not in by_method or VANILLA not in by_method:
                continue
            v_dml, g_dml = by_method[DML]
            v_van, g_van = by_method[VANILLA]
            log_v.append(np.log10(v_dml / v_van))
            log_g.append(np.log10(g_dml / g_van))
        if not log_v:
            continue
        out[cfg] = {
            "ratio_value": float(10.0 ** np.median(log_v)),
            "ratio_grad":  float(10.0 ** np.median(log_g)),
            "n_pairs":     len(log_v),
        }
    return out


def perf_profile(ratios, taus):
    """For each tau, fraction of configs with ratio <= tau."""
    arr = np.asarray(ratios, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros_like(taus)
    return np.array([float(np.mean(arr <= t)) for t in taus])


def plot_panel(metric_key: str, title_metric: str, out_path: Path):
    cells = load_pairs()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    taus = np.logspace(-4, 4, 200)
    data_dump = {}
    for ax, func in zip(axes.ravel(), FAMILIES):
        ax.set_title(func)
        ax.set_xscale("log")
        ax.axvline(1.0, color="grey", alpha=0.4, linestyle="--")
        for sigma in SIGMAS:
            ratios = [cells[c][metric_key]
                      for c in cells if c[0] == func and c[3] == sigma]
            if not ratios:
                continue
            curve = perf_profile(ratios, taus)
            ax.plot(taus, curve, color=WONG[sigma],
                     label=f"σ={sigma}", linewidth=1.4)
            data_dump.setdefault(func, {})[str(sigma)] = {
                "n_configs": len(ratios),
                "median_ratio": float(np.median(ratios)),
            }
        ax.set_xlim(1e-4, 1e4)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"fraction of configs with $r \leq \tau$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\tau$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(SIGMAS),
                    bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(
        f"Performance profile (Dolan-Moré 2002): per-cell paired-log-ratio "
        f"(R1) of {title_metric} for dml_fixed vs vanilla", y=1.10
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return data_dump


def main():
    OUT_FIG_VAL.parent.mkdir(parents=True, exist_ok=True)
    val_dump = plot_panel("ratio_value", "value MSE", OUT_FIG_VAL)
    grad_dump = plot_panel("ratio_grad", "gradient MSE", OUT_FIG_GRAD)
    OUT_DATA.write_text(json.dumps({"value": val_dump, "grad": grad_dump}, indent=2))
    print(f"wrote {OUT_FIG_VAL}, {OUT_FIG_GRAD}, {OUT_DATA}")


if __name__ == "__main__":
    main()
