#!/usr/bin/env python3
"""
S2: Critical Difference (Demšar 2006) diagrams for the 11-method
discontinuous-payoff comparison and the 6-method rMD17 PaiNN
comparison. Friedman omnibus + Nemenyi post-hoc, plotted via
autorank.

We report TWO diagrams per setting:
- primary: block = dataset (mean over seeds), n_blocks small but
  matches Demšar §3.1.1's independence assumption.
- supplement: block = (dataset, seed), n_blocks larger (50 / 45),
  anti-conservative but agrees with the more powerful test if the
  dataset-blocked diagram does.

Output:
  papers/neurips_DB/latex/figures/fig_cd_disc_payoff_dataset.pdf
  papers/neurips_DB/latex/figures/fig_cd_disc_payoff_dataset_seed.pdf
  papers/neurips_DB/latex/figures/fig_cd_rmd17_dataset.pdf
  papers/neurips_DB/latex/figures/fig_cd_rmd17_dataset_seed.pdf
  papers/neurips_DB/evidence/cd_diagram_summary.json
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json
DISC_PAYOFF_DIR = ROOT / "results" / "unified_comparison"
RMD17_DIR = ROOT / "results" / "molecular_painn"
FIG_DIR = ROOT / "papers" / "neurips_DB" / "latex" / "figures"
OUT_JSON = ROOT / "papers" / "neurips_DB" / "evidence" / "cd_diagram_summary.json"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_unified_disc_payoff():
    """Load (method, dataset, seed) -> grad MSE from
    results/unified_comparison/multi_seed/. Files are flat (named
    {dataset}_{method}_s{seed}.json)."""
    rows = []
    for p in (DISC_PAYOFF_DIR / "multi_seed").glob("*.json"):
        r = load_result_json(p)
        if r is None:
            continue
        method = r.get("method")
        ds = r.get("dataset")
        seed = r.get("seed")
        g = r.get("test_grad_mse")
        if None in (method, ds, seed, g):
            continue
        rows.append({"method": method, "dataset": ds, "seed": int(seed),
                      "grad_mse": float(g)})
    return pd.DataFrame(rows)


def _load_rmd17():
    rows = []
    for p in RMD17_DIR.glob("*.json"):
        r = load_result_json(p)
        if r is None:
            continue
        method = r.get("method")
        mol = r.get("molecule")
        split = r.get("split_id") or r.get("split") or r.get("seed")
        f = r.get("test_force_mae_mev")
        if None in (method, mol, split, f):
            continue
        rows.append({"method": method, "dataset": mol, "seed": int(split),
                      "force_mae": float(f)})
    return pd.DataFrame(rows)


def _make_pivot(df, value_col):
    """Return a wide pivot (rows = blocks, columns = methods, values =
    metric). Rows are unique (dataset, seed) tuples; missing cells are
    dropped to keep autorank happy."""
    df = df.dropna()
    pivot = df.pivot_table(index=["dataset", "seed"], columns="method",
                            values=value_col, aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any")  # complete blocks only
    return pivot


def _make_dataset_blocked(df, value_col):
    """Mean over seeds first; rows = datasets."""
    df = df.dropna()
    g = df.groupby(["dataset", "method"])[value_col].mean().unstack("method")
    g = g.dropna(axis=0, how="any")
    return g


def _save_cd(pivot, title, out_path):
    """Run autorank → CD diagram. Returns dict summary."""
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return {"error": f"shape {pivot.shape} too small for CD"}
    from autorank import autorank, plot_stats
    # Lower is better. autorank's default ranks higher-better; we negate.
    res = autorank(-pivot, alpha=0.05, verbose=False, force_mode="nonparametric")
    try:
        plt.figure(figsize=(7, 3.5))
        ax = plot_stats(res, allow_insignificant=True)
        fig = plt.gcf() if ax is None else ax.figure
        fig.suptitle(title, y=1.04)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    except Exception as e:
        return {"error": f"plot_stats failed: {e}"}
    return {
        "n_blocks": int(pivot.shape[0]),
        "n_methods": int(pivot.shape[1]),
        "p_omnibus": float(getattr(res, "pvalue", float("nan"))),
        "rankdf_means": {m: float(r) for m, r in res.rankdf["meanrank"].items()},
    }


def main():
    summary = {}
    # --- discontinuous-payoff, 11-method
    df = _load_unified_disc_payoff()
    if not df.empty:
        ds_pivot = _make_dataset_blocked(df, "grad_mse")
        seed_pivot = _make_pivot(df, "grad_mse")
        summary["disc_payoff_dataset"] = _save_cd(
            ds_pivot,
            "CD diagram (Demšar): 11-method discontinuous-payoff "
            "(blocks = dataset, mean over seeds)",
            FIG_DIR / "fig_cd_disc_payoff_dataset.pdf",
        )
        summary["disc_payoff_dataset_seed"] = _save_cd(
            seed_pivot,
            "CD diagram (Demšar): 11-method discontinuous-payoff "
            "(blocks = dataset×seed, anti-conservative)",
            FIG_DIR / "fig_cd_disc_payoff_dataset_seed.pdf",
        )
    else:
        summary["disc_payoff_error"] = f"no data in {DISC_PAYOFF_DIR}/multi_seed/"

    # --- rMD17 PaiNN
    df = _load_rmd17()
    if not df.empty:
        ds_pivot = _make_dataset_blocked(df, "force_mae")
        seed_pivot = _make_pivot(df, "force_mae")
        summary["rmd17_dataset"] = _save_cd(
            ds_pivot,
            "CD diagram: rMD17 PaiNN methods (blocks = molecule, "
            "mean over splits)",
            FIG_DIR / "fig_cd_rmd17_dataset.pdf",
        )
        summary["rmd17_dataset_seed"] = _save_cd(
            seed_pivot,
            "CD diagram: rMD17 PaiNN methods (blocks = molecule×split, "
            "anti-conservative)",
            FIG_DIR / "fig_cd_rmd17_dataset_seed.pdf",
        )
    else:
        summary["rmd17_error"] = f"no data in {RMD17_DIR}"

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
