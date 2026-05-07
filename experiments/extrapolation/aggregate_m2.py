"""
Aggregate M2 results into per-(function, d, n_train, mode, method) summary.

Inputs:  results/extrapolation_M2/m2_*.json (360 cells)
Outputs:
  results/extrapolation_M2/m2_summary.csv      (per-method, per-config mean MSE + std)
  results/extrapolation_M2/m2_delta_pct.csv    (Δ% vs vanilla per (func, d, n_train, mode))
  results/extrapolation_M2/m2_summary.md       (markdown table for paper)
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M2_DIR = ROOT / "results" / "extrapolation_M2"


def load_all() -> pd.DataFrame:
    rows = []
    for f in sorted(glob(str(M2_DIR / "m2_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows.append({
            "func": d.get("func_type"),
            "d": d.get("dim"),
            "n_train": d.get("n_train"),
            "n_test": d.get("n_test"),
            "mode": d.get("split_mode"),
            "method": d.get("method"),
            "seed": d.get("seed"),
            "test_value_mse": d.get("test_value_mse"),
            "test_grad_mse": d.get("test_grad_mse"),
            "best_epoch": d.get("best_epoch"),
            "n_epochs_actual": d.get("n_epochs_actual"),
            "dist_nn_p25": d.get("dist_nn_p25"),
            "dist_nn_p50": d.get("dist_nn_p50"),
            "dist_nn_p75": d.get("dist_nn_p75"),
            "dist_nn_max": d.get("dist_nn_max"),
        })
    return pd.DataFrame(rows)


def per_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["func", "d", "n_train", "mode", "method"])
    out = g.agg(
        val_mse_mean=("test_value_mse", "mean"),
        val_mse_std=("test_value_mse", "std"),
        val_mse_median=("test_value_mse", "median"),
        grad_mse_mean=("test_grad_mse", "mean"),
        grad_mse_std=("test_grad_mse", "std"),
        n_seeds=("seed", "nunique"),
        dist_nn_p50_mean=("dist_nn_p50", "mean"),
        dist_nn_max_mean=("dist_nn_max", "mean"),
    ).reset_index()
    return out


def delta_pct_vs_vanilla(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(func, d, n_train, mode) Δ% of each DML method vs vanilla.

    Δ% = 100 * (mean_method - mean_vanilla) / mean_vanilla. Negative = DML helps.
    Also computes paired-seed log10 ratio for the canonical R1 metric.
    """
    rows = []
    for (func, d, n, mode), sub in df.groupby(["func", "d", "n_train", "mode"]):
        van = sub[sub["method"] == "vanilla"]
        if len(van) == 0:
            continue
        van_val = van["test_value_mse"].to_numpy()
        van_grad = van["test_grad_mse"].to_numpy()
        van_seeds = van["seed"].to_numpy()
        van_val_mean = float(np.mean(van_val))
        van_grad_mean = float(np.mean(van_grad))

        for method in ("dml_fixed", "dml_gradnorm"):
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            v_mean = float(msub["test_value_mse"].mean())
            g_mean = float(msub["test_grad_mse"].mean())
            row = {
                "func": func, "d": d, "n_train": n, "mode": mode, "method": method,
                "vanilla_val_mse": van_val_mean,
                "method_val_mse": v_mean,
                "val_pct": 100.0 * (v_mean - van_val_mean) / van_val_mean,
                "vanilla_grad_mse": van_grad_mean,
                "method_grad_mse": g_mean,
                "grad_pct": (100.0 * (g_mean - van_grad_mean) / van_grad_mean
                             if van_grad_mean > 0 else np.nan),
                "n_seeds": int(msub["seed"].nunique()),
            }
            # Paired log10 ratio (R1 metric)
            paired = []
            for s in msub["seed"]:
                msr = msub[msub["seed"] == s]["test_value_mse"]
                vsr = van[van["seed"] == s]["test_value_mse"]
                if len(msr) and len(vsr) and float(vsr.iloc[0]) > 0:
                    paired.append(np.log10(float(msr.iloc[0]) / float(vsr.iloc[0])))
            if paired:
                row["median_paired_log10_ratio"] = float(np.median(paired))
                row["mean_paired_log10_ratio"] = float(np.mean(paired))
                row["dml_wins"] = int(sum(1 for v in paired if v < 0))
            rows.append(row)
    return pd.DataFrame(rows)


def write_md(per_method: pd.DataFrame, deltas: pd.DataFrame):
    lines = ["# M2 — Extrapolation split aggregate", ""]
    lines += [
        "Source: `results/extrapolation_M2/m2_*.json` (360 cells = 2 funcs × 3 d × 2 N_train × 2 modes × 3 methods × 5 seeds).",
        "Replication: `python experiments/extrapolation/aggregate_m2.py`",
        "Each test set is the OOS half of the cube; train is the other half.",
        "Δ% < 0 means DML reduces value MSE vs vanilla.",
        "",
        "## Δ% per (function, d, n_train, mode), 5 seeds",
        "",
    ]
    for func in sorted(deltas["func"].unique()):
        lines.append(f"### {func}")
        lines.append("")
        sub = deltas[deltas["func"] == func].copy()
        sub = sub.sort_values(["d", "n_train", "mode", "method"])
        lines.append(
            sub[["d", "n_train", "mode", "method",
                 "val_pct", "grad_pct", "median_paired_log10_ratio", "dml_wins", "n_seeds"]]
            .to_markdown(index=False, floatfmt=".2f")
        )
        lines.append("")
    lines += [
        "## Per-method aggregate value MSE",
        "",
        per_method[["func", "d", "n_train", "mode", "method",
                    "val_mse_mean", "val_mse_std", "val_mse_median",
                    "n_seeds"]]
        .sort_values(["func", "d", "n_train", "mode", "method"])
        .to_markdown(index=False, floatfmt=".4e"),
        "",
    ]
    return "\n".join(lines)


def main():
    df = load_all()
    print(f"Loaded {len(df)} cells from {M2_DIR}")
    if len(df) == 0:
        raise SystemExit("No cells found.")
    print(f"  funcs: {sorted(df['func'].unique())}")
    print(f"  d: {sorted(df['d'].unique())}")
    print(f"  n_train: {sorted(df['n_train'].unique())}")
    print(f"  modes: {sorted(df['mode'].unique())}")
    print(f"  methods: {sorted(df['method'].unique())}")
    print(f"  seeds: {sorted(df['seed'].unique())}")

    per_method = per_method_summary(df)
    per_method.to_csv(M2_DIR / "m2_summary.csv", index=False)
    print(f"\nSaved: {M2_DIR}/m2_summary.csv  ({len(per_method)} rows)")

    deltas = delta_pct_vs_vanilla(df)
    deltas.to_csv(M2_DIR / "m2_delta_pct.csv", index=False)
    print(f"Saved: {M2_DIR}/m2_delta_pct.csv  ({len(deltas)} rows)")

    md = write_md(per_method, deltas)
    (M2_DIR / "m2_summary.md").write_text(md)
    print(f"Saved: {M2_DIR}/m2_summary.md")

    # Headline numbers for paper
    print("\n===== HEADLINE Δ% (negative = DML helps) =====")
    print(deltas[["func", "d", "n_train", "mode", "method",
                  "val_pct", "median_paired_log10_ratio", "dml_wins", "n_seeds"]]
          .sort_values(["func", "d", "n_train", "mode", "method"])
          .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
