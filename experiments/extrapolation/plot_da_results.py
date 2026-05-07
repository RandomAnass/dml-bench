"""Plot DA results: log_ratio vs sigma for each region, comparing arms.

Output:
  results/extrapolation_M1_DA/da_log_ratio_per_region.{pdf,png}
  results/extrapolation_M1_DA/da_log_ratio_near_extrap.{pdf,png}  (key plot)
  results/extrapolation_M1_DA/comparison_table.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_COLORS = {
    "vanilla": "#888888",
    "dml_train": "#2176ae",
    "dml_da": "#c73e1d",
}


def plot_per_region(out: Path, df: pd.DataFrame) -> None:
    regions = ["interpolation", "near_extrap", "far_extrap"]
    arms = ["dml_train", "dml_da"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, region in zip(axes, regions):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        for arm in arms:
            sub = df[(df["region"] == region) & (df["arm"] == arm)]
            for seed in sorted(sub["seed"].unique()):
                ss = sub[sub["seed"] == seed].sort_values("sigma_rel")
                ax.plot(ss["sigma_rel"], ss["log_ratio"],
                        color=ARM_COLORS[arm], alpha=0.25, linewidth=0.8)
            mean = (sub.groupby("sigma_rel")["log_ratio"]
                      .mean().reset_index().sort_values("sigma_rel"))
            ax.plot(mean["sigma_rel"], mean["log_ratio"],
                    color=ARM_COLORS[arm], linewidth=2.2,
                    label=arm.replace("_", "-"))
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_title(f"region = {region}")
        ax.set_xlabel("σ_rel  (gradient noise)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("log(MSE_arm / MSE_vanilla)")
    axes[-1].legend(loc="best", frameon=False)
    fig.suptitle("SIREN  —  domain-adaptation arm vs DML-train arm")
    fig.tight_layout()
    fig.savefig(out / "da_log_ratio_per_region.pdf")
    fig.savefig(out / "da_log_ratio_per_region.png", dpi=150)
    plt.close(fig)


def plot_near_extrap_focus(out: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    for arm in ["dml_train", "dml_da"]:
        sub = df[(df["region"] == "near_extrap") & (df["arm"] == arm)]
        for seed in sorted(sub["seed"].unique()):
            ss = sub[sub["seed"] == seed].sort_values("sigma_rel")
            ax.plot(ss["sigma_rel"], ss["log_ratio"],
                    color=ARM_COLORS[arm], alpha=0.25, linewidth=0.8)
        mean = (sub.groupby("sigma_rel")["log_ratio"]
                  .mean().reset_index().sort_values("sigma_rel"))
        ax.plot(mean["sigma_rel"], mean["log_ratio"],
                color=ARM_COLORS[arm], linewidth=2.5,
                label=f"{arm.replace('_', '-')}  (mean across 5 seeds)")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("σ_rel  (relative gradient-label noise)")
    ax.set_ylabel("log(MSE_arm / MSE_vanilla)   [near-extrap]")
    ax.set_title("SIREN near-extrap  —  DA training improves σ*")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "da_log_ratio_near_extrap.pdf")
    fig.savefig(out / "da_log_ratio_near_extrap.png", dpi=150)
    plt.close(fig)


def comparison_table(out: Path, df_rows: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side comparison table."""
    out_rows = []
    for region in ["interpolation", "near_extrap", "far_extrap", "all"]:
        s = df_summary[df_summary["region"] == region]
        s_dt = s[s["arm"] == "dml_train"]
        s_da = s[s["arm"] == "dml_da"]
        if len(s_dt) == 0 or len(s_da) == 0:
            continue
        sigstar_dt = s_dt["sigma_star"].iloc[0]
        sigstar_da = s_da["sigma_star"].iloc[0]
        lr0_dt = s_dt["mean_log_ratio_sigma0"].iloc[0]
        lr0_da = s_da["mean_log_ratio_sigma0"].iloc[0]
        std0_dt = s_dt["std_log_ratio_sigma0"].iloc[0]
        std0_da = s_da["std_log_ratio_sigma0"].iloc[0]
        out_rows.append({
            "region": region,
            "sigma_star_dml_train": sigstar_dt,
            "sigma_star_dml_da": sigstar_da,
            "delta_sigma_star": sigstar_da - sigstar_dt,
            "mean_log_ratio_sigma0_dml_train": lr0_dt,
            "mean_log_ratio_sigma0_dml_da": lr0_da,
            "delta_mean_log_ratio_sigma0": lr0_da - lr0_dt,
            "std_log_ratio_sigma0_dml_train": std0_dt,
            "std_log_ratio_sigma0_dml_da": std0_da,
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out / "comparison_table.csv", index=False)
    print("\n===== DA vs DML comparison =====")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(out_df.to_string(index=False))
    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="results/extrapolation_M1_DA")
    args = p.parse_args()
    out = Path(args.out_dir)
    df = pd.read_csv(out / "rows.csv")
    summary = pd.read_csv(out / "sigma_star_summary.csv")
    plot_per_region(out, df)
    plot_near_extrap_focus(out, df)
    print(f"[saved] {out / 'da_log_ratio_per_region.pdf'}")
    print(f"[saved] {out / 'da_log_ratio_near_extrap.pdf'}")
    comparison_table(out, df, summary)


if __name__ == "__main__":
    main()
