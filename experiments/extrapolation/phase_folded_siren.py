"""
Phase-folded SIREN figure for Appendix H.

Question: is SIREN's positive near-extrapolation σ* a result of *periodic continuation*
(SIREN has learned the target's period and reproduces it across the eval domain) or
*smooth boundary fit* (the train-domain solution extends locally with the right slope,
which DML helps anchor)?

Diagnostic: plot SIREN predictions on the eval domain [-3, 3], folded by the
target's true period (= 1.0 for our K=5 target). If predictions collapse onto a
single curve, SIREN is doing periodic continuation. If they fan out, the
benefit is a local smoothness effect.

Run:
    python experiments/extrapolation/phase_folded_siren.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pilot_periodic_extrap import (  # noqa: E402
    Config, make_train_data, make_eval_grid, standardize_eval,
    make_grad_noise, apply_grad_noise, make_model, clone_state,
    train_model, predict_values, stable_seed, target_value,
)


OUT = Path("results/extrapolation_M1/phase_folded_siren")
OUT.mkdir(parents=True, exist_ok=True)


def collect_siren_predictions(seeds=range(5), sigmas=(0.0, 0.5, 1.5)) -> pd.DataFrame:
    cfg = Config()
    cfg.models = ("siren",)
    rows = []
    x_eval, y_raw, g_raw = make_eval_grid(cfg.n_eval, cfg.eval_domain)
    x_eval_flat = x_eval.squeeze()

    for target_seed in seeds:
        train = make_train_data(target_seed, cfg.n_train, cfg.train_domain)
        y_eval, _ = standardize_eval(y_raw, g_raw, train["y_mean"], train["y_std"])
        init_model = make_model("siren", target_seed, cfg)
        init_state = clone_state(init_model)
        g_scale, eps = make_grad_noise(train["g"], target_seed)

        # Vanilla baseline
        vanilla_seed = stable_seed("vanilla-train", "siren", target_seed)
        vanilla = train_model("siren", init_state, train["x"], train["y"], cfg,
                              mode="vanilla", train_seed=vanilla_seed)
        pred_v = predict_values(vanilla, x_eval, cfg.device)

        # DML at each σ (same dml_seed across σ for clean isolation)
        dml_seed_fixed = stable_seed("dml-train", "siren", target_seed)
        for sigma in sigmas:
            g_noisy = apply_grad_noise(train["g"], g_scale, eps, float(sigma))
            dml = train_model("siren", init_state, train["x"], train["y"], cfg,
                              g_train=g_noisy, g_scale=g_scale,
                              mode="dml", train_seed=dml_seed_fixed)
            pred_d = predict_values(dml, x_eval, cfg.device)

            # un-standardize so we can compare against target_value
            pred_v_unscaled = pred_v * train["y_std"] + train["y_mean"]
            pred_d_unscaled = pred_d * train["y_std"] + train["y_mean"]

            for i, x in enumerate(x_eval_flat):
                rows.append({
                    "seed": target_seed,
                    "sigma": float(sigma),
                    "x": float(x),
                    "y_true": float(target_value(np.array([x]))[0]),
                    "y_vanilla": float(pred_v_unscaled[i]),
                    "y_dml": float(pred_d_unscaled[i]),
                })
        print(f"  siren seed={target_seed} done")

    return pd.DataFrame(rows)


def plot_phase_folded(df: pd.DataFrame, period: float = 1.0):
    """Two-panel plot:
    (top) standard predictions vs x with the train domain shaded
    (bottom) phase-folded — predictions vs (x mod period) — periodic recovery test
    """
    df = df.copy()
    df["x_phase"] = df["x"] % period
    df["region"] = pd.cut(np.abs(df["x"]), bins=[-1e-9, 1.0, 1.5, np.inf],
                          labels=["interp", "near", "far"])

    sigmas = sorted(df["sigma"].unique())
    n_sigma = len(sigmas)

    fig, axes = plt.subplots(2, n_sigma, figsize=(5.0 * n_sigma, 7.5),
                             sharex="row", sharey="row")
    if n_sigma == 1:
        axes = axes.reshape(2, 1)

    region_colors = {"interp": "#2176ae", "near": "#f18f01", "far": "#c73e1d"}

    for col, sigma in enumerate(sigmas):
        sub = df[df["sigma"] == sigma]
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: standard prediction curves vs x
        ax_top.axvspan(-1, 1, color="#d8ecff", alpha=0.4)
        ax_top.axvline(-1, color="#999", linewidth=1)
        ax_top.axvline(+1, color="#999", linewidth=1)
        x_dense = np.linspace(-3, 3, 1000)
        ax_top.plot(x_dense, target_value(x_dense), color="black",
                    linewidth=2.0, label="true", zorder=5)
        for seed in sorted(sub["seed"].unique()):
            ss = sub[sub["seed"] == seed].sort_values("x")
            ax_top.plot(ss["x"], ss["y_dml"], color="#14a66a",
                        linewidth=1.0, alpha=0.6,
                        label="DML" if seed == 0 else None)
            ax_top.plot(ss["x"], ss["y_vanilla"], color="#6b7280",
                        linewidth=1.0, alpha=0.6, linestyle="--",
                        label="vanilla" if seed == 0 else None)
        ax_top.set_title(f"σ = {sigma}")
        ax_top.set_ylabel("y")
        ax_top.set_ylim(-3, 3)
        ax_top.grid(True, alpha=0.2)
        if col == 0:
            ax_top.legend(loc="upper left", frameon=False, fontsize=9)

        # Bottom: phase-folded — DML predictions vs (x mod period), colored by region
        ax_bot.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        x_phase_dense = np.linspace(0, period, 200)
        x_full = x_phase_dense  # one period of the target
        ax_bot.plot(x_full, target_value(x_full), color="black",
                    linewidth=2.5, label="true (one period)", zorder=10)
        for region in ("interp", "near", "far"):
            rsub = sub[sub["region"] == region]
            for seed in sorted(rsub["seed"].unique()):
                ss = rsub[rsub["seed"] == seed].sort_values("x_phase")
                ax_bot.scatter(ss["x_phase"], ss["y_dml"],
                               s=2.5, color=region_colors[region], alpha=0.4)
        ax_bot.set_xlabel("x mod 1 (target period)")
        ax_bot.set_ylabel("DML prediction")
        ax_bot.set_ylim(-3, 3)
        ax_bot.grid(True, alpha=0.2)
        if col == n_sigma - 1:
            for region, color in region_colors.items():
                ax_bot.scatter([], [], s=20, color=color, label=region)
            ax_bot.scatter([], [], s=0, label="(black: target)")
            ax_bot.legend(loc="upper right", frameon=False, fontsize=9, markerscale=2)

    fig.suptitle("SIREN phase-folded test — does extrap recovery match the target's period?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "phase_folded_siren.pdf")
    fig.savefig(OUT / "phase_folded_siren.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT}/phase_folded_siren.pdf")


def main():
    print("Collecting SIREN predictions (5 seeds × 3 σ)...")
    df = collect_siren_predictions(seeds=range(5), sigmas=(0.0, 0.5, 1.5))
    df.to_csv(OUT / "predictions.csv", index=False)
    print(f"Saved: {OUT}/predictions.csv  rows={len(df)}")
    plot_phase_folded(df, period=1.0)

    # Quick diagnostic stat: dispersion of DML predictions per (x_phase bin) within
    # each region. Low dispersion in extrap regions = periodic continuation.
    df["x_phase"] = df["x"] % 1.0
    df["region"] = pd.cut(np.abs(df["x"]),
                          bins=[-1e-9, 1.0, 1.5, np.inf],
                          labels=["interp", "near", "far"])
    df["phase_bin"] = pd.cut(df["x_phase"], bins=20).astype(str)
    disp = (df.groupby(["region", "phase_bin"])["y_dml"]
            .agg(["mean", "std"]).reset_index())
    region_mean_std = disp.groupby("region", observed=True)["std"].mean()
    print("\nPhase-folded DML prediction std per region (averaged over phase bins):")
    print(region_mean_std.to_string())
    print("\nIf periodic continuation worked, near/far std should be close to interp std.")
    print("If smooth-boundary effect only, near/far should have larger std.")


if __name__ == "__main__":
    main()
