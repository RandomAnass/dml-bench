#!/usr/bin/env python3
"""
Noise-curriculum supporting experiment for the warmup qualitative claim.

Hypothesis: warmup helps in proportion to the noise on the gradient labels.
We artificially scale the **pathwise** Heston-barrier gradient labels by
(1 + N(0, σ²)) for σ ∈ {0, 0.1, 0.5, 1.0}, train `dml_fixed` (no warmup)
and `dml_warmup` (2-stage), and check whether the warmup-vs-fixed-λ MSE
ratio is monotone-decreasing in σ.

Output: results/heston_barrier_4way/noise_curriculum/
    {method}_sigma{σ}_seed{s}.json   per fit
    analysis.json                     aggregate summary

Usage:
    python experiments/heston_barrier_4way/run_noise_curriculum.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference
from dml_benchmark.trainer import train_single_experiment
import experiments.heston_barrier_4way.run_pilot as rp

SEEDS = [42, 123, 456]                  # 3 seeds (cost control)
SIGMAS = [0.0, 0.1, 0.5, 1.0]
METHODS = ["dml_fixed", "dml_warmup"]   # the two we compare
SPOT_LOW_MULT = 0.5                     # wide range (where DML can win)
SPOT_HIGH_MULT = 1.5

HPARAMS = dict(rp.HPARAMS)
HPARAMS["early_stopping_patience"] = 200

GROUND_TRUTH_N_TEST = 200
GROUND_TRUTH_N_PATHS = 100_000
GROUND_TRUTH_SEED = 999


def gen_pathwise_with_noise(seed: int, sigma: float, gt: dict) -> dict:
    """Generate pathwise data, then add multiplicative N(0, σ²) noise to dydx."""
    rp.SPOT_LOW_MULT = SPOT_LOW_MULT
    rp.SPOT_HIGH_MULT = SPOT_HIGH_MULT
    data = rp._pathwise_barrier_heston(seed=seed)
    if sigma > 0.0:
        noise_rng = np.random.RandomState(seed + 1_000_000)
        noise = noise_rng.normal(0.0, sigma, size=data["dydx_pw"].shape)
        data["dydx_pw"] = data["dydx_pw"] * (1.0 + noise)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = REPO_ROOT / "results" / "heston_barrier_4way" / "noise_curriculum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT (re-use wide-range cached if present, else compute)
    gt_cache = REPO_ROOT / "results" / "heston_barrier_4way" / "predictions" / "wide" / "gt.npz"
    if gt_cache.exists():
        print(f"Loading wide-range GT from {gt_cache}")
        gt = dict(np.load(gt_cache))
        ground_truth = {
            "x": gt["x"].reshape(-1, 1),
            "y": gt["y"].reshape(-1, 1),
            "dydx": gt["dydx"].reshape(-1, 1, 1),
        }
    else:
        print("Computing wide-range GT (200 spots × 100k MC paths)…")
        rng_gt = np.random.RandomState(GROUND_TRUTH_SEED)
        S0_test = rng_gt.uniform(rp.HESTON_PARAMS["strike"] * SPOT_LOW_MULT,
                                  rp.HESTON_PARAMS["strike"] * SPOT_HIGH_MULT,
                                  GROUND_TRUTH_N_TEST)
        gt_full = heston_barrier_doc_mc_reference(
            S0=S0_test, n_paths=GROUND_TRUTH_N_PATHS, seed=GROUND_TRUTH_SEED,
            **rp.HESTON_PARAMS,
        )
        ground_truth = {"x": gt_full["x"], "y": gt_full["y"], "dydx": gt_full["dydx"]}

    rp.SPOT_LOW_MULT = SPOT_LOW_MULT
    rp.SPOT_HIGH_MULT = SPOT_HIGH_MULT
    rp.HPARAMS = HPARAMS
    rp.SEEDS = SEEDS

    if not args.analyze_only:
        for sigma in SIGMAS:
            for seed in SEEDS:
                # generate ONCE per (seed, sigma); reuse for both methods
                t_g = time.time()
                data = gen_pathwise_with_noise(seed=seed, sigma=sigma, gt=ground_truth)
                gen_time = time.time() - t_g

                data_split = {
                    "x_train": data["x"], "y_train": data["y"],
                    "dydx_train": data["dydx_pw"],
                    "x_test": ground_truth["x"], "y_test": ground_truth["y"],
                    "dydx_test": ground_truth["dydx"],
                }

                for method in METHODS:
                    out_json = out_dir / f"{method}_sigma{sigma}_seed{seed}.json"
                    if out_json.exists():
                        print(f"  SKIP cached: {method} σ={sigma} seed={seed}")
                        continue
                    t0 = time.time()
                    try:
                        result = train_single_experiment(
                            x_train=data_split["x_train"],
                            y_train=data_split["y_train"],
                            dydx_train=data_split["dydx_train"],
                            x_test=data_split["x_test"],
                            y_test=data_split["y_test"],
                            dydx_test=data_split["dydx_test"],
                            method=method, seed=seed, pbar=False,
                            **HPARAMS,
                        )
                        elapsed = time.time() - t0
                        rec = {
                            "method": method, "sigma": sigma, "seed": seed,
                            "test_value_mse": float(result.test_value_mse),
                            "test_grad_mse": float(result.test_grad_mse),
                            "best_epoch": int(result.best_epoch),
                            "time_s": round(elapsed, 2),
                            "data_gen_s": round(gen_time, 2),
                            "spot_range": [SPOT_LOW_MULT, SPOT_HIGH_MULT],
                        }
                        with open(out_json, "w") as f:
                            json.dump(rec, f, indent=2)
                        print(f"  {method} σ={sigma} seed={seed}: "
                              f"val_mse={result.test_value_mse:.4e} "
                              f"({elapsed:.1f}s)", flush=True)
                    except Exception as e:
                        print(f"  ERROR {method} σ={sigma} seed={seed}: {e}", flush=True)
                        with open(out_json, "w") as f:
                            json.dump({"method": method, "sigma": sigma,
                                       "seed": seed, "error": str(e)}, f, indent=2)

    # Aggregate
    summary = {}
    for sigma in SIGMAS:
        summary[str(sigma)] = {}
        for method in METHODS:
            v = []; g = []
            for seed in SEEDS:
                f = out_dir / f"{method}_sigma{sigma}_seed{seed}.json"
                if not f.exists(): continue
                with open(f) as fh:
                    d = json.load(fh)
                if "error" in d: continue
                v.append(d["test_value_mse"]); g.append(d["test_grad_mse"])
            if v:
                summary[str(sigma)][method] = {
                    "n_seeds": len(v),
                    "value_mse_mean": float(np.mean(v)),
                    "value_mse_std":  float(np.std(v)),
                    "grad_mse_mean":  float(np.mean(g)),
                    "grad_mse_std":   float(np.std(g)),
                }
    # ratio table
    ratio_table = {}
    for sigma in SIGMAS:
        ks = sorted(summary[str(sigma)].keys())
        if "dml_fixed" in summary[str(sigma)] and "dml_warmup" in summary[str(sigma)]:
            ratio_table[str(sigma)] = {
                "value_mse_warmup_over_fixed":
                    summary[str(sigma)]["dml_warmup"]["value_mse_mean"]
                    / summary[str(sigma)]["dml_fixed"]["value_mse_mean"],
                "grad_mse_warmup_over_fixed":
                    summary[str(sigma)]["dml_warmup"]["grad_mse_mean"]
                    / summary[str(sigma)]["dml_fixed"]["grad_mse_mean"],
            }

    out = {"summary": summary, "ratio": ratio_table,
           "interpretation": ("If hypothesis holds, "
                              "value_mse_warmup_over_fixed should DECREASE "
                              "as σ increases (warmup gain grows with noise)")}
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== Ratio summary (warmup / fixed-λ) ===")
    print(f"  {'σ':>5}  {'value MSE ratio':>18}  {'grad MSE ratio':>18}")
    for sigma in SIGMAS:
        if str(sigma) in ratio_table:
            r = ratio_table[str(sigma)]
            print(f"  {sigma:>5}  {r['value_mse_warmup_over_fixed']:>18.4f}  "
                  f"{r['grad_mse_warmup_over_fixed']:>18.4f}")
    print(f"\nWrote {out_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
