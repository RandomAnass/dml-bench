#!/usr/bin/env python3
"""
Chen-Glasserman 2007 multi-step LRM variance scaling check (= run #4 / VERIFICATION).

Per code review final summary: verify CG 2007 Theorem 4.6 prediction —
multi-step LRM variance should be BOUNDED in n_steps (vs single-step LRM
variance growing as 1/Δt). If our smoke-test finding (variance reduction
of 113× at n=168) is real, increasing n further should NOT inflate variance.

Sweep n_substeps_to_T_1 ∈ {32, 64, 128, 256} (Δt ∈ ~{0.01, 0.005, 0.003, 0.001}).
For each, measure:
    - CG multi-step LRM variance (should be ~constant)
    - single-step LRM variance (should grow as 1/Δt for reference)
    - Trained model val_mse and grad_mse vs ground truth

Output: 2 methods × 4 n_steps × 5 seeds = 40 runs.

Usage:
    python experiments/heston_barrier_4way/run_cg_variance.py --gpu 1
    python experiments/heston_barrier_4way/run_cg_variance.py --gpu 1 --resume

Expected runtime: ~45 min on 1 A6000.

References:
    - Chen & Glasserman (2007) Theorem 4.6.
    - Code review verification: cg2007_pdf_verification.md.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.lrm_labels import (
    lrm_barrier_heston,
    lrm_multistep_heston_barrier,
)
from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference

from experiments.heston_barrier_4way.run_pilot import HESTON_PARAMS, HPARAMS

# Sweep grid (each value applied to BOTH n_substeps_to_T1 and n_substeps_T1_to_T2)
N_STEPS_VALUES = [32, 64, 128, 256]
SEEDS = [42, 123, 456, 789, 1337]
METHODS = ["dml_lrm_fixed", "dml_lrm_multistep_fixed"]
N_SAMPLES = 1024
K_PATHS = 10
GROUND_TRUTH_N_TEST = 200
GROUND_TRUTH_N_PATHS = 100_000
GROUND_TRUTH_SEED = 999

TRAINER_METHOD_MAP = {
    "dml_lrm_fixed":          "dml_fixed",
    "dml_lrm_multistep_fixed": "dml_fixed",
}


def make_key(method: str, n_steps: int, seed: int) -> str:
    return f"cg_variance_{method}_nsteps{n_steps}_s{seed}"


def load_existing(results_dir: Path) -> dict:
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    existing[data.get("key", f.stem)] = data
            except Exception:
                pass
    return existing


def save_result(results_dir: Path, key: str, result_dict: dict):
    result_dict["key"] = key
    result_dict["timestamp"] = datetime.now().isoformat()
    path = results_dir / f"{key}.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp_path.rename(path)


def run_cg_variance(results_dir: Path, existing: dict, resume: bool):
    print("\n" + "=" * 70)
    print("CG MULTI-STEP LRM VARIANCE SCALING CHECK")
    print("=" * 70)
    print(f"n_steps:  {N_STEPS_VALUES}")
    print(f"Seeds:    {SEEDS}")
    print(f"Methods:  {METHODS}")
    print(f"Output:   {results_dir}")

    # Generate a SEPARATE ground truth per n_steps (matched discretization).
    # Audit caught that using a single-finest-grid GT confounds intrinsic LRM
    # variance with Euler discretization bias when comparing val_mse across
    # n_steps. With matched GT per n_steps, val_mse is now apples-to-apples.
    print(f"\nGenerating matched-grid Heston barrier ground truth per n_steps "
          f"(n_paths={GROUND_TRUTH_N_PATHS})...")
    rng_gt = np.random.RandomState(GROUND_TRUTH_SEED)
    S0_test = rng_gt.uniform(
        HESTON_PARAMS["strike"] * 0.7, HESTON_PARAMS["strike"] * 1.3,
        GROUND_TRUTH_N_TEST,
    )
    ground_truth_per_n = {}
    for n_steps in N_STEPS_VALUES:
        t_gt = time.time()
        gt_params = dict(HESTON_PARAMS)
        gt_params["n_substeps_to_T1"] = n_steps
        gt_params["n_substeps_T1_to_T2"] = n_steps
        gt = heston_barrier_doc_mc_reference(
            S0=S0_test, n_paths=GROUND_TRUTH_N_PATHS,
            seed=GROUND_TRUTH_SEED + n_steps,  # disjoint seeds across grids
            **gt_params,
        )
        ground_truth_per_n[n_steps] = gt
        print(f"  n={n_steps}: ready in {time.time() - t_gt:.1f}s, "
              f"mean SE_price={gt['std_err_price'].mean():.2e}")

    for n_steps in N_STEPS_VALUES:
        for method in METHODS:
            for seed in SEEDS:
                key = make_key(method, n_steps, seed)
                if resume and key in existing:
                    print(f"  SKIP: {method} n={n_steps} seed={seed}")
                    continue

                print(f"\n--- {method} n_steps={n_steps} seed={seed} ---")

                # Override n_substeps in HESTON_PARAMS for this run
                run_params = dict(HESTON_PARAMS)
                run_params["n_substeps_to_T1"] = n_steps
                run_params["n_substeps_T1_to_T2"] = n_steps

                t_gen = time.time()
                if method == "dml_lrm_fixed":
                    data = lrm_barrier_heston(
                        n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **run_params,
                    )
                    dydx_key = "dydx_lrm"
                else:  # multistep
                    data = lrm_multistep_heston_barrier(
                        n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed, **run_params,
                    )
                    dydx_key = "dydx_lrm"
                gen_time = time.time() - t_gen
                lrm_var = float(data["lrm_var"].mean())

                t0 = time.time()
                try:
                    trainer_method = TRAINER_METHOD_MAP[method]
                    gt = ground_truth_per_n[n_steps]  # matched-grid GT
                    result = train_single_experiment(
                        x_train=data["x"], y_train=data["y"],
                        dydx_train=data[dydx_key],
                        x_test=gt["x"], y_test=gt["y"],
                        dydx_test=gt["dydx"],
                        method=trainer_method, seed=seed, pbar=False, **HPARAMS,
                    )
                    elapsed = time.time() - t0
                    save_result(results_dir, key, {
                        "method": method,
                        "n_steps": n_steps,
                        "delta_t": HESTON_PARAMS["T1"] / n_steps,
                        "seed": seed,
                        "lrm_var_mean": lrm_var,
                        "test_value_mse": float(result.test_value_mse),
                        "test_grad_mse": float(result.test_grad_mse),
                        "best_epoch": int(result.best_epoch),
                        "time_s": round(elapsed, 2),
                        "data_gen_s": round(gen_time, 2),
                        "hparams": dict(HPARAMS),
                        "heston_params": dict(run_params),
                    })
                    print(f"  lrm_var={lrm_var:.4e}, "
                          f"val_mse={result.test_value_mse:.4e}, "
                          f"grad_mse={result.test_grad_mse:.4e}, t={elapsed:.1f}s")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    traceback.print_exc()


def analyze(results_dir: Path):
    existing = load_existing(results_dir)
    if not existing:
        print("No results.")
        return
    print("\n" + "=" * 90)
    print("CG VARIANCE SCALING — RESULTS")
    print("=" * 90)
    print(f"  {'method':<28} {'n_steps':>8} {'lrm_var':>14} "
          f"{'val_mse':>14} {'grad_mse':>14} {'n':>4}")
    by_key = {}
    for k, res in existing.items():
        gkey = (res["method"], res["n_steps"])
        by_key.setdefault(gkey, []).append(res)
    for (method, n_steps) in sorted(by_key.keys(), key=lambda x: (x[0], x[1])):
        vals = by_key[(method, n_steps)]
        var_mean = np.mean([r["lrm_var_mean"] for r in vals])
        v_mean = np.mean([r["test_value_mse"] for r in vals])
        g_mean = np.mean([r["test_grad_mse"] for r in vals])
        n = len(vals)
        print(f"  {method:<28} {n_steps:>8} {var_mean:14.4e} "
              f"{v_mean:14.4e} {g_mean:14.4e} {n:>4}")
    print("\nExpected (CG 2007 Theorem 4.6): multi-step lrm_var ~ constant in n_steps;")
    print("                                  single-step lrm_var ~ 1/Δt = n_steps/T_1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = Path("results/heston_barrier_4way/cg_variance_check")
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir)
        return
    existing = load_existing(results_dir) if args.resume else {}
    run_cg_variance(results_dir, existing, args.resume)
    analyze(results_dir)
    print("\nCG variance check complete!")


if __name__ == "__main__":
    main()
