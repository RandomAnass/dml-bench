#!/usr/bin/env python3
"""
ε-sensitivity sweep on fuzzy-Heston-barrier (= run #3 / SENSITIVITY).

Per code review LOW #3 + G&K v2 §3.3 caveat: fuzzy bandwidth ε is
sensitive, especially for path-dependent payoffs. Sweep
ε_barrier_mult ∈ {0.1, 0.25, 0.5, 0.75, 1.0} (Savine's recommended range)
to characterize the bias-variance tradeoff.

Setup: same as run_pilot.py (Heston, single check at T_1=1/3) but with
`dml_fuzzy_fixed` only and varying ε_barrier_mult.

Output: 1 method × 5 ε × 5 seeds = 25 runs.

Usage:
    python experiments/heston_barrier_4way/run_eps_sweep.py --gpu 1
    python experiments/heston_barrier_4way/run_eps_sweep.py --gpu 1 --resume

Expected runtime: ~30 min on 1 A6000.

References:
    - Savine (2024) "Fuzzy Payoff Evaluation".
    - Glasserman & Karmarkar (2025/2026) v2 §3.3 (path-dependent caveat).
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
from dml_benchmark.fuzzy_smoothing import fuzzy_barrier_heston
from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference

from experiments.heston_barrier_4way.run_pilot import HESTON_PARAMS, HPARAMS

EPS_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 456, 789, 1337]
N_SAMPLES = 1024
K_PATHS = 10
GROUND_TRUTH_N_TEST = 200
GROUND_TRUTH_N_PATHS = 100_000
GROUND_TRUTH_SEED = 999


def make_key(eps: float, seed: int) -> str:
    return f"fuzzy_eps_sweep_eps{eps:.2f}_s{seed}"


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


def run_eps_sweep(results_dir: Path, existing: dict, resume: bool):
    print("\n" + "=" * 70)
    print("ε-SENSITIVITY SWEEP — fuzzy-Heston-barrier")
    print("=" * 70)
    print(f"ε_barrier_mult: {EPS_VALUES}")
    print(f"Seeds:          {SEEDS}")
    print(f"Method:         dml_fuzzy_fixed")
    print(f"Output:         {results_dir}")

    # Method-independent ground truth (same as pilot v3)
    print(f"\nGenerating Heston barrier ground truth ({GROUND_TRUTH_N_TEST} spots × "
          f"{GROUND_TRUTH_N_PATHS} paths)...")
    t_gt = time.time()
    rng_gt = np.random.RandomState(GROUND_TRUTH_SEED)
    S0_test = rng_gt.uniform(
        HESTON_PARAMS["strike"] * 0.7, HESTON_PARAMS["strike"] * 1.3,
        GROUND_TRUTH_N_TEST,
    )
    ground_truth = heston_barrier_doc_mc_reference(
        S0=S0_test, n_paths=GROUND_TRUTH_N_PATHS, seed=GROUND_TRUTH_SEED,
        **HESTON_PARAMS,
    )
    print(f"  Ready in {time.time() - t_gt:.1f}s")

    for eps in EPS_VALUES:
        for seed in SEEDS:
            key = make_key(eps, seed)
            if resume and key in existing:
                print(f"  SKIP: ε={eps} seed={seed}")
                continue

            print(f"\n--- ε={eps} seed={seed} ---")
            t_gen = time.time()
            data = fuzzy_barrier_heston(
                n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
                eps_barrier_mult=eps, **HESTON_PARAMS,
            )
            gen_time = time.time() - t_gen

            t0 = time.time()
            try:
                result = train_single_experiment(
                    x_train=data["x"],
                    y_train=data["y"],
                    dydx_train=data["dydx_fuzzy"],
                    x_test=ground_truth["x"],
                    y_test=ground_truth["y"],
                    dydx_test=ground_truth["dydx"],
                    method="dml_fixed", seed=seed, pbar=False,
                    **HPARAMS,
                )
                elapsed = time.time() - t0
                save_result(results_dir, key, {
                    "method": "dml_fuzzy_fixed",
                    "eps_barrier_mult": eps,
                    "epsilon_barrier_used": data["epsilon_barrier"],
                    "seed": seed,
                    "payoff": "barrier_doc_call",
                    "model": "heston_full_truncation_euler",
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "data_gen_s": round(gen_time, 2),
                    "hparams": dict(HPARAMS),
                    "heston_params": dict(HESTON_PARAMS),
                })
                print(f"  ε_used={data['epsilon_barrier']:.4f}, "
                      f"val_mse={result.test_value_mse:.4e}, "
                      f"grad_mse={result.test_grad_mse:.4e}, t={elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()


def analyze(results_dir: Path):
    existing = load_existing(results_dir)
    if not existing:
        print("No results found.")
        return
    print("\n" + "=" * 70)
    print("ε-SWEEP RESULTS")
    print("=" * 70)
    print(f"  {'ε_mult':>8} {'val_mse mean':>14} {'val_mse std':>14} "
          f"{'grad_mse mean':>14} {'n':>4}")
    by_eps = {}
    for key, res in existing.items():
        by_eps.setdefault(res.get("eps_barrier_mult", 0), []).append(res)
    for eps in sorted(by_eps.keys()):
        vals = by_eps[eps]
        v_mean = np.mean([r["test_value_mse"] for r in vals])
        v_std = np.std([r["test_value_mse"] for r in vals])
        g_mean = np.mean([r["test_grad_mse"] for r in vals])
        print(f"  {eps:>8.2f} {v_mean:14.4e} {v_std:14.4e} "
              f"{g_mean:14.4e} {len(vals):>4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = Path("results/heston_barrier_4way/eps_sweep_fuzzy")
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(results_dir)
        return
    existing = load_existing(results_dir) if args.resume else {}
    run_eps_sweep(results_dir, existing, args.resume)
    analyze(results_dir)
    print("\nε-sweep complete!")


if __name__ == "__main__":
    main()
