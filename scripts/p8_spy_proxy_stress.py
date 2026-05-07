#!/usr/bin/env python3
"""
Phase 8 — Priority F: SPY proxy-label stress test.

Trains DML methods on SPY options where the proxy Black-Scholes Greeks
(used as derivative labels) are perturbed along four axes:

  1. sigma_staleness_days       k ∈ {5, 10, 20}     (k=0 = baseline)
  2. sigma_misspec_delta        Δ ∈ {0.01, 0.05, 0.10}
  3. greek_additive_noise       ε ∈ {0.05, 0.1, 0.2, 0.5}
  4. greek_multiplicative_noise ε ∈ {0.05, 0.1, 0.2}

The grid is one-axis-at-a-time (other axes at zero) so the marginal
effect of each perturbation type is identifiable. A small "combined"
sub-grid at the end pairs misspec + additive at moderate levels to
catch interaction effects.

Output: results/p8_spy_proxy_stress/spy_p8_<axis>_<level>_<method>_s<seed>.json

Usage:
  python scripts/p8_spy_proxy_stress.py --gpu 0 --resume
  python scripts/p8_spy_proxy_stress.py --gpu 0 --smoke      # 1 axis × 1 level × 1 method × 1 seed
  python scripts/p8_spy_proxy_stress.py --gpu 0 --axes misspec staleness
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.real_data_spy.spy_data_loader import load_spy_data
from experiments.real_data_spy.spy_perturbations import perturb_spy_data
from experiments.real_data_spy.run_spy_experiment import HPARAMS, train_warmup_spy
from dml_benchmark.trainer import train_single_experiment


METHODS_DEFAULT = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"]
SEEDS_DEFAULT = [42, 123, 456, 789, 1337]
N_TRAIN = 10000
N_TEST = 10000
WARMUP_FRACTION = 0.5

# One-axis-at-a-time grid. Each axis includes a baseline level (= clean) so
# we can normalize degradation later.
AXES = {
    "baseline":      [{"sigma_staleness_days": 0, "sigma_misspec_delta": 0.0,
                       "greek_additive_noise": 0.0, "greek_multiplicative_noise": 0.0}],
    "staleness":     [{"sigma_staleness_days": k, "sigma_misspec_delta": 0.0,
                       "greek_additive_noise": 0.0, "greek_multiplicative_noise": 0.0}
                      for k in (5, 10, 20)],
    "misspec":       [{"sigma_staleness_days": 0, "sigma_misspec_delta": d,
                       "greek_additive_noise": 0.0, "greek_multiplicative_noise": 0.0}
                      for d in (0.01, 0.05, 0.10)],
    "additive":      [{"sigma_staleness_days": 0, "sigma_misspec_delta": 0.0,
                       "greek_additive_noise": e, "greek_multiplicative_noise": 0.0}
                      for e in (0.05, 0.10, 0.20, 0.50)],
    "multiplicative":[{"sigma_staleness_days": 0, "sigma_misspec_delta": 0.0,
                       "greek_additive_noise": 0.0, "greek_multiplicative_noise": e}
                      for e in (0.05, 0.10, 0.20)],
    "combined":      [{"sigma_staleness_days": 0, "sigma_misspec_delta": 0.05,
                       "greek_additive_noise": 0.10, "greek_multiplicative_noise": 0.0},
                      {"sigma_staleness_days": 10, "sigma_misspec_delta": 0.0,
                       "greek_additive_noise": 0.10, "greek_multiplicative_noise": 0.0}],
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def axis_label(axis: str, cfg: dict) -> str:
    """Slug for filename like 'misspec_d0.05' or 'staleness_k10'."""
    if axis == "baseline":
        return "baseline"
    if axis == "staleness":
        return f"staleness_k{cfg['sigma_staleness_days']}"
    if axis == "misspec":
        return f"misspec_d{cfg['sigma_misspec_delta']:g}"
    if axis == "additive":
        return f"additive_e{cfg['greek_additive_noise']:g}"
    if axis == "multiplicative":
        return f"multiplicative_e{cfg['greek_multiplicative_noise']:g}"
    if axis == "combined":
        bits = []
        if cfg["sigma_staleness_days"]:
            bits.append(f"k{cfg['sigma_staleness_days']}")
        if cfg["sigma_misspec_delta"]:
            bits.append(f"d{cfg['sigma_misspec_delta']:g}")
        if cfg["greek_additive_noise"]:
            bits.append(f"a{cfg['greek_additive_noise']:g}")
        if cfg["greek_multiplicative_noise"]:
            bits.append(f"m{cfg['greek_multiplicative_noise']:g}")
        return "combined_" + "_".join(bits)
    raise ValueError(f"unknown axis: {axis}")


def run_one(axis: str, cfg: dict, method: str, seed: int,
            results_dir: Path, baseline_data: dict) -> dict:
    """Train one (axis, level, method, seed) configuration."""
    label = axis_label(axis, cfg)
    key = f"spy_p8_{label}_{method}_s{seed}"
    path = results_dir / f"{key}.json"

    # Apply perturbation deterministically (per-config seed for the perturbation
    # noise; train seed independent).
    perturbation_seed = hash((axis, label, seed)) & 0xFFFFFFFF
    data = perturb_spy_data(
        baseline_data,
        sigma_staleness_days=cfg["sigma_staleness_days"],
        sigma_misspec_delta=cfg["sigma_misspec_delta"],
        greek_additive_noise=cfg["greek_additive_noise"],
        greek_multiplicative_noise=cfg["greek_multiplicative_noise"],
        include_volume=False,
        seed=perturbation_seed,
    )

    t0 = time.time()
    if method == "dml_warmup":
        result = train_warmup_spy(
            x_train=data["x_train"], y_train=data["y_train"], dydx_train=data["dydx_train"],
            x_test=data["x_test"], y_test=data["y_test"], dydx_test=data["dydx_test"],
            warmup_fraction=WARMUP_FRACTION, seed=seed, pbar=False, **HPARAMS,
        )
    else:
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"], dydx_train=data["dydx_train"],
            x_test=data["x_test"], y_test=data["y_test"], dydx_test=data["dydx_test"],
            method=method, seed=seed, pbar=False, **HPARAMS,
        )
    elapsed = time.time() - t0

    res = {
        "key": key,
        "method": method,
        "dataset": "spy_options",
        "axis": axis,
        "axis_label": label,
        "perturbation": cfg,
        "perturbation_seed": int(perturbation_seed),
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "dim": 4,
        "split_mode": "temporal",
        "seed": seed,
        "test_value_mse": float(result.test_value_mse),
        "test_grad_mse": float(result.test_grad_mse),
        "best_epoch": int(result.best_epoch),
        "time_s": round(elapsed, 2),
        "git_hash": git_hash(),
        "hparams": {k: v for k, v in HPARAMS.items()},
    }
    path.write_text(json.dumps(res, indent=2))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--axes", nargs="+", default=list(AXES.keys()))
    p.add_argument("--methods", nargs="+", default=METHODS_DEFAULT)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    p.add_argument("--results_dir", default="results/p8_spy_proxy_stress")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 axis × 1 level × 1 method × 1 seed")
    args = p.parse_args()

    if args.smoke:
        args.axes = ["misspec"]
        args.methods = ["dml_fixed"]
        args.seeds = [42]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load clean SPY data ONCE per (seed) — train/test indices depend on seed.
    # Hot-cache by seed so we don't reload 1.5M records for every config.
    spy_cache: dict[int, dict] = {}

    def get_baseline(seed: int):
        if seed not in spy_cache:
            spy_cache[seed] = load_spy_data(
                n_train=N_TRAIN, n_test=N_TEST,
                include_volume=False, stratify_by_moneyness=True,
                seed=seed, split_mode="temporal",
            )
        return spy_cache[seed]

    # Build full config list
    configs: List[Tuple[str, dict, str, int]] = []
    for axis in args.axes:
        if axis not in AXES:
            print(f"[P8] unknown axis: {axis}; skipping")
            continue
        for cfg in AXES[axis]:
            if args.smoke and cfg["sigma_misspec_delta"] not in (0.05,):
                continue
            for method in args.methods:
                for seed in args.seeds:
                    configs.append((axis, cfg, method, seed))

    print(f"[P8] {len(configs)} configs | gpu={args.gpu} | git={git_hash()[:8]}",
          flush=True)

    n_ok = n_fail = n_skip = 0
    for i, (axis, cfg, method, seed) in enumerate(configs):
        label = axis_label(axis, cfg)
        key = f"spy_p8_{label}_{method}_s{seed}"
        path = results_dir / f"{key}.json"
        if args.resume and path.exists():
            n_skip += 1
            continue
        try:
            baseline = get_baseline(seed)
            _ = run_one(axis, cfg, method, seed, results_dir, baseline)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"[P8] FAIL {key}: {e}", file=sys.stderr)
                traceback.print_exc()
        if (i + 1) % 10 == 0 or (i + 1) == len(configs):
            print(f"[P8] progress {i+1}/{len(configs)} ok={n_ok} fail={n_fail} skip={n_skip}",
                  flush=True)

    print(f"[P8] DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
