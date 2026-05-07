#!/usr/bin/env python3
"""
Phase 7 — Fuzzy robustness (1D ε_mult sweep, 3 methods × 7 eps × 5 seeds = 105).

Paper §7 Priority E: "report a robust region, not one lucky point".

The earlier inline-python launcher silently passed eps_mult via **kwargs to
generate_dataset() which did not accept it, so all 7 ε values produced identical
data. Fixed in run_unified_experiment.py: generate_dataset now takes an
eps_mult kwarg and threads it into the fuzzy_* generators.

This script uses that fix. Also runs post-cff4862a so GradNorm actually
updates weights (was silent-broken pre-fix).

Usage:
    python scripts/p7_fuzzy_2d_run.py --gpu 0
    python scripts/p7_fuzzy_2d_run.py --gpu 0 --resume
    python scripts/p7_fuzzy_2d_run.py --gpu 0 --smoke    # 1 eps, 1 seed, 1 method
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
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.unified_comparison.run_unified_experiment import generate_dataset
from dml_benchmark.trainer import train_single_experiment


DATASET = "barrier_bs"  # canonical fuzzy-smoothing target
EPS_MULT_SWEEP = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]  # 7 values, logspace
SEEDS = [42, 123, 456, 789, 1000]
METHODS = ["dml_fuzzy", "dml_gradnorm_fuzzy", "dml_warmup_fuzzy"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def _method_to_trainer(method: str) -> str:
    """Map paper-level fuzzy method name → trainer.train_single_experiment method kwarg.

    The fuzzy-label variants use the same trainer but with the fuzzy-derivative
    inputs rather than pathwise / LRM. So train_single_experiment is invoked
    with the underlying balancing method (dml_fixed / dml_gradnorm / dml_warmup).
    """
    return {
        "dml_fuzzy": "dml_fixed",
        "dml_gradnorm_fuzzy": "dml_gradnorm",
        "dml_warmup_fuzzy": "dml_warmup",
    }[method]


def run_one(method: str, eps_mult: float, seed: int, results_dir: Path) -> dict:
    key = f"p7_{DATASET}_eps{eps_mult:g}_{method}_s{seed}"
    path = results_dir / f"{key}.json"

    data = generate_dataset(DATASET, seed=seed, eps_mult=eps_mult)

    # Fuzzy training: x, y = LRM-average inputs, dydx = fuzzy-smoothed derivatives
    x_tr, y_tr, dydx_tr = data["x_train"], data["y_train"], data["dydx_fuzzy_train"]
    x_te, y_te, dydx_te = data["x_test"], data["y_test"], data["dydx_fuzzy_test"]

    # val split: last 10% of train
    n_val = int(len(x_tr) * 0.1)
    x_val, y_val, dydx_val = x_tr[-n_val:], y_tr[-n_val:], dydx_tr[-n_val:]
    x_tr, y_tr, dydx_tr = x_tr[:-n_val], y_tr[:-n_val], dydx_tr[:-n_val]

    trainer_method = _method_to_trainer(method)
    t0 = time.time()
    result = train_single_experiment(
        x_tr, y_tr, dydx_tr, x_te, y_te, dydx_te,
        x_val=x_val, y_val=y_val, dydx_val=dydx_val,
        method=trainer_method, seed=seed, pbar=False,
        n_epochs=500, batch_size=256, n_layers=4, hidden_size=128,
        lr=0.005, activation="softplus", lambda_=1.0,
    )
    res = {
        "key": key,
        "method": method,
        "trainer_method": trainer_method,
        "dataset": DATASET,
        "eps_mult": eps_mult,
        "seed": seed,
        "test_value_mse": float(result.test_value_mse),
        "test_grad_mse": float(result.test_grad_mse),
        "best_epoch": int(result.best_epoch),
        "time_s": time.time() - t0,
        "git_hash": git_hash(),
    }
    path.write_text(json.dumps(res, indent=2))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eps_mults", nargs="+", type=float, default=EPS_MULT_SWEEP)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--results_dir", default="results/p7_fuzzy_2d")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 eps, 1 seed, 1 method")
    args = p.parse_args()

    if args.smoke:
        args.eps_mults = [args.eps_mults[0]]
        args.seeds = [args.seeds[0]]
        args.methods = [args.methods[0]]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    configs: List[tuple] = []
    for method in args.methods:
        for eps in args.eps_mults:
            for seed in args.seeds:
                configs.append((method, eps, seed))

    print(f"[P7] {len(configs)} configs | gpu={args.gpu} | git={git_hash()[:8]}",
          flush=True)

    n_ok = n_fail = n_skip = 0
    for i, (method, eps, seed) in enumerate(configs):
        key = f"p7_{DATASET}_eps{eps:g}_{method}_s{seed}"
        path = results_dir / f"{key}.json"
        if args.resume and path.exists():
            n_skip += 1
            continue
        try:
            _ = run_one(method, eps, seed, results_dir)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                traceback.print_exc()
        if (n_ok + n_fail) % 10 == 0 and (n_ok + n_fail) > 0:
            print(f"[P7] progress {i+1}/{len(configs)} ok={n_ok} fail={n_fail} "
                  f"skip={n_skip}", flush=True)

    print(f"[P7] DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
