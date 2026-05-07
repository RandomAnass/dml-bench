#!/usr/bin/env python3
"""
Phase 9 — Dim-normalized GradNorm sweep.

Hypothesis (Priority G): GradNorm's underperformance vs fixed-λ at high d is
partially a dimension-calibration artifact — dividing the derivative gradient
norm by d (or √d) should rebalance.

Grid: 5 dims × 3 methods × 5 seeds = 75 runs.
Runs post-cff4862a where GradNorm's weight gradient path is restored.

Note on expected finding: Adam used to step task_weights is approximately
per-parameter scale-invariant, so a constant dim factor applied to the
gradient-w.r.t.-w_i may be (nearly) absorbed. Document outcome either way.

Usage:
    python scripts/p9_dimnorm_run.py --gpu 0
    python scripts/p9_dimnorm_run.py --gpu 0 --resume
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

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.functions import generate_data


DATASET = "trig"
DIMS = [2, 5, 10, 20, 50]
SEEDS = [42, 123, 456, 789, 1000]
METHODS = ["dml_gradnorm", "dml_dimnorm_gradnorm", "dml_sqrtdimnorm_gradnorm"]
N_SAMPLES = 1024
N_EPOCHS = 500


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_one(dim: int, method: str, seed: int, results_dir: Path) -> dict:
    key = f"p9_{DATASET}_d{dim}_{method}_s{seed}"
    path = results_dir / f"{key}.json"

    tr = generate_data(DATASET, n_dim=dim, n_samples=N_SAMPLES, seed=seed)
    va = generate_data(DATASET, n_dim=dim, n_samples=N_SAMPLES // 4, seed=seed + 10_000)
    te = generate_data(DATASET, n_dim=dim, n_samples=N_SAMPLES // 4, seed=seed + 20_000)

    t0 = time.time()
    result = train_single_experiment(
        tr.x, tr.y, tr.dydx, te.x, te.y, te.dydx,
        x_val=va.x, y_val=va.y, dydx_val=va.dydx,
        method=method, seed=seed, pbar=False,
        n_epochs=N_EPOCHS, batch_size=256, n_layers=4, hidden_size=256,
        lr=0.005, activation="softplus", lambda_=1.0,
    )
    res = {
        "key": key,
        "method": method,
        "dataset": DATASET,
        "dim": dim,
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
    p.add_argument("--dims", nargs="+", type=int, default=DIMS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--results_dir", default="results/p9_dimnorm_gradnorm")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    configs: List[tuple] = []
    for d in args.dims:
        for m in args.methods:
            for s in args.seeds:
                configs.append((d, m, s))

    print(f"[P9] {len(configs)} configs | gpu={args.gpu} | git={git_hash()[:8]}",
          flush=True)

    n_ok = n_fail = n_skip = 0
    for i, (d, m, s) in enumerate(configs):
        key = f"p9_{DATASET}_d{d}_{m}_s{s}"
        path = results_dir / f"{key}.json"
        if args.resume and path.exists():
            n_skip += 1
            continue
        try:
            _ = run_one(d, m, s, results_dir)
            n_ok += 1
        except Exception:
            n_fail += 1
            if n_fail <= 3:
                traceback.print_exc()
        if (n_ok + n_fail) % 10 == 0 and (n_ok + n_fail) > 0:
            print(f"[P9] progress {i+1}/{len(configs)} ok={n_ok} fail={n_fail} "
                  f"skip={n_skip}", flush=True)

    print(f"[P9] DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
