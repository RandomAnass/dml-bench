#!/usr/bin/env python3
"""
Phase 6 — Corruption / missing-label regime map.

Grid: 2 funcs × 5 corruption types × 5 severity levels × 5 seeds × 3 methods = 750.

Reproducible replacement for the earlier inline-python launcher that was used
pre-incident. Records config + git hash in each result JSON so the run is
traceable.

Usage:
    # full grid on GPU 0
    python scripts/p6_corruption_run.py --gpu 0

    # restart / skip existing files
    python scripts/p6_corruption_run.py --gpu 0 --resume

    # subset (smoke)
    python scripts/p6_corruption_run.py --gpu 0 --funcs poly_trig --seeds 42 --methods dml_fixed
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


FUNCS = ["poly_trig", "trig"]
CORR_TYPES = ["gaussian_additive", "multiplicative", "dropout", "sparse_coord", "mixed"]
SEVERITY_LEVELS = [0.01, 0.05, 0.1, 0.3, 1.0]
SEEDS = [42, 123, 456, 789, 1000]
METHODS = ["vanilla", "dml_fixed", "dml_gradnorm"]
DIM_DEFAULT = 5
N_SAMPLES_DEFAULT = 2048
N_EPOCHS_DEFAULT = 300


def corrupt(dydx: np.ndarray, corr_type: str, severity: float, rng: np.random.RandomState) -> np.ndarray:
    """Deterministic corruption of derivative labels."""
    out = dydx.copy()
    d = out.shape[-1]
    if corr_type == "gaussian_additive":
        out += (rng.randn(*out.shape) * severity * np.std(out)).astype(np.float32)
    elif corr_type == "multiplicative":
        out *= (1 + severity * rng.randn(*out.shape)).astype(np.float32)
    elif corr_type == "dropout":
        out *= (rng.random(out.shape) > severity).astype(np.float32)
    elif corr_type == "sparse_coord":
        n_keep = max(1, int(d * (1 - severity)))
        keep_dims = rng.choice(d, n_keep, replace=False)
        mask = np.zeros(d, dtype=np.float32)
        mask[keep_dims] = 1.0
        out = out * mask
    elif corr_type == "mixed":
        out += (rng.randn(*out.shape) * severity * 0.5 * np.std(out)).astype(np.float32)
        out *= (rng.random(out.shape) > (severity * 0.5)).astype(np.float32)
    else:
        raise ValueError(f"Unknown corr_type: {corr_type}")
    return out


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_one(func: str, corr_type: str, severity: float, seed: int, method: str,
            dim: int, n_samples: int, n_epochs: int, results_dir: Path) -> dict:
    key = f"p6_{func}_d{dim}_{corr_type}_sev{severity}_{method}_s{seed}"
    path = results_dir / f"{key}.json"

    rng = np.random.RandomState(seed)
    data = generate_data(func, n_dim=dim, n_samples=n_samples, seed=seed)
    x = data.x.astype(np.float32)
    y = data.y.astype(np.float32)
    dydx_clean = data.dydx.astype(np.float32)
    dydx_corr = corrupt(dydx_clean, corr_type, severity, rng)

    n_train = int(len(x) * 0.8)
    t0 = time.time()
    result = train_single_experiment(
        x_train=x[:n_train], y_train=y[:n_train], dydx_train=dydx_corr[:n_train],
        x_test=x[n_train:], y_test=y[n_train:], dydx_test=dydx_clean[n_train:],
        method=method, seed=seed, pbar=False,
        n_epochs=n_epochs, batch_size=256, n_layers=4, hidden_size=256,
        lr=0.005, activation="softplus", lambda_=1.0,
    )
    res = {
        "key": key,
        "method": method,
        "dataset": func,
        "corruption_type": corr_type,
        "severity": severity,
        "dim": dim,
        "seed": seed,
        "n_samples": n_samples,
        "n_epochs_target": n_epochs,
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
    p.add_argument("--funcs", nargs="+", default=FUNCS)
    p.add_argument("--corr_types", nargs="+", default=CORR_TYPES)
    p.add_argument("--severities", nargs="+", type=float, default=SEVERITY_LEVELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--dim", type=int, default=DIM_DEFAULT)
    p.add_argument("--n_samples", type=int, default=N_SAMPLES_DEFAULT)
    p.add_argument("--n_epochs", type=int, default=N_EPOCHS_DEFAULT)
    p.add_argument("--results_dir", default="results/p6_corruption")
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose result JSON already exists.")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    configs: List[tuple] = []
    for func in args.funcs:
        for corr in args.corr_types:
            for sev in args.severities:
                for seed in args.seeds:
                    for method in args.methods:
                        configs.append((func, corr, sev, seed, method))

    print(f"[P6] {len(configs)} configs | gpu={args.gpu} | git={git_hash()[:8]}",
          flush=True)

    n_ok = n_fail = n_skip = 0
    for i, (func, corr, sev, seed, method) in enumerate(configs):
        key = f"p6_{func}_d{args.dim}_{corr}_sev{sev}_{method}_s{seed}"
        path = results_dir / f"{key}.json"
        if args.resume and path.exists():
            n_skip += 1
            continue
        try:
            _ = run_one(func, corr, sev, seed, method,
                        args.dim, args.n_samples, args.n_epochs, results_dir)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                traceback.print_exc()
        if (n_ok + n_fail) % 25 == 0 and (n_ok + n_fail) > 0:
            print(f"[P6] progress {i+1}/{len(configs)} ok={n_ok} fail={n_fail} "
                  f"skip={n_skip}", flush=True)

    print(f"[P6] DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
