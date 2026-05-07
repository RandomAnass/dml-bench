#!/usr/bin/env python3
"""
Targeted bachelier noise-sweep extension.

run_full_benchmark.py:152-161 scoped the gradient-noise axis (sigma in
{0.05, 0.10, 0.20, 0.50}) to trig + poly_trig only; bachelier is
present at sigma=0 only. This script fills the bachelier noise sweep
at a single representative cell so the sigma_star figure can show a
bachelier curve. Output JSONs follow the existing schema (results
land at results/tier3_benchmark/) so sigma_star_lowess.py picks them
up without changes.

Cell grid: d=10, n_train=1024, sigma in {0.05, 0.10, 0.20, 0.50},
seeds = [42, 123, 456, 789, 1000], methods = [vanilla, dml_fixed].
Total: 40 cells (~20 min on a free GPU).

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_bachelier_noise_extension.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dml_benchmark.functions import (    # noqa: E402
    generate_data,
    train_test_split,
    corrupt_derivatives,
)
from dml_benchmark.trainer import train_single_experiment   # noqa: E402

OUT_DIR = ROOT / "results" / "tier3_benchmark"

DIM = 10
N_TRAIN = 1024
NOISE_LEVELS = [0.05, 0.10, 0.20, 0.50]
SEEDS = [42, 123, 456, 789, 1000]
METHODS = ["vanilla", "dml_fixed"]

HPARAMS = {
    "n_epochs": 500,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 5e-3,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}


def cell_key(method: str, seed: int, noise: float) -> str:
    return f"bachelier_d{DIM}_n{N_TRAIN}_noise{noise}_s{seed}_{method}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = [
        (method, seed, noise)
        for seed in SEEDS
        for noise in NOISE_LEVELS
        for method in METHODS
    ]
    pending = []
    for method, seed, noise in grid:
        out = OUT_DIR / f"{cell_key(method, seed, noise)}.json"
        if not out.exists():
            pending.append((method, seed, noise, out))

    print(
        f"[bach-ext] grid: {len(grid)} cells; pending: {len(pending)} "
        f"(skipped {len(grid) - len(pending)} already-done).",
        flush=True,
    )

    t0 = time.time()
    for i, (method, seed, noise, out) in enumerate(pending, 1):
        print(
            f"[{i}/{len(pending)}] method={method} seed={seed} "
            f"noise={noise} ...",
            flush=True,
            end=" ",
        )
        ti = time.time()
        try:
            data = generate_data(
                "bachelier",
                n_dim=DIM,
                n_samples=N_TRAIN,
                seed=seed,
            )
            train, test = train_test_split(data, train_ratio=0.8, seed=seed)
            dydx_train_noisy = (
                corrupt_derivatives(train.dydx, noise_level=noise, seed=seed)
                if noise > 0
                else train.dydx
            )
            r = train_single_experiment(
                x_train=train.x,
                y_train=train.y,
                dydx_train=dydx_train_noisy,
                x_test=test.x,
                y_test=test.y,
                dydx_test=test.dydx,
                lambda_=1.0,
                method=method,
                seed=seed,
                pbar=False,
                **HPARAMS,
            )
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            continue

        result_dict = {
            "method": method,
            "func_type": "bachelier",
            "dim": DIM,
            "n_samples": N_TRAIN,
            "noise_level": noise,
            "seed": seed,
            "lambda": 1.0,
            "test_value_mse": r.test_value_mse,
            "test_grad_mse": r.test_grad_mse,
            "best_epoch": r.best_epoch,
            "time_s": time.time() - ti,
            "n_epochs_actual": len(r.training_logs),
        }
        with open(out, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(
            f"OK val={r.test_value_mse:.3e} grad={r.test_grad_mse:.3e} "
            f"({result_dict['time_s']:.0f}s)"
        )

    print(
        f"[bach-ext] done. wall {(time.time() - t0) / 60:.1f} min.",
        flush=True,
    )


if __name__ == "__main__":
    main()
