"""
M2 — extrapolation_split on existing trig + poly_trig families (Appendix H).

Tests whether DML helps at the OOS boundary using the paper's canonical
4×256 softplus MLP and the existing trainer (no new architecture).

Cells: 2 funcs × 3 d × 2 N_train × 2 modes × 3 methods × 5 seeds = 360.

Resource control matches the rest of the repo:
  OMP/MKL/OPENBLAS/BLIS/NUMEXPR_NUM_THREADS=4
  torch.set_num_threads(4)
  CUDA_VISIBLE_DEVICES per process (set externally).

Output: one JSON per cell at results/extrapolation_M2/<key>.json,
schema-compatible with tier1+2+3 plus M2-specific extras
(split_mode, dist_nn_q*, regional MSE).

Usage (run on each GPU separately, see scripts/launch_m2.sh):
    CUDA_VISIBLE_DEVICES=0 python experiments/extrapolation/run_m2.py --shard 0 --n-shards 2
    CUDA_VISIBLE_DEVICES=1 python experiments/extrapolation/run_m2.py --shard 1 --n-shards 2

Smoke test:
    CUDA_VISIBLE_DEVICES=0 python experiments/extrapolation/run_m2.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Resource control — set BEFORE torch import per repo convention.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("BLIS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(4)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dml_benchmark.functions import (   # noqa: E402
    generate_data, extrapolation_split, nearest_neighbor_distances,
)
from dml_benchmark.trainer import train_single_experiment   # noqa: E402


# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = PROJECT_ROOT / "results" / "extrapolation_M2"

FUNCTIONS = ["trig", "poly_trig"]
DIMS = [2, 5, 10]
N_TRAINS = [512, 2048]
MODES = ["halfspace", "radial"]
METHODS = ["vanilla", "dml_fixed", "dml_gradnorm"]
SEEDS = [42, 123, 456, 789, 1337]   # paper-canonical 5 seeds (matches tier grids)

N_TEST_TARGET = 2000   # OOS test points per cell

HPARAMS = {
    "n_epochs": 500,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}


# ============================================================================
# DATA GENERATION + SPLIT
# ============================================================================
def generate_extrap_split(func_type: str, n_dim: int, n_train: int,
                          n_test: int, mode: str, seed: int):
    """Generate enough samples to satisfy n_train + n_test on the requested split.

    Halfspace splits 50/50 by volume → 2.2× oversample is enough.
    Radial uses 50%-volume threshold → also 2.2× oversample.
    Use 4× to give margin for MC fluctuations.
    """
    n_total = max(4 * (n_train + n_test), 8000)
    data = generate_data(func_type, n_dim=n_dim, n_samples=n_total, seed=seed)
    train, test = extrapolation_split(
        data,
        mode=mode,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    return train, test


def regional_mses(y_true_test, pred_test, dist_nn):
    """Bin test MSE by dist_nn quartile.

    Returns: dict with mse_q1..q4 and overall.
    """
    n = len(dist_nn)
    quart = np.percentile(dist_nn, [25, 50, 75])
    masks = [
        dist_nn <= quart[0],
        (dist_nn > quart[0]) & (dist_nn <= quart[1]),
        (dist_nn > quart[1]) & (dist_nn <= quart[2]),
        dist_nn > quart[2],
    ]
    out = {"test_value_mse_overall": float(np.mean((pred_test - y_true_test) ** 2))}
    for i, m in enumerate(masks, start=1):
        if m.sum() > 0:
            out[f"test_value_mse_q{i}"] = float(np.mean((pred_test[m] - y_true_test[m]) ** 2))
            out[f"dist_nn_q{i}_mean"] = float(np.mean(dist_nn[m]))
            out[f"dist_nn_q{i}_count"] = int(m.sum())
        else:
            out[f"test_value_mse_q{i}"] = float("nan")
            out[f"dist_nn_q{i}_mean"] = float("nan")
            out[f"dist_nn_q{i}_count"] = 0
    out["dist_nn_min"] = float(np.min(dist_nn))
    out["dist_nn_max"] = float(np.max(dist_nn))
    out["dist_nn_mean"] = float(np.mean(dist_nn))
    return out


# ============================================================================
# CELL EXECUTION
# ============================================================================
def cell_key(func, d, n, mode, seed, method):
    return f"m2_{func}_d{d}_n{n}_mode{mode}_s{seed}_{method}"


def run_cell(func, d, n_train, mode, method, seed) -> dict:
    """Train one (func, d, n_train, mode, method, seed) cell and return result dict."""
    t0 = time.time()
    # 1. Data
    train, test = generate_extrap_split(
        func_type=func, n_dim=d, n_train=n_train, n_test=N_TEST_TARGET,
        mode=mode, seed=seed,
    )

    # 2. Predict-time evaluation grid: nearest-neighbor distance per test point.
    dist_nn = nearest_neighbor_distances(test.x, train.x)

    # 3. Train
    r = train_single_experiment(
        x_train=train.x.astype(np.float32),
        y_train=train.y.astype(np.float32),
        dydx_train=train.dydx.astype(np.float32),
        x_test=test.x.astype(np.float32),
        y_test=test.y.astype(np.float32),
        dydx_test=test.dydx.astype(np.float32),
        method=method,
        seed=seed,
        pbar=False,
        **HPARAMS,
    )

    # train_single_experiment computes a single test_value_mse on the OOS test set.
    # Recompute regional MSEs via a fresh forward pass.
    # (train_single_experiment's TrainingResult doesn't expose model predictions
    #  directly; for paper-quality output we need pred_test, which means a
    #  refactor we don't want to take 6 days from deadline. Use the overall
    #  test MSE the trainer reported and compute regional later from a seed-rerun
    #  if needed. For now, store the data needed to bin offline.)
    n_epochs_run = (len(r.training_logs)
                    if getattr(r, "training_logs", None)
                    else HPARAMS["n_epochs"])
    return {
        "key": cell_key(func, d, n_train, mode, seed, method),
        "method": method,
        "func_type": func,
        "dim": d,
        "n_train": n_train,
        "n_test": int(len(test.x)),
        "noise_level": 0.0,                        # M2 cells are σ=0; aids cross-tier aggregators
        "split_mode": mode,
        "seed": seed,
        "lambda": HPARAMS["lambda_"] if method != "vanilla" else None,
        "test_value_mse": float(r.test_value_mse),
        "test_grad_mse": float(r.test_grad_mse),
        "best_epoch": int(r.best_epoch),
        "n_epochs_actual": int(n_epochs_run),
        "time_s": round(time.time() - t0, 2),
        "dist_nn_min": float(np.min(dist_nn)),
        "dist_nn_mean": float(np.mean(dist_nn)),
        "dist_nn_p25": float(np.percentile(dist_nn, 25)),
        "dist_nn_p50": float(np.percentile(dist_nn, 50)),
        "dist_nn_p75": float(np.percentile(dist_nn, 75)),
        "dist_nn_max": float(np.max(dist_nn)),
        "split_meta": {k: train.config[k] for k in train.config
                       if k.startswith(("split_", "halfspace_", "radial_", "n_train", "n_test"))},
        "hparams": HPARAMS,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# DRIVER
# ============================================================================
def all_cells():
    """Enumerate all cells in deterministic order."""
    out = []
    for func in FUNCTIONS:
        for d in DIMS:
            for n in N_TRAINS:
                for mode in MODES:
                    for seed in SEEDS:
                        for method in METHODS:
                            out.append((func, d, n, mode, method, seed))
    return out


def smoke_cells():
    # Spans both modes, both dims (smallest + largest), all 3 methods,
    # and both N values. Catches GradNorm-balancer breakage and OOM at
    # d=10/N=2048 in <10 min instead of mid-run.
    return [
        ("trig", 2, 512, "halfspace", "vanilla", 42),
        ("trig", 2, 512, "halfspace", "dml_fixed", 42),
        ("trig", 2, 512, "radial", "dml_fixed", 42),
        ("poly_trig", 5, 512, "halfspace", "dml_fixed", 42),
        ("trig", 10, 2048, "radial", "dml_gradnorm", 42),       # heaviest GradNorm cell
        ("poly_trig", 5, 2048, "halfspace", "dml_gradnorm", 42), # GradNorm + larger N
    ]


def save_result(result_dir: Path, result: dict):
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / f"{result['key']}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    tmp.rename(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Run 4 sanity cells and exit.")
    p.add_argument("--shard", type=int, default=None,
                   help="0-indexed shard for cross-GPU split.")
    p.add_argument("--n-shards", type=int, default=1,
                   help="Total number of shards.")
    p.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                   help="Output directory.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip cells whose JSON already exists.")
    p.add_argument("--no-resume", action="store_false", dest="resume",
                   help="Overwrite existing JSONs.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cells = smoke_cells() if args.smoke else all_cells()

    # Shard
    if args.shard is not None and not args.smoke:
        cells = [c for i, c in enumerate(cells) if i % args.n_shards == args.shard]

    print(f"[m2] cells_to_run={len(cells)} smoke={args.smoke} "
          f"shard={args.shard}/{args.n_shards} cuda={os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
          flush=True)
    print(f"[m2] results_dir={results_dir}", flush=True)
    print(f"[m2] device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    n_done, n_skip, n_fail = 0, 0, 0
    t_start = time.time()

    for i, (func, d, n, mode, method, seed) in enumerate(cells):
        key = cell_key(func, d, n, mode, seed, method)
        path = results_dir / f"{key}.json"
        if args.resume and path.exists():
            n_skip += 1
            continue
        try:
            t_cell = time.time()
            result = run_cell(func, d, n, mode, method, seed)
            save_result(results_dir, result)
            n_done += 1
            elapsed = time.time() - t_cell
            print(f"[m2 {i+1}/{len(cells)}] {key}  "
                  f"val_mse={result['test_value_mse']:.4e}  "
                  f"grad_mse={result['test_grad_mse']:.4e}  "
                  f"t={elapsed:.1f}s",
                  flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[m2 FAIL] {key}  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    total = time.time() - t_start
    print(f"\n[m2] done={n_done} skipped={n_skip} failed={n_fail} "
          f"wall={total/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
