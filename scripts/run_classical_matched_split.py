#!/usr/bin/env python3
"""
Classical-vs-neural comparison at matched training fraction (#179).

The original tier3/tier5 classical-baseline runs gave classical methods
(GP/KRR/RF) all 80% of the data, while neural-DML methods used the same
80% but internally carved off 20% for validation/early-stopping
(`val_split=0.2` in dml_benchmark.trainer.train_single_experiment).
That gave classical a 25% data advantage (80/64 = 1.25x).

This script reruns the comparison on a matched 64/16/20 split:
classical fits on the 64% train slice; neural fits on the same 64% train
+ 16% val (i.e., gets the 80% the classical baselines also see, but
spends 16% of it on early stopping). Test is the same 20% for both.

GP scaling: O(n^3). We cap GP to small (n_train, d) cells where it
actually finishes in reasonable time. KRR, RF run everywhere.

Output: results/classical_matched_split/<func>_d<d>_n<n>_s<seed>_<method>.json
Logs:   logs/classical_matched_split.log

Concurrency: --workers N. Default uses min(8, cpu_count // 2). Workers
pin to disjoint CPU sets to avoid thrashing.

Usage:
  python scripts/run_classical_matched_split.py            # default workers
  python scripts/run_classical_matched_split.py --workers 6
  python scripts/run_classical_matched_split.py --resume
  python scripts/run_classical_matched_split.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Imports deferred until inside worker so child processes do not reload
# heavy GPU libraries.

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

FUNCTIONS = ["poly_trig", "trig", "bachelier", "black_scholes"]
DIMS = [5, 10, 20, 50, 100]
N_SAMPLES = [256, 1024, 4096]
SEEDS = [42, 123, 456, 789, 1024]

CLASSICAL_METHODS = ["gp", "krr", "rf"]
NEURAL_METHODS = ["dml_fixed", "vanilla"]
ALL_METHODS = CLASSICAL_METHODS + NEURAL_METHODS

VAL_FRAC = 0.2          # of the 80% pre-test pool ⇒ 16% of total
TEST_FRAC = 0.2         # of total
TRAIN_FRAC = 1.0 - TEST_FRAC - (1.0 - TEST_FRAC) * VAL_FRAC  # 0.64

OUT_DIR = ROOT / "results" / "classical_matched_split"
LOG_FILE = ROOT / "logs" / "classical_matched_split.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def gp_admissible(n_train: int, dim: int) -> bool:
    """Skip GP where O(n^3) cost dominates. ~656 train points
    (the 64% slice of n=1024) at dim<=20 finishes in 2-5 minutes;
    higher n or higher d makes GP impractical and brings nothing
    over KRR/RF that we don't already have."""
    return n_train <= 656 and dim <= 20


def make_key(func: str, dim: int, n: int, seed: int, method: str) -> str:
    return f"{func}_d{dim}_n{n}_s{seed}_{method}"


def out_path(key: str) -> Path:
    return OUT_DIR / f"{key}.json"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(spec: dict) -> dict:
    """Run a single (function, dim, n, seed, method) cell and write
    JSON. Returns a small status dict for the dispatcher's log."""
    # Late imports keep the child small.
    from dml_benchmark.functions import generate_data, train_test_split
    from dml_benchmark.baselines import run_baseline_experiment

    func = spec["func"]
    dim = spec["dim"]
    n = spec["n"]
    seed = spec["seed"]
    method = spec["method"]
    key = spec["key"]

    if spec.get("cpu_affinity") is not None:
        try:
            os.sched_setaffinity(0, set(spec["cpu_affinity"]))
        except Exception:
            pass

    np.random.seed(seed)
    t0 = time.time()
    try:
        # Reproducible 64/16/20 split: same RNG seed everywhere.
        data = generate_data(func, n_dim=dim, n_samples=n, seed=seed)
        # First carve off the 20% test set.
        rest, test_data = train_test_split(data, train_ratio=1.0 - TEST_FRAC, seed=seed)
        # From the remaining 80%, carve off 16/(16+64) = 20% as val.
        train_data, val_data = train_test_split(rest, train_ratio=1.0 - VAL_FRAC, seed=seed)

        x_train, y_train, dydx_train = train_data.x, train_data.y, train_data.dydx
        x_val, y_val, dydx_val = val_data.x, val_data.y, val_data.dydx
        x_test, y_test, dydx_test = test_data.x, test_data.y, test_data.dydx

        if method in CLASSICAL_METHODS:
            # Classical methods get the 64% train only; they don't need val.
            r = run_baseline_experiment(
                method, x_train, y_train, dydx_train,
                x_test, y_test, dydx_test,
            )
            value_mse = r["value_mse"]
            grad_mse = r["grad_mse"]
            extra = {"baseline_fit_time_s": r.get("fit_time_s")}
        elif method in NEURAL_METHODS:
            # Neural methods get 64% train + 16% val explicitly.
            from dml_benchmark.trainer import train_single_experiment
            lambda_ = 1.0 if method == "dml_fixed" else 0.0
            res = train_single_experiment(
                x_train=x_train, y_train=y_train, dydx_train=dydx_train,
                x_val=x_val, y_val=y_val, dydx_val=dydx_val,
                x_test=x_test, y_test=y_test, dydx_test=dydx_test,
                lambda_=lambda_, n_epochs=500, batch_size=min(256, len(x_train)),
                seed=seed, method=method,
            )
            value_mse = res.test_value_mse
            grad_mse = res.test_grad_mse
            extra = {"best_epoch": getattr(res, "best_epoch", None),
                     "early_stopped": getattr(res, "early_stopped", None)}
        else:
            raise ValueError(f"unknown method: {method}")

        elapsed = time.time() - t0
        out = {
            "method": "baseline_" + method if method in CLASSICAL_METHODS else method,
            "func_type": func,
            "dim": dim,
            "n_samples": n,
            "n_train": len(x_train),
            "n_val": len(x_val),
            "n_test": len(x_test),
            "noise_level": 0.0,
            "seed": seed,
            "lambda": 1.0 if method in ("dml_fixed",) else None,
            "split_train_frac": TRAIN_FRAC,
            "test_value_mse": float(value_mse),
            "test_grad_mse": float(grad_mse),
            "time_s": elapsed,
            **extra,
        }
        out_path(key).write_text(json.dumps(out, indent=2))
        return {"key": key, "ok": True, "time_s": elapsed}
    except Exception as e:
        tb = traceback.format_exc()
        return {"key": key, "ok": False, "error": str(e), "tb": tb,
                "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def build_grid(skip_gp_large: bool = True) -> list:
    n_train_of_n = lambda n: int(round(n * (1.0 - TEST_FRAC) * (1.0 - VAL_FRAC)))
    grid = []
    for func in FUNCTIONS:
        for dim in DIMS:
            for n in N_SAMPLES:
                n_train = n_train_of_n(n)
                for seed in SEEDS:
                    for method in ALL_METHODS:
                        if method == "gp" and skip_gp_large and not gp_admissible(n_train, dim):
                            continue
                        grid.append({
                            "func": func, "dim": dim, "n": n, "seed": seed,
                            "method": method,
                            "key": make_key(func, dim, n, seed, method),
                        })
    return grid


def cpu_partition(num_workers: int, cores_per_worker: int = 4) -> list:
    """Disjoint CPU lists for each worker. Avoids workers thrashing."""
    total = os.cpu_count() or 16
    take = min(num_workers * cores_per_worker, total)
    pool = list(range(take))
    return [pool[i*cores_per_worker:(i+1)*cores_per_worker] for i in range(num_workers)]


def already_done(key: str) -> bool:
    p = out_path(key)
    if not p.exists():
        return False
    try:
        json.load(open(p))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 16) // 4)))
    ap.add_argument("--cores-per-worker", type=int, default=4)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-gp-large", action="store_true",
                    help="Run GP at n_train > 656 or d > 20 too (slow; default skips).")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Subset of methods (gp, krr, rf, dml_fixed, vanilla).")
    args = ap.parse_args()

    grid = build_grid(skip_gp_large=not args.include_gp_large)
    if args.methods:
        grid = [g for g in grid if g["method"] in set(args.methods)]
    log(f"Built grid: {len(grid)} cells across {len(FUNCTIONS)} funcs, {len(DIMS)} dims, "
        f"{len(N_SAMPLES)} n values, {len(SEEDS)} seeds, "
        f"{len(args.methods or ALL_METHODS)} methods.")
    log(f"Train fraction = {TRAIN_FRAC} (matched 64/16/20 split).")
    log(f"Output dir: {OUT_DIR}")
    log(f"Log file:   {LOG_FILE}")

    pending = [g for g in grid if not (args.resume and already_done(g["key"]))]
    log(f"Pending: {len(pending)} (resume skipped {len(grid) - len(pending)} existing).")

    if args.dry_run:
        from collections import Counter
        c = Counter(g["method"] for g in pending)
        for m, n in c.items():
            log(f"  pending method={m}: {n}")
        return

    # CPU partition for workers (disjoint cpusets prevents OS thrash).
    partitions = cpu_partition(args.workers, args.cores_per_worker)
    log(f"Spawning {args.workers} workers; cores per worker = {args.cores_per_worker}.")
    log(f"CPU partitions: {partitions}")

    t_start = time.time()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for i, spec in enumerate(pending):
            spec["cpu_affinity"] = partitions[i % args.workers]
            fut = ex.submit(run_one, spec)
            futures[fut] = spec
        for fut in as_completed(futures):
            res = fut.result()
            if res["ok"]:
                completed += 1
            else:
                failed += 1
            elapsed = time.time() - t_start
            done = completed + failed
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else float("inf")
            status = "✓" if res["ok"] else "✗"
            log(f"[{done}/{len(pending)}] {status} {res['key']} "
                f"({res['time_s']:.1f}s)  eta={eta/60:.1f}min")
            if not res["ok"]:
                log(f"   error: {res['error']}")

    log(f"DONE. ok={completed} fail={failed} total_time={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
