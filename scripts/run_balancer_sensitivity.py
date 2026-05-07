#!/usr/bin/env python3
"""
Task #197 — GradNorm α and ReLoBRaLo τ sensitivity sweep (figure F28).

Sweeps:
  - GradNorm.alpha   ∈ {0.5, 1.0, 1.5, 2.0}     (default α=1.5)
  - ReLoBRaLo.tau    ∈ {0.10, 0.25, 0.50, 1.00} (default τ=1.0)

On a small but representative subset of the synthetic Tier-3 corpus
plus one Burgers PDE setting:

  funcs = [poly_trig, trig, step]    (3 — covers smooth + discontinuous)
  d     = [5]                        (single dim, fixes data-volume across cells)
  n     = [1024]                     (single sample size)
  noise = [0.0]                      (clean — isolates balancer effect from σ*)
  seeds = [42, 123, 456, 789, 1000]  (5 seeds = matches paper convention)

Total grid: 3 funcs × 4 (α | τ) × 5 seeds × 2 methods = 120 cells synthetic.
PDE: same 4 (α | τ) × 5 seeds × 2 methods × 1 dataset = 40 cells (Burgers IC).

Output:
  results/balancer_sensitivity/synthetic/*.json
  results/balancer_sensitivity/burgers_ic/*.json

Usage:
  python scripts/run_balancer_sensitivity.py --resume
  python scripts/run_balancer_sensitivity.py --resume --gpus 1 \\
      --workers-per-gpu 2 --start-core 16
  python scripts/run_balancer_sensitivity.py --dry-run  # list cells
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_SYN = ROOT / "results" / "balancer_sensitivity" / "synthetic"
OUT_PDE = ROOT / "results" / "balancer_sensitivity" / "burgers_ic"
LOG_FILE = ROOT / "logs" / "balancer_sensitivity.log"
OUT_SYN.mkdir(parents=True, exist_ok=True)
OUT_PDE.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


FUNCS = ["poly_trig", "trig", "step"]
SEEDS = [42, 123, 456, 789, 1000]
DIM = 5
N_SAMPLES = 1024
NOISE = 0.0
GRADNORM_ALPHAS = [0.5, 1.0, 1.5, 2.0]
RELOBRALO_TAUS = [0.10, 0.25, 0.50, 1.00]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def cell_key_syn(func: str, seed: int, method: str, hp: float) -> str:
    return (
        f"sens_{func}_d{DIM}_n{N_SAMPLES}_sigma0_s{seed}_"
        f"{method}_hp{hp:.2f}"
    )


def cell_key_pde(seed: int, method: str, hp: float) -> str:
    return f"sens_burgers_ic_s{seed}_{method}_hp{hp:.2f}"


def synthetic_specs() -> list:
    out = []
    for func in FUNCS:
        for seed in SEEDS:
            for alpha in GRADNORM_ALPHAS:
                out.append({
                    "kind": "synthetic", "func": func, "seed": seed,
                    "method": "dml_gradnorm", "hp_name": "alpha", "hp_value": alpha,
                })
            for tau in RELOBRALO_TAUS:
                out.append({
                    "kind": "synthetic", "func": func, "seed": seed,
                    "method": "dml_relobralo", "hp_name": "tau", "hp_value": tau,
                })
    return out


def pde_specs() -> list:
    out = []
    for seed in SEEDS:
        for alpha in GRADNORM_ALPHAS:
            out.append({
                "kind": "pde", "seed": seed,
                "method": "dml_gradnorm", "hp_name": "alpha", "hp_value": alpha,
            })
        for tau in RELOBRALO_TAUS:
            out.append({
                "kind": "pde", "seed": seed,
                "method": "dml_relobralo", "hp_name": "tau", "hp_value": tau,
            })
    return out


def out_path(spec: dict) -> Path:
    if spec["kind"] == "synthetic":
        return OUT_SYN / (cell_key_syn(spec["func"], spec["seed"],
                                          spec["method"], spec["hp_value"]) + ".json")
    return OUT_PDE / (cell_key_pde(spec["seed"], spec["method"],
                                      spec["hp_value"]) + ".json")


def already_done(spec) -> bool:
    p = out_path(spec)
    if not p.exists():
        return False
    try:
        json.load(open(p))
        return True
    except Exception:
        return False


def run_synthetic_cell(spec: dict, gpu: int, cpu_set: list) -> dict:
    """Run a single synthetic Tier-3-style cell with a swept hyperparameter."""
    out = out_path(spec)
    key = out.stem
    code = f"""
import json, os, sys, time
sys.path.insert(0, "{ROOT}")
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("OMP_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("MKL_NUM_THREADS", "{len(cpu_set)}")
import numpy as np, torch
torch.set_num_threads({len(cpu_set)})
from dml_benchmark.functions import generate_data, corrupt_derivatives
from dml_benchmark.trainer import train_single_experiment
np.random.seed({spec['seed']}); torch.manual_seed({spec['seed']})
tr = generate_data("{spec['func']}", n_dim={DIM}, n_samples={N_SAMPLES}, seed={spec['seed']})
te = generate_data("{spec['func']}", n_dim={DIM}, n_samples=512, seed={spec['seed'] + 1})
dydx_train_corrupted = corrupt_derivatives(
    tr.dydx, noise_level={NOISE}, seed={spec['seed']}
)
# BUGFIX 2026-05-04 — see scripts/run_balancer_sensitivity.py docstring.
balancer_kwargs = {{"{spec['hp_name']}": {spec['hp_value']}}}
t0 = time.time()
r = train_single_experiment(
    tr.x, tr.y, dydx_train_corrupted, te.x, te.y, te.dydx,
    method="{spec['method']}", lambda_=1.0, n_epochs=200,
    batch_size=64, lr=1e-3, n_layers=4, hidden_size=64,
    seed={spec['seed']}, pbar=False, balancer_kwargs=balancer_kwargs,
)
out = {{
    "key": "{key}",
    "kind": "synthetic",
    "func_type": "{spec['func']}",
    "dim": {DIM},
    "n_samples": {N_SAMPLES},
    "noise_level": {NOISE},
    "seed": {spec['seed']},
    "method": "{spec['method']}",
    "hp_name": "{spec['hp_name']}",
    "hp_value": {spec['hp_value']},
    "test_value_mse": float(r.test_value_mse),
    "test_grad_mse":  float(r.test_grad_mse),
    "best_epoch": int(r.best_epoch),
    "early_stopped": bool(r.early_stopped),
    "time_s": time.time() - t0,
}}
with open("{out}", "w") as f:
    json.dump(out, f, indent=2)
print(f"OK {{out['key']}} val={{out['test_value_mse']:.4e}} grad={{out['test_grad_mse']:.4e}}")
"""

    py = os.environ.get("PYTHON", "python")
    cpu_list = ",".join(str(c) for c in cpu_set)
    cmd = ["taskset", "-c", cpu_list, py, "-c", code]
    t0 = time.time()
    log_path = LOG_FILE.parent / f"sens_{key}.log"
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                   timeout=2 * 3600)
        return {"key": key, "ok": proc.returncode == 0 and out.exists(),
                "rc": proc.returncode, "time_s": time.time() - t0}
    except subprocess.TimeoutExpired:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0, "error": "timeout"}
    except Exception as e:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0,
                "error": str(e), "tb": traceback.format_exc()}


def run_pde_cell(spec: dict, gpu: int, cpu_set: list) -> dict:
    """Run a single Burgers IC cell with a swept balancer hyperparameter."""
    out = out_path(spec)
    key = out.stem
    code = f"""
import json, os, sys, time
sys.path.insert(0, "{ROOT}")
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("OMP_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("MKL_NUM_THREADS", "{len(cpu_set)}")
import torch
torch.set_num_threads({len(cpu_set)})
from scripts.run_burgers import run_one_cell
t0 = time.time()
r = run_one_cell(
    cache_path="{ROOT}/results/burgers/ic/_cache/burgers_K16_nu0.01_smoke0.npz",
    n_layers=4, hidden=256,
    method="{spec['method']}", seed={spec['seed']},
    smoke=False, input_mode="ic",
    {spec['hp_name']}={spec['hp_value']},
)
out = {{
    "key": "{key}", "kind": "pde", "dataset": "burgers_ic",
    "seed": {spec['seed']}, "method": "{spec['method']}",
    "hp_name": "{spec['hp_name']}", "hp_value": {spec['hp_value']},
    "test_value_mse": float(r["test_value_mse"]),
    "test_grad_mse":  float(r["test_grad_mse"]),
    "time_s": time.time() - t0,
}}
with open("{out}", "w") as f:
    json.dump(out, f, indent=2)
print(f"OK {{out['key']}}")
"""
    py = os.environ.get("PYTHON",
                         "python")
    cpu_list = ",".join(str(c) for c in cpu_set)
    cmd = ["taskset", "-c", cpu_list, py, "-c", code]
    t0 = time.time()
    log_path = LOG_FILE.parent / f"sens_{key}.log"
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                   timeout=4 * 3600)
        return {"key": key, "ok": proc.returncode == 0 and out.exists(),
                "rc": proc.returncode, "time_s": time.time() - t0}
    except subprocess.TimeoutExpired:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0, "error": "timeout"}
    except Exception as e:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0,
                "error": str(e), "tb": traceback.format_exc()}


def cpu_partition(num_workers: int, cores_per_worker: int = 2,
                   start_core: int = 0) -> list:
    total = os.cpu_count() or 16
    end = min(start_core + num_workers * cores_per_worker, total)
    pool = list(range(start_core, end))
    return [pool[i*cores_per_worker:(i+1)*cores_per_worker] for i in range(num_workers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--cores-per-worker", type=int, default=2)
    ap.add_argument("--start-core", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-pde", action="store_true",
                    help="Skip PDE cells (synthetic-only).")
    args = ap.parse_args()

    grid = synthetic_specs()
    if not args.skip_pde:
        grid += pde_specs()
    pending = [g for g in grid if not (args.resume and already_done(g))]

    workers_per_gpu = args.workers_per_gpu
    num_workers = workers_per_gpu * len(args.gpus)
    parts = cpu_partition(num_workers, args.cores_per_worker, args.start_core)

    log(f"Grid: {len(grid)} cells. Pending: {len(pending)} (resume skipped {len(grid) - len(pending)}).")
    log(f"GPUs: {args.gpus}; workers/GPU: {workers_per_gpu}; total workers: {num_workers}")
    log(f"Output dirs: {OUT_SYN}, {OUT_PDE}")

    if args.dry_run:
        for g in pending[:8]:
            log(f"  pending {out_path(g).stem}")
        if len(pending) > 8:
            log(f"  ... and {len(pending) - 8} more")
        return

    t_start = time.time()
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {}
        for i, spec in enumerate(pending):
            slot = i % num_workers
            gpu = args.gpus[slot // workers_per_gpu]
            cpu_set = parts[slot]
            runner = run_synthetic_cell if spec["kind"] == "synthetic" else run_pde_cell
            futures[ex.submit(runner, spec, gpu, cpu_set)] = spec
        for fut in as_completed(futures):
            res = fut.result()
            done = completed + failed + 1
            if res["ok"]:
                completed += 1
                status = "✓"
            else:
                failed += 1
                status = "✗"
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else float("inf")
            log(f"[{done}/{len(pending)}] {status} {res['key']} "
                f"({res['time_s']:.0f}s, rc={res.get('rc')})  eta={eta/60:.1f}min")
            if not res["ok"]:
                log(f"  error: {res.get('error', 'unknown')}")

    log(f"DONE. ok={completed} fail={failed} total_time={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
