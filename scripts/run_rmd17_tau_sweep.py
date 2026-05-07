#!/usr/bin/env python3
"""
rMD17 warmup-fraction (τ) sweep on aspirin (#195).

The current paper attributes the dml_warmup catastrophic failure on
PaiNN-rMD17 to τ = 0.5 allocating 500 epochs to energy-only training
on 950 frames (overfit). This sweep tests whether smaller τ recovers
performance.

Grid:
  molecule = aspirin
  method   = dml_warmup
  τ        = 0.1, 0.25, 0.5
  splits   = canonical Figshare 1..5 (seed = split_id)
  ⇒ 15 PaiNN runs

Concurrency: each cell runs as its own subprocess so CUDA contexts do
not interfere. By default, 3 workers per GPU × 2 GPUs = 6 simultaneous,
falling back to 2/GPU if --light. Each worker pinned to disjoint CPU
set to prevent thrashing on the 24-core host.

Output: results/rmd17_tau_sweep/painn_md17_aspirin_split{1..5}_dml_warmup_tau{0.10,0.25,0.50}.json
Logs:   logs/rmd17_tau_sweep.log

Usage:
  python scripts/run_rmd17_tau_sweep.py             # default 3 workers/GPU
  python scripts/run_rmd17_tau_sweep.py --light     # 2 workers/GPU
  python scripts/run_rmd17_tau_sweep.py --resume    # skip done cells
  python scripts/run_rmd17_tau_sweep.py --dry-run
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

OUT_DIR = ROOT / "results" / "rmd17_tau_sweep"
LOG_FILE = ROOT / "logs" / "rmd17_tau_sweep.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MOLECULES = ["aspirin"]
METHOD = "dml_warmup"
TAUS = [0.10, 0.25, 0.50]
SPLIT_IDS = [1, 2, 3, 4, 5]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def cell_key(mol: str, split_id: int, tau: float) -> str:
    return f"painn_md17_{mol}_split{split_id}_{METHOD}_tau{tau:.2f}"


def out_path(key: str) -> Path:
    return OUT_DIR / f"{key}.json"


def cpu_partition(num_workers: int, cores_per_worker: int = 4,
                   start_core: int = 0) -> list:
    total = os.cpu_count() or 16
    end = min(start_core + num_workers * cores_per_worker, total)
    pool = list(range(start_core, end))
    return [pool[i*cores_per_worker:(i+1)*cores_per_worker] for i in range(num_workers)]


def run_cell(spec: dict) -> dict:
    """Run one cell as a fresh subprocess to keep CUDA contexts isolated."""
    mol = spec["molecule"]
    split_id = spec["split_id"]
    tau = spec["tau"]
    gpu = spec["gpu"]
    cpu_set = spec["cpu_set"]
    timeout_s = spec.get("timeout_s", 8 * 3600)
    key = cell_key(mol, split_id, tau)
    out = out_path(key)

    # Build the worker command. We invoke a small inline runner via -c that
    # imports train_one and writes the JSON itself. This avoids spawning the
    # main() of run_painn.py, which would set its own CUDA_VISIBLE_DEVICES
    # and process all combinations.
    code = f"""
import json, os, sys, time
sys.path.insert(0, "{ROOT}")
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("OMP_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("MKL_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "{len(cpu_set)}")
import torch
torch.set_num_threads({len(cpu_set)})
from experiments.molecular.run_painn import train_one, HPARAMS_CANONICAL
hparams = dict(HPARAMS_CANONICAL)
result = train_one(
    molecule="{mol}", method="{METHOD}", seed={split_id},
    hparams=hparams, smoke=False, gpu={gpu}, split_id={split_id},
    warmup_fraction={tau},
)
result["sweep_tau"] = {tau}
result["key"] = "{key}"
with open("{out}", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"OK key={{result['key']}} time={{result['time_s']:.1f}}s "
      f"E_MAE={{result['test_energy_mae_mev']:.1f}} F_MAE={{result['test_force_mae_mev']:.1f}}")
"""

    env = os.environ.copy()
    # Pin the subprocess to its CPU set
    cpu_list = ",".join(str(c) for c in cpu_set)
    # Use the conda env that has schnetpack/pytorch_lightning installed.
    py = os.environ.get("PAINN_PYTHON",
                        "python")
    cmd = ["taskset", "-c", cpu_list, py, "-c", code]

    t0 = time.time()
    log_path = LOG_FILE.parent / f"rmd17_tau_sweep_{key}.log"
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                   timeout=timeout_s)
        elapsed = time.time() - t0
        ok = (proc.returncode == 0) and out.exists()
        return {"key": key, "ok": ok, "rc": proc.returncode,
                "time_s": elapsed, "log": str(log_path)}
    except subprocess.TimeoutExpired:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0, "error": "timeout"}
    except Exception as e:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0, "error": str(e),
                "tb": traceback.format_exc()}


def build_grid() -> list:
    return [{"molecule": mol, "split_id": sid, "tau": tau}
            for mol in MOLECULES for sid in SPLIT_IDS for tau in TAUS]


def already_done(spec) -> bool:
    p = out_path(cell_key(spec["molecule"], spec["split_id"], spec["tau"]))
    if not p.exists():
        return False
    try:
        json.load(open(p))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    ap.add_argument("--light", action="store_true",
                    help="Use 2 workers/GPU instead of default 3.")
    ap.add_argument("--cores-per-worker", type=int, default=4)
    ap.add_argument("--start-core", type=int, default=0,
                    help="First CPU index to allocate; useful when other "
                         "workers occupy the lower range (e.g., 24 if cores 0-23 are taken).")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout-hours", type=float, default=8.0,
                    help="Per-cell wall-clock timeout. Default 8h; bump to 14 for "
                         "low-τ cells that hit max_epochs without ES triggering.")
    args = ap.parse_args()

    workers_per_gpu = 2 if args.light else args.workers_per_gpu
    num_workers = workers_per_gpu * len(args.gpus)
    partitions = cpu_partition(num_workers, args.cores_per_worker, args.start_core)

    grid = build_grid()
    pending = [g for g in grid if not (args.resume and already_done(g))]
    log(f"Grid: {len(grid)} cells. Pending: {len(pending)} (resume skipped {len(grid) - len(pending)}).")
    log(f"GPUs: {args.gpus}; workers/GPU: {workers_per_gpu}; total workers: {num_workers}")
    log(f"CPU partitions per worker: {partitions}")
    log(f"Output dir: {OUT_DIR}")

    if args.dry_run:
        for g in pending:
            log(f"  pending {cell_key(g['molecule'], g['split_id'], g['tau'])}")
        return

    # Round-robin GPU assignment per worker slot. Each cell gets the
    # next-free worker; we use a thread pool of size num_workers and round-robin
    # the CPU set. The round-robin is per-launch order; if a cell completes
    # quickly the slot's CPU set is reused for the next cell.
    t_start = time.time()
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {}
        for i, spec in enumerate(pending):
            slot = i % num_workers
            spec["gpu"] = args.gpus[slot // workers_per_gpu]
            spec["cpu_set"] = partitions[slot]
            spec["timeout_s"] = int(args.timeout_hours * 3600)
            fut = ex.submit(run_cell, spec)
            futures[fut] = spec
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
                log(f"   error: {res.get('error', 'see log')}")

    log(f"DONE. ok={completed} fail={failed} total_time={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
