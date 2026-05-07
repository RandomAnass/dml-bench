#!/usr/bin/env python3
"""
rMD17 salicylic-acid PaiNN runner — fills the missing molecule (#205).

Grid:
  molecule = salicylic
  method   = vanilla, dml_fixed, dml_fixed_half, dml_gradnorm, dml_warmup, native_EF
  splits   = canonical Figshare 1..5 (seed = split_id)
  ⇒ 30 PaiNN runs, ~30-60 min each → ~5-6 h with 4-5 parallel workers.

Output filename matches the other 9 molecules in results/molecular_painn/
(`painn_md17_salicylic_split{1..5}_{method}.json`) so existing aggregators
(F11 force-MAE bars, F12 CD diagram, tab:md17) pick it up automatically.

Concurrency mirrors `scripts/run_rmd17_tau_sweep.py`: each cell runs as
its own subprocess to keep CUDA contexts isolated, with explicit CPU
pinning to avoid thrashing.

Usage:
  python scripts/run_painn_salicylic.py --gpus 0 --workers-per-gpu 4 \\
      --cores-per-worker 4 --start-core 0 --resume

  # Optionally launch a second worker on GPU 1 (cores 20-22):
  python scripts/run_painn_salicylic.py --gpus 1 --workers-per-gpu 1 \\
      --cores-per-worker 3 --start-core 20 --resume
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

OUT_DIR = ROOT / "results" / "molecular_painn"
LOG_FILE = ROOT / "logs" / "painn_salicylic.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MOLECULE = "salicylic"
METHODS = ["vanilla", "dml_fixed", "dml_fixed_half", "dml_gradnorm", "dml_warmup", "native_EF"]
SPLIT_IDS = [1, 2, 3, 4, 5]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def cell_key(split_id: int, method: str) -> str:
    return f"painn_md17_{MOLECULE}_split{split_id}_{method}"


def out_path(key: str) -> Path:
    return OUT_DIR / f"{key}.json"


def run_cell(spec: dict) -> dict:
    """Run one cell as a fresh subprocess to keep CUDA contexts isolated."""
    split_id = spec["split_id"]
    method = spec["method"]
    gpu = spec["gpu"]
    cpu_set = spec["cpu_set"]
    timeout_s = spec.get("timeout_s", 14 * 3600)
    key = cell_key(split_id, method)
    out = out_path(key)

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
    molecule="{MOLECULE}", method="{method}", seed={split_id},
    hparams=hparams, smoke=False, gpu={gpu}, split_id={split_id},
)
result["key"] = "{key}"
with open("{out}", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"OK key={{result['key']}} time={{result['time_s']:.1f}}s "
      f"E_MAE={{result.get('test_energy_mae_mev', float('nan')):.1f}} "
      f"F_MAE={{result.get('test_force_mae_mev', float('nan')):.1f}}")
"""

    env = os.environ.copy()
    cpu_list = ",".join(str(c) for c in cpu_set)
    py = os.environ.get("PAINN_PYTHON",
                        "python")
    cmd = ["taskset", "-c", cpu_list, py, "-c", code]

    t0 = time.time()
    log_path = LOG_FILE.parent / f"painn_salicylic_{key}.log"
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
    return [{"split_id": sid, "method": m}
            for sid in SPLIT_IDS for m in METHODS]


def already_done(spec) -> bool:
    p = out_path(cell_key(spec["split_id"], spec["method"]))
    if not p.exists():
        return False
    try:
        json.load(open(p))
        return True
    except Exception:
        return False


def cpu_partition(num_workers: int, cores_per_worker: int = 4,
                   start_core: int = 0) -> list:
    total = os.cpu_count() or 16
    end = min(start_core + num_workers * cores_per_worker, total)
    pool = list(range(start_core, end))
    return [pool[i*cores_per_worker:(i+1)*cores_per_worker] for i in range(num_workers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--cores-per-worker", type=int, default=4)
    ap.add_argument("--start-core", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout-hours", type=float, default=14.0)
    args = ap.parse_args()

    workers_per_gpu = args.workers_per_gpu
    num_workers = workers_per_gpu * len(args.gpus)
    partitions = cpu_partition(num_workers, args.cores_per_worker, args.start_core)

    grid = build_grid()
    pending = [g for g in grid if not (args.resume and already_done(g))]
    log(f"Grid: {len(grid)} cells. Pending: {len(pending)} (resume skipped {len(grid) - len(pending)}).")
    log(f"GPUs: {args.gpus}; workers/GPU: {workers_per_gpu}; total workers: {num_workers}")
    log(f"CPU partitions per worker: {partitions}")
    log(f"Output dir: {OUT_DIR}")
    log(f"Per-cell timeout: {args.timeout_hours} h")

    if args.dry_run:
        for g in pending:
            log(f"  pending {cell_key(g['split_id'], g['method'])}")
        return

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
                log(f"   error: {res.get('error', 'rc!=0')}")

    log(f"DONE. ok={completed} fail={failed} total_time={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
