#!/usr/bin/env python3
"""
PDEBench 2D Darcy flow — DML benchmark runner.

Two input modes (negative-control + canonical):

  --input-mode bare  : input = (x, y, a(x, y)).  Pointwise diffusion
                        coefficient only.  The simulation's full a-field is
                        not visible to the network; sims with the same a(x, y)
                        at this point but different a elsewhere give different
                        u values.  TOST equivalence is the predicted outcome.

  --input-mode ic    : input = (x, y, a(x, y), Re/Im of low-freq 2D-FFT
                        coefficients of the full a-field for the simulation).
                        Default K=4 (4×4 modes → 32 real-valued IC features),
                        for a 35-dim input.  Gradient labels for the IC slots
                        are zero and a per-sample mask masks them out.

In both modes the gradient label has 3 physical components
(∂u/∂x, ∂u/∂y, 0) and the third (∂u/∂a_pt) is masked out — there is no
ground-truth derivative w.r.t. the parameter-field channel from the dataset.

Output:
  results/darcy/{mode}/darcy_beta{BETA}_{ARCH}_{METHOD}_s{SEED}.json
  results/darcy/{mode}/_cache/darcy_{...}_beta{BETA}_smoke{0/1}.npz
  logs/darcy_{mode}.log

Architecture: 4x256 / 6x512 softplus MLP. Training mechanics (loss balancing,
dml_warmup, GradNorm, ReLoBRaLo) are unchanged from the rest of the benchmark.

Usage:
  python scripts/run_darcy.py --smoke                       # all 6 methods, 1 seed, 4x256
  python scripts/run_darcy.py                               # full grid, bare mode
  python scripts/run_darcy.py --input-mode ic               # full grid, IC mode (K=4)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

DATA_FILES = {
    1.0: "data/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5",
}
ARCHS = [("4x256", 4, 256), ("6x512", 6, 512)]
METHODS = [
    "vanilla",
    "dml_fixed",
    "dml_fixed_half",
    "dml_gradnorm",
    "dml_relobralo",
    "dml_warmup",
]
DEFAULT_N_SEEDS = 20

N_TRAIN_SIMS = 1800
N_VAL_SIMS = 100
N_TEST_SIMS = 100
SAMPLES_PER_TRAIN_SIM = 300
SAMPLES_PER_VAL_SIM = 100
SAMPLES_PER_TEST_SIM = 100

DEFAULT_K_PER_DIM = 4   # IC mode only; 4×4 = 16 complex modes → 32 IC features


def log(msg: str, log_file: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Fourier IC encoding (--input-mode ic only)
# ---------------------------------------------------------------------------

def compute_darcy_ic_features(a_all: np.ndarray, k_per_dim: int) -> np.ndarray:
    """2D FFT of a(x, y) per simulation; return real+imag of the k×k low-freq corner.

    a_all: (n_sim, n_x, n_y).  Returns (n_sim, 2 * k_per_dim**2), float32.
    """
    coeffs = np.fft.fft2(a_all)[:, :k_per_dim, :k_per_dim]
    flat_real = coeffs.real.reshape(coeffs.shape[0], -1)
    flat_imag = coeffs.imag.reshape(coeffs.shape[0], -1)
    return np.concatenate([flat_real, flat_imag], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Data preparation (cached per (mode, K, beta, smoke))
# ---------------------------------------------------------------------------

def prepare_data_cache(beta: float, smoke: bool, cache_dir: Path,
                       input_mode: str, k_per_dim: int,
                       log_file: Path) -> Path:
    """Read PDEBench HDF5 once, build pointwise sample arrays, save .npz."""
    import h5py

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if input_mode == "bare" else f"_K{k_per_dim}"
    cache_path = cache_dir / f"darcy{suffix}_beta{beta}_smoke{int(smoke)}.npz"
    if cache_path.exists():
        log(f"data cache present: {cache_path}", log_file)
        return cache_path

    src = ROOT / DATA_FILES[beta]
    if not src.exists():
        raise FileNotFoundError(f"PDEBench Darcy data not found: {src}")

    log(f"preparing data cache from {src} (mode={input_mode}, smoke={smoke})", log_file)
    with h5py.File(src, "r") as f:
        u_all = np.asarray(f["tensor"][:, 0, :, :], dtype=np.float32)  # (n_sim, n_x, n_y)
        a_all = np.asarray(f["nu"][:], dtype=np.float32)               # diffusion coefficient field
        x_coords = np.asarray(f["x-coordinate"][:], dtype=np.float32)
        y_coords = np.asarray(f["y-coordinate"][:], dtype=np.float32)
    n_sim_full, n_x, n_y = u_all.shape
    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])

    if smoke:
        n_sim = min(50, n_sim_full)
        n_train, n_val, n_test = 30, 10, 10
        per_train, per_val, per_test = 5, 2, 2
        u_all = u_all[:n_sim]
        a_all = a_all[:n_sim]
    else:
        n_sim = n_sim_full
        n_train, n_val, n_test = N_TRAIN_SIMS, N_VAL_SIMS, N_TEST_SIMS
        per_train = SAMPLES_PER_TRAIN_SIM
        per_val = SAMPLES_PER_VAL_SIM
        per_test = SAMPLES_PER_TEST_SIM

    if n_train + n_val + n_test > n_sim:
        raise ValueError(f"requested {n_train + n_val + n_test} sims > available {n_sim}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_sim)
    train_sims = np.sort(perm[:n_train])
    val_sims = np.sort(perm[n_train:n_train + n_val])
    test_sims = np.sort(perm[n_train + n_val:n_train + n_val + n_test])

    log("computing central-FD gradient labels (du/dx, du/dy)", log_file)
    dudx = np.zeros_like(u_all)
    dudx[:, 1:-1, :] = (u_all[:, 2:, :] - u_all[:, :-2, :]) / (2 * dx)
    dudx[:, 0, :]  = (u_all[:, 1, :] - u_all[:, 0, :]) / dx
    dudx[:, -1, :] = (u_all[:, -1, :] - u_all[:, -2, :]) / dx
    dudy = np.zeros_like(u_all)
    dudy[:, :, 1:-1] = (u_all[:, :, 2:] - u_all[:, :, :-2]) / (2 * dy)
    dudy[:, :, 0]  = (u_all[:, :, 1] - u_all[:, :, 0]) / dy
    dudy[:, :, -1] = (u_all[:, :, -1] - u_all[:, :, -2]) / dy

    if input_mode == "ic":
        ic_features = compute_darcy_ic_features(a_all, k_per_dim)
        log(f"IC features: {ic_features.shape} (K_per_dim={k_per_dim})", log_file)
    else:
        ic_features = None

    def assemble(sim_idx, per_sim, sub_seed):
        rng_s = np.random.default_rng(sub_seed)
        x_lo, x_hi = 4, n_x - 4
        y_lo, y_hi = 4, n_y - 4
        n = len(sim_idx) * per_sim
        sim_ids = np.repeat(sim_idx, per_sim)
        x_ids = rng_s.integers(x_lo, x_hi, size=n)
        y_ids = rng_s.integers(y_lo, y_hi, size=n)
        x_phys = x_coords[x_ids]
        y_phys = y_coords[y_ids]
        a_phys = a_all[sim_ids, x_ids, y_ids].astype(np.float32)
        X_base = np.stack([x_phys, y_phys, a_phys], axis=1).astype(np.float32)
        y = u_all[sim_ids, x_ids, y_ids].reshape(-1, 1).astype(np.float32)
        gx = dudx[sim_ids, x_ids, y_ids].astype(np.float32)
        gy = dudy[sim_ids, x_ids, y_ids].astype(np.float32)
        ga = np.zeros_like(gx)  # ground-truth ∂u/∂a unavailable
        dydx_base = np.stack([gx, gy, ga], axis=-1).reshape(n, 1, 3).astype(np.float32)
        # Mask: physics channels (∂u/∂x, ∂u/∂y) are supervised; the a_pt channel
        # is masked because we have no ground-truth ∂u/∂a from the dataset.
        mask_base = np.ones_like(dydx_base)
        mask_base[:, :, 2] = 0.0

        if input_mode == "bare":
            return X_base, y, dydx_base, mask_base

        ic_per_sample = ic_features[sim_ids]
        X = np.concatenate([X_base, ic_per_sample], axis=1).astype(np.float32)
        n_ic = 2 * k_per_dim * k_per_dim
        zero_grad = np.zeros((n, 1, n_ic), dtype=np.float32)
        dydx = np.concatenate([dydx_base, zero_grad], axis=-1).astype(np.float32)
        mask_ic = np.zeros((n, 1, n_ic), dtype=np.float32)
        mask = np.concatenate([mask_base, mask_ic], axis=-1).astype(np.float32)
        return X, y, dydx, mask

    Xtr, ytr, gtr, mtr = assemble(train_sims, per_train, 1)
    Xva, yva, gva, mva = assemble(val_sims, per_val, 2)
    Xte, yte, gte, mte = assemble(test_sims, per_test, 3)
    log(f"train: {Xtr.shape}, val: {Xva.shape}, test: {Xte.shape}", log_file)
    log(f"y train mean/var: {ytr.mean():.4f} / {ytr.var():.4f}", log_file)
    log(f"y test  mean/var: {yte.mean():.4f} / {yte.var():.4f}", log_file)

    np.savez_compressed(
        cache_path,
        Xtr=Xtr, ytr=ytr, gtr=gtr, mtr=mtr,
        Xva=Xva, yva=yva, gva=gva, mva=mva,
        Xte=Xte, yte=yte, gte=gte, mte=mte,
        beta=np.array([beta]),
        train_sims=train_sims, val_sims=val_sims, test_sims=test_sims,
    )
    log(f"saved data cache: {cache_path}", log_file)
    return cache_path


def _cache_sha16(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Single-cell entry — calls dml_benchmark.trainer.train_single_experiment directly.
# ---------------------------------------------------------------------------

def run_one_cell(cache_path: str, n_layers: int, hidden: int,
                 method: str, seed: int, smoke: bool,
                 input_mode: str) -> dict:
    from dml_benchmark.trainer import train_single_experiment, get_run_metadata

    t0 = time.time()
    cache_sha = _cache_sha16(cache_path)
    data = np.load(cache_path, allow_pickle=False)
    Xtr, ytr, gtr, mtr = data["Xtr"], data["ytr"], data["gtr"], data["mtr"]
    Xva, yva, gva, mva = data["Xva"], data["yva"], data["gva"], data["mva"]
    Xte, yte, gte, mte = data["Xte"], data["yte"], data["gte"], data["mte"]

    n_epochs = 30 if smoke else 500
    sched_pat = 10 if smoke else 20

    res = train_single_experiment(
        x_train=Xtr, y_train=ytr, dydx_train=gtr, dydx_train_mask=mtr,
        x_test=Xte, y_test=yte, dydx_test=gte, dydx_test_mask=mte,
        x_val=Xva, y_val=yva, dydx_val=gva, dydx_val_mask=mva,
        method=method,
        lambda_=1.0,
        n_epochs=n_epochs,
        batch_size=256,
        n_layers=n_layers,
        hidden_size=hidden,
        lr=5e-3,
        activation="softplus",
        seed=seed,
        pbar=False,
        max_grad_norm=1.0,
        scheduler_patience=sched_pat,
        scheduler_factor=0.5,
    )

    return {
        "test_value_mse": float(res.test_value_mse),
        "test_grad_mse":  float(res.test_grad_mse),
        "final_train_loss": float(res.final_train_loss),
        "final_val_loss":   float(res.final_val_loss),
        "best_epoch": int(getattr(res, "best_epoch", 0)),
        "early_stopped": bool(getattr(res, "early_stopped", False)),
        "n_epochs_logged": int(len(res.training_logs)) if getattr(res, "training_logs", None) else int(n_epochs),
        "time_s": time.time() - t0,
        "cache_sha16": cache_sha,
        "hparams": {
            "n_layers": n_layers, "hidden_size": hidden,
            "input_dim": int(Xtr.shape[1]),
            "input_mode": input_mode,
            "lr": 5e-3, "batch_size": 256, "n_epochs": n_epochs,
            "scheduler_patience": sched_pat, "scheduler_factor": 0.5,
            "max_grad_norm": 1.0, "activation": "softplus",
            "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
            "n_test": int(Xte.shape[0]),
        },
        "run_metadata": get_run_metadata(),
    }


# ---------------------------------------------------------------------------
# Dispatcher (subprocess-per-cell, taskset CPU pinning)
# ---------------------------------------------------------------------------

def cell_key(beta: float, arch_name: str, method: str, seed: int) -> str:
    return f"darcy_beta{beta}_{arch_name}_{method}_s{seed}"


def _spawn_cell(spec: dict) -> dict:
    beta, arch_name = spec["beta"], spec["arch_name"]
    n_layers, hidden = spec["n_layers"], spec["hidden"]
    method, seed = spec["method"], spec["seed"]
    gpu, cpu_set = spec["gpu"], spec["cpu_set"]
    cache_path, smoke = spec["cache_path"], spec["smoke"]
    input_mode = spec["input_mode"]
    out_dir = Path(spec["out_dir"])
    sub_log_dir = Path(spec["sub_log_dir"])
    key = cell_key(beta, arch_name, method, seed)
    out_path = out_dir / f"{key}.json"

    code = f"""
import json, os, sys
sys.path.insert(0, "{ROOT}")
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("OMP_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("MKL_NUM_THREADS", "{len(cpu_set)}")
import torch
torch.set_num_threads({len(cpu_set)})
from scripts.run_darcy import run_one_cell
result = run_one_cell(
    cache_path="{cache_path}",
    n_layers={n_layers}, hidden={hidden},
    method="{method}", seed={seed}, smoke={smoke},
    input_mode="{input_mode}",
)
result["beta"] = {beta}
result["arch"] = "{arch_name}"
result["method"] = "{method}"
result["seed"] = {seed}
result["key"] = "{key}"
result["input_mode"] = "{input_mode}"
with open("{out_path}", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"OK key={{result['key']}} val={{result['test_value_mse']:.4e}} "
      f"grad={{result['test_grad_mse']:.4e}} time={{result['time_s']:.0f}}s")
"""
    cpu_list = ",".join(str(c) for c in cpu_set)
    py = sys.executable
    cmd = ["taskset", "-c", cpu_list, py, "-c", code]
    sub_log = sub_log_dir / f"darcy_{input_mode}_{key}.log"
    sub_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with open(sub_log, "w") as logf:
            proc = subprocess.run(cmd, env=os.environ.copy(),
                                  stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=3 * 3600)
        ok = (proc.returncode == 0) and out_path.exists()
        return {"key": key, "ok": ok, "rc": proc.returncode,
                "time_s": time.time() - t0, "log": str(sub_log)}
    except Exception as e:
        return {"key": key, "ok": False, "rc": -1,
                "time_s": time.time() - t0,
                "error": str(e), "tb": traceback.format_exc()}


def cpu_partition(num_workers: int, cores_per_worker: int, start_core: int) -> list:
    total = os.cpu_count() or 16
    end = min(start_core + num_workers * cores_per_worker, total)
    pool = list(range(start_core, end))
    return [pool[i*cores_per_worker:(i+1)*cores_per_worker] for i in range(num_workers)]


def already_done(out_dir: Path, key: str) -> bool:
    p = out_dir / f"{key}.json"
    if not p.exists():
        return False
    try:
        json.load(open(p))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-mode", choices=["bare", "ic"], default="bare",
                    help="bare: (x,y,a_pt) → u; "
                         "ic: append low-freq 2D FFT of full a-field.")
    ap.add_argument("--n-fourier-modes", type=int, default=DEFAULT_K_PER_DIM,
                    help="K_per_dim for 2D FFT in --input-mode ic "
                         "(K×K complex modes → 2*K**2 real-valued IC features).")
    ap.add_argument("--smoke", action="store_true",
                    help="50-sim smoke run, all 6 methods × 1 seed × 4x256; "
                         "outputs to /tmp; not committed.")
    ap.add_argument("--betas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--archs", nargs="+", default=[a[0] for a in ARCHS])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--light", action="store_true",
                    help="2 workers/GPU instead of 4.")
    ap.add_argument("--cores-per-worker", type=int, default=4)
    ap.add_argument("--start-core", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mode = args.input_mode

    if args.smoke:
        out_dir = Path(f"/tmp/darcy_{mode}_smoke")
        log_file = Path(f"/tmp/darcy_{mode}_smoke.log")
        cache_dir = out_dir / "_cache"
        sub_log_dir = out_dir / "logs"
        n_seeds = 1
        # Smoke now exercises every method (not just vanilla + dml_fixed)
        # so loss-class signature/regression bugs surface here, not mid-grid.
        methods = list(args.methods)
        archs = [a for a in ARCHS if a[0] == "4x256"]
    else:
        out_dir = ROOT / "results" / "darcy" / mode
        log_file = ROOT / "logs" / f"darcy_{mode}.log"
        cache_dir = out_dir / "_cache"
        sub_log_dir = ROOT / "logs"
        n_seeds = args.n_seeds
        methods = args.methods
        archs = [a for a in ARCHS if a[0] in args.archs]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sub_log_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = {}
    for beta in args.betas:
        cache_paths[beta] = prepare_data_cache(
            beta, smoke=args.smoke, cache_dir=cache_dir,
            input_mode=mode, k_per_dim=args.n_fourier_modes,
            log_file=log_file,
        )

    workers_per_gpu = 2 if args.light else args.workers_per_gpu
    num_workers = workers_per_gpu * len(args.gpus)
    partitions = cpu_partition(num_workers, args.cores_per_worker, args.start_core)

    grid = []
    for beta in args.betas:
        for arch_name, n_layers, hidden in archs:
            for method in methods:
                for seed in range(1, n_seeds + 1):
                    grid.append({
                        "beta": beta, "arch_name": arch_name,
                        "n_layers": n_layers, "hidden": hidden,
                        "method": method, "seed": seed,
                    })
    pending = [g for g in grid
               if not (args.resume and already_done(out_dir, cell_key(g["beta"], g["arch_name"], g["method"], g["seed"])))]
    log(f"Mode: {mode} (K_per_dim={args.n_fourier_modes})", log_file)
    log(f"Grid: {len(grid)} cells. Pending: {len(pending)} (resume skipped {len(grid) - len(pending)}).", log_file)
    log(f"GPUs: {args.gpus}; workers/GPU: {workers_per_gpu}; total workers: {num_workers}", log_file)
    log(f"CPU partitions: {partitions}", log_file)
    log(f"Output dir: {out_dir}", log_file)

    if args.dry_run:
        log("DRY RUN — exiting.", log_file)
        return

    t_start = time.time()
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {}
        for i, spec in enumerate(pending):
            slot = i % num_workers
            spec["gpu"] = args.gpus[slot // workers_per_gpu]
            spec["cpu_set"] = partitions[slot]
            spec["cache_path"] = str(cache_paths[spec["beta"]])
            spec["smoke"] = args.smoke
            spec["input_mode"] = mode
            spec["out_dir"] = str(out_dir)
            spec["sub_log_dir"] = str(sub_log_dir)
            fut = ex.submit(_spawn_cell, spec)
            futures[fut] = spec
        for fut in as_completed(futures):
            res = fut.result()
            done = completed + failed + 1
            if res["ok"]:
                completed += 1; status = "OK"
            else:
                failed += 1; status = "FAIL"
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else float("inf")
            log(f"[{done}/{len(pending)}] {status} {res['key']} "
                f"({res['time_s']:.0f}s, rc={res.get('rc')})  eta={eta/60:.1f}min",
                log_file)
            if not res["ok"]:
                log(f"   error: {res.get('error', 'see log')}", log_file)

    log(f"DONE. ok={completed} fail={failed} total_time={(time.time()-t_start)/60:.1f}min",
        log_file)


if __name__ == "__main__":
    main()
