#!/usr/bin/env python3
"""
PDEBench 1D Burgers — DML benchmark runner.

Two input modes (negative-control + canonical):

  --input-mode bare  : input = (x, t).  No IC information.
                        u(x, t) is multi-valued across IC simulations,
                        so the regression target is E_IC[u(x, t)] and
                        MSE is bounded below by Var_IC[u(x, t)].
                        TOST equivalence is the predicted outcome
                        (vanilla and DML both saturate at the same floor).

  --input-mode ic    : input = (x, t, Re(â_0..K-1), Im(â_0..K-1))
                        where â are the first K Fourier coefficients
                        of u(x, t=0) for the simulation containing the
                        sample.  Default K=16 → 32 IC features → 34-dim
                        input.  Gradient labels for the IC slots are zero
                        and a per-sample mask masks them out of the loss.

Both modes train all six DML methods. Output is one JSON per cell at:
  results/burgers/{mode}/burgers_nu{NU}_{ARCH}_{METHOD}_s{SEED}.json
  results/burgers/{mode}/_cache/burgers_{...}_nu{NU}_smoke{0/1}.npz
  logs/burgers_{mode}.log

Architecture: 4x256 / 6x512 softplus MLP. Training mechanics (loss balancing,
dml_warmup, GradNorm, ReLoBRaLo) are unchanged from the rest of the benchmark.

Usage:
  python scripts/run_burgers.py --smoke                       # all 6 methods, 1 seed, 4x256
  python scripts/run_burgers.py                               # full grid, bare mode
  python scripts/run_burgers.py --input-mode ic               # full grid, IC mode (K=16)
  python scripts/run_burgers.py --input-mode ic --n-fourier-modes 32
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
    0.01: "data/pdebench/1D_Burgers_Sols_Nu0.01.hdf5",
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

DEFAULT_N_FOURIER_MODES = 16  # IC mode only; ignored for bare


def log(msg: str, log_file: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Fourier IC encoding (--input-mode ic only)
# ---------------------------------------------------------------------------

def compute_burgers_ic_features(u_at_t0: np.ndarray, n_modes: int) -> np.ndarray:
    """1D rFFT of u(x, t=0) per simulation; return real+imag of first n_modes.

    u_at_t0: (n_sim, n_x).  Returns (n_sim, 2 * n_modes), float32.
    """
    coeffs = np.fft.rfft(u_at_t0, axis=1)[:, :n_modes]  # (n_sim, n_modes) complex
    return np.concatenate([coeffs.real, coeffs.imag], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Data preparation (cached per (mode, K, nu, smoke))
# ---------------------------------------------------------------------------

def prepare_data_cache(nu: float, smoke: bool, cache_dir: Path,
                       input_mode: str, n_fourier_modes: int,
                       log_file: Path) -> Path:
    """Read PDEBench HDF5 once, build pointwise sample arrays, save .npz."""
    import h5py

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if input_mode == "bare" else f"_K{n_fourier_modes}"
    cache_path = cache_dir / f"burgers{suffix}_nu{nu}_smoke{int(smoke)}.npz"
    if cache_path.exists():
        log(f"data cache present: {cache_path}", log_file)
        return cache_path

    src = ROOT / DATA_FILES[nu]
    if not src.exists():
        raise FileNotFoundError(f"PDEBench Burgers data not found: {src}")

    log(f"preparing data cache from {src} (mode={input_mode}, smoke={smoke})", log_file)
    with h5py.File(src, "r") as f:
        u_all = np.asarray(f["tensor"][:], dtype=np.float32)
        x_coords = np.asarray(f["x-coordinate"][:], dtype=np.float32)
        t_coords = np.asarray(f["t-coordinate"][:], dtype=np.float32)
    n_sim_full, n_t, n_x = u_all.shape
    dx = float(x_coords[1] - x_coords[0])
    dt = float(t_coords[1] - t_coords[0])

    if smoke:
        n_sim = min(50, n_sim_full)
        n_train, n_val, n_test = 30, 10, 10
        per_train, per_val, per_test = 5, 2, 2
        u_all = u_all[:n_sim]
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

    log("computing central-FD gradient labels (du/dx, du/dt)", log_file)
    dudx = np.zeros_like(u_all)
    dudx[:, :, 1:-1] = (u_all[:, :, 2:] - u_all[:, :, :-2]) / (2 * dx)
    dudx[:, :, 0]  = (u_all[:, :, 1] - u_all[:, :, 0]) / dx
    dudx[:, :, -1] = (u_all[:, :, -1] - u_all[:, :, -2]) / dx
    dudt = np.zeros_like(u_all)
    dudt[:, 1:-1, :] = (u_all[:, 2:, :] - u_all[:, :-2, :]) / (2 * dt)
    dudt[:, 0, :]  = (u_all[:, 1, :] - u_all[:, 0, :]) / dt
    dudt[:, -1, :] = (u_all[:, -1, :] - u_all[:, -2, :]) / dt

    if input_mode == "ic":
        ic_features = compute_burgers_ic_features(u_all[:, 0, :], n_fourier_modes)
        log(f"IC features: {ic_features.shape} (K={n_fourier_modes})", log_file)
    else:
        ic_features = None

    def assemble(sim_idx, per_sim, sub_seed):
        rng_s = np.random.default_rng(sub_seed)
        t_lo, t_hi = 4, n_t - 4
        x_lo, x_hi = 4, n_x - 4
        n = len(sim_idx) * per_sim
        sim_ids = np.repeat(sim_idx, per_sim)
        t_ids = rng_s.integers(t_lo, t_hi, size=n)
        x_ids = rng_s.integers(x_lo, x_hi, size=n)
        x_phys = x_coords[x_ids]
        t_phys = t_coords[t_ids]
        X_base = np.stack([x_phys, t_phys], axis=1).astype(np.float32)
        y = u_all[sim_ids, t_ids, x_ids].reshape(-1, 1).astype(np.float32)
        gx = dudx[sim_ids, t_ids, x_ids].astype(np.float32)
        gt = dudt[sim_ids, t_ids, x_ids].astype(np.float32)
        dydx_base = np.stack([gx, gt], axis=-1).reshape(n, 1, 2).astype(np.float32)

        if input_mode == "bare":
            return X_base, y, dydx_base, None

        ic_per_sample = ic_features[sim_ids]
        X = np.concatenate([X_base, ic_per_sample], axis=1).astype(np.float32)
        zero_grad = np.zeros((n, 1, 2 * n_fourier_modes), dtype=np.float32)
        dydx = np.concatenate([dydx_base, zero_grad], axis=-1).astype(np.float32)
        mask_phys = np.ones_like(dydx_base)
        mask_ic = np.zeros_like(zero_grad)
        mask = np.concatenate([mask_phys, mask_ic], axis=-1).astype(np.float32)
        return X, y, dydx, mask

    Xtr, ytr, gtr, mtr = assemble(train_sims, per_train, 1)
    Xva, yva, gva, mva = assemble(val_sims, per_val, 2)
    Xte, yte, gte, mte = assemble(test_sims, per_test, 3)
    log(f"train: {Xtr.shape}, val: {Xva.shape}, test: {Xte.shape}", log_file)
    log(f"y train mean/var: {ytr.mean():.4f} / {ytr.var():.4f}", log_file)
    log(f"y test  mean/var: {yte.mean():.4f} / {yte.var():.4f}", log_file)

    save_kwargs = dict(
        Xtr=Xtr, ytr=ytr, gtr=gtr,
        Xva=Xva, yva=yva, gva=gva,
        Xte=Xte, yte=yte, gte=gte,
        nu=np.array([nu]),
        train_sims=train_sims, val_sims=val_sims, test_sims=test_sims,
    )
    if mtr is not None:
        save_kwargs.update(mtr=mtr, mva=mva, mte=mte)
    np.savez_compressed(cache_path, **save_kwargs)
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
                 input_mode: str,
                 balancer_kwargs: dict | None = None,
                 **extra_balancer_kwargs) -> dict:
    """
    Run one Burgers PDE training cell.

    `balancer_kwargs` (and the loose `**extra_balancer_kwargs` form) accept
    balancer hyperparameters (e.g. `alpha` for GradNorm, `tau` for ReLoBRaLo).
    Default `None`/empty dict preserves byte-identical behaviour for the 100+
    existing cells in `results/burgers/`. See #197 sensitivity sweep.
    """
    from dml_benchmark.trainer import train_single_experiment, get_run_metadata
    bk = dict(balancer_kwargs or {})
    bk.update(extra_balancer_kwargs)

    t0 = time.time()
    cache_sha = _cache_sha16(cache_path)
    data = np.load(cache_path, allow_pickle=False)
    Xtr, ytr, gtr = data["Xtr"], data["ytr"], data["gtr"]
    Xva, yva, gva = data["Xva"], data["yva"], data["gva"]
    Xte, yte, gte = data["Xte"], data["yte"], data["gte"]
    has_mask = "mtr" in data.files
    if has_mask:
        mtr, mva, mte = data["mtr"], data["mva"], data["mte"]

    n_epochs = 30 if smoke else 500
    sched_pat = 10 if smoke else 20

    train_kwargs = dict(
        x_train=Xtr, y_train=ytr, dydx_train=gtr,
        x_test=Xte, y_test=yte, dydx_test=gte,
        x_val=Xva, y_val=yva, dydx_val=gva,
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
    if has_mask:
        train_kwargs.update(
            dydx_train_mask=mtr, dydx_val_mask=mva, dydx_test_mask=mte
        )
    if bk:
        train_kwargs["balancer_kwargs"] = bk
    res = train_single_experiment(**train_kwargs)

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

def cell_key(nu: float, arch_name: str, method: str, seed: int) -> str:
    return f"burgers_nu{nu}_{arch_name}_{method}_s{seed}"


def _spawn_cell(spec: dict) -> dict:
    nu, arch_name = spec["nu"], spec["arch_name"]
    n_layers, hidden = spec["n_layers"], spec["hidden"]
    method, seed = spec["method"], spec["seed"]
    gpu, cpu_set = spec["gpu"], spec["cpu_set"]
    cache_path, smoke = spec["cache_path"], spec["smoke"]
    input_mode = spec["input_mode"]
    out_dir = Path(spec["out_dir"])
    sub_log_dir = Path(spec["sub_log_dir"])
    key = cell_key(nu, arch_name, method, seed)
    out_path = out_dir / f"{key}.json"

    code = f"""
import json, os, sys
sys.path.insert(0, "{ROOT}")
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("OMP_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("MKL_NUM_THREADS", "{len(cpu_set)}")
import torch
torch.set_num_threads({len(cpu_set)})
from scripts.run_burgers import run_one_cell
result = run_one_cell(
    cache_path="{cache_path}",
    n_layers={n_layers}, hidden={hidden},
    method="{method}", seed={seed}, smoke={smoke},
    input_mode="{input_mode}",
)
result["nu"] = {nu}
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
    sub_log = sub_log_dir / f"burgers_{input_mode}_{key}.log"
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
                    help="bare: (x,t) → u, no IC info; "
                         "ic: append Fourier features of u(x,t=0).")
    ap.add_argument("--n-fourier-modes", type=int, default=DEFAULT_N_FOURIER_MODES,
                    help="number of complex modes per simulation in --input-mode ic "
                         "(2*K real-valued IC features).")
    ap.add_argument("--smoke", action="store_true",
                    help="50-sim smoke run, all 6 methods × 1 seed × 4x256; "
                         "outputs to /tmp; not committed.")
    ap.add_argument("--viscosities", type=float, nargs="+", default=[0.01])
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
        out_dir = Path(f"/tmp/burgers_{mode}_smoke")
        log_file = Path(f"/tmp/burgers_{mode}_smoke.log")
        cache_dir = out_dir / "_cache"
        sub_log_dir = out_dir / "logs"
        n_seeds = 1
        # Smoke now exercises every method (not just vanilla + dml_fixed)
        # so loss-class signature/regression bugs surface here, not mid-grid.
        methods = list(args.methods)
        archs = [a for a in ARCHS if a[0] == "4x256"]
    else:
        out_dir = ROOT / "results" / "burgers" / mode
        log_file = ROOT / "logs" / f"burgers_{mode}.log"
        cache_dir = out_dir / "_cache"
        sub_log_dir = ROOT / "logs"
        n_seeds = args.n_seeds
        methods = args.methods
        archs = [a for a in ARCHS if a[0] in args.archs]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sub_log_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = {}
    for nu in args.viscosities:
        cache_paths[nu] = prepare_data_cache(
            nu, smoke=args.smoke, cache_dir=cache_dir,
            input_mode=mode, n_fourier_modes=args.n_fourier_modes,
            log_file=log_file,
        )

    workers_per_gpu = 2 if args.light else args.workers_per_gpu
    num_workers = workers_per_gpu * len(args.gpus)
    partitions = cpu_partition(num_workers, args.cores_per_worker, args.start_core)

    grid = []
    for nu in args.viscosities:
        for arch_name, n_layers, hidden in archs:
            for method in methods:
                for seed in range(1, n_seeds + 1):
                    grid.append({
                        "nu": nu, "arch_name": arch_name,
                        "n_layers": n_layers, "hidden": hidden,
                        "method": method, "seed": seed,
                    })
    pending = [g for g in grid
               if not (args.resume and already_done(out_dir, cell_key(g["nu"], g["arch_name"], g["method"], g["seed"])))]
    log(f"Mode: {mode} (K={args.n_fourier_modes})", log_file)
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
            spec["cache_path"] = str(cache_paths[spec["nu"]])
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
