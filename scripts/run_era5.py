#!/usr/bin/env python3
"""
ERA5 Z500 sub-pillar — DML benchmark runner.

  --regime bare   : input = (lat_rad_norm, sin_lon, cos_lon, sin_doy, cos_doy)
  --regime state  : input = bare ⊕ EOF context (K=16) of the same day's anomaly

Output: results/era5/{regime}/era5_{ARCH}_{METHOD}_s{SEED}.json
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

ARCHS = [("4x256", 4, 256), ("6x512", 6, 512)]
METHODS = ["vanilla", "dml_fixed", "dml_fixed_half"]
DEFAULT_N_SEEDS = 5


def log(msg: str, log_file: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def _cache_sha16(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_one_cell(
    cache_path: str, n_layers: int, hidden: int,
    method: str, seed: int, regime: str,
    n_points_per_epoch: int = 100_000,
    n_epochs: int = 200,
    pilot: bool = False,
) -> dict:
    """Train one ERA5 cell. Returns the JSON-schema dict."""
    import torch
    from torch.utils.data import DataLoader
    from dml_benchmark.model import (
        DmlFeedForward, DmlLoss, VanillaLoss,
    )
    from dml_benchmark.trainer import (
        DmlTrainer, set_deterministic, get_run_metadata,
    )
    from dml_benchmark.era5_dataset import (
        ERA5Dataset, make_era5_directional_projection, G_GRAVITY,
    )

    set_deterministic(seed)
    t0 = time.time()
    cache_sha = _cache_sha16(cache_path)

    train_ds = ERA5Dataset(cache_path, regime=regime, split="train",
                           n_points_per_epoch=n_points_per_epoch, seed=seed)
    val_ds = ERA5Dataset(cache_path, regime=regime, split="val",
                         n_points_per_epoch=n_points_per_epoch // 5,
                         seed=seed + 1)
    test_ds = ERA5Dataset(cache_path, regime=regime, split="test",
                          n_points_per_epoch=n_points_per_epoch // 5,
                          seed=seed + 2)

    sample0 = train_ds[0]
    input_dim = sample0["x"].shape[0]

    model = DmlFeedForward(input_dim=input_dim, output_dim=1,
                           n_layers=n_layers, hidden_size=hidden,
                           activation="softplus")

    if method == "vanilla":
        loss_fn = VanillaLoss()
        use_dml = False
        proj_fn = None
    elif method in {"dml_fixed", "dml_fixed_half"}:
        scheme = "hs" if method == "dml_fixed" else "half"
        proj_fn = make_era5_directional_projection()
        loss_fn = DmlLoss(
            lambda_=1.0,
            input_dim=2,
            lambda_j=np.ones(2, dtype=np.float32),
            weight_scheme=scheme,
            gradient_projection_fn=proj_fn,
        )
        use_dml = True
    else:
        raise ValueError(f"Unsupported method for ERA5 pilot: {method}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6,
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0, drop_last=False)

    class Era5Trainer(DmlTrainer):
        def train_epoch(self, dataloader, epoch, pbar=False):
            for ds_ in (train_ds, val_ds):
                ds_.reseed(seed * 100_000 + epoch)
            self.model.train()
            tot = tot_v = tot_d = 0.0
            n_batches = 0
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                dydx = batch["dydx"].to(self.device)
                mask = batch["dydx_mask"].to(self.device)
                self.optimizer.zero_grad()
                if self.use_dml:
                    y_pred, dydx_pred = self.model.forward_with_greek(x)
                    lc = self.loss_fn(y_pred, y, dydx_pred, dydx,
                                      self.model, dydx_mask=mask, x_query=x)
                else:
                    y_pred = self.model(x)
                    lc = self.loss_fn(y_pred, y)
                lc.total.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm,
                    )
                self.optimizer.step()
                tot += lc.total.item()
                tot_v += lc.value_loss.item()
                tot_d += lc.deriv_loss.item()
                n_batches += 1
            return {"loss": tot / n_batches,
                    "value_loss": tot_v / n_batches,
                    "deriv_loss": tot_d / n_batches}

        def validate(self, dataloader):
            self.model.eval()
            tot = tot_v = tot_d = 0.0
            n_batches = 0
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                dydx = batch["dydx"].to(self.device)
                mask = batch["dydx_mask"].to(self.device)
                if self.use_dml:
                    with torch.enable_grad():
                        y_pred, dydx_pred = self.model.forward_with_greek(x)
                        lc = self.loss_fn(y_pred, y, dydx_pred, dydx,
                                          self.model, dydx_mask=mask, x_query=x)
                else:
                    with torch.no_grad():
                        y_pred = self.model(x)
                        lc = self.loss_fn(y_pred, y)
                tot += lc.total.item()
                tot_v += lc.value_loss.item()
                tot_d += lc.deriv_loss.item()
                n_batches += 1
            self.model.train()
            return {"loss": tot / n_batches,
                    "value_loss": tot_v / n_batches,
                    "deriv_loss": tot_d / n_batches}

        def evaluate(self, dataloader, unscale=False):
            """Masked-aware evaluate: grad MSE on the directional slots only."""
            self.model.eval()
            v_sum = g_sum = 0.0
            v_n = g_n = 0
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                dydx = batch["dydx"].to(self.device)
                mask = batch["dydx_mask"].to(self.device)
                with torch.enable_grad():
                    y_pred, dydx_pred = self.model.forward_with_greek(x)
                # Project to directional slots (lat-tangent + lon-tangent)
                if proj_fn is not None:
                    proj_pred, proj_target, proj_mask = proj_fn(
                        dydx_pred, dydx, x, mask
                    )
                else:
                    proj_pred = dydx_pred[:, :, :2]
                    proj_target = dydx[:, :, :2]
                    proj_mask = mask[:, :, :2]
                v_err = ((y_pred - y) ** 2).detach()
                v_sum += float(v_err.sum().item())
                v_n += int(v_err.numel())
                g_err = ((proj_pred - proj_target) ** 2 * proj_mask).detach()
                g_sum += float(g_err.sum().item())
                g_n += int(proj_mask.sum().item())
            return {
                "value_mse": v_sum / max(v_n, 1),
                "grad_mse": g_sum / max(g_n, 1),
            }

    trainer = Era5Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, normalizer=None,
        max_grad_norm=1.0, use_dml=use_dml,
    )

    n_ep = 5 if pilot else n_epochs
    res = trainer.train(train_loader, val_loader, n_epochs=n_ep,
                        pbar=False, early_stopping_patience=50)
    eval_metrics = trainer.evaluate(test_loader, unscale=False)

    Omega = 7.2921e-5
    R_EARTH_M = 6_371_000.0
    g_wind_mae = float("nan")
    try:
        # Physical geostrophic wind from un-normalised gradients.
        # The model autograd df_norm/d(x_norm) needs chain-rule conversion:
        #   dZ/d(lat_rad) = (s_Z / s_lat_rad) * df/d(lat_rad_norm)
        #   dZ/d(lon_rad) = s_Z * df/d(lambda)         (lon was sin/cos, no lat_norm scale)
        # Then dZ/dy = dZ/d(lat_rad) / R, dZ/dx_east = dZ/d(lon_rad) / (R cos lat)
        # u_g = -(g/f) dZ/dy ;  v_g = (g/f) dZ/dx_east
        s_Z = train_ds.s_Z
        s_lat_rad = train_ds.s_lat_rad
        model.eval()
        with torch.enable_grad():
            batch = next(iter(test_loader))
            x = batch["x"].to(trainer.device)
            y_pred, dydx_pred = model.forward_with_greek(x)
            sin_lon = x[:, 1].unsqueeze(-1)
            cos_lon = x[:, 2].unsqueeze(-1)
            df_dlat_norm = dydx_pred[:, :, 0]
            df_dlambda = dydx_pred[:, :, 1] * cos_lon - dydx_pred[:, :, 2] * sin_lon
            dZ_dlatrad = (s_Z / s_lat_rad) * df_dlat_norm
            dZ_dlonrad = s_Z * df_dlambda
            f_mid = 2.0 * Omega * np.sin(np.radians(45.0))
            cos_lat_mid = np.cos(np.radians(45.0))
            u_mag = (G_GRAVITY / f_mid) * (dZ_dlatrad.abs().mean().item() / R_EARTH_M)
            v_mag = (G_GRAVITY / f_mid) * (dZ_dlonrad.abs().mean().item() / (R_EARTH_M * cos_lat_mid))
            g_wind_mae = 0.5 * (u_mag + v_mag)
    except Exception:
        g_wind_mae = float("nan")

    return {
        "method": method,
        "regime": regime,
        "seed": seed,
        "dataset": "era5_z500",
        "resolution_deg": 1.0,
        "n_points_per_epoch": n_points_per_epoch,
        "MSEvalue": float(eval_metrics["value_mse"]),
        "MSEgradient": float(eval_metrics["grad_mse"]),
        "geostrophic_wind_MAE_ms": float(g_wind_mae),
        "epoch_stopped": int(getattr(res, "best_epoch", n_ep)),
        "training_time_s": time.time() - t0,
        "cache_sha16": cache_sha,
        "hparams": {
            "n_layers": n_layers, "hidden_size": hidden,
            "input_dim": int(input_dim),
            "regime": regime, "n_epochs": n_ep,
            "lr": 5e-3, "batch_size": 256,
            "scheduler_patience": 20, "scheduler_factor": 0.5,
            "max_grad_norm": 1.0, "activation": "softplus",
            "n_points_per_epoch": int(n_points_per_epoch),
        },
        "run_metadata": get_run_metadata(),
    }


def cell_key(arch_name: str, method: str, seed: int) -> str:
    return f"era5_{arch_name}_{method}_s{seed}"


def already_done(out_dir: Path, key: str) -> bool:
    p = out_dir / f"{key}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except Exception:
        return False


def cpu_partition(num_workers: int, cores_per_worker: int, start_core: int) -> list:
    total = os.cpu_count() or 16
    end = min(start_core + num_workers * cores_per_worker, total)
    pool = list(range(start_core, end))
    return [pool[i * cores_per_worker:(i + 1) * cores_per_worker] for i in range(num_workers)]


def _spawn_cell(spec: dict) -> dict:
    arch_name = spec["arch_name"]
    n_layers, hidden = spec["n_layers"], spec["hidden"]
    method, seed = spec["method"], spec["seed"]
    gpu, cpu_set = spec["gpu"], spec["cpu_set"]
    cache_path, regime = spec["cache_path"], spec["regime"]
    n_points = spec["n_points_per_epoch"]
    n_epochs = spec["n_epochs"]
    pilot = spec["pilot"]
    out_dir = Path(spec["out_dir"])
    sub_log_dir = Path(spec["sub_log_dir"])
    key = cell_key(arch_name, method, seed)
    out_path = out_dir / f"{key}.json"
    tmp_path = out_dir / f"{key}.json.tmp"

    code = f"""
import json, os, sys
sys.path.insert(0, "{ROOT}")
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("OMP_NUM_THREADS", "{len(cpu_set)}")
os.environ.setdefault("MKL_NUM_THREADS", "{len(cpu_set)}")
import torch
torch.set_num_threads({len(cpu_set)})
from scripts.run_era5 import run_one_cell
result = run_one_cell(
    cache_path="{cache_path}",
    n_layers={n_layers}, hidden={hidden},
    method="{method}", seed={seed}, regime="{regime}",
    n_points_per_epoch={n_points}, n_epochs={n_epochs}, pilot={pilot},
)
result["arch"] = "{arch_name}"
result["key"] = "{key}"
with open("{tmp_path}", "w") as f:
    json.dump(result, f, indent=2, default=str)
os.replace("{tmp_path}", "{out_path}")
print(f"OK key={{result['key']}} val={{result['MSEvalue']:.4e}} "
      f"grad={{result['MSEgradient']:.4e}}")
"""
    cpu_list = ",".join(str(c) for c in cpu_set)
    py = sys.executable
    cmd = ["taskset", "-c", cpu_list, py, "-c", code]
    sub_log = sub_log_dir / f"era5_{regime}_{key}.log"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", default="data/era5/pilot/era5_pilot_cache.npz")
    ap.add_argument("--regime", choices=["bare", "state"], default="bare")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--archs", nargs="+", default=[a[0] for a in ARCHS])
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--cores-per-worker", type=int, default=4)
    ap.add_argument("--start-core", type=int, default=0)
    ap.add_argument("--n-points-per-epoch", type=int, default=100_000)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--pilot", action="store_true",
                    help="5-epoch smoke run; output to /tmp/era5_smoke.")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.pilot:
        out_dir = Path(f"/tmp/era5_{args.regime}_smoke")
        log_file = Path(f"/tmp/era5_{args.regime}_smoke.log")
        sub_log_dir = out_dir / "logs"
    else:
        out_dir = ROOT / "results" / "era5" / args.regime
        log_file = ROOT / "logs" / f"era5_{args.regime}.log"
        sub_log_dir = ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_log_dir.mkdir(parents=True, exist_ok=True)

    archs = [a for a in ARCHS if a[0] in args.archs]
    grid = []
    for arch_name, n_layers, hidden in archs:
        for method in args.methods:
            for seed in range(1, args.n_seeds + 1):
                grid.append({
                    "arch_name": arch_name, "n_layers": n_layers, "hidden": hidden,
                    "method": method, "seed": seed,
                })
    pending = [
        g for g in grid
        if not (args.resume and already_done(out_dir, cell_key(g["arch_name"], g["method"], g["seed"])))
    ]
    log(f"Regime: {args.regime}", log_file)
    log(f"Grid: {len(grid)} cells; pending: {len(pending)}", log_file)

    if args.dry_run:
        log("DRY RUN — exiting.", log_file)
        return

    workers_per_gpu = args.workers_per_gpu
    num_workers = workers_per_gpu * len(args.gpus)
    partitions = cpu_partition(num_workers, args.cores_per_worker, args.start_core)

    t_start = time.time()
    completed = failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {}
        for i, spec in enumerate(pending):
            slot = i % num_workers
            spec["gpu"] = args.gpus[slot // workers_per_gpu]
            spec["cpu_set"] = partitions[slot]
            spec["cache_path"] = str(args.cache_path)
            spec["regime"] = args.regime
            spec["n_points_per_epoch"] = args.n_points_per_epoch
            spec["n_epochs"] = args.n_epochs
            spec["pilot"] = bool(args.pilot)
            spec["out_dir"] = str(out_dir)
            spec["sub_log_dir"] = str(sub_log_dir)
            futures[ex.submit(_spawn_cell, spec)] = spec
        for fut in as_completed(futures):
            res = fut.result()
            done = completed + failed + 1
            if res["ok"]:
                completed += 1
                status = "OK"
            else:
                failed += 1
                status = "FAIL"
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else float("inf")
            log(f"[{done}/{len(pending)}] {status} {res['key']} "
                f"({res['time_s']:.0f}s, rc={res.get('rc')})  eta={eta / 60:.1f}min",
                log_file)
            if not res["ok"]:
                log(f"   error: {res.get('error', 'see log')}", log_file)

    log(f"DONE. ok={completed} fail={failed} total_time={(time.time() - t_start) / 60:.1f}min", log_file)


if __name__ == "__main__":
    main()
