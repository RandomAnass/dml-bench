#!/usr/bin/env python3
"""
Phase 4 pilot driver.

Runs 6 representative configurations on aspirin (largest rMD17 molecule, 21
atoms) at seed 42, 100 epochs each. Sequential on GPU 1 (Tier 3+4 still on
GPU 0 and must not be disturbed).

Configurations:
  - MLP × {vanilla, dml_fixed, dml_warmup}     — 3 runs
  - GATv2 × {vanilla, dml_fixed}                — 2 runs
  - PaiNN × native_EF                            — 1 run
  Total: 6 runs.

Goal of pilot per AGENT_PRINCIPLES §6 Stage B:
  1. Estimate runtime per (architecture, method).
  2. Verify metrics behave plausibly (E/F MAE roughly compatible with
     literature expectation at this epoch budget).
  3. Catch any pathological behavior (OOM, NaN explosion, divergence).

Output: results/pilot_p4/*.json + a summary table to stdout and to
logs/pilot_p4_<timestamp>.log.

Usage: python scripts/pilot_p4.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set GPU + thread caps before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("BLIS_NUM_THREADS", "6")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")

import torch  # noqa: E402

torch.set_num_threads(6)

PILOT_MOL = "aspirin"
PILOT_SEED = 42
PILOT_EPOCHS = 100
RESULTS_DIR = ROOT / "results" / "pilot_p4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = ROOT / "logs" / f"pilot_p4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_f = open(LOG_PATH, "w")


def log(msg: str):
    print(msg, flush=True)
    log_f.write(msg + "\n")
    log_f.flush()


# ============================================================================
# RUN MLP
# ============================================================================
def run_mlp(method):
    log(f"\n=== MLP {method} {PILOT_MOL} s{PILOT_SEED} {PILOT_EPOCHS}ep ===")
    from experiments.molecular.run_mlp_molecular import train_one, HPARAMS
    hp = dict(HPARAMS)
    hp["n_epochs"] = PILOT_EPOCHS
    t0 = time.time()
    try:
        result = train_one(PILOT_MOL, method, PILOT_SEED, hp, device_idx=0,
                           n_train=1000, n_val=1000, n_test=1000)
        result["key"] = f"mlp_pilot_{PILOT_MOL}_s{PILOT_SEED}_{method}"
        p = RESULTS_DIR / f"{result['key']}.json"
        with open(p, "w") as fp:
            json.dump(result, fp, indent=2, default=str)
        e = result.get("test_energy_mae_approx_mev", 0)
        f = result.get("test_force_mae_approx_mev", 0)
        log(f"  OK ({result['time_s']:.1f}s) E≈{e:7.1f}meV F≈{f:7.1f}meV/Å -> {p.name}")
        return result
    except Exception as e:
        log(f"  FAIL ({time.time()-t0:.1f}s) {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(file=log_f)
        return None


# ============================================================================
# RUN GATv2
# ============================================================================
def run_gatv2(method):
    log(f"\n=== GATv2 {method} {PILOT_MOL} s{PILOT_SEED} {PILOT_EPOCHS}ep ===")
    from experiments.gnn_md17 import (
        GATv2EnergyModel, load_rmd17_graphs, train_gnn_md17,
        set_deterministic, HPARAMS,
    )
    set_deterministic(PILOT_SEED)
    train_data, val_data, test_data, meta = load_rmd17_graphs(
        PILOT_MOL, n_train=1000, n_val=1000, n_test=1000,
        seed=PILOT_SEED, r_cut=HPARAMS["r_cut"],
    )
    set_deterministic(PILOT_SEED)
    model = GATv2EnergyModel(
        hidden_dim=HPARAMS["hidden_dim"], n_heads=HPARAMS["n_heads"],
        n_layers=HPARAMS["n_layers"], n_rbf=HPARAMS["n_rbf"],
        r_cut=HPARAMS["r_cut"], max_z=HPARAMS["max_z"],
    )
    t0 = time.time()
    try:
        metrics = train_gnn_md17(
            model=model, train_data=train_data, val_data=val_data, test_data=test_data,
            method=method, n_epochs=PILOT_EPOCHS,
            batch_size=HPARAMS["batch_size"], lr=HPARAMS["lr"],
            weight_decay=HPARAMS["weight_decay"],
            patience=HPARAMS["patience"], min_lr=HPARAMS["min_lr"],
            lambda_force=HPARAMS["lambda_force"], device="cuda:0",
        )
        elapsed = time.time() - t0
        result = {
            "key": f"gatv2_pilot_{PILOT_MOL}_s{PILOT_SEED}_{method}",
            "method": method, "model": "GATv2",
            "molecule": PILOT_MOL, "seed": PILOT_SEED,
            "n_epochs_actual": metrics["n_epochs_actual"],
            "best_epoch": metrics["best_epoch"],
            "test_value_mse": metrics["test_energy_mse"],
            "test_grad_mse": metrics["test_force_mse"],
            "test_energy_mae_mev": metrics["test_energy_mae_mev"],
            "test_force_mae_mev": metrics["test_force_mae_mev"],
            "time_s": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }
        p = RESULTS_DIR / f"{result['key']}.json"
        with open(p, "w") as fp:
            json.dump(result, fp, indent=2, default=str)
        log(f"  OK ({elapsed:.1f}s) E={metrics['test_energy_mae_mev']:7.1f}meV "
            f"F={metrics['test_force_mae_mev']:7.1f}meV/Å -> {p.name}")
        return result
    except Exception as e:
        log(f"  FAIL ({time.time()-t0:.1f}s) {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(file=log_f)
        return None


# ============================================================================
# RUN PaiNN
# ============================================================================
def run_painn(method):
    log(f"\n=== PaiNN {method} {PILOT_MOL} s{PILOT_SEED} {PILOT_EPOCHS}ep ===")
    from experiments.molecular.run_painn import train_one, HPARAMS_CANONICAL
    hp = dict(HPARAMS_CANONICAL)
    hp["n_epochs"] = PILOT_EPOCHS
    hp["num_train"] = 1000
    hp["num_val"] = 1000
    t0 = time.time()
    try:
        result = train_one(PILOT_MOL, method, PILOT_SEED, hp, smoke=False, gpu=1)
        result["key"] = f"painn_pilot_{PILOT_MOL}_s{PILOT_SEED}_{method}"
        p = RESULTS_DIR / f"{result['key']}.json"
        with open(p, "w") as fp:
            json.dump(result, fp, indent=2, default=str)
        log(f"  OK ({result['time_s']:.1f}s) E={result['test_energy_mae_mev']:7.1f}meV "
            f"F={result['test_force_mae_mev']:7.1f}meV/Å -> {p.name}")
        return result
    except Exception as e:
        log(f"  FAIL ({time.time()-t0:.1f}s) {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(file=log_f)
        return None


# ============================================================================
# MAIN
# ============================================================================
def main():
    log("=" * 70)
    log(f"Phase 4 Pilot — {PILOT_MOL} × seed {PILOT_SEED} × {PILOT_EPOCHS} epochs")
    log(f"Started: {datetime.now().isoformat()}")
    log(f"GPU 1 (Tier 3+4 untouched on GPU 0)")
    log(f"Log: {LOG_PATH}")
    log("=" * 70)

    results = []

    # MLP
    for m in ["vanilla", "dml_fixed", "dml_warmup"]:
        r = run_mlp(m)
        if r:
            results.append(r)

    # GATv2
    for m in ["vanilla", "dml_fixed"]:
        r = run_gatv2(m)
        if r:
            results.append(r)

    # PaiNN
    for m in ["native_EF"]:
        r = run_painn(m)
        if r:
            results.append(r)

    # Summary
    log("\n" + "=" * 70)
    log("PILOT SUMMARY")
    log("=" * 70)
    for r in results:
        e = r.get("test_energy_mae_mev",
                  r.get("test_energy_mae_approx_mev", float("nan")))
        f = r.get("test_force_mae_mev",
                  r.get("test_force_mae_approx_mev", float("nan")))
        t = r.get("time_s", 0)
        log(f"  {r['key']:55s}  E={e:7.1f}meV  F={f:7.1f}meV/Å  t={t:6.1f}s")

    log(f"\n{len(results)}/6 successful")
    log(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
    log_f.close()
