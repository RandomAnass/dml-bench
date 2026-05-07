#!/usr/bin/env python3
"""
Smoke test for Phase 4 GATv2 extensions.

Runs all 5 methods (vanilla, dml_fixed, dml_gradnorm, dml_relobralo, dml_warmup)
on ethanol with 5 epochs each, seed 42.

Goal: catch crashes / shape errors / NaN / OOM. Not for benchmarking.
Per AGENT_PRINCIPLES §6 Stage A.

Usage: python scripts/smoke_p4_gatv2.py --gpu 1
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from experiments.gnn_md17 import (
    GATv2EnergyModel, load_rmd17_graphs, train_gnn_md17, set_deterministic, METHODS, HPARAMS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--molecule", default="ethanol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Per AGENT_PRINCIPLES §11: explicit thread caps
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    torch.set_num_threads(4)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== Smoke test: GATv2 on rMD17 {args.molecule} ===")
    print(f"Device: {device}; n_train={args.n_train}, n_epochs={args.n_epochs}, seed={args.seed}")
    print(f"Methods: {METHODS}")
    print()

    # Load tiny subset
    set_deterministic(args.seed)
    train_data, val_data, test_data, metadata = load_rmd17_graphs(
        args.molecule,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.seed, r_cut=HPARAMS["r_cut"],
    )
    print(f"Loaded {args.molecule}: n_atoms={metadata['n_atoms']}\n")

    results = {}
    for method in METHODS:
        print(f"--- {method} ---")
        set_deterministic(args.seed)
        model = GATv2EnergyModel(
            hidden_dim=HPARAMS["hidden_dim"],
            n_heads=HPARAMS["n_heads"],
            n_layers=HPARAMS["n_layers"],
            n_rbf=HPARAMS["n_rbf"],
            r_cut=HPARAMS["r_cut"],
            max_z=HPARAMS["max_z"],
        )
        t0 = time.time()
        try:
            metrics = train_gnn_md17(
                model=model,
                train_data=train_data, val_data=val_data, test_data=test_data,
                method=method,
                n_epochs=args.n_epochs,
                batch_size=HPARAMS["batch_size"],
                lr=HPARAMS["lr"],
                weight_decay=HPARAMS["weight_decay"],
                patience=args.n_epochs + 1,  # disable early stopping for smoke
                min_lr=HPARAMS["min_lr"],
                lambda_force=HPARAMS["lambda_force"],
                device=device,
            )
            elapsed = time.time() - t0
            e_mae = metrics["test_energy_mae_mev"]
            f_mae = metrics["test_force_mae_mev"]
            print(f"    OK ({elapsed:.1f}s): E_MAE={e_mae:.1f} meV, F_MAE={f_mae:.1f} meV/Å\n")
            results[method] = {"ok": True, "e_mae": e_mae, "f_mae": f_mae, "time_s": elapsed}
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            traceback.print_exc()
            print(f"    FAILED ({elapsed:.1f}s): {e}\n")
            results[method] = {"ok": False, "error": str(e), "time_s": elapsed}

    print("=" * 60)
    print("SMOKE-TEST SUMMARY")
    print("=" * 60)
    for method, r in results.items():
        if r["ok"]:
            print(f"  {method:20s} OK   {r['time_s']:5.1f}s   E={r['e_mae']:6.1f} meV  F={r['f_mae']:6.1f} meV/Å")
        else:
            print(f"  {method:20s} FAIL {r['time_s']:5.1f}s   {r['error']}")

    n_ok = sum(1 for r in results.values() if r["ok"])
    print(f"\n{n_ok}/{len(METHODS)} passed")
    sys.exit(0 if n_ok == len(METHODS) else 1)


if __name__ == "__main__":
    main()
