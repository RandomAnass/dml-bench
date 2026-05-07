#!/usr/bin/env python3
"""
V1/P7 ablation — dml_fixed weight scheme A/B test (H&S 1/(1+λd) vs 0.5/0.5).

Compares MLP performance under the two weight conventions on aspirin (largest
rMD17 molecule, stress test) across all 5 canonical Figshare splits. Goal:
inform the cross-architecture decision in P4 (Option I/II/III/IV).

Output: results/ablation_v1_p7/mlp_md17_aspirin_split{1..5}_dml_fixed_{hs,half}.json

Usage: python scripts/ablation_v1_p7_weights.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--molecule", default="aspirin")
    parser.add_argument("--split_ids", type=str, default="1,2,3,4,5")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Per-method epochs (200 = decent convergence on aspirin MLP)")
    parser.add_argument("--results_dir", default="results/ablation_v1_p7")
    args = parser.parse_args()

    split_ids = [int(s) for s in args.split_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(4)

    from experiments.molecular.run_mlp_molecular import train_one, HPARAMS

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"V1/P7 ablation: dml_fixed weights (H&S vs 0.5/0.5)")
    print(f"Molecule: {args.molecule}, splits: {split_ids}, n_epochs: {args.n_epochs}")
    print("=" * 70)

    hparams = dict(HPARAMS)
    hparams["n_epochs"] = args.n_epochs

    methods = [("dml_fixed", "hs_1_over_1plus_ld"), ("dml_fixed_half", "half_05_05")]

    n_done = 0
    for split_id in split_ids:
        for method, label in methods:
            key = f"mlp_md17_{args.molecule}_split{split_id}_{method}"
            print(f"\n--- {key} ({label}) ---")
            t0 = time.time()
            try:
                # J-H1 (2026-04-16): train_one signature narrowed in the MLP
                # rewrite to (molecule, method, split_id, hparams). The old
                # seed/device_idx/n_train/n_val/n_test/use_canonical_splits args
                # are now baked in (canonical 950/50/1000 always).
                result = train_one(
                    args.molecule, method, split_id, hparams,
                )
                result["key"] = key
                result["weight_scheme_label"] = label
                save_path = results_dir / f"{key}.json"
                with open(save_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                e = result["test_value_mse"]; g = result["test_grad_mse"]
                print(f"  OK ({result['time_s']:.1f}s)  val_MSE={e:.4e}  grad_MSE={g:.4e}")
                n_done += 1
            except Exception as e:
                import traceback
                print(f"  FAIL: {e}")
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("V1/P7 SUMMARY (mean over splits)")
    print("=" * 70)
    import numpy as np
    for method, label in methods:
        e_vals, g_vals = [], []
        for split_id in split_ids:
            p = results_dir / f"mlp_md17_{args.molecule}_split{split_id}_{method}.json"
            if p.exists():
                d = json.load(open(p))
                e_vals.append(d["test_value_mse"])
                g_vals.append(d["test_grad_mse"])
        if e_vals:
            print(f"  {method:18s} ({label:25s}) "
                  f" val_MSE = {np.mean(e_vals):.4e} ± {np.std(e_vals):.4e}, "
                  f" grad_MSE = {np.mean(g_vals):.4e} ± {np.std(g_vals):.4e}, "
                  f" n={len(e_vals)}")
    print(f"\nDone. {n_done}/{2 * len(split_ids)} succeeded.")


if __name__ == "__main__":
    main()
