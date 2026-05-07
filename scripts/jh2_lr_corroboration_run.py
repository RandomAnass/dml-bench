#!/usr/bin/env python3
"""
J-H2 corroboration: does the lr/10 drop (used in warmup phase-2) transfer to
standalone dml_gradnorm on the pairwise-MLP backbone?

Grid: aspirin × splits {1,2,3} × {vanilla baseline, dml_gradnorm with lr_div ∈
{5, 10, 20}} = 12 runs total. Runs post-cff4862a where GradNorm actually
updates weights (was silent-broken 2026-04-16 → 2026-04-23 per I-H5).

Reuses experiments/molecular/run_mlp_molecular.train_one for the actual
training so we inherit every fix (canonical splits, Cartesian-force MSE,
J-M5 per-method seed, etc.). Only the hparams dict varies.

Usage:
    python scripts/jh2_lr_corroboration_run.py --gpu 1
    python scripts/jh2_lr_corroboration_run.py --gpu 1 --resume
    python scripts/jh2_lr_corroboration_run.py --gpu 1 --variants lr_div_10
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.molecular.run_mlp_molecular import HPARAMS, train_one


DEFAULT_MOLECULE = "aspirin"
DEFAULT_SPLITS = [1, 2, 3]
LR_DROP_FACTORS = [5, 10, 20]
BASE_LR = HPARAMS["lr"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_one(molecule: str, split_id: int, variant: str, results_dir: Path) -> dict:
    """variant ∈ {'vanilla', 'lr_div_5', 'lr_div_10', 'lr_div_20'}."""
    if variant == "vanilla":
        method = "vanilla"
        hparams = dict(HPARAMS)  # base lr
        lr_factor = None
    else:
        factor = int(variant.split("_")[-1])
        method = "dml_gradnorm"
        hparams = dict(HPARAMS)
        hparams["lr"] = BASE_LR / factor
        lr_factor = factor

    res = train_one(molecule, method, split_id, hparams)

    # Stamp ablation-specific fields + git hash
    res["key"] = f"corroboration_{molecule}_split{split_id}_{variant}"
    res["effective_lr"] = hparams["lr"]
    res["lr_factor"] = lr_factor
    res["git_hash"] = git_hash()

    # Use Cartesian force MSE as headline metric (matches the MLP runner's
    # convention post-I-H1).
    path = results_dir / f"{res['key']}.json"
    path.write_text(json.dumps(res, indent=2))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--molecule", default=DEFAULT_MOLECULE)
    p.add_argument("--splits", nargs="+", type=int, default=DEFAULT_SPLITS)
    p.add_argument(
        "--variants", nargs="+",
        default=["vanilla"] + [f"lr_div_{f}" for f in LR_DROP_FACTORS],
    )
    p.add_argument("--results_dir", default="results/ablation_lr_corroboration")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    configs: List[Tuple[str, int, str]] = []
    for split_id in args.splits:
        for variant in args.variants:
            configs.append((args.molecule, split_id, variant))

    print(f"[J-H2] {len(configs)} configs | gpu={args.gpu} | git={git_hash()[:8]}",
          flush=True)

    n_ok = n_fail = n_skip = 0
    for i, (mol, split_id, variant) in enumerate(configs):
        key = f"corroboration_{mol}_split{split_id}_{variant}"
        path = results_dir / f"{key}.json"
        if args.resume and path.exists():
            n_skip += 1
            continue
        try:
            _ = run_one(mol, split_id, variant, results_dir)
            n_ok += 1
        except Exception:
            n_fail += 1
            traceback.print_exc()
        print(f"[J-H2] {i+1}/{len(configs)} ok={n_ok} fail={n_fail} skip={n_skip}",
              flush=True)

    print(f"[J-H2] DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
