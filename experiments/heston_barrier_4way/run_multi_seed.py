#!/usr/bin/env python3
"""
Heston barrier multi-seed run (= run #1 / HEADLINE).

Drops 5 seeds × 12 methods = 60 runs into `results/heston_barrier_4way/multi_seed/`.
Uses the same method-independent MC ground truth (200 spots × 100k paths) as
the single-seed pilot v3, but with a fresh seed disjoint from training.

Seeds: [42, 123, 456, 789, 1337] — matches the existing 5-seed convention used
in the synthetic benchmark and the discontinuous-payoff sub-benchmark.

Output schema matches `run_pilot.py` exactly. Resume-safe (will skip existing
JSONs by key).

Usage:
    python experiments/heston_barrier_4way/run_multi_seed.py --gpu 0
    python experiments/heston_barrier_4way/run_multi_seed.py --gpu 0 --resume
    python experiments/heston_barrier_4way/run_multi_seed.py --analyze-only

Expected runtime: ~2-2.5h on 1 A6000 (60 runs × ~2.5 min average).

References (same as run_pilot.py):
    - Heston, Andersen, G&K v2, Chen-Glasserman 2007, Savine.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import everything from run_pilot, then override SEEDS and dirs.
import experiments.heston_barrier_4way.run_pilot as rp


def main():
    parser = argparse.ArgumentParser(description="Heston barrier multi-seed (5 seeds)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=None,
        help="Override default ES patience (default 50). Use a larger value "
             "(e.g. 200) or 999999 (effectively disabled) to avoid the cross-seed "
             "ES artifact identified in multi_seed_deep_analysis.md §1.",
    )
    parser.add_argument(
        "--results-subdir", default="multi_seed",
        help="Output directory under results/heston_barrier_4way/.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Override pilot's SEEDS for the multi-seed run
    rp.SEEDS = [42, 123, 456, 789, 1337]

    # Configurable ES patience (preserves backward compat: omit flag → default 50)
    if args.early_stopping_patience is not None:
        rp.HPARAMS = dict(rp.HPARAMS)
        rp.HPARAMS["early_stopping_patience"] = args.early_stopping_patience
        print(f"Overriding early_stopping_patience to {args.early_stopping_patience}")

    results_dir = Path(f"results/heston_barrier_4way/{args.results_subdir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        rp.analyze(results_dir)
        return

    existing = rp.load_existing(results_dir) if args.resume else {}
    rp.run_pilot(results_dir, existing, args.resume)
    rp.analyze(results_dir)
    print("\nMulti-seed Heston barrier run complete!")


if __name__ == "__main__":
    main()
