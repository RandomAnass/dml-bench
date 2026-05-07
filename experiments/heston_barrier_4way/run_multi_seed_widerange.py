#!/usr/bin/env python3
"""
Heston barrier multi-seed at WIDE spot range [0.5K, 1.5K] (G&K-style).

Same setup as run_multi_seed.py except S_0 ~ Uniform[0.5K, 1.5K] instead of
[0.7K, 1.3K]. The wider range includes the deep-OTM region (S_0 < B = 0.85K)
where the option is dead at t=0, plus the near-barrier region where the
pricing function has highest curvature. This is the regime where polynomial
regression should fail to capture the function.

Goal: test whether DML methods beat polynomial baseline when the spot range
is wide enough to include the boundary regime — replicating G&K v2 §3.4's
[0.5K, 1.5K] convention on our Heston SV barrier.

Usage:
    python experiments/heston_barrier_4way/run_multi_seed_widerange.py --gpu 0
    python experiments/heston_barrier_4way/run_multi_seed_widerange.py --analyze-only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import experiments.heston_barrier_4way.run_pilot as rp


def main():
    parser = argparse.ArgumentParser(description="Heston barrier multi-seed (wide range, 5 seeds)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=200)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # WIDE spot range — matches G&K v2 §3.4 / Table 2 [0.5K, 1.5K] convention
    rp.SPOT_LOW_MULT = 0.5
    rp.SPOT_HIGH_MULT = 1.5
    rp.SEEDS = [42, 123, 456, 789, 1337]
    rp.HPARAMS = dict(rp.HPARAMS)
    rp.HPARAMS["early_stopping_patience"] = args.early_stopping_patience

    print(f"Wide-range Heston barrier multi-seed:")
    print(f"  spot range: [{rp.SPOT_LOW_MULT} * K, {rp.SPOT_HIGH_MULT} * K] "
          f"= [{rp.SPOT_LOW_MULT * rp.HESTON_PARAMS['strike']}, "
          f"{rp.SPOT_HIGH_MULT * rp.HESTON_PARAMS['strike']}]")
    print(f"  early_stopping_patience: {args.early_stopping_patience}")

    results_dir = Path("results/heston_barrier_4way/multi_seed_widerange")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        rp.analyze(results_dir)
        return

    existing = rp.load_existing(results_dir) if args.resume else {}
    rp.run_pilot(results_dir, existing, args.resume)
    rp.analyze(results_dir)
    print("\nWide-range Heston barrier multi-seed complete!")


if __name__ == "__main__":
    main()
