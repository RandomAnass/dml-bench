#!/usr/bin/env python3
"""
Reproducible win-rate computation for DML-Bench paper.
Computes DML vs vanilla win rates from tier1-4 benchmark JSONs.

Usage:
    conda run -n dml-bench-env python3 scripts/compute_win_rates.py

Criteria: noise=0, dml_fixed vs vanilla, gradient MSE, averaged over seeds per config.
"""
import json
import glob
import numpy as np
from collections import defaultdict
import os

RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_win_rates(metric="test_grad_mse",
                       target_method="dml_fixed",
                       baseline_method="vanilla"):
    """Compute per-function PAIRED-SEED win rates from tier1-4 results.

    J-M4 (2026-04-16): paired by (func, dim, n, seed). A config/seed counts
    once iff BOTH target and baseline completed; target wins if its metric
    is lower than baseline's. Previously this averaged over seeds per config
    without checking equal coverage, biasing toward whichever method had more
    successful seeds.
    """
    files = []
    for tier in ["tier1_benchmark", "tier2_benchmark", "tier3_benchmark", "tier4_benchmark"]:
        files.extend(glob.glob(os.path.join(RESULTS_DIR, f"results/{tier}/*.json")))

    # Group by (func, dim, n_samples) → seed → method → metric
    config_data = defaultdict(dict)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue

        func = d.get("func_type", "")
        dim = d.get("dim", 0)
        n = d.get("n_samples", 0)
        noise = d.get("noise_level", 0.0)
        method = d.get("method", "")
        seed = d.get("seed", -1)

        if noise != 0.0 or method not in (baseline_method, target_method):
            continue

        val = d.get(metric)
        if val is not None:
            # J-M4 (2026-04-16): key by (config, seed, method) so we can do
            # PAIRED-SEED comparison (only a seed where BOTH methods completed
            # counts toward the win rate). Prior mean-per-config comparison
            # silently biased toward whichever method had more completed seeds.
            config_data[(func, dim, n)].setdefault(seed, {})[method] = val

    # J-M4: paired-seed win rate — a config counts once per seed where both
    # target and baseline completed; DML "wins" when target metric < baseline.
    wins = defaultdict(lambda: {"total": 0, "dml_wins": 0})
    for (func, dim, n), seed_map in config_data.items():
        for seed, methods in seed_map.items():
            if baseline_method in methods and target_method in methods:
                wins[func]["total"] += 1
                if methods[target_method] < methods[baseline_method]:
                    wins[func]["dml_wins"] += 1

    return wins


def main():
    for metric_name, metric_key in [
        ("Gradient MSE", "test_grad_mse"),
        ("Value MSE", "test_value_mse"),
    ]:
        print(f"\n{'='*60}")
        print(f"Win rates by {metric_name} (noise=0, dml_fixed vs vanilla)")
        print(f"{'='*60}")
        wins = compute_win_rates(metric_key)
        print(f"{'Function':<20} {'Wins/Total':<15} {'Rate':<10}")
        print("-" * 45)
        tw, tt = 0, 0
        for func in sorted(wins.keys()):
            w = wins[func]
            rate = w["dml_wins"] / w["total"] * 100 if w["total"] > 0 else 0
            print(f"{func:<20} {w['dml_wins']}/{w['total']:<13} {rate:.1f}%")
            tw += w["dml_wins"]
            tt += w["total"]
        print("-" * 45)
        print(f"{'TOTAL':<20} {tw}/{tt:<13} {tw / tt * 100:.1f}%")


if __name__ == "__main__":
    main()
