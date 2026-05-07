#!/usr/bin/env python3
"""
Full Benchmark Runner — Publishable Results

Designed for parallel execution across GPUs and CPU cores.
Saves each experiment atomically (crash-resilient).
Tiers are COMPLEMENTARY: Tier1 ∪ Tier2 ∪ Tier3 ≥ Original Full Plan.

Usage:
    # Tier 1 — Minimum Publishable (~5-6h on 1 GPU):
    python run_full_benchmark.py --tier 1 --gpu 0

    # Tier 2 — Finance/Quant + Noise + Step:
    python run_full_benchmark.py --tier 2 --gpu 0

    # Tier 3 — Fill remaining gaps (λ sweep, dim=1, 8192, etc.):
    python run_full_benchmark.py --tier 3 --gpu 0

    # Tier 4 — Statistical power (10 seeds) + training logs:
    python run_full_benchmark.py --tier 4 --gpu 0 --save-logs --resume

    # Baselines only (CPU-bound, run in parallel):
    python run_full_benchmark.py --tier 1 --baselines-only

    # NN only (GPU-bound):
    python run_full_benchmark.py --tier 1 --gpu 0 --nn-only

    # Resume after crash:
    python run_full_benchmark.py --tier 1 --gpu 0 --resume

Tier definitions (COMPLEMENTARY — each adds NEW experiments only):
    Tier 1 (Minimum Publishable):
        - Functions: trig, poly_trig
        - Dims: 2, 5, 10, 20, 50, 100
        - Samples: 256, 1024, 4096
        - Seeds: 5, Noise: 0.0, λ=1, Epochs: 500
        - Methods: vanilla, dml_fixed, dml_gradnorm, dml_relobralo
        - Baselines: GP, KRR, RF (dim ≤ 20)

    Tier 2 (Finance/Quant + Noise + Step — complements Tier 1):
        - NEW functions: step, bachelier, black_scholes, heston
        - Noise axis on trig/poly_trig: 0.05, 0.10, 0.20, 0.50
        - NEW sample sizes for trig/poly_trig: 512, 2048
        - Hedging backtest experiment (Black-Scholes)
        - Finance dims: bachelier [1,2,5,10,20,50], BS/heston dim=1
        - Epochs: 500

    Tier 3 (Fill remaining gaps — completes original plan):
        - dim=1 for trig, poly_trig, step
        - Sample size 8192 everywhere
        - λ sweep: [0.001, 0.01, 0.1, 10] for dml_fixed (λ=1 in Tier1/2)
        - Epochs: 1000 (extended training)
        - Everything remaining to cover FullBenchmarkConfig

    Tier 4 (Statistical power — 10 seeds for significance):
        - 5 extra seeds (2000-6000) for key configs
        - poly_trig, trig (all dims), step, bachelier, BS, heston
        - n=1024 noise=0 (core comparisons) + 256/4096 for poly_trig
        - --save-logs flag captures epoch-by-epoch training curves
        - Enables Wilcoxon p < 0.01 (min 0.002 with 10 seeds)
"""

import sys
import os
import time
import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dml_benchmark.functions import generate_data, train_test_split, corrupt_derivatives
from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.baselines import run_baseline_experiment


# ============================================================================
# TIER CONFIGURATIONS — COMPLEMENTARY (union ≥ original FullBenchmarkConfig)
#
# Each tier generates its OWN experiment list. No overlap between tiers.
# Tier 1: core synthetic (trig, poly_trig), clean derivatives, λ=1
# Tier 2: finance functions + noise axis + step + fill samples + hedging
# Tier 3: dim=1, 8192 samples, λ sweep, 1000-epoch reruns
# ============================================================================

def build_tier1_experiments():
    """Tier 1 — Minimum Publishable: core synthetic, clean, λ=1."""
    experiments = []
    for func in ["trig", "poly_trig"]:
        for dim in [2, 5, 10, 20, 50, 100]:
            for ns in [256, 1024, 4096]:
                for seed in [42, 123, 456, 789, 1000]:
                    experiments.append({
                        "func_type": func, "dim": dim, "n_samples": ns,
                        "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                    })
    return experiments


def build_tier2_experiments():
    """Tier 2 — Finance/Quant + Noise + Step (complementary to Tier 1).
    
    Adds:
    A) step function across all Tier-1 dims/samples
    B) Finance: bachelier (multi-dim), black_scholes (dim=1), heston (dim=1)
    C) Noise axis on trig/poly_trig: noise ∈ {0.05, 0.10, 0.20, 0.50}
    D) Fill sample sizes 512, 2048 for trig/poly_trig (noise=0.0)
    E) Hedging backtest flag
    """
    experiments = []
    seeds = [42, 123, 456, 789, 1000]
    
    # A) Step function — same grid as Tier 1 but for step
    for dim in [2, 5, 10, 20, 50, 100]:
        for ns in [256, 1024, 4096]:
            for seed in seeds:
                experiments.append({
                    "func_type": "step", "dim": dim, "n_samples": ns,
                    "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                })
    
    # B) Finance functions
    # Bachelier basket — multi-dim (n_assets = dim)
    for dim in [1, 2, 5, 10, 20, 50]:
        for ns in [256, 1024, 4096]:
            for seed in seeds:
                experiments.append({
                    "func_type": "bachelier", "dim": dim, "n_samples": ns,
                    "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                })
    
    # Black-Scholes (dim=1 only)
    for ns in [256, 1024, 4096]:
        for seed in seeds:
            experiments.append({
                "func_type": "black_scholes", "dim": 1, "n_samples": ns,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    # Heston (dim=1 only, internally 2D)
    for ns in [256, 1024, 4096]:
        for seed in seeds:
            experiments.append({
                "func_type": "heston", "dim": 1, "n_samples": ns,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    # C) Noise axis on trig/poly_trig (noise > 0, NOT 0.0 which is Tier 1)
    for func in ["trig", "poly_trig"]:
        for dim in [2, 5, 10, 20, 50, 100]:
            for ns in [256, 1024, 4096]:
                for noise in [0.05, 0.10, 0.20, 0.50]:
                    for seed in seeds:
                        experiments.append({
                            "func_type": func, "dim": dim, "n_samples": ns,
                            "noise_level": noise, "seed": seed, "lambda_": 1.0,
                        })
    
    # D) Fill sample sizes 512, 2048 for trig/poly_trig at noise=0.0
    for func in ["trig", "poly_trig"]:
        for dim in [2, 5, 10, 20, 50, 100]:
            for ns in [512, 2048]:
                for seed in seeds:
                    experiments.append({
                        "func_type": func, "dim": dim, "n_samples": ns,
                        "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                    })
    
    return experiments


def build_tier3_experiments():
    """Tier 3 — Fill remaining gaps to cover original FullBenchmarkConfig.
    
    Adds:
    A) dim=1 for trig, poly_trig, step
    B) Sample size 8192 for all functions/dims
    C) λ sweep for dml_fixed: λ ∈ {0.001, 0.01, 0.1, 10}
       (λ=1 already covered in Tier 1/2, λ=0 is just vanilla)
    D) Noise axis on step + finance functions
    E) Fill any remaining gaps (512/2048 for step/finance, etc.)
    """
    experiments = []
    seeds = [42, 123, 456, 789, 1000]
    all_synthetic = ["trig", "poly_trig", "step"]
    
    # A) dim=1 for synthetic functions
    for func in all_synthetic:
        for ns in [256, 512, 1024, 2048, 4096, 8192]:
            for noise in [0.0, 0.05, 0.10, 0.20, 0.50]:
                for seed in seeds:
                    experiments.append({
                        "func_type": func, "dim": 1, "n_samples": ns,
                        "noise_level": noise, "seed": seed, "lambda_": 1.0,
                    })
    
    # B) Sample size 8192 for all existing dim/func combos
    for func in all_synthetic:
        for dim in [2, 5, 10, 20, 50, 100]:
            for noise in [0.0, 0.05, 0.10, 0.20, 0.50]:
                for seed in seeds:
                    experiments.append({
                        "func_type": func, "dim": dim, "n_samples": 8192,
                        "noise_level": noise, "seed": seed, "lambda_": 1.0,
                    })
    # 8192 for finance
    for dim in [1, 2, 5, 10, 20, 50]:
        for seed in seeds:
            experiments.append({
                "func_type": "bachelier", "dim": dim, "n_samples": 8192,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    for seed in seeds:
        experiments.append({
            "func_type": "black_scholes", "dim": 1, "n_samples": 8192,
            "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
        })
        experiments.append({
            "func_type": "heston", "dim": 1, "n_samples": 8192,
            "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
        })
    
    # C) λ sweep for dml_fixed (core synthetic only, representative grid)
    # Run on subset of dims/samples to keep it tractable
    for func in ["trig", "poly_trig"]:
        for dim in [2, 10, 50, 100]:
            for ns in [1024, 4096]:
                for lam in [0.001, 0.01, 0.1, 10.0]:
                    for seed in seeds:
                        experiments.append({
                            "func_type": func, "dim": dim, "n_samples": ns,
                            "noise_level": 0.0, "seed": seed, "lambda_": lam,
                            "methods_override": ["dml_fixed"],  # λ only affects fixed
                        })
    
    # D) Noise on step function
    for dim in [2, 5, 10, 20, 50, 100]:
        for ns in [256, 1024, 4096]:
            for noise in [0.05, 0.10, 0.20, 0.50]:
                for seed in seeds:
                    experiments.append({
                        "func_type": "step", "dim": dim, "n_samples": ns,
                        "noise_level": noise, "seed": seed, "lambda_": 1.0,
                    })
    
    # E) Fill 512, 2048 for step and finance at noise=0.0
    for dim in [2, 5, 10, 20, 50, 100]:
        for ns in [512, 2048]:
            for seed in seeds:
                experiments.append({
                    "func_type": "step", "dim": dim, "n_samples": ns,
                    "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                })
    for dim in [1, 2, 5, 10, 20, 50]:
        for ns in [512, 2048]:
            for seed in seeds:
                experiments.append({
                    "func_type": "bachelier", "dim": dim, "n_samples": ns,
                    "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                })
    for ns in [512, 2048]:
        for seed in seeds:
            experiments.append({
                "func_type": "black_scholes", "dim": 1, "n_samples": ns,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
            experiments.append({
                "func_type": "heston", "dim": 1, "n_samples": ns,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    return experiments


def build_tier4_experiments():
    """Tier 4 — Statistical Power + Training Logs (complementary to Tier 1-3).
    
    Purpose: Bring Wilcoxon p-values below 0.01 and capture learning curves.
    
    Adds 5 extra seeds (2000,3000,4000,5000,6000) for key configurations, giving
    10 total seeds → Wilcoxon min p ≈ 0.002 (two-sided).
    
    Key configs chosen for their scientific importance:
    - poly_trig: DML's strongest advantage domain
    - trig: contrast with poly_trig (lower derivative information)
    - step: negative/mixed result — honest reporting
    - bachelier: finance multi-dim scaling
    - black_scholes/heston: finance dim=1 (paper's application story)
    """
    experiments = []
    extra_seeds = [2000, 3000, 4000, 5000, 6000]
    
    # --- Core configs: key dims at n=1024, noise=0 ---
    
    # poly_trig: strongest DML story
    for dim in [2, 5, 10, 20, 50, 100]:
        for seed in extra_seeds:
            experiments.append({
                "func_type": "poly_trig", "dim": dim, "n_samples": 1024,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    # trig: contrast case
    for dim in [2, 5, 10, 20, 50, 100]:
        for seed in extra_seeds:
            experiments.append({
                "func_type": "trig", "dim": dim, "n_samples": 1024,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    # step: negative/mixed result (honest reporting)
    for dim in [2, 10, 50, 100]:
        for seed in extra_seeds:
            experiments.append({
                "func_type": "step", "dim": dim, "n_samples": 1024,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    # bachelier: multi-dim finance
    for dim in [1, 2, 5, 10, 20, 50]:
        for seed in extra_seeds:
            experiments.append({
                "func_type": "bachelier", "dim": dim, "n_samples": 1024,
                "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
            })
    
    # black_scholes: dim=1
    for seed in extra_seeds:
        experiments.append({
            "func_type": "black_scholes", "dim": 1, "n_samples": 1024,
            "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
        })
    
    # heston: dim=1 (investigate failure mode)
    for seed in extra_seeds:
        experiments.append({
            "func_type": "heston", "dim": 1, "n_samples": 1024,
            "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
        })
    
    # --- Additional sample sizes for poly_trig (dimension scaling completeness) ---
    for dim in [2, 5, 10, 20, 50, 100]:
        for ns in [256, 4096]:
            for seed in extra_seeds:
                experiments.append({
                    "func_type": "poly_trig", "dim": dim, "n_samples": ns,
                    "noise_level": 0.0, "seed": seed, "lambda_": 1.0,
                })
    
    return experiments


BUILD_TIER_FNS = {1: build_tier1_experiments, 2: build_tier2_experiments, 3: build_tier3_experiments, 4: build_tier4_experiments}

# Shared training hyperparams (consistent across all tiers)
TIER_HPARAMS = {
    1: {"n_epochs": 500,  "batch_size": 512,  "n_layers": 4, "hidden_size": 256, "activation": "softplus", "lr": 0.005},
    2: {"n_epochs": 500,  "batch_size": 512,  "n_layers": 4, "hidden_size": 256, "activation": "softplus", "lr": 0.005},
    3: {"n_epochs": 1000, "batch_size": 1024, "n_layers": 4, "hidden_size": 256, "activation": "softplus", "lr": 0.005},
    4: {"n_epochs": 500,  "batch_size": 512,  "n_layers": 4, "hidden_size": 256, "activation": "softplus", "lr": 0.005},
}

TIER_NAMES = {
    1: "Minimum Publishable",
    2: "Finance/Quant + Noise + Step",
    3: "Fill Remaining (λ sweep, dim=1, 8192, 1000ep)",
    4: "Statistical Power (10 seeds) + Training Logs",
}

# Default NN methods and baselines — all 7 DML-Bench methods.
# I-L1 (2026-04-16): extended from 4 to cover the full method grid.
# L-H1 fix (2026-04-16): train_single_experiment now dispatches
# method="dml_warmup" to train_warmup (lazy import from
# experiments.unified_comparison.run_unified_experiment), so warmup can
# be included in tier 1-4 runs without special-casing in run_full_benchmark.
DEFAULT_NN_METHODS = [
    "vanilla", "dml_fixed", "dml_fixed_half",
    "dml_gradnorm", "dml_softmax_balance", "dml_relobralo", "dml_warmup",
]
DEFAULT_BASELINE_METHODS = ["gp", "krr", "rf"]
BASELINE_MAX_DIM = 20

# NOTE ON BASELINE DATA ADVANTAGE (document in paper):
# NNs use an 80/20 train/val split (val for early stopping + model selection),
# so they effectively train on 64% of the data (80% × 80%).
# Baselines (GP, KRR, RF) use the full 80% training set — no validation needed.
# This gives baselines ~25% more training data.
# This is STANDARD PRACTICE — baselines don't require validation sets.
# Must be acknowledged in the paper's experimental setup section.


def parse_args():
    parser = argparse.ArgumentParser(description="Full DML Benchmark Runner")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Experiment tier (1=core, 2=finance+noise, 3=gaps, 4=stat power)")
    parser.add_argument("--gpu", type=int, default=None, help="Pin to specific GPU")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--nn-only", action="store_true")
    parser.add_argument("--hedging", action="store_true",
                        help="Run hedging backtest (Tier 2 only, after NN training)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Override NN methods to run (e.g., --methods dml_gradnorm)")
    parser.add_argument("--save-logs", action="store_true",
                        help="Save full training logs (epoch-by-epoch) to result JSON files")
    return parser.parse_args()


def make_key(func_type, dim, n_samples, noise, seed, method):
    return f"{func_type}_d{dim}_n{n_samples}_noise{noise}_s{seed}_{method}"


def load_existing_results(results_dir):
    """Load all individual result files for resume support."""
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            if f.name == "summary.json":
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    existing[data.get("key", f.stem)] = data
            except Exception:
                pass
    return existing


def save_single_result(results_dir, key, result_dict):
    """Save a single experiment result atomically."""
    result_dict["key"] = key
    result_dict["timestamp"] = datetime.now().isoformat()
    path = results_dir / f"{key}.json"
    # Atomic write via temp file
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp_path.rename(path)


def save_summary(results_dir, config, all_results, errors, total_time):
    """Save overall summary."""
    summary = {
        "config": config,
        "n_experiments": len(all_results),
        "errors": errors,
        "total_time_s": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    # Aggregate stats
    method_groups = {}
    for key, r in all_results.items():
        m = r["method"]
        if m not in method_groups:
            method_groups[m] = {"val": [], "grad": [], "times": []}
        method_groups[m]["val"].append(r["test_value_mse"])
        method_groups[m]["grad"].append(r["test_grad_mse"])
        method_groups[m]["times"].append(r.get("time_s", 0))
    
    summary["method_summary"] = {}
    for m, d in method_groups.items():
        summary["method_summary"][m] = {
            "mean_value_mse": float(np.mean(d["val"])),
            "std_value_mse": float(np.std(d["val"])),
            "mean_grad_mse": float(np.mean(d["grad"])),
            "std_grad_mse": float(np.std(d["grad"])),
            "mean_time_s": float(np.mean(d["times"])),
            "count": len(d["val"]),
        }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def estimate_runtime(experiments, hparams, nn_only, baselines_only):
    """Estimate total runtime based on Phase 2 timing."""
    # Per-experiment timing estimates (NN per 200 epochs).
    # I-L9 (2026-04-16): these constants were measured in an earlier iteration
    # with 200-epoch tier. Tier 3 now uses 1000 epochs and tier 4 uses 500 —
    # the constant below is scaled by epoch_ratio below for rough estimation.
    # Recalibration via `--dry-run` profiling is recommended if a user wants
    # a tight runtime estimate.
    avg_nn_per_200ep = 11.1  # seconds (measured on A6000, legacy calibration)
    avg_gp_per_exp = 192.0
    avg_krr_per_exp = 4.4
    avg_rf_per_exp = 11.6
    
    n_epochs = hparams["n_epochs"]
    epoch_ratio = n_epochs / 200.0
    
    total_nn = 0
    total_bl = 0
    
    for exp in experiments:
        methods = exp.get("methods_override", DEFAULT_NN_METHODS)
        n_methods = len(methods)
        if not baselines_only:
            total_nn += n_methods * avg_nn_per_200ep * epoch_ratio
        if not nn_only and exp["dim"] <= BASELINE_MAX_DIM:
            total_bl += avg_gp_per_exp + avg_krr_per_exp + avg_rf_per_exp
    
    return total_nn, total_bl


def run_benchmark():
    args = parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    hparams = TIER_HPARAMS[args.tier]
    tier_name = TIER_NAMES[args.tier]
    results_dir = Path(f"results/tier{args.tier}_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Build experiment list for this tier
    experiments = BUILD_TIER_FNS[args.tier]()
    
    # Estimate runtime
    est_nn, est_bl = estimate_runtime(experiments, hparams, args.nn_only, args.baselines_only)
    
    # Collect unique values for display
    funcs = sorted(set(e["func_type"] for e in experiments))
    dims = sorted(set(e["dim"] for e in experiments))
    samples = sorted(set(e["n_samples"] for e in experiments))
    noises = sorted(set(e["noise_level"] for e in experiments))
    lambdas = sorted(set(e["lambda_"] for e in experiments))
    
    print("=" * 70)
    print(f"DML BENCHMARK — Tier {args.tier}: {tier_name}")
    print("=" * 70)
    print(f"Functions:     {funcs}")
    print(f"Dimensions:    {dims}")
    print(f"Sample sizes:  {samples}")
    print(f"Noise levels:  {noises}")
    print(f"Lambda values: {lambdas}")
    print(f"Experiments:   {len(experiments)} data points")
    print(f"Epochs:        {hparams['n_epochs']}")
    print(f"Architecture:  {hparams['n_layers']}L x {hparams['hidden_size']}H, {hparams['activation']}")
    if args.gpu is not None:
        print(f"GPU:           {args.gpu}")
    
    if args.nn_only:
        print(f"\nRunning NN ONLY — Est. {est_nn/3600:.1f}h")
    elif args.baselines_only:
        print(f"\nRunning BASELINES ONLY — Est. {est_bl/3600:.1f}h")
    else:
        print(f"\nEst. NN: {est_nn/3600:.1f}h, Baselines: {est_bl/3600:.1f}h, Total: {(est_nn+est_bl)/3600:.1f}h")
    
    # Resume support — load from ALL tier dirs (cross-tier dedup)
    all_results = {}
    if args.resume:
        for t in [1, 2, 3, 4]:
            tier_dir = Path(f"results/tier{t}_benchmark")
            all_results.update(load_existing_results(tier_dir))
        if all_results:
            print(f"Resuming: {len(all_results)} experiments found across all tiers")
    
    # Group experiments by (func, dim, n_samples, noise) to generate data once
    from itertools import groupby
    experiments_sorted = sorted(experiments, key=lambda e: (e["func_type"], e["dim"], e["n_samples"], e["noise_level"]))
    
    data_groups = {}
    for key_tuple, group_items in groupby(
        experiments_sorted,
        key=lambda e: (e["func_type"], e["dim"], e["n_samples"], e["noise_level"])
    ):
        data_groups[key_tuple] = list(group_items)
    
    n_groups = len(data_groups)
    
    errors = []
    total_start = time.time()
    completed = 0
    skipped = 0
    
    for gi, ((func_type, dim, n_samples, noise_level), group_exps) in enumerate(data_groups.items()):
        
        banner = f"[{gi+1}/{n_groups}] {func_type} | dim={dim} | n={n_samples} | noise={noise_level}"
        elapsed_so_far = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"{banner}  [{elapsed_so_far/60:.0f}m elapsed, {completed} done, {skipped} skipped]")
        print(f"{'='*70}")
        
        # Unique seeds in this group
        seeds_in_group = sorted(set(e["seed"] for e in group_exps))
        # Collect per-seed experiment params (lambda, methods_override)
        seed_params = {}
        for e in group_exps:
            s = e["seed"]
            if s not in seed_params:
                seed_params[s] = []
            seed_params[s].append(e)
        
        for seed in seeds_in_group:
            # Generate data once per (func, dim, n_samples, seed)
            try:
                data = generate_data(func_type, n_dim=dim, n_samples=n_samples, seed=seed)
                train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
            except Exception as e:
                msg = f"DATA GEN FAILED: {func_type} d={dim} n={n_samples} s={seed}: {e}"
                print(f"  ❌ {msg}")
                errors.append(msg)
                traceback.print_exc()
                continue
            
            x_train, y_train, dydx_train = train_data.x, train_data.y, train_data.dydx
            x_test, y_test, dydx_test = test_data.x, test_data.y, test_data.dydx
            
            # Apply derivative noise if needed
            if noise_level > 0:
                dydx_train_noisy = corrupt_derivatives(
                    dydx_train, noise_level=noise_level, seed=seed
                )
            else:
                dydx_train_noisy = dydx_train
            
            # Process each experiment config for this seed
            for exp in seed_params[seed]:
                lambda_val = exp["lambda_"]
                nn_methods = exp.get("methods_override", DEFAULT_NN_METHODS)
                
                # CLI override: filter to only requested methods
                if args.methods:
                    nn_methods = [m for m in nn_methods if m in args.methods]
                
                # ---- NN Methods ----
                if not args.baselines_only:
                    for method in nn_methods:
                        lam_tag = f"_lam{lambda_val}" if lambda_val != 1.0 else ""
                        key = make_key(func_type, dim, n_samples, noise_level, seed, f"{method}{lam_tag}")
                        
                        if args.resume and key in all_results:
                            skipped += 1
                            continue
                        
                        lam_str = f" λ={lambda_val}" if lambda_val != 1.0 else ""
                        print(f"  {method}{lam_str} s={seed}...", end=" ", flush=True)
                        t0 = time.time()
                        
                        try:
                            result = train_single_experiment(
                                x_train=x_train,
                                y_train=y_train,
                                dydx_train=dydx_train_noisy,
                                x_test=x_test,
                                y_test=y_test,
                                dydx_test=dydx_test,  # Always eval against clean gradients
                                lambda_=lambda_val,
                                n_epochs=hparams["n_epochs"],
                                batch_size=hparams["batch_size"],
                                n_layers=hparams["n_layers"],
                                hidden_size=hparams["hidden_size"],
                                lr=hparams["lr"],
                                activation=hparams["activation"],
                                seed=seed,
                                method=method,
                                pbar=False,
                            )
                            elapsed = time.time() - t0
                            
                            result_dict = {
                                "method": method,
                                "func_type": func_type,
                                "dim": dim,
                                "n_samples": n_samples,
                                "noise_level": noise_level,
                                "seed": seed,
                                "lambda": lambda_val,
                                "test_value_mse": result.test_value_mse,
                                "test_grad_mse": result.test_grad_mse,
                                "best_epoch": result.best_epoch,
                                "time_s": elapsed,
                                "n_epochs_actual": len(result.training_logs),
                            }
                            
                            # Optionally save full training logs (for learning curves)
                            if args.save_logs:
                                result_dict["training_logs"] = result.training_logs
                            
                            all_results[key] = result_dict
                            save_single_result(results_dir, key, result_dict)
                            
                            loss_ok = result.training_logs[-1]["train_loss"] < result.training_logs[0]["train_loss"]
                            status = "✅" if loss_ok else "⚠️"
                            print(f"{status} MSE={result.test_value_mse:.6f} grad={result.test_grad_mse:.6f} "
                                  f"best@{result.best_epoch} ({elapsed:.1f}s)")
                            completed += 1
                            
                        except Exception as e:
                            elapsed = time.time() - t0
                            msg = f"TRAIN FAILED: {key}: {e}"
                            print(f"❌ ({elapsed:.1f}s) {msg}")
                            errors.append(msg)
                            traceback.print_exc()
                
                # ---- Baselines (dim <= threshold, only for default λ=1) ----
                if not args.nn_only and dim <= BASELINE_MAX_DIM and lambda_val == 1.0:
                    for bl_name in DEFAULT_BASELINE_METHODS:
                        key = make_key(func_type, dim, n_samples, noise_level, seed, f"baseline_{bl_name}")
                        
                        if args.resume and key in all_results:
                            skipped += 1
                            continue
                        
                        print(f"  baseline_{bl_name} s={seed}...", end=" ", flush=True)
                        t0 = time.time()
                        
                        try:
                            bl_result = run_baseline_experiment(
                                bl_name,
                                x_train, y_train, dydx_train_noisy,
                                x_test, y_test, dydx_test,
                            )
                            elapsed = time.time() - t0
                            
                            result_dict = {
                                "method": f"baseline_{bl_name}",
                                "func_type": func_type,
                                "dim": dim,
                                "n_samples": n_samples,
                                "noise_level": noise_level,
                                "seed": seed,
                                "lambda": 1.0,
                                "test_value_mse": bl_result["value_mse"],
                                "test_grad_mse": bl_result["grad_mse"],
                                "time_s": elapsed,
                            }
                            
                            all_results[key] = result_dict
                            save_single_result(results_dir, key, result_dict)
                            
                            print(f"✅ MSE={bl_result['value_mse']:.6f} grad={bl_result['grad_mse']:.6f} ({elapsed:.1f}s)")
                            completed += 1
                            
                        except Exception as e:
                            elapsed = time.time() - t0
                            msg = f"BASELINE FAILED: {key}: {e}"
                            print(f"❌ ({elapsed:.1f}s) {msg}")
                            errors.append(msg)
                            traceback.print_exc()
    
    # ---- Hedging backtest (Tier 2 only, after NN training) ----
    if args.tier == 2 and args.hedging and not args.baselines_only:
        print("\n" + "=" * 70)
        print("HEDGING BACKTEST (Black-Scholes)")
        print("=" * 70)
        try:
            from dml_benchmark.finance.hedging import HedgingBacktest
            _run_hedging_experiment(results_dir, hparams, args.gpu)
        except Exception as e:
            msg = f"HEDGING FAILED: {e}"
            print(f"  ❌ {msg}")
            errors.append(msg)
            traceback.print_exc()
    
    # ---- Final Summary ----
    total_time = time.time() - total_start
    
    # Only count results from THIS tier for the summary
    tier_results = load_existing_results(results_dir)
    
    print("\n" + "=" * 70)
    print(f"TIER {args.tier} BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total time:     {total_time/3600:.1f}h ({total_time/60:.0f} min)")
    print(f"Completed:      {completed}")
    print(f"Skipped (resume): {skipped}")
    print(f"Errors:         {len(errors)}")
    print(f"Results dir:    {results_dir}")
    
    if errors:
        print("\nFAILED:")
        for e in errors[:20]:
            print(f"  ❌ {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
    
    # Per-method summary (this tier only)
    print(f"\n{'Method':<25} {'Mean Val MSE':>14} {'Mean Grad MSE':>14} {'Count':>6}")
    print("-" * 65)
    
    method_groups = {}
    for key, r in tier_results.items():
        m = r["method"]
        if m not in method_groups:
            method_groups[m] = {"val": [], "grad": []}
        method_groups[m]["val"].append(r["test_value_mse"])
        method_groups[m]["grad"].append(r["test_grad_mse"])
    
    for m in sorted(method_groups.keys()):
        vals = method_groups[m]["val"]
        grads = method_groups[m]["grad"]
        print(f"{m:<25} {np.mean(vals):>14.6f} {np.mean(grads):>14.6f} {len(vals):>6}")
    
    # Save summary
    save_summary(results_dir, {"tier": args.tier, "name": tier_name}, tier_results, errors, total_time)
    
    if len(errors) == 0:
        print(f"\n🟢 ALL {completed} EXPERIMENTS PASSED")
    else:
        print(f"\n🟡 DONE with {len(errors)} error(s)")
    
    return len(errors) == 0


def _run_hedging_experiment(results_dir, hparams, gpu):
    """Run hedging backtest — scaffolded, not implemented.

    I-L8 (2026-04-16): dead code after `raise NotImplementedError` removed.
    The previous 65+ lines used `result.model` (doesn't exist on TrainingResult)
    and passed `train_data` as `test_data` (data leakage). If this is ever
    implemented, rewrite from scratch against the current trainer API.
    """
    raise NotImplementedError(
        "Hedging backtest is not yet implemented. "
        "The finance.hedging module is scaffolded only. "
        "Remove the --hedging flag or implement the module first."
    )


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
