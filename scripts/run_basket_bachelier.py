#!/usr/bin/env python3
"""
Bachelier Basket Options — H&S 2020 canonical setup with analytical closed-form.

Cross-checked against the official Huge & Savine 2020 reference implementation
at `repos/differential-ml-eval/experiments/paper_examples.py` (the
differential-machine-learning GitHub notebooks' Bachelier class). Our setup
mirrors H&S's canonical scaling:

  - Initial spots S₀ = 1.0 per asset (all symmetric)
  - Strike K = 1.10 (10% OTM)
  - Basket volatility target bktVol = 0.2 (achieved by normalizing individual
    vols against a random correlation matrix)
  - Random weights a ~ U(1, 10) normalized to sum=1
  - Random individual vols ~ U(5, 50) normalized so basket vol == bktVol
  - Random PSD correlation matrix via Y^T Y + diag normalization (genCorrel)
  - Uniform test spots in dimension-adjusted range [0.5, 1.5] (H&S's testSet)

Divergence from H&S: we use ANALYTICAL closed-form prices/deltas as training
data (clean labels) rather than H&S's Monte Carlo antithetic simulation. This
is a deliberate benchmark-diagnostic choice: clean labels isolate the DML
effect without MC-noise confounds. H&S's MC setup is preserved in their
`Bachelier.trainingSet(anti=True)` method for reference.

Math (verified):
  B(0) = Σᵢ wᵢ Sᵢ(0)
  σ_B² = (w⊙σ)ᵀ ρ (w⊙σ) = Σᵢⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
  d = (B(0) − K) / (σ_B √T)
  Price: C = σ_B √T · [d · Φ(d) + φ(d)]
  Delta: ∂C/∂Sᵢ = wᵢ · Φ(d)

Audit: EVIDENCE/basket_audit.md (2026-04-16).

Usage:
  python scripts/run_basket_bachelier.py --gpu 0
  python scripts/run_basket_bachelier.py --gpu 0 --resume
  python scripts/run_basket_bachelier.py --dims 5 20 50 --seeds 42 123 456
"""

import sys
import os
import time
import json
import argparse
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dml_benchmark.trainer import train_single_experiment
from experiments.unified_comparison.run_unified_experiment import train_warmup

# Full 7-method grid to match cross-arch / cross-method parity in DML-Bench.
METHODS = [
    "vanilla",
    "dml_fixed",
    "dml_fixed_half",
    "dml_gradnorm",
    "dml_softmax_balance",
    "dml_relobralo",
    "dml_warmup",
]

TRAIN_HPARAMS = {
    "n_epochs": 1000,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
    "warmup_fraction": 0.5,
}
DEFAULT_SEEDS = [42, 123, 456, 789, 1000]
DEFAULT_DIMS = [5, 10, 20, 50]
N_TRAIN_DEFAULT = 8192
N_TEST_DEFAULT = 1024


# ============================================================================
# Bachelier closed-form (math verified against paper_examples.py:13-20)
# ============================================================================

def bach_price(basket_spot, K, basket_vol, T):
    """C = σ_B √T · [d Φ(d) + φ(d)]"""
    from scipy.stats import norm
    d = (basket_spot - K) / (basket_vol * np.sqrt(T))
    return basket_vol * np.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d))


def bach_delta(basket_spot, K, basket_vol, T):
    """∂C/∂B = Φ(d)"""
    from scipy.stats import norm
    d = (basket_spot - K) / (basket_vol * np.sqrt(T))
    return norm.cdf(d)


def gen_correl(n, rng):
    """Random PSD correlation matrix (H&S canonical, paper_examples.py:31-35).

    Builds C = Y^T Y from random Gaussian Y, then normalizes rows/cols to
    produce a proper correlation matrix (unit diagonal, bounded off-diagonal).
    """
    randoms = rng.uniform(low=-1.0, high=1.0, size=(2 * n, n))
    cov = randoms.T @ randoms
    inv_diag = np.diag(1.0 / np.sqrt(np.diagonal(cov)))
    return inv_diag @ cov @ inv_diag


def build_hs_bachelier_setup(d, bkt_vol, rng):
    """Mirror H&S's Bachelier.__init__ + trainingSet setup:
        S₀=1, K=1.10, random weights ~U(1,10), random vols ~U(5,50),
        random PSD correlation, vols normalized so basket vol == bkt_vol.
    """
    S0 = np.ones(d)                              # all assets start at 1.0
    corr = gen_correl(d, rng)                    # random PSD correlation
    a = rng.uniform(1.0, 10.0, size=d)
    a = a / a.sum()                              # normalized weights
    vols = rng.uniform(5.0, 50.0, size=d)
    # Normalize individual vols so basket vol == bkt_vol:
    #   avols = a ⊙ vols; basket_var = avols^T ρ avols; rescale s.t. √var = bkt_vol
    avols = (a * vols).reshape(-1, 1)
    v = float(np.sqrt((avols.T @ corr @ avols)[0, 0]))
    vols = vols * bkt_vol / v
    return {"S0": S0, "K": 1.10, "T": 1.0, "weights": a, "vols": vols,
            "corr": corr, "bkt_vol": bkt_vol}


# ============================================================================
# Data generation (analytical closed-form; uniform test-spot sampling per H&S)
# ============================================================================

def generate_basket_data(d: int, n_train: int, n_test: int, seed: int,
                          bkt_vol: float = 0.2):
    """Generate DML-Bench basket data under H&S canonical setup.

    Train + test uses the SAME analytical closed-form (clean labels).
    Test spots are uniform in [adj_lower, adj_upper] per H&S's testSet
    dimension adjustment (paper_examples.py:122-136).
    """
    rng = np.random.RandomState(seed)
    setup = build_hs_bachelier_setup(d, bkt_vol, rng)

    # H&S's testSet domain adjustment to scale with d (paper_examples.py:125-128).
    lower, upper = 0.5, 1.5
    adj = 1 + 0.5 * np.sqrt((d - 1) * (upper - lower) / 12.0)
    adj_lower = 1.0 - (1.0 - lower) * adj
    adj_upper = 1.0 + (upper - 1.0) * adj

    # Draw train and test spots uniformly per asset
    total = n_train + n_test
    spots = rng.uniform(adj_lower, adj_upper, size=(total, d))

    basket_spot = spots @ setup["weights"]      # (total,)
    prices = bach_price(basket_spot, setup["K"], setup["bkt_vol"], setup["T"])
    prices = prices.reshape(total, 1)
    # N_d = Φ(d) (CDF). Avoid using `phi_d` which per convention is the PDF.
    N_d = bach_delta(basket_spot, setup["K"], setup["bkt_vol"], setup["T"])
    deltas = N_d[:, None] * setup["weights"][None, :]
    deltas = deltas.reshape(total, 1, d)

    # Split train/test after sampling (spots drawn from same distribution)
    # Use seed-based deterministic order: first n_train = train, rest = test.
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, total)

    metadata = {
        "dataset": "bachelier_basket",
        "d": d, "K": setup["K"], "T": setup["T"],
        "bkt_vol": setup["bkt_vol"], "n_samples": total,
        "n_train": n_train, "n_test": n_test,
        "sampling": "uniform_adjusted",
        "weights_scheme": "random_U1_10_normalized",
        "vols_scheme": f"random_U5_50_normalized_to_bkt_vol={bkt_vol}",
        "correlation_scheme": "random_PSD_via_Y^T Y",
        "training_labels": "analytical_closed_form",
    }

    return {
        "x_train": spots[train_idx].astype(np.float32),
        "y_train": prices[train_idx].astype(np.float32),
        "dydx_train": deltas[train_idx].astype(np.float32),
        "x_test": spots[test_idx].astype(np.float32),
        "y_test": prices[test_idx].astype(np.float32),
        "dydx_test": deltas[test_idx].astype(np.float32),
        "metadata": metadata,
    }


# ============================================================================
# Result persistence
# ============================================================================

def make_key(d, method, seed):
    return f"basket_d{d}_s{seed}_{method}"


def load_existing(results_dir):
    existing = set()
    if results_dir.exists():
        for f in results_dir.glob("basket_d*_s*_*.json"):
            existing.add(f.stem)
    return existing


def save_result(results_dir, key, result_dict):
    result_dict["key"] = key
    out = results_dir / f"{key}.json"
    with open(out, "w") as fh:
        json.dump(result_dict, fh, indent=2, default=str)


# ============================================================================
# Runner
# ============================================================================

def run_one(d, method, seed, hparams, n_train, n_test):
    data = generate_basket_data(d, n_train=n_train, n_test=n_test, seed=seed)

    # I-H8 (2026-04-16): use an IDENTICAL 80/20 train/val split for ALL methods.
    # Previously warmup used a 90/10 carve while non-warmup methods used the
    # trainer's internal 80/20, giving warmup ~12.5% more training data per run.
    rng = np.random.RandomState(seed + 1)
    perm = rng.permutation(n_train)
    n_val = max(1, int(n_train * 0.2))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_tr = data["x_train"][tr_idx]; y_tr = data["y_train"][tr_idx]; dy_tr = data["dydx_train"][tr_idx]
    x_va = data["x_train"][val_idx]; y_va = data["y_train"][val_idx]; dy_va = data["dydx_train"][val_idx]

    if method == "dml_warmup":
        res = train_warmup(
            x_train=x_tr, y_train=y_tr, dydx_train=dy_tr,
            x_val=x_va, y_val=y_va, dydx_val=dy_va,
            x_test=data["x_test"], y_test=data["y_test"],
            dydx_test=data["dydx_test"],
            warmup_fraction=hparams.get("warmup_fraction", 0.5),
            seed=seed, pbar=False,
            n_epochs=hparams["n_epochs"],
            batch_size=hparams["batch_size"],
            lr=hparams["lr"],
            n_layers=hparams["n_layers"],
            hidden_size=hparams["hidden_size"],
            activation=hparams["activation"],
            max_grad_norm=hparams["max_grad_norm"],
            scheduler_patience=hparams["scheduler_patience"],
            scheduler_factor=hparams["scheduler_factor"],
        )
    else:
        res = train_single_experiment(
            x_train=x_tr, y_train=y_tr, dydx_train=dy_tr,
            x_val=x_va, y_val=y_va, dydx_val=dy_va,
            x_test=data["x_test"], y_test=data["y_test"],
            dydx_test=data["dydx_test"],
            method=method, seed=seed, pbar=False,
            n_epochs=hparams["n_epochs"],
            batch_size=hparams["batch_size"],
            lr=hparams["lr"],
            n_layers=hparams["n_layers"],
            hidden_size=hparams["hidden_size"],
            activation=hparams["activation"],
            lambda_=hparams["lambda_"],
            max_grad_norm=hparams["max_grad_norm"],
            scheduler_patience=hparams["scheduler_patience"],
            scheduler_factor=hparams["scheduler_factor"],
        )
    return {
        "method": method,
        "dataset": f"basket_d{d}",
        "d": d,
        "seed": seed,
        "test_value_mse": float(res.test_value_mse),
        "test_grad_mse": float(res.test_grad_mse),
        "best_epoch": int(res.best_epoch) if res.best_epoch is not None else -1,
        "time_s": float(res.total_time_s) if hasattr(res, "total_time_s") else 0.0,
        "n_epochs_actual": len(res.training_logs) if getattr(res, "training_logs", None) else 0,
        "metadata": data["metadata"],
        "hparams": dict(hparams),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--n_train", type=int, default=N_TRAIN_DEFAULT)
    parser.add_argument("--n_test", type=int, default=N_TEST_DEFAULT)
    parser.add_argument("--n_epochs", type=int, default=TRAIN_HPARAMS["n_epochs"])
    parser.add_argument("--results_dir", default="results/basket_bachelier")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(4)

    hparams = dict(TRAIN_HPARAMS)
    hparams["n_epochs"] = args.n_epochs

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(results_dir) if args.resume else set()

    total = len(args.dims) * len(args.seeds) * len(args.methods)
    print("=" * 70)
    print("Basket (H&S Bachelier canonical) — DML-Bench")
    print(f"Dims: {args.dims}")
    print(f"Seeds: {args.seeds}")
    print(f"Methods: {args.methods}")
    print(f"n_train={args.n_train}, n_test={args.n_test}, n_epochs={args.n_epochs}")
    print(f"Total: {total} runs → {results_dir}")
    print("=" * 70)

    n_done = n_failed = n_skipped = 0
    for d in args.dims:
        for seed in args.seeds:
            for method in args.methods:
                key = make_key(d, method, seed)
                if args.resume and key in existing:
                    n_skipped += 1
                    continue
                print(f"\n--- {key} ---")
                t0 = time.time()
                try:
                    res = run_one(d, method, seed, hparams,
                                   args.n_train, args.n_test)
                    save_result(results_dir, key, res)
                    n_done += 1
                    print(f"  OK ({time.time()-t0:.1f}s)  val={res['test_value_mse']:.4e}  "
                          f"grad={res['test_grad_mse']:.4e}  best_epoch={res['best_epoch']}")
                except Exception as e:
                    n_failed += 1
                    print(f"  FAIL: {e}")
                    traceback.print_exc()

    print(f"\nDone. {n_done} OK, {n_failed} failed, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
