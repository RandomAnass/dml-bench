#!/usr/bin/env python3
"""
Heston barrier (down-and-out call) — pilot run with 1 seed, all methods.

The single-fresh-experiment block of the Heston extension. Combines our
existing 11-method discontinuous-payoff panel with two new label paradigms
(ρ-corrected single-step LRM, fuzzy-on-Heston-barrier) on the Heston SV
model — a cell that G&K v2 explicitly does NOT cover (their §3.4 is BS-only
barrier; their §3.6 is Heston-only digital).

Setup follows G&K v2 §3.4 (barrier with single intermediate check) generalised
to Heston-Euler full-truncation per §3.6:
- spot in [0.7, 1.3], K=1, B=0.85
- T1 = 1/3 (barrier check), T2 = 2/3 (expiry)
- Heston: V_0 = θ = 0.04, κ = 1, σ_v = 0.15, ρ = -0.7, r = 0
- m = 1024 training spots, k = 10 paths per spot
- ground truth: 100k-path MC reference (full-truncation Euler)

Methods (13 candidates; pilot runs 1 seed each, prunes after based on
predicted_performance.md decision tree):
    1.  vanilla
    2.  dml_fixed         (Pathwise + λ=1)
    3.  dml_gradnorm      (Pathwise + GradNorm)
    4.  dml_relobralo     (Pathwise + ReLoBRaLo)
    5.  dml_warmup        (Pathwise + Warmup)
    6.  dml_lrm_fixed     (LRM single-step ρ-corrected + λ=1)        [NEW Heston barrier]
    7.  dml_lrm_gradnorm  (LRM single-step + GradNorm)               [NEW]
    8.  dml_lrm_warmup    (LRM single-step + Warmup)                 [NEW]
    9.  dml_fuzzy_fixed   (Fuzzy + λ=1)                              [NEW Heston barrier]
    10. dml_fuzzy_warmup  (Fuzzy + Warmup)                           [NEW]

Drops (from the original 13-method plan):
- dml_fixed_half (redundant; existing data covers λ-sensitivity)
- dml_softmax_balance (redundant with ReLoBRaLo)
- multi-step LRM (Chen-Glasserman is degenerate for ∂/∂S_0 — see
  docs/heston_extension/multistep_lrm_derivation.md)

Outputs to results/heston_barrier_4way/pilot/. JSON schema matches
results/lrm_comparison/.

Usage:
    python experiments/heston_barrier_4way/run_pilot.py --gpu 0
    python experiments/heston_barrier_4way/run_pilot.py --gpu 0 --resume
    python experiments/heston_barrier_4way/run_pilot.py --analyze-only

Expected runtime: ~10-15 min on 1 GPU (10 methods × 1 seed × ~1-2 min/run).

References:
    - Glasserman, P., and S. H. Karmarkar (2025/2026). Differential ML with
      a Difference. arXiv:2512.05301 v2 §3.4, §3.6.
    - Heston, S. L. (1993). RFS 6(2), 327-343.
    - Andersen, L. (2008). JCF 11(3), 1-42 (full-truncation Euler).
    - Savine, A. (2018/2024). Modern Computational Finance: Fuzzy Payoff Eval.
    - Derivation notes: docs/heston_extension/heston_lrm_score_derivation.md.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dml_benchmark.trainer import train_single_experiment
from dml_benchmark.lrm_labels import (
    lrm_barrier_heston,
    lrm_multistep_heston_barrier,
    bel_barrier_heston,
)
from dml_benchmark.lrm_labels import prepare_for_training as prepare_lrm
from dml_benchmark.fuzzy_smoothing import fuzzy_barrier_heston
from dml_benchmark.fuzzy_smoothing import prepare_for_training as prepare_fuzzy
from dml_benchmark.high_fidelity_references import heston_barrier_doc_mc_reference


# ============================================================================
# CONFIGURATION
# ============================================================================

METHODS_PATHWISE = ["vanilla", "dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"]
METHODS_LRM = ["dml_lrm_fixed", "dml_lrm_gradnorm", "dml_lrm_warmup"]
METHODS_LRM_MULTISTEP = ["dml_lrm_multistep_fixed", "dml_lrm_multistep_warmup"]
METHODS_FUZZY = ["dml_fuzzy_fixed", "dml_fuzzy_warmup"]
METHODS_BEL = ["dml_bel_fixed", "dml_bel_warmup"]
ALL_METHODS = (METHODS_PATHWISE + METHODS_LRM + METHODS_LRM_MULTISTEP
               + METHODS_FUZZY + METHODS_BEL)

# Explicit map: pilot method name -> trainer balancer name (used by
# train_single_experiment to dispatch the loss/balancer logic). Replaces
# fragile str.split-based parsing (code-review LOW #2).
TRAINER_METHOD_MAP = {
    # Pathwise (label paradigm dispatched via generate_data; trainer just
    # reads the method to pick the balancer)
    "vanilla":           "vanilla",
    "dml_fixed":         "dml_fixed",
    "dml_gradnorm":      "dml_gradnorm",
    "dml_relobralo":     "dml_relobralo",
    "dml_warmup":        "dml_warmup",
    # LRM single-step
    "dml_lrm_fixed":     "dml_fixed",
    "dml_lrm_gradnorm":  "dml_gradnorm",
    "dml_lrm_warmup":    "dml_warmup",
    # LRM multi-step (Chen-Glasserman 2007)
    "dml_lrm_multistep_fixed":  "dml_fixed",
    "dml_lrm_multistep_warmup": "dml_warmup",
    # Fuzzy (Savine call-spread)
    "dml_fuzzy_fixed":   "dml_fixed",
    "dml_fuzzy_warmup":  "dml_warmup",
    # BEL (Fournié-localised Malliavin)
    "dml_bel_fixed":     "dml_fixed",
    "dml_bel_warmup":    "dml_warmup",
}

# Heston barrier setup (matches G&K v2 §3.4 + §3.6 generalisation)
HESTON_PARAMS = {
    "strike": 1.0,
    "barrier": 0.85,
    "v0": 0.04,
    "kappa": 1.0,
    "theta": 0.04,
    "sigma_v": 0.15,
    "rho": -0.7,
    "r": 0.0,
    "T1": 1.0 / 3.0,
    "T2": 2.0 / 3.0,
    "n_substeps_to_T1": 84,
    "n_substeps_T1_to_T2": 84,
}

# Training hyperparameters (consistent with existing benchmark)
HPARAMS = {
    "n_epochs": 500,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "lambda_": 1.0,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 50,  # explicit ES patience (was implicit before)
}

N_SAMPLES = 1024
K_PATHS = 10
N_REF_PATHS = 100_000
SEEDS = [42]  # Pilot: single seed

# Spot range multipliers (default = our original [0.7K, 1.3K]; runners can
# override e.g. for G&K-style [0.5K, 1.5K] wider-range setup).
SPOT_LOW_MULT = 0.7
SPOT_HIGH_MULT = 1.3

# Method-independent ground truth test set: we evaluate ALL methods on the
# SAME (x_test, y_test, dydx_test) drawn from a 100k-path MC reference, NOT
# on each method's own (biased) noisy labels. Without this, pathwise's
# missing-Dirac bias is structurally undetectable on the test set.
# See repos/docs/heston_extension/gk_barrier_discrepancy_investigation.md.
GROUND_TRUTH_N_TEST = 200
GROUND_TRUTH_N_PATHS = 100_000
GROUND_TRUTH_SEED = 999  # disjoint from training seeds


# ============================================================================
# UTILITIES
# ============================================================================

def make_key(method: str, seed: int) -> str:
    return f"heston_barrier_doc_{method}_s{seed}"


def load_existing(results_dir: Path) -> dict:
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            if f.name in ("summary.json", "analysis.json", "ground_truth.json"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    existing[data.get("key", f.stem)] = data
            except Exception:
                pass
    return existing


def save_result(results_dir: Path, key: str, result_dict: dict):
    result_dict["key"] = key
    result_dict["timestamp"] = datetime.now().isoformat()
    path = results_dir / f"{key}.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp_path.rename(path)


def generate_data(method: str, seed: int):
    """Generate (x, y, dydx) labels per the method's paradigm."""
    if method in METHODS_PATHWISE:
        # For pathwise on Heston barrier, we use the analytical pathwise
        # (which ignores the barrier indicator's discontinuity — biased on purpose).
        # Generate via the LRM function but replace the score with pathwise.
        # We treat pathwise as: dydx = 1{S_T2 > K} * (S_T2/S_0) * survival_indicator
        # — i.e., AAD through the discontinuous payoff (Dirac at barrier missed).
        data = _pathwise_barrier_heston(seed=seed)
    elif method in METHODS_LRM:
        data = lrm_barrier_heston(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            **HESTON_PARAMS,
        )
    elif method in METHODS_LRM_MULTISTEP:
        data = lrm_multistep_heston_barrier(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            **HESTON_PARAMS,
        )
    elif method in METHODS_FUZZY:
        data = fuzzy_barrier_heston(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            **HESTON_PARAMS,
        )
    elif method in METHODS_BEL:
        data = bel_barrier_heston(
            n_samples=N_SAMPLES, k_paths=K_PATHS, seed=seed,
            spot_low_mult=SPOT_LOW_MULT, spot_high_mult=SPOT_HIGH_MULT,
            **HESTON_PARAMS,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return data


def _pathwise_barrier_heston(seed: int) -> dict:
    """
    Pathwise (AAD) labels for Heston barrier — biased on the indicator.
    The standard Huge-Savine pathwise approach: differentiate through the
    payoff, ignoring the Dirac at the barrier.

    Spot range honors module-level SPOT_LOW_MULT/SPOT_HIGH_MULT (default
    0.7-1.3 = our original setup; runner can override for wider-range runs).
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(
        HESTON_PARAMS["strike"] * SPOT_LOW_MULT,
        HESTON_PARAMS["strike"] * SPOT_HIGH_MULT,
        (N_SAMPLES, 1)
    )
    dt1 = HESTON_PARAMS["T1"] / HESTON_PARAMS["n_substeps_to_T1"]
    dt2 = (HESTON_PARAMS["T2"] - HESTON_PARAMS["T1"]) / HESTON_PARAMS["n_substeps_T1_to_T2"]
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-HESTON_PARAMS["r"] * HESTON_PARAMS["T2"])

    y_all = np.zeros((N_SAMPLES, K_PATHS))
    dydx_all = np.zeros((N_SAMPLES, K_PATHS))

    for p in range(K_PATHS):
        log_S = np.log(S0.flatten())
        v = np.full(N_SAMPLES, HESTON_PARAMS["v0"])

        for step in range(HESTON_PARAMS["n_substeps_to_T1"]):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(N_SAMPLES)
            Z_indep = rng.standard_normal(N_SAMPLES)
            Z2 = HESTON_PARAMS["rho"] * Z1 + np.sqrt(1.0 - HESTON_PARAMS["rho"] ** 2) * Z_indep
            log_S = log_S + (HESTON_PARAMS["r"] - 0.5 * v_pos) * dt1 + sqrt_v * sqrt_dt1 * Z1
            v = v + HESTON_PARAMS["kappa"] * (HESTON_PARAMS["theta"] - v_pos) * dt1 \
                + HESTON_PARAMS["sigma_v"] * sqrt_v * sqrt_dt1 * Z2

        S_T1 = np.exp(log_S)
        alive = (S_T1 > HESTON_PARAMS["barrier"]).astype(np.float64)

        for step in range(HESTON_PARAMS["n_substeps_T1_to_T2"]):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(N_SAMPLES)
            Z_indep = rng.standard_normal(N_SAMPLES)
            Z2 = HESTON_PARAMS["rho"] * Z1 + np.sqrt(1.0 - HESTON_PARAMS["rho"] ** 2) * Z_indep
            log_S = log_S + (HESTON_PARAMS["r"] - 0.5 * v_pos) * dt2 + sqrt_v * sqrt_dt2 * Z1
            v = v + HESTON_PARAMS["kappa"] * (HESTON_PARAMS["theta"] - v_pos) * dt2 \
                + HESTON_PARAMS["sigma_v"] * sqrt_v * sqrt_dt2 * Z2

        S_T2 = np.exp(log_S)
        call_payoff = np.maximum(S_T2 - HESTON_PARAMS["strike"], 0.0)
        call_indicator = (S_T2 > HESTON_PARAMS["strike"]).astype(np.float64)

        S0_flat = S0.flatten()
        # Pathwise delta: dπ/dS_0 = 1{alive} * 1{S_T2 > K} * (S_T2 / S_0) * discount
        # NOTE: this misses the Dirac at the barrier — known bias for discontinuous payoffs
        pathwise_delta = alive * call_indicator * (S_T2 / S0_flat) * discount
        payoff = call_payoff * alive * discount

        y_all[:, p] = payoff
        dydx_all[:, p] = pathwise_delta

    y = y_all.mean(axis=1, keepdims=True)
    dydx_pw = dydx_all.mean(axis=1).reshape(N_SAMPLES, 1, 1)

    return {
        "x": S0,
        "y": y,
        "dydx_pw": dydx_pw,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "heston_full_truncation_euler",
            "label_method": "pathwise",
            **HESTON_PARAMS,
            "k_paths": K_PATHS,
            "n_samples": N_SAMPLES,
            "seed": seed,
        },
    }


def prepare_data_dict(data: dict, method: str, seed: int, ground_truth: dict) -> dict:
    """
    Convert generator output to train_single_experiment-compatible dict.

    CRITICAL: training labels are method-specific (each method uses its own
    pathwise/LRM/fuzzy estimator); test labels come from the
    method-INDEPENDENT 100k-path MC reference (`ground_truth` arg). Without
    the latter, each method is evaluated against its own biased estimator,
    which structurally hides pathwise's missing-Dirac bias. See
    repos/docs/heston_extension/gk_barrier_discrepancy_investigation.md.
    """
    if method in METHODS_PATHWISE:
        dydx_key = "dydx_pw"
    elif method in METHODS_LRM or method in METHODS_LRM_MULTISTEP or method in METHODS_BEL:
        dydx_key = "dydx_lrm"
    elif method in METHODS_FUZZY:
        dydx_key = "dydx_fuzzy"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Train: ALL of `data` (no train/test split needed since test comes
    # from the disjoint ground-truth grid). This gives us more training data.
    return {
        "x_train": data["x"],
        "y_train": data["y"],
        "dydx_train": data[dydx_key],
        "x_test": ground_truth["x"],
        "y_test": ground_truth["y"],
        "dydx_test": ground_truth["dydx"],
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_pilot(results_dir: Path, existing: dict, resume: bool):
    print("\n" + "=" * 70)
    print("HESTON BARRIER (DOWN-AND-OUT CALL) — PILOT (1 SEED)")
    print("=" * 70)
    print(f"Methods:  {len(ALL_METHODS)} ({', '.join(ALL_METHODS)})")
    print(f"Seeds:    {SEEDS}")
    print(f"Heston:   ρ={HESTON_PARAMS['rho']}, V_0={HESTON_PARAMS['v0']}, "
          f"T_1={HESTON_PARAMS['T1']:.3f}, T_2={HESTON_PARAMS['T2']:.3f}")
    print(f"Output:   {results_dir}")

    # Generate method-INDEPENDENT ground truth for test evaluation
    print(f"\nGenerating ground truth: {GROUND_TRUTH_N_TEST} spots × "
          f"{GROUND_TRUTH_N_PATHS} MC paths (full-truncation Euler)...")
    t_gt = time.time()
    rng_gt = np.random.RandomState(GROUND_TRUTH_SEED)
    S0_test = rng_gt.uniform(
        HESTON_PARAMS["strike"] * SPOT_LOW_MULT,
        HESTON_PARAMS["strike"] * SPOT_HIGH_MULT,
        GROUND_TRUTH_N_TEST,
    )
    ground_truth = heston_barrier_doc_mc_reference(
        S0=S0_test,
        n_paths=GROUND_TRUTH_N_PATHS,
        seed=GROUND_TRUTH_SEED,
        **HESTON_PARAMS,
    )
    print(f"  Ground truth ready in {time.time() - t_gt:.1f}s. "
          f"Mean SE_price={ground_truth['std_err_price'].mean():.2e}, "
          f"mean SE_delta={ground_truth['std_err_delta'].mean():.2e}")

    results = {}

    for seed_idx, seed in enumerate(SEEDS):
        for method_idx, method in enumerate(ALL_METHODS):
            key = make_key(method, seed)

            if resume and key in existing:
                print(f"  SKIP (exists): {method} seed={seed}")
                results[key] = existing[key]
                continue

            print(f"\n--- {method} seed={seed} "
                  f"({method_idx + 1}/{len(ALL_METHODS)}) ---")

            t_gen = time.time()
            data = generate_data(method, seed)
            data_split = prepare_data_dict(data, method, seed, ground_truth)
            gen_time = time.time() - t_gen
            print(f"  Data gen: {gen_time:.1f}s")

            t0 = time.time()
            try:
                # Explicit dispatch (code-review LOW #2): str.split with
                # positional indices was fragile to renames. Map every method
                # name to its trainer balancer explicitly.
                trainer_method = TRAINER_METHOD_MAP.get(method)
                if trainer_method is None:
                    raise ValueError(
                        f"Unknown method {method!r}; add it to "
                        f"TRAINER_METHOD_MAP at the top of this file."
                    )

                result = train_single_experiment(
                    x_train=data_split["x_train"],
                    y_train=data_split["y_train"],
                    dydx_train=data_split["dydx_train"],
                    x_test=data_split["x_test"],
                    y_test=data_split["y_test"],
                    dydx_test=data_split["dydx_test"],
                    method=trainer_method,
                    seed=seed,
                    pbar=False,
                    **HPARAMS,
                )
                elapsed = time.time() - t0

                result_dict = {
                    "method": method,
                    "trainer_method": trainer_method,
                    "seed": seed,
                    "payoff": "barrier_doc_call",
                    "model": "heston_full_truncation_euler",
                    "n_samples": N_SAMPLES,
                    "k_paths": K_PATHS,
                    "test_value_mse": float(result.test_value_mse),
                    "test_grad_mse": float(result.test_grad_mse),
                    "best_epoch": int(result.best_epoch),
                    "time_s": round(elapsed, 2),
                    "data_gen_s": round(gen_time, 2),
                    "hparams": dict(HPARAMS),
                    "heston_params": dict(HESTON_PARAMS),
                }

                save_result(results_dir, key, result_dict)
                results[key] = result_dict

                print(f"  val_mse={result.test_value_mse:.6e}, "
                      f"grad_mse={result.test_grad_mse:.6e}, "
                      f"ep={result.best_epoch}, t={elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED ({elapsed:.1f}s): {e}")
                traceback.print_exc()

    return results


def analyze(results_dir: Path):
    existing = load_existing(results_dir)
    if not existing:
        print("No results found.")
        return

    print("\n" + "=" * 80)
    print("HESTON BARRIER PILOT — RESULTS")
    print("=" * 80)
    print(f"  {'Method':<25} {'val_mse':>12} {'grad_mse':>12} {'best_ep':>8} {'time_s':>8}")
    print("  " + "-" * 75)

    by_method = {}
    for key, res in existing.items():
        method = res.get("method", "?")
        by_method.setdefault(method, []).append(res)

    for method in ALL_METHODS:
        if method not in by_method:
            continue
        vals = by_method[method]
        v = vals[0]  # 1 seed in pilot
        print(f"  {method:<25} {v['test_value_mse']:12.4e} "
              f"{v['test_grad_mse']:12.4e} {v['best_epoch']:>8} {v['time_s']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Heston barrier pilot")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results_dir = Path("results/heston_barrier_4way/pilot")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analyze(results_dir)
        return

    existing = load_existing(results_dir) if args.resume else {}
    run_pilot(results_dir, existing, args.resume)
    analyze(results_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
