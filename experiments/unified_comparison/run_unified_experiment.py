#!/usr/bin/env python3
"""
Unified Discontinuous-Payoff Experiment: All Methods Side-by-Side.

Methods compared (11 total):
  Pathwise labels (biased ≡ 0 for digital payoffs):
    1. vanilla            — Value-only, no derivative supervision
    2. dml_fixed          — DML λ=1 with pathwise labels (Huge & Savine 2020)
    3. dml_gradnorm       — GradNorm adaptive balancing with pathwise labels
    4. dml_relobralo      — ReLoBRaLo adaptive balancing with pathwise labels
    5. dml_warmup         — Vanilla warmup → DML GradNorm with pathwise labels
  LRM labels (Glasserman & Karmarkar 2025):
    6. dml_lrm            — DML λ=1 with LRM labels
    7. dml_gradnorm_lrm   — GradNorm with LRM labels
    8. dml_warmup_lrm     — Warmup → DML GradNorm with LRM labels
  Fuzzy-smoothed labels (Savine 2018 + ours):
    9. dml_fuzzy           — DML λ=1 with fuzzy call-spread labels
   10. dml_gradnorm_fuzzy  — GradNorm with fuzzy labels
   11. dml_warmup_fuzzy    — Warmup → DML GradNorm with fuzzy labels (NOVEL)

Datasets:
  A. Digital BS          (1D, clean theory case — G&K's showcase)
  B. Barrier BS          (1D, multi-step — noisy labels)
  C. Heston-Euler digital (1D, stochastic vol — where LRM variance explodes)
  D. Basket digital      (multi-dim: d=1,7 — scaling test)

Process (per user directive):
  smoke_test → single_seed → multi_seed

Usage:
  python experiments/unified_comparison/run_unified_experiment.py --mode smoke_test --gpu 0
  python experiments/unified_comparison/run_unified_experiment.py --mode single_seed --gpu 0
  python experiments/unified_comparison/run_unified_experiment.py --mode multi_seed --gpu 0
  python experiments/unified_comparison/run_unified_experiment.py --analyze-only
"""

import sys
import os
import time
import json
import copy
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from dml_benchmark.trainer import train_single_experiment, set_deterministic
from dml_benchmark.model import (
    DmlFeedForward, DmlLoss, VanillaLoss, DmlDataset,
    DataNormalizer, LossComponents, get_device
)
from dml_benchmark.loss_balancing import GradNormDmlLoss
from dml_benchmark.trainer import DmlTrainer, create_data_loaders, TrainingResult

# Data generators
from dml_benchmark.lrm_labels import (
    lrm_digital_bs,
    lrm_barrier_bs,
    lrm_basket_bachelier,
    lrm_euler_heston,
    prepare_for_training as lrm_prepare,
)
from dml_benchmark.fuzzy_smoothing import (
    fuzzy_digital_bs,
    fuzzy_barrier_bs,
    fuzzy_basket_bachelier,
    fuzzy_euler_heston,
    prepare_for_training as fuzzy_prepare,
)
from dml_benchmark.high_fidelity_references import (
    barrier_bs_analytical_delta,
    heston_digital_cos_delta,
    basket_high_k_lrm_delta,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("results/unified_comparison")

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999]

# Shared hyperparameters (consistent across methods for fair comparison)
HPARAMS = {
    "n_epochs": 500,
    "batch_size": 256,
    "n_layers": 4,
    "hidden_size": 256,
    "activation": "softplus",
    "lr": 0.005,
    "max_grad_norm": 1.0,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
}

# Smoke test uses fewer epochs/samples
SMOKE_HPARAMS = {
    **HPARAMS,
    "n_epochs": 50,
}

# Data generation parameters
DATA_CONFIG = {
    "digital_bs": {
        "n_samples": 2048,
        "k_paths": 10,
        "strike": 100.0,
        "vol": 0.2,
        "r": 0.05,
        "T": 1.0,
    },
    "barrier_bs": {
        "n_samples": 2048,
        "k_paths": 10,
        "strike": 100.0,
        "barrier": 80.0,
        "vol": 0.2,
        "r": 0.05,
        "T": 1.0,
        "n_steps": 252,
    },
    "heston_digital": {
        "n_samples": 1024,
        "k_paths": 10,
        "strike": 100.0,
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma_v": 0.3,
        "rho": -0.7,
        "r": 0.05,
        "T": 1.0,
        "n_steps": 50,
    },
    "basket_d1": {
        "n_samples": 2048,
        "d": 1,
        "k_paths": 10,
        "strike": 100.0,
        "base_vol": 20.0,
        "T": 1.0,
        "rho": 0.5,
    },
    "basket_d7": {
        "n_samples": 4096,
        "d": 7,
        "k_paths": 10,
        "strike": 100.0,
        "base_vol": 20.0,
        "T": 1.0,
        "rho": 0.5,
    },
}

# Fuzzy smoothing ε multipliers to test
EPS_MULTS = [0.5]

# Warmup fraction
WARMUP_FRACTION = 0.5

# Methods to run (11 total: 5 pathwise + 3 LRM + 3 fuzzy)
ALL_METHODS = [
    # Pathwise labels (biased — ≡ 0 for digital payoffs)
    "vanilla",
    "dml_fixed",
    "dml_gradnorm",
    "dml_relobralo",
    "dml_warmup",
    # LRM labels (Glasserman & Karmarkar 2025)
    "dml_lrm",
    "dml_gradnorm_lrm",
    "dml_warmup_lrm",
    # Fuzzy-smoothed labels (Savine 2018 + ours)
    "dml_fuzzy",
    "dml_gradnorm_fuzzy",
    "dml_warmup_fuzzy",
]


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_dataset(dataset_name: str, seed: int, eps_mult: Optional[float] = None):
    """
    Generate data for a given dataset with ALL label types.

    Args:
        dataset_name: One of digital_bs / barrier_bs / heston_digital / basket_d{k}
        seed: RNG seed for data generation (LRM k_paths MC + fuzzy eps calibration).
        eps_mult: Optional fuzzy-smoothing bandwidth multiplier. When None, uses
            EPS_MULTS[0] (the canonical choice for main-paper runs). When passed,
            threads into the fuzzy_* generator so P7-style ε sweeps work. This
            is ONLY applied to the fuzzy branch; LRM labels are independent of
            eps_mult. See fuzzy_smoothing.calibrate_epsilon for the formula.

    Returns: dict with keys like 'x_train', 'y_train', 'dydx_lrm_train',
             'dydx_fuzzy_train', 'x_test', 'y_test', 'dydx_lrm_test',
             'dydx_fuzzy_test', 'dydx_exact_test', etc.
    """
    cfg = DATA_CONFIG[dataset_name]
    _eps = EPS_MULTS[0] if eps_mult is None else float(eps_mult)

    if dataset_name == "digital_bs":
        # LRM labels
        lrm_data = lrm_digital_bs(
            n_samples=cfg["n_samples"], strike=cfg["strike"], vol=cfg["vol"],
            r=cfg["r"], T=cfg["T"], k_paths=cfg["k_paths"], seed=seed,
            return_pathwise=True,
        )
        # Fuzzy labels
        fuzzy_data = fuzzy_digital_bs(
            n_samples=cfg["n_samples"], strike=cfg["strike"], vol=cfg["vol"],
            r=cfg["r"], T=cfg["T"], k_paths=cfg["k_paths"],
            eps_mult=_eps, seed=seed,
        )
        return _combine_data(lrm_data, fuzzy_data, seed, dataset_name)

    elif dataset_name == "barrier_bs":
        lrm_data = lrm_barrier_bs(
            n_samples=cfg["n_samples"], strike=cfg["strike"], barrier=cfg["barrier"],
            vol=cfg["vol"], r=cfg["r"], T=cfg["T"],
            n_steps=cfg["n_steps"], k_paths=cfg["k_paths"], seed=seed,
        )
        fuzzy_data = fuzzy_barrier_bs(
            n_samples=cfg["n_samples"], strike=cfg["strike"], barrier=cfg["barrier"],
            vol=cfg["vol"], r=cfg["r"], T=cfg["T"],
            n_steps=cfg["n_steps"], k_paths=cfg["k_paths"],
            eps_mult=_eps, seed=seed,
        )
        return _combine_data(lrm_data, fuzzy_data, seed, dataset_name)

    elif dataset_name == "heston_digital":
        lrm_data = lrm_euler_heston(
            n_samples=cfg["n_samples"], strike=cfg["strike"],
            v0=cfg["v0"], kappa=cfg["kappa"], theta=cfg["theta"],
            sigma_v=cfg["sigma_v"], rho=cfg["rho"], r=cfg["r"], T=cfg["T"],
            n_steps=cfg["n_steps"], k_paths=cfg["k_paths"],
            payoff_type="digital", seed=seed,
        )
        fuzzy_data = fuzzy_euler_heston(
            n_samples=cfg["n_samples"], strike=cfg["strike"],
            v0=cfg["v0"], kappa=cfg["kappa"], theta=cfg["theta"],
            sigma_v=cfg["sigma_v"], rho=cfg["rho"], r=cfg["r"], T=cfg["T"],
            n_steps=cfg["n_steps"], k_paths=cfg["k_paths"],
            payoff_type="digital", eps_mult=_eps, seed=seed,
        )
        return _combine_data(lrm_data, fuzzy_data, seed, dataset_name)

    elif dataset_name.startswith("basket_d"):
        d = cfg["d"]
        lrm_data = lrm_basket_bachelier(
            n_samples=cfg["n_samples"], d=d, strike=cfg["strike"],
            base_vol=cfg["base_vol"], T=cfg["T"], rho=cfg["rho"],
            k_paths=cfg["k_paths"], seed=seed,
        )
        fuzzy_data = fuzzy_basket_bachelier(
            n_samples=cfg["n_samples"], d=d, strike=cfg["strike"],
            base_vol=cfg["base_vol"], T=cfg["T"], rho=cfg["rho"],
            k_paths=cfg["k_paths"], eps_mult=_eps, seed=seed,
        )
        return _combine_data(lrm_data, fuzzy_data, seed, dataset_name)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _combine_data(lrm_data, fuzzy_data, seed, dataset_name=""):
    """
    Combine LRM and fuzzy data into a unified dict for all methods.

    Both generators use the same seed but produce different labels.
    We use the LRM generator's x, y (MC-averaged payoff) as canonical.

    For evaluation references:
      - digital_bs, basket_d1: use analytical deltas (already in lrm_data["dydx_exact"])
      - barrier_bs: use analytical delta via reflection principle (Reiner & Rubinstein 1991)
      - heston_digital: use COS method delta (Fang & Oosterlee 2008)
      - basket_d7: use high-k MC LRM (k=100,000)
    """
    rng = np.random.RandomState(seed)
    n = lrm_data["x"].shape[0]
    indices = rng.permutation(n)
    n_test = int(n * 0.2)

    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    result = {
        # Inputs (same for all methods)
        "x_train": lrm_data["x"][train_idx],
        "x_test": lrm_data["x"][test_idx],

        # Value labels: use LRM-generator's MC payoff (unsmoothed)
        "y_train": lrm_data["y"][train_idx],
        "y_test": lrm_data["y"][test_idx],

        # LRM derivative labels
        "dydx_lrm_train": lrm_data["dydx_lrm"][train_idx],
        "dydx_lrm_test": lrm_data["dydx_lrm"][test_idx],

        # Fuzzy-smoothed derivative labels
        "dydx_fuzzy_train": fuzzy_data["dydx_fuzzy"][train_idx],
        "dydx_fuzzy_test": fuzzy_data["dydx_fuzzy"][test_idx],

        # Fuzzy-smoothed value labels (for pathwise-only training)
        "y_fuzzy_train": fuzzy_data["y"][train_idx],
        "y_fuzzy_test": fuzzy_data["y"][test_idx],

        # Pathwise (biased) labels: 0 everywhere for digital payoffs
        "dydx_pw_train": np.zeros_like(lrm_data["dydx_lrm"][train_idx]),
        "dydx_pw_test": np.zeros_like(lrm_data["dydx_lrm"][test_idx]),

        # Metadata
        "lrm_var": lrm_data.get("lrm_var"),
        "lrm_config": lrm_data.get("config", {}),
        "fuzzy_config": fuzzy_data.get("config", {}),
        "epsilon": fuzzy_data.get("epsilon", fuzzy_data.get("epsilon_barrier", None)),
    }

    # Exact test values if available
    for key in ["y_exact", "dydx_exact"]:
        if key in lrm_data and lrm_data[key] is not None:
            result[f"{key}_test"] = lrm_data[key][test_idx]

    # ── Common evaluation ground truth (CRITICAL for fair cross-method comparison) ──
    # All methods must be evaluated against the SAME gradient reference.
    # Priority: analytical/semi-analytical > high-k MC > noisy LRM
    #
    # Audit fix A1: For barrier_bs, heston_digital, basket_d7, we now use
    # high-fidelity references instead of the noisy LRM (k=10) that was
    # previously used for both training and evaluation.
    if "dydx_exact" in lrm_data and lrm_data["dydx_exact"] is not None:
        # digital_bs, basket_d1: analytical delta already available
        result["dydx_eval_test"] = lrm_data["dydx_exact"][test_idx]
        result["eval_source"] = "analytical"
    elif dataset_name == "barrier_bs":
        # Audit A1: Analytical delta via reflection principle
        x_test = result["x_test"]
        cfg = DATA_CONFIG["barrier_bs"]
        hf_delta = barrier_bs_analytical_delta(
            x_test.flatten(),
            strike=cfg["strike"], barrier=cfg["barrier"],
            vol=cfg["vol"], r=cfg["r"], T=cfg["T"],
        )
        result["dydx_eval_test"] = hf_delta.reshape(-1, 1, 1)
        result["eval_source"] = "analytical_reflection_principle"
    elif dataset_name == "heston_digital":
        # Audit A1: COS method delta (Fang & Oosterlee 2008)
        x_test = result["x_test"]
        cfg = DATA_CONFIG["heston_digital"]
        hf_delta = heston_digital_cos_delta(
            x_test.flatten(),
            strike=cfg["strike"], v0=cfg["v0"],
            kappa=cfg["kappa"], theta=cfg["theta"],
            sigma_v=cfg["sigma_v"], rho=cfg["rho"],
            r=cfg["r"], T=cfg["T"],
        )
        result["dydx_eval_test"] = hf_delta.reshape(-1, 1, 1)
        result["eval_source"] = "cos_method_semi_analytical"
    elif dataset_name == "basket_d7":
        # Audit A1: High-k MC LRM (k=100,000)
        x_test = result["x_test"]
        cfg = DATA_CONFIG["basket_d7"]
        hf_dydx, _, hf_var = basket_high_k_lrm_delta(
            x_test, d=cfg["d"], strike=cfg["strike"],
            base_vol=cfg["base_vol"], T=cfg["T"], rho=cfg["rho"],
            k_paths=100_000, seed=seed + 7777,
        )
        result["dydx_eval_test"] = hf_dydx.reshape(-1, 1, cfg["d"])
        result["eval_source"] = "high_k_mc_lrm_100k"
        result["hf_lrm_var"] = float(np.mean(hf_var))
    else:
        result["dydx_eval_test"] = lrm_data["dydx_lrm"][test_idx]
        result["eval_source"] = "lrm"

    # For value: always evaluate against unsmoothed MC payoff (= y_test)
    # y_exact_test is also available for digital_bs/basket_d1 but MC payoff is
    # the more realistic target (it's what the model would see in production).

    return result


# ============================================================================
# TWO-PHASE (WARMUP) TRAINING
# ============================================================================

def train_warmup(
    x_train, y_train, dydx_train,
    x_test, y_test, dydx_test,
    warmup_fraction=0.5,
    seed=42, pbar=False,
    x_val=None, y_val=None, dydx_val=None,
    **hparams,
):
    """
    Two-phase training: vanilla warmup → DML GradNorm fine-tuning.

    E-H3 (2026-04-13): explicit val args. If `x_val`/`y_val`/`dydx_val` are all
    provided, they are passed through to `create_data_loaders`, bypassing the
    trainer's internal 80/20 split (which would otherwise reduce a 1000-frame
    training set to 800/200). Required for the rMD17 canonical 950/50 protocol
    in the P4 molecular pillar.
    """
    set_deterministic(seed)
    input_dim = x_train.shape[1]
    n_epochs = hparams.get("n_epochs", 500)
    warmup_epochs = int(n_epochs * warmup_fraction)
    finetune_epochs = n_epochs - warmup_epochs

    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        batch_size=hparams.get("batch_size", 256), seed=seed,
        x_val=x_val, y_val=y_val, dydx_val=dydx_val,
    )
    device = get_device()

    # Phase 1: Vanilla warmup
    model = DmlFeedForward(
        input_dim=input_dim, output_dim=1,
        n_layers=hparams.get("n_layers", 4),
        hidden_size=hparams.get("hidden_size", 256),
        activation=hparams.get("activation", "softplus"),
    )

    vanilla_loss_fn = VanillaLoss()
    opt_p1 = torch.optim.AdamW(model.parameters(), lr=hparams.get("lr", 0.005), weight_decay=0.0)
    sched_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p1, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20), min_lr=1e-6,
    )

    trainer_p1 = DmlTrainer(
        model=model, loss_fn=vanilla_loss_fn, optimizer=opt_p1,
        normalizer=normalizer, scheduler=sched_p1, use_dml=False,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )
    # 2026-04-14 (phase-1 ES ablation, 75 runs): enable ES in phase 1 with
    # patience=50 (same as phase 2). All three configs tested (off / pat50 /
    # pat20) produce identical test metrics within noise (<2% variation),
    # pat50 saves 15-55% phase-1 compute. See EVIDENCE/ablation_outcomes.md.
    result_p1 = trainer_p1.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=warmup_epochs, config={"phase": "vanilla_warmup"},
        pbar=pbar, early_stopping_patience=50,
    )

    if trainer_p1.best_model_state is not None:
        model.load_state_dict(trainer_p1.best_model_state)
        model = model.to(device)

    # Phase 2: DML GradNorm fine-tuning
    # 2026-04-14: lr drop factor changed from /5 → /10 after warmup-LR ablation
    # (7 strategies × 3 targets × 5 splits/seeds = 105 runs); lr_div_10 had
    # lowest mean rank across all 6 (target×metric) cells. See
    # EVIDENCE/ablation_outcomes.md and EVIDENCE/warmup_definition.md.
    dml_loss_fn = GradNormDmlLoss(input_dim=input_dim)
    finetune_lr = hparams.get("lr", 0.005) / 10.0
    opt_p2 = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.0)
    sched_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p2, mode="min",
        factor=hparams.get("scheduler_factor", 0.5),
        patience=hparams.get("scheduler_patience", 20), min_lr=1e-7,
    )

    trainer_p2 = DmlTrainer(
        model=model, loss_fn=dml_loss_fn, optimizer=opt_p2,
        normalizer=normalizer, scheduler=sched_p2, use_dml=True,
        max_grad_norm=hparams.get("max_grad_norm", 1.0),
    )
    result_p2 = trainer_p2.train(
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=finetune_epochs, config={"phase": "dml_gradnorm_finetune"},
        pbar=pbar, early_stopping_patience=50,
    )

    if trainer_p2.best_model_state is not None:
        model.load_state_dict(trainer_p2.best_model_state)
        model = model.to(device)

    # Evaluate
    eval_trainer = DmlTrainer(
        model=model, loss_fn=dml_loss_fn, optimizer=opt_p2,
        normalizer=normalizer, use_dml=True,
    )
    test_metrics = eval_trainer.evaluate(test_loader)

    total_time = result_p1.total_time_s + result_p2.total_time_s
    result = TrainingResult(
        config={"method": "dml_warmup", "warmup_fraction": warmup_fraction},
        final_train_loss=result_p2.final_train_loss,
        final_val_loss=float(trainer_p2.best_val_loss),
        test_value_mse=test_metrics["value_mse"],
        test_grad_mse=test_metrics["grad_mse"],
        training_logs=result_p1.training_logs + result_p2.training_logs,
        total_time_s=total_time,
        # J9 (2026-04-16, G-L2): report absolute epoch of the returned model
        # (phase-2 best offset by the phase-1 epoch count). Prior reporting of
        # phase-1 best_epoch made dml_warmup's best_epoch metadata apples-to-
        # oranges vs other methods that reported full-training best.
        best_epoch=warmup_epochs + result_p2.best_epoch,
        # I-H1 (2026-04-16): expose phase-2 best state for post-eval hooks
        # (e.g. MLP-pairwise Cartesian-force MSE reconstruction).
        best_model_state=trainer_p2.best_model_state,
    )
    return result


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_method(
    method: str,
    data: dict,
    seed: int,
    hparams: dict,
    pbar: bool = False,
) -> dict:
    """
    Run a single method on the unified dataset.

    CRITICAL: All methods are evaluated against the SAME test ground truth
    (analytical delta if available, otherwise LRM). This ensures fair
    cross-method comparison. Each method uses its own training labels but
    shares evaluation labels.

    Returns dict with test_value_mse, test_grad_mse, time_s.
    """
    t0 = time.time()

    # Common evaluation references (same for ALL methods)
    eval_y_test = data["y_test"]              # unsmoothed MC payoff
    eval_dydx_test = data["dydx_eval_test"]   # analytical or LRM (see _combine_data)

    # ── Pathwise-label methods ─────────────────────────────────────────
    if method == "vanilla":
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_pw_train"],  # Not used, but required
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="vanilla", seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_fixed":
        # Huge & Savine (2020) baseline: pathwise labels (≡ 0 for digitals)
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_pw_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_fixed", lambda_=1.0,
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_gradnorm":
        # GradNorm with pathwise labels
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_pw_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_gradnorm",
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_relobralo":
        # ReLoBRaLo with pathwise labels
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_pw_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_relobralo",
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_warmup":
        # Warmup → DML GradNorm with pathwise labels
        result = train_warmup(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_pw_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            warmup_fraction=WARMUP_FRACTION,
            seed=seed, pbar=pbar, **hparams,
        )

    # ── LRM-label methods (Glasserman & Karmarkar 2025) ──────────────
    elif method == "dml_lrm":
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_lrm_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_fixed", lambda_=1.0,
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_gradnorm_lrm":
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_lrm_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_gradnorm",
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_warmup_lrm":
        result = train_warmup(
            x_train=data["x_train"], y_train=data["y_train"],
            dydx_train=data["dydx_lrm_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            warmup_fraction=WARMUP_FRACTION,
            seed=seed, pbar=pbar, **hparams,
        )

    # ── Fuzzy-label methods (Savine 2018 + ours) ─────────────────────
    elif method == "dml_fuzzy":
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_fuzzy_train"],
            dydx_train=data["dydx_fuzzy_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_fixed", lambda_=1.0,
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_gradnorm_fuzzy":
        result = train_single_experiment(
            x_train=data["x_train"], y_train=data["y_fuzzy_train"],
            dydx_train=data["dydx_fuzzy_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            method="dml_gradnorm",
            seed=seed, pbar=pbar, **hparams,
        )
    elif method == "dml_warmup_fuzzy":
        # NOVEL COMBINATION: Vanilla warmup → DML GradNorm with fuzzy labels
        result = train_warmup(
            x_train=data["x_train"], y_train=data["y_fuzzy_train"],
            dydx_train=data["dydx_fuzzy_train"],
            x_test=data["x_test"], y_test=eval_y_test,
            dydx_test=eval_dydx_test,
            warmup_fraction=WARMUP_FRACTION,
            seed=seed, pbar=pbar, **hparams,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0

    return {
        "test_value_mse": float(result.test_value_mse),
        "test_grad_mse": float(result.test_grad_mse),
        "time_s": round(elapsed, 2),
        "best_epoch": int(result.best_epoch),
    }


# ============================================================================
# I/O HELPERS
# ============================================================================

def make_key(dataset, method, seed):
    return f"{dataset}_{method}_s{seed}"


def load_existing(results_dir):
    existing = {}
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            if f.name.startswith("summary") or f.name.startswith("analysis"):
                continue
            try:
                with open(f) as fh:
                    d = json.load(fh)
                    existing[d.get("key", f.stem)] = d
            except Exception:
                pass
    return existing


def save_result(results_dir, key, result_dict):
    result_dict["key"] = key
    result_dict["timestamp"] = datetime.now().isoformat()
    path = results_dir / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    tmp.rename(path)


# ============================================================================
# MAIN EXPERIMENT LOOPS
# ============================================================================

def run_experiments(
    mode: str = "smoke_test",
    resume: bool = True,
    datasets: list = None,
    methods: list = None,
):
    """
    Run the unified comparison experiment.

    Modes:
        smoke_test  — 1 dataset, 1 seed, 50 epochs (verify everything works)
        single_seed — all datasets, 1 seed, 500 epochs
        multi_seed  — all datasets, 10 seeds, 500 epochs
    """
    results_dir = RESULTS_DIR / mode
    results_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(results_dir) if resume else {}

    if datasets is None:
        if mode == "smoke_test":
            datasets = ["digital_bs"]
        else:
            datasets = list(DATA_CONFIG.keys())

    if methods is None:
        methods = ALL_METHODS

    if mode == "smoke_test":
        seeds = [42]
        hparams = SMOKE_HPARAMS
    elif mode == "single_seed":
        seeds = [42]
        hparams = HPARAMS
    else:  # multi_seed
        seeds = SEEDS
        hparams = HPARAMS

    n_total = len(datasets) * len(methods) * len(seeds)
    n_done = 0
    n_skip = 0

    print(f"\n{'='*80}")
    print(f"UNIFIED DISCONTINUOUS-PAYOFF COMPARISON — mode: {mode}")
    print(f"{'='*80}")
    print(f"Datasets:  {datasets}")
    print(f"Methods:   {methods}")
    print(f"Seeds:     {seeds}")
    print(f"Epochs:    {hparams['n_epochs']}")
    print(f"Total:     {n_total} experiments")
    print(f"{'='*80}\n")

    for dataset in datasets:
        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset}")
        print(f"{'─'*60}")

        for seed_idx, seed in enumerate(seeds):
            print(f"\n  Seed {seed} ({seed_idx + 1}/{len(seeds)})")

            # Generate data once per (dataset, seed) — shared across methods
            print(f"    Generating data ...", end=" ", flush=True)
            t_gen = time.time()
            try:
                data = generate_dataset(dataset, seed=seed)
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                continue
            print(f"done ({time.time() - t_gen:.1f}s)")

            # Print data stats
            n_train = data["x_train"].shape[0]
            n_test = data["x_test"].shape[0]
            dim = data["x_train"].shape[1]
            lrm_var = float(np.mean(data["lrm_var"])) if data.get("lrm_var") is not None else None
            eps = data.get("epsilon")
            eval_src = data.get("eval_source", "unknown")
            print(f"    n_train={n_train}, n_test={n_test}, dim={dim}, eps={eps}, lrm_var={lrm_var}, eval_ref={eval_src}")

            for method in methods:
                key = make_key(dataset, method, seed)

                if resume and key in existing:
                    print(f"    {method:<25} SKIP (exists)")
                    n_skip += 1
                    n_done += 1
                    continue

                print(f"    {method:<25}", end=" ", flush=True)

                try:
                    metrics = run_single_method(
                        method=method, data=data, seed=seed,
                        hparams=hparams, pbar=False,
                    )

                    result_dict = {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "dim": dim,
                        "n_train": n_train,
                        "n_test": n_test,
                        "test_value_mse": metrics["test_value_mse"],
                        "test_grad_mse": metrics["test_grad_mse"],
                        "time_s": metrics["time_s"],
                        "best_epoch": metrics["best_epoch"],
                        "hparams": hparams,
                        "epsilon": eps,
                        "lrm_var_mean": lrm_var,
                        "eval_source": eval_src,
                        "mode": mode,
                    }
                    save_result(results_dir, key, result_dict)

                    print(
                        f"val={metrics['test_value_mse']:.4e}  "
                        f"grad={metrics['test_grad_mse']:.4e}  "
                        f"t={metrics['time_s']:.1f}s"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()

                n_done += 1

    print(f"\n{'='*80}")
    print(f"Completed: {n_done}/{n_total} ({n_skip} skipped)")
    print(f"Results in: {results_dir}")
    print(f"{'='*80}")


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(mode: str = None):
    """Analyze unified experiment results."""
    if mode is None:
        # Auto-detect: prefer multi_seed > single_seed > smoke_test
        for m in ["multi_seed", "single_seed", "smoke_test"]:
            if (RESULTS_DIR / m).exists():
                mode = m
                break
    if mode is None:
        print("No results found.")
        return

    results_dir = RESULTS_DIR / mode
    results = load_existing(results_dir)
    if not results:
        print(f"No results in {results_dir}")
        return

    print(f"\n{'='*80}")
    print(f"UNIFIED COMPARISON RESULTS — mode: {mode} ({len(results)} experiments)")
    print(f"{'='*80}")

    # Group by dataset
    by_dataset = defaultdict(list)
    for r in results.values():
        by_dataset[r["dataset"]].append(r)

    for dataset in sorted(by_dataset.keys()):
        recs = by_dataset[dataset]
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset} (dim={recs[0]['dim']})")
        print(f"{'─'*70}")

        # Group by method
        by_method = defaultdict(list)
        for r in recs:
            by_method[r["method"]].append(r)

        # Table header
        print(f"\n  {'Method':<25} {'Mean Val MSE':>14} {'Std':>12} {'Mean Grad MSE':>14} {'Std':>12} {'N':>4}")
        print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*14} {'-'*12} {'-'*4}")

        # Compute vanilla baseline for relative comparisons
        vanilla_vals = [r["test_value_mse"] for r in by_method.get("vanilla", [])]
        vanilla_grads = [r["test_grad_mse"] for r in by_method.get("vanilla", [])]
        van_val = np.mean(vanilla_vals) if vanilla_vals else None
        van_grad = np.mean(vanilla_grads) if vanilla_grads else None

        method_stats = {}
        for method in ALL_METHODS:
            if method not in by_method:
                continue
            vals = [r["test_value_mse"] for r in by_method[method]]
            grads = [r["test_grad_mse"] for r in by_method[method]]
            mv, sv = np.mean(vals), np.std(vals)
            mg, sg = np.mean(grads), np.std(grads)
            method_stats[method] = (mv, sv, mg, sg, len(vals))
            print(f"  {method:<25} {mv:14.6e} {sv:12.6e} {mg:14.6e} {sg:12.6e} {len(vals):4d}")

        # Relative comparison to vanilla
        if van_val is not None and van_grad is not None:
            print(f"\n  Relative to vanilla:")
            print(f"  {'Method':<25} {'Val Penalty':>14} {'Grad Improvement':>18}")
            print(f"  {'-'*25} {'-'*14} {'-'*18}")
            for method in ALL_METHODS:
                if method == "vanilla" or method not in method_stats:
                    continue
                mv, sv, mg, sg, n = method_stats[method]
                val_pen = (mv - van_val) / van_val * 100 if van_val > 0 else 0
                if van_grad > 0 and mg > 0:
                    grad_imp = van_grad / mg
                    grad_str = f"{grad_imp:17.1f}x"
                elif mg == 0 and van_grad > 0:
                    grad_str = "             inf x"
                else:
                    grad_str = "              N/A"
                print(f"  {method:<25} {val_pen:+13.1f}% {grad_str}")

    # ---- Summary: best method per dataset ----
    print(f"\n{'='*80}")
    print("BEST METHOD PER DATASET (lowest gradient MSE with ≤10% value penalty)")
    print(f"{'='*80}")

    for dataset in sorted(by_dataset.keys()):
        recs = by_dataset[dataset]
        by_method = defaultdict(list)
        for r in recs:
            by_method[r["method"]].append(r)

        vanilla_vals = [r["test_value_mse"] for r in by_method.get("vanilla", [])]
        vanilla_grads = [r["test_grad_mse"] for r in by_method.get("vanilla", [])]
        van_val = np.mean(vanilla_vals) if vanilla_vals else None
        van_grad = np.mean(vanilla_grads) if vanilla_grads else None

        best_method = None
        best_grad = float("inf")

        for method in ALL_METHODS:
            if method not in by_method:
                continue
            records = by_method[method]
            mv = np.mean([r["test_value_mse"] for r in records])
            mg = np.mean([r["test_grad_mse"] for r in records])

            # Allow up to 10% value penalty
            if van_val is not None and (mv - van_val) / van_val > 0.10:
                continue

            if mg < best_grad:
                best_grad = mg
                best_method = method

        if best_method and van_val and van_grad:
            mv = np.mean([r["test_value_mse"] for r in by_method[best_method]])
            mg = np.mean([r["test_grad_mse"] for r in by_method[best_method]])
            val_pen = ((mv - van_val) / van_val * 100) if van_val > 0 else 0
            if van_grad > 0 and mg > 0:
                grad_imp = f"{van_grad / mg:.1f}x"
            else:
                grad_imp = "N/A"
            print(f"  {dataset:<20} → {best_method:<25} grad_mse={mg:.4e} val_penalty={val_pen:+.1f}% grad_imp={grad_imp}")
        else:
            print(f"  {dataset:<20} → vanilla (no method improves gradients within 10% value penalty)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Discontinuous-Payoff Comparison Experiment"
    )
    parser.add_argument(
        "--mode", choices=["smoke_test", "single_seed", "multi_seed"],
        default="smoke_test",
        help="Experiment mode (default: smoke_test)"
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Specific datasets to run (default: all for mode)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Specific methods to run (default: all)"
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.analyze_only:
        analyze_results()
        return

    run_experiments(
        mode=args.mode,
        resume=args.resume,
        datasets=args.datasets,
        methods=args.methods,
    )

    # Auto-analyze
    analyze_results(mode=args.mode)

    print("\nDone!")


if __name__ == "__main__":
    main()
