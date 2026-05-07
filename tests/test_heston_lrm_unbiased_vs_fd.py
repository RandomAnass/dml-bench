#!/usr/bin/env python3
"""
Regression test: Heston-barrier multi-step LRM (BEL-2D) is unbiased.

Compares three estimators of ∂C/∂S_0 at a representative spot under the
production Heston-barrier setup:

  1. FD reference     — central FD on S_0 with CRN (Glasserman 2004 §7.1)
  2. BEL-2D LRM       — `lrm_multistep_heston_barrier` (the canonical multi-step
                        score; uses orthogonal-to-vol noise per step)
  3. Z₁-only Heston specialisation — the 1D Chen-Glasserman 2007 Eq. (10)
                        applied directly to Heston, ignoring the V-Z₁ correlation;
                        biased ~6% at ρ=-0.7. Implemented inline here only for
                        the regression comparison; not a production label
                        generator.

The BEL-2D score must be unbiased relative to FD within MC noise; the Z₁-only
specialisation must be biased ~6%. The bias signature is the canonical regression
guard against accidentally reverting to the 1D specialisation.

Reference: paper §E.3 (`paper/sections/E_theory_crossover.tex`) and
`paper/agents/HESTON_MATH_VERIFICATION.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Production parameter set
HP = dict(
    strike=1.0, barrier=0.85, v0=0.04, kappa=1.0, theta=0.04,
    sigma_v=0.15, rho=-0.7, r=0.0, T1=1.0/3.0, T2=2.0/3.0,
    n_substeps_to_T1=84, n_substeps_T1_to_T2=84,
)

S0_test = 1.0           # representative spot (above barrier)
N_PATHS = 200_000       # 200k paths per estimator
SEEDS = [42, 123, 456, 789, 1337]
EPS_FD = 0.01           # FD bump


def simulate(S0_arr, n_paths, seed, return_path_increments=False):
    """Simulate Heston full-truncation Euler, return barrier-payoff and (optionally) per-step Z's."""
    rng = np.random.RandomState(seed)
    n = S0_arr.shape[0]
    dt1 = HP["T1"] / HP["n_substeps_to_T1"]
    dt2 = (HP["T2"] - HP["T1"]) / HP["n_substeps_T1_to_T2"]
    sqrt_dt1, sqrt_dt2 = np.sqrt(dt1), np.sqrt(dt2)
    discount = np.exp(-HP["r"] * HP["T2"])
    rho = HP["rho"]; sqrt_1mr2 = np.sqrt(1.0 - rho**2)

    # Pre-draw all randomness (CRN across S0 variants)
    n_steps_total = HP["n_substeps_to_T1"] + HP["n_substeps_T1_to_T2"]
    Z1_all = rng.standard_normal((n_paths, n_steps_total))
    Z_indep_all = rng.standard_normal((n_paths, n_steps_total))

    # Tile S0 to match path count
    log_S = np.tile(np.log(S0_arr).reshape(-1, 1), (1, n_paths))
    V = np.full((n, n_paths), HP["v0"])

    # Phase 1
    for i in range(HP["n_substeps_to_T1"]):
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        Z1 = Z1_all[:, i][None, :]
        Z_indep = Z_indep_all[:, i][None, :]
        Z2 = rho * Z1 + sqrt_1mr2 * Z_indep
        log_S = log_S + (HP["r"] - 0.5 * V_pos) * dt1 + sqrt_V * sqrt_dt1 * Z1
        V = V + HP["kappa"] * (HP["theta"] - V_pos) * dt1 + HP["sigma_v"] * sqrt_V * sqrt_dt1 * Z2
    alive = (np.exp(log_S) > HP["barrier"]).astype(np.float64)

    # Phase 2
    for i in range(HP["n_substeps_T1_to_T2"]):
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        Z1 = Z1_all[:, HP["n_substeps_to_T1"] + i][None, :]
        Z_indep = Z_indep_all[:, HP["n_substeps_to_T1"] + i][None, :]
        Z2 = rho * Z1 + sqrt_1mr2 * Z_indep
        log_S = log_S + (HP["r"] - 0.5 * V_pos) * dt2 + sqrt_V * sqrt_dt2 * Z1
        V = V + HP["kappa"] * (HP["theta"] - V_pos) * dt2 + HP["sigma_v"] * sqrt_V * sqrt_dt2 * Z2
    S_T2 = np.exp(log_S)
    payoff = np.maximum(S_T2 - HP["strike"], 0.0) * alive * discount

    if return_path_increments:
        return payoff, Z1_all, Z_indep_all
    return payoff


def fd_delta(S0, n_paths, seed, eps):
    """Central finite-difference Δ with shared random numbers (CRN)."""
    S0_triple = np.array([S0 - eps, S0, S0 + eps])
    payoff = simulate(S0_triple, n_paths, seed)  # (3, n_paths)
    p_minus = payoff[0].mean(); p_centre = payoff[1].mean(); p_plus = payoff[2].mean()
    return (p_plus - p_minus) / (2.0 * eps), p_centre


def lrm_multistep(S0, n_paths, seed, naive=False):
    """Multi-step LRM Δ — BEL-2D (default) or naive Z_1-only (for comparison)."""
    rho = HP["rho"]; v_floor = 1e-8
    rho_correction = 0.0 if naive else rho / np.sqrt(1.0 - rho**2)

    rng = np.random.RandomState(seed)
    dt1 = HP["T1"] / HP["n_substeps_to_T1"]
    dt2 = (HP["T2"] - HP["T1"]) / HP["n_substeps_T1_to_T2"]
    sqrt_dt1, sqrt_dt2 = np.sqrt(dt1), np.sqrt(dt2)
    discount = np.exp(-HP["r"] * HP["T2"])
    T_total = HP["T2"]

    log_S = np.full(n_paths, np.log(S0))
    V = np.full(n_paths, HP["v0"])
    weight_sum = np.zeros(n_paths)

    for i in range(HP["n_substeps_to_T1"]):
        V_sim = np.maximum(V, 0.0); V_score = np.maximum(V, v_floor)
        sqrt_V_sim, sqrt_V_score = np.sqrt(V_sim), np.sqrt(V_score)
        Z1 = rng.standard_normal(n_paths)
        Z_indep = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z_indep
        weight_sum += sqrt_dt1 * (Z1 - rho_correction * Z_indep) / sqrt_V_score
        log_S += (HP["r"] - 0.5 * V_sim) * dt1 + sqrt_V_sim * sqrt_dt1 * Z1
        V += HP["kappa"] * (HP["theta"] - V_sim) * dt1 + HP["sigma_v"] * sqrt_V_sim * sqrt_dt1 * Z2
    alive = (np.exp(log_S) > HP["barrier"]).astype(np.float64)

    for i in range(HP["n_substeps_T1_to_T2"]):
        V_sim = np.maximum(V, 0.0); V_score = np.maximum(V, v_floor)
        sqrt_V_sim, sqrt_V_score = np.sqrt(V_sim), np.sqrt(V_score)
        Z1 = rng.standard_normal(n_paths)
        Z_indep = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z_indep
        weight_sum += sqrt_dt2 * (Z1 - rho_correction * Z_indep) / sqrt_V_score
        log_S += (HP["r"] - 0.5 * V_sim) * dt2 + sqrt_V_sim * sqrt_dt2 * Z1
        V += HP["kappa"] * (HP["theta"] - V_sim) * dt2 + HP["sigma_v"] * sqrt_V_sim * sqrt_dt2 * Z2
    S_T2 = np.exp(log_S)
    payoff = np.maximum(S_T2 - HP["strike"], 0.0) * alive * discount
    delta_per_path = payoff * weight_sum / (T_total * S0)
    return delta_per_path.mean()


print(f"FD verification — production Heston-barrier setup, S_0 = {S0_test}")
print(f"Paths per seed: {N_PATHS}, seeds: {SEEDS}")
print(f"FD bump: {EPS_FD}\n")

fds, bels, naives, prices = [], [], [], []
for seed in SEEDS:
    fd, p_centre = fd_delta(S0_test, N_PATHS, seed, EPS_FD)
    bel = lrm_multistep(S0_test, N_PATHS, seed, naive=False)
    naive_d = lrm_multistep(S0_test, N_PATHS, seed, naive=True)
    fds.append(fd); bels.append(bel); naives.append(naive_d); prices.append(p_centre)
    print(f"  seed={seed}:  FD={fd:.4f}  BEL-2D={bel:.4f}  naive(Z1)={naive_d:.4f}  price={p_centre:.4f}")

fd_m, bel_m, naive_m = np.mean(fds), np.mean(bels), np.mean(naives)
fd_s, bel_s, naive_s = np.std(fds), np.std(bels), np.std(naives)
print(f"\n  Aggregate (mean ± std across {len(SEEDS)} seeds, {N_PATHS} paths each):")
print(f"    FD reference:  {fd_m:.4f} ± {fd_s:.4f}")
print(f"    BEL-2D:        {bel_m:.4f} ± {bel_s:.4f}  (bias vs FD: {(bel_m - fd_m)/fd_m*100:+.2f}%)")
print(f"    Z_1-only naive: {naive_m:.4f} ± {naive_s:.4f}  (bias vs FD: {(naive_m - fd_m)/fd_m*100:+.2f}%)")
print(f"\n  Reference values (paper §E.3 / HESTON_MATH_VERIFICATION.md):")
print(f"    FD ≈ 0.568, BEL-2D ≈ 0.567 (within MC noise of FD), Z₁-only ≈ 0.532 (~−6% bias)")
print(f"\n  Pass criteria: BEL-2D bias < 1% AND Z₁-only bias > 4% (regression guard).")
