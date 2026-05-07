"""
Heston barrier label generators with 2-D inputs (S_0, V_0).

Mirrors the 1-D pipeline (pathwise / fuzzy / BEL) for the V_0-extension
experiment described in
    repos/docs/heston_extension/writing/POLYNOMIAL_BASELINE_FINDING.md (Option C)
    repos/docs/heston_extension/agents/...

We sample V_0 per training spot from Uniform[0.5θ, 1.5θ] and condition the
network on the 2-D feature (S_0, V_0). Delta supervision is on ∂C/∂S_0 only
(no Vega supervision in this first cut). Path generation is identical to the
1-D code modulo the per-sample initial variance.

Backward compatibility: this module is purely additive. The existing 1-D
generators in `lrm_labels.py` and `fuzzy_smoothing.py` are untouched.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_v0_grid(
    n_samples: int,
    theta: float,
    rng: np.random.RandomState,
    v0_low_mult: float = 0.5,
    v0_high_mult: float = 1.5,
) -> np.ndarray:
    """V_0 ~ Uniform[v0_low_mult · θ, v0_high_mult · θ], shape (n_samples,)."""
    return rng.uniform(theta * v0_low_mult, theta * v0_high_mult, n_samples)


# ---------------------------------------------------------------------------
# Pathwise (Huge–Savine, ignores barrier Dirac)
# ---------------------------------------------------------------------------

def pathwise_barrier_heston_2d(
    n_samples: int,
    strike: float = 1.0,
    barrier: float = 0.85,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma_v: float = 0.15,
    rho: float = -0.7,
    r: float = 0.0,
    T1: float = 1.0 / 3.0,
    T2: float = 2.0 / 3.0,
    n_substeps_to_T1: int = 84,
    n_substeps_T1_to_T2: int = 84,
    k_paths: int = 10,
    seed: int = 42,
    spot_low_mult: float = 0.7,
    spot_high_mult: float = 1.3,
    v0_low_mult: float = 0.5,
    v0_high_mult: float = 1.5,
    **_: Any,  # absorb HESTON_PARAMS["v0"] for API compat
) -> Dict[str, Any]:
    """Pathwise (AAD) labels with (S_0, V_0) varied per sample."""
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, n_samples)
    V0 = sample_v0_grid(n_samples, theta, rng, v0_low_mult, v0_high_mult)

    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0)
        v = V0.copy()

        for step in range(n_substeps_to_T1):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S = log_S + (r - 0.5 * v_pos) * dt1 + sqrt_v * sqrt_dt1 * Z1
            v = v + kappa * (theta - v_pos) * dt1 + sigma_v * sqrt_v * sqrt_dt1 * Z2

        S_T1 = np.exp(log_S)
        alive = (S_T1 > barrier).astype(np.float64)

        for step in range(n_substeps_T1_to_T2):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S = log_S + (r - 0.5 * v_pos) * dt2 + sqrt_v * sqrt_dt2 * Z1
            v = v + kappa * (theta - v_pos) * dt2 + sigma_v * sqrt_v * sqrt_dt2 * Z2

        S_T2 = np.exp(log_S)
        call_payoff = np.maximum(S_T2 - strike, 0.0)
        call_indicator = (S_T2 > strike).astype(np.float64)

        # ∂π/∂S_0 = 1{alive} · 1{S_T2 > K} · (S_T2 / S_0) · discount  (Dirac missed)
        pathwise_delta = alive * call_indicator * (S_T2 / S0) * discount
        payoff = call_payoff * alive * discount

        y_all[:, p] = payoff
        dydx_all[:, p] = pathwise_delta

    x = np.column_stack([S0, V0])  # (n, 2)
    y = y_all.mean(axis=1, keepdims=True)
    # Output dydx with shape (n, 1, 2): for now ∂/∂V_0 channel is zero (no Vega
    # supervision in this experiment). We keep the 2-D last-dim for shape
    # symmetry with the input feature dim.
    dydx_pw = np.zeros((n_samples, 1, 2))
    dydx_pw[:, 0, 0] = dydx_all.mean(axis=1)

    return {
        "x": x,
        "y": y,
        "dydx_pw": dydx_pw,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "heston_full_truncation_euler",
            "label_method": "pathwise_2d",
            "v0_low_mult": v0_low_mult, "v0_high_mult": v0_high_mult,
            "spot_low_mult": spot_low_mult, "spot_high_mult": spot_high_mult,
            "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho,
            "r": r, "T1": T1, "T2": T2,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "k_paths": k_paths, "n_samples": n_samples, "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# Fuzzy (Savine call-spread on barrier indicator)
# ---------------------------------------------------------------------------

def fuzzy_barrier_heston_2d(
    n_samples: int,
    strike: float = 1.0,
    barrier: float = 0.85,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma_v: float = 0.15,
    rho: float = -0.7,
    r: float = 0.0,
    T1: float = 1.0 / 3.0,
    T2: float = 2.0 / 3.0,
    n_substeps_to_T1: int = 84,
    n_substeps_T1_to_T2: int = 84,
    k_paths: int = 10,
    eps: float = 0.05,
    seed: int = 42,
    spot_low_mult: float = 0.7,
    spot_high_mult: float = 1.3,
    v0_low_mult: float = 0.5,
    v0_high_mult: float = 1.5,
    **_: Any,
) -> Dict[str, Any]:
    """Savine-style fuzzy (call-spread on barrier indicator) for 2-D (S_0, V_0)."""
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, n_samples)
    V0 = sample_v0_grid(n_samples, theta, rng, v0_low_mult, v0_high_mult)

    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0)
        v = V0.copy()

        for step in range(n_substeps_to_T1):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S = log_S + (r - 0.5 * v_pos) * dt1 + sqrt_v * sqrt_dt1 * Z1
            v = v + kappa * (theta - v_pos) * dt1 + sigma_v * sqrt_v * sqrt_dt1 * Z2

        S_T1 = np.exp(log_S)
        # Savine call-spread on the barrier indicator: smooth 1{S_T1 > B}
        # via [B-ε/2, B+ε/2] ramp; derivative is 1/ε on the ramp interval.
        alive_smooth = np.clip((S_T1 - (barrier - eps / 2)) / eps, 0.0, 1.0)
        d_alive_dS_T1 = ((S_T1 > barrier - eps / 2) & (S_T1 < barrier + eps / 2)).astype(np.float64) / eps
        # ∂S_T1 / ∂S_0 = S_T1 / S_0 (log-spot pathwise derivative)
        d_alive_dS0 = d_alive_dS_T1 * (S_T1 / S0)

        for step in range(n_substeps_T1_to_T2):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S = log_S + (r - 0.5 * v_pos) * dt2 + sqrt_v * sqrt_dt2 * Z1
            v = v + kappa * (theta - v_pos) * dt2 + sigma_v * sqrt_v * sqrt_dt2 * Z2

        S_T2 = np.exp(log_S)
        call_payoff = np.maximum(S_T2 - strike, 0.0)
        call_indicator = (S_T2 > strike).astype(np.float64)

        payoff = call_payoff * alive_smooth * discount
        # Product rule: ∂(C(S_T2) · ϕ_ε(S_T1)) / ∂S_0
        #             = ϕ_ε(S_T1) · 1{S_T2>K} · (S_T2/S_0)
        #               + C(S_T2) · ϕ'_ε(S_T1) · (S_T1/S_0)
        fuzzy_delta = (
            alive_smooth * call_indicator * (S_T2 / S0)
            + call_payoff * d_alive_dS0
        ) * discount

        y_all[:, p] = payoff
        dydx_all[:, p] = fuzzy_delta

    x = np.column_stack([S0, V0])
    y = y_all.mean(axis=1, keepdims=True)
    dydx_fuzzy = np.zeros((n_samples, 1, 2))
    dydx_fuzzy[:, 0, 0] = dydx_all.mean(axis=1)

    return {
        "x": x,
        "y": y,
        "dydx_fuzzy": dydx_fuzzy,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "heston_full_truncation_euler",
            "label_method": "fuzzy_2d",
            "eps": eps,
            "v0_low_mult": v0_low_mult, "v0_high_mult": v0_high_mult,
            "spot_low_mult": spot_low_mult, "spot_high_mult": spot_high_mult,
            "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho,
            "r": r, "T1": T1, "T2": T2,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "k_paths": k_paths, "n_samples": n_samples, "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# BEL (Fournié-localised Malliavin)
# ---------------------------------------------------------------------------

def bel_barrier_heston_2d(
    n_samples: int,
    strike: float = 1.0,
    barrier: float = 0.85,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma_v: float = 0.15,
    rho: float = -0.7,
    r: float = 0.0,
    T1: float = 1.0 / 3.0,
    T2: float = 2.0 / 3.0,
    n_substeps_to_T1: int = 84,
    n_substeps_T1_to_T2: int = 84,
    k_paths: int = 10,
    v_floor: float = 1e-8,
    seed: int = 42,
    spot_low_mult: float = 0.7,
    spot_high_mult: float = 1.3,
    v0_low_mult: float = 0.5,
    v0_high_mult: float = 1.5,
    **_: Any,
) -> Dict[str, Any]:
    """Fournié-localised Malliavin Δ with 2-D inputs (S_0, V_0)."""
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, n_samples)
    V0 = sample_v0_grid(n_samples, theta, rng, v0_low_mult, v0_high_mult)

    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)
    T_total = T2

    if abs(rho) >= 1.0 - 1e-10:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")
    rho_correction = rho / np.sqrt(1.0 - rho ** 2)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0)
        v = V0.copy()
        weight_sum = np.zeros(n_samples)

        for step in range(n_substeps_to_T1):
            v_pos_sim = np.maximum(v, 0.0)
            v_pos_score = np.maximum(v, v_floor)
            sqrt_v_sim = np.sqrt(v_pos_sim)
            sqrt_v_score = np.sqrt(v_pos_score)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            increment = sqrt_dt1 * (Z1 - rho_correction * Z_indep) / sqrt_v_score
            weight_sum = weight_sum + increment

            log_S = log_S + (r - 0.5 * v_pos_sim) * dt1 + sqrt_v_sim * sqrt_dt1 * Z1
            v = v + kappa * (theta - v_pos_sim) * dt1 + sigma_v * sqrt_v_sim * sqrt_dt1 * Z2

        S_T1 = np.exp(log_S)
        alive = (S_T1 > barrier).astype(np.float64)

        for step in range(n_substeps_T1_to_T2):
            v_pos_sim = np.maximum(v, 0.0)
            v_pos_score = np.maximum(v, v_floor)
            sqrt_v_sim = np.sqrt(v_pos_sim)
            sqrt_v_score = np.sqrt(v_pos_score)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            increment = sqrt_dt2 * (Z1 - rho_correction * Z_indep) / sqrt_v_score
            weight_sum = weight_sum + increment

            log_S = log_S + (r - 0.5 * v_pos_sim) * dt2 + sqrt_v_sim * sqrt_dt2 * Z1
            v = v + kappa * (theta - v_pos_sim) * dt2 + sigma_v * sqrt_v_sim * sqrt_dt2 * Z2

        S_T2 = np.exp(log_S)
        payoff = np.maximum(S_T2 - strike, 0.0) * alive * discount
        bel_delta = payoff * weight_sum / (T_total * S0)

        y_all[:, p] = payoff
        dydx_all[:, p] = bel_delta

    x = np.column_stack([S0, V0])
    y = y_all.mean(axis=1, keepdims=True)
    dydx_lrm = np.zeros((n_samples, 1, 2))
    dydx_lrm[:, 0, 0] = dydx_all.mean(axis=1)

    return {
        "x": x,
        "y": y,
        "dydx_lrm": dydx_lrm,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "heston_full_truncation_euler",
            "label_method": "bel_2d",
            "v0_low_mult": v0_low_mult, "v0_high_mult": v0_high_mult,
            "spot_low_mult": spot_low_mult, "spot_high_mult": spot_high_mult,
            "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho,
            "r": r, "T1": T1, "T2": T2,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "v_floor": v_floor,
            "k_paths": k_paths, "n_samples": n_samples, "seed": seed,
        },
    }
