"""
BS down-and-out barrier — properly-monitored LRM and pathwise label generators.

These functions complement the existing `lrm_barrier_bs` (which monitors the
barrier at every Euler step including expiry — a different convention than
G&K v2 §3.4 / Hull Ch. 26).

The functions here implement the G&K v2 §3.4 convention:
    - Barrier monitored at intermediate dates T_1, ..., T_{n-1} only
    - Expiry at T_n: payoff is (S_T_n - K)^+ * 1{path-survived}, NO barrier
      check at T_n
    - LRM score uses ONLY the first-step Brownian increment (Markov property:
      given S_T_1, the distribution of (S_T_2,...,S_T_n) is independent of S_0
      under GBM, so only the first transition density depends on S_0)

References:
    - Glasserman, P., and S. H. Karmarkar (2025/2026). "Differential ML
      with a Difference". arXiv:2512.05301 v2 §3.4 + Eq. (barrierlrm).
    - Hull, J. (2018). Options, Futures, and Other Derivatives, Ch. 26.
    - Reiner & Rubinstein (1991). Risk 4(8), 28-35.

Implementation notes:
    - Number of stochastic time steps = n_monitor (the n in G&K v2 Table 2);
      barrier checked at the first n_monitor-1 of them.
    - Last step is the expiry; no barrier check at expiry.
    - All step lengths uniform: Δt = T_total / n_monitor.
    - For G&K v2 Table 2 row n=2: n_monitor=2 means 1 barrier check at T_1
      followed by 1 expiry step to T_2. Total time T_total = n * Δt = 2 * (1/3) = 2/3.

Convention (matches our codebase):
    Z_1 = primary normal draw, drives spot directly:
        log S_{i+1} = log S_i + (r - σ²/2) Δt + σ √Δt · Z_{1,i}

LRM score (verified vs G&K v2 §3.4):
    Δ̂ = π(path) · ξ_1 / (S_0 · σ · √Δt_first)
    where ξ_1 is the first step's Brownian increment.
"""

import numpy as np
from typing import Dict, Any, Optional


def _bs_barrier_doc_simulate(
    S0: np.ndarray,
    strike: float,
    barrier: float,
    vol: float,
    r: float,
    T_total: float,
    n_monitor: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """
    Simulate BS GBM paths with barrier monitoring at the first n_monitor-1
    dates only (barrier check excludes expiry).

    Returns:
        S_path: (n_monitor+1, n_samples) — S at each monitoring date including S_0 and S_T
        alive: (n_samples,) — 1.0 if path survived, 0.0 if knocked out
        Z_first: (n_samples,) — first-step normal increment (for LRM score)
    """
    n = S0.shape[0]
    dt = T_total / n_monitor
    sqrt_dt = np.sqrt(dt)

    # All path normals: shape (n_samples, n_monitor)
    Z = rng.standard_normal((n, n_monitor))

    log_S = np.log(S0.flatten())  # (n,)
    alive = np.ones(n, dtype=np.float64)
    S_path = [np.exp(log_S)]

    for step in range(n_monitor):
        log_S = log_S + (r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z[:, step]
        S_at_step = np.exp(log_S)
        S_path.append(S_at_step)
        # Barrier check ONLY at intermediate dates (step < n_monitor - 1).
        # The final step is expiry; no barrier check there.
        if step < n_monitor - 1:
            alive = alive * (S_at_step > barrier).astype(np.float64)

    return {
        "S_path": np.array(S_path),  # (n_monitor+1, n)
        "S_T": np.exp(log_S),  # final spot at expiry
        "alive": alive,
        "Z_first": Z[:, 0],
    }


def lrm_barrier_bs_intermediate_only(
    n_samples: int,
    strike: float = 100.0,
    barrier: float = 85.0,
    vol: float = 0.20,
    r: float = 0.0,
    T_total: float = 2.0 / 3.0,
    n_monitor: int = 2,
    k_paths: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    BS down-and-out call with proper G&K v2 monitoring (intermediate dates only,
    NOT expiry) and LRM training labels.

    Defaults match G&K v2 Table 2 row n=2: T_total=2/3, n_monitor=2 (single
    intermediate barrier check at T_1=1/3, then expiry at T_2=2/3).

    Args:
        n_samples: number of training spots.
        strike: K.
        barrier: B (B < K expected).
        vol: σ.
        r: risk-free rate.
        T_total: total maturity.
        n_monitor: number of stochastic time steps = number of monitoring
            dates (the last is expiry, not a barrier check).
        k_paths: MC paths per spot.
        seed: RNG seed.

    Returns:
        dict with x, y, dydx_lrm, lrm_var, config.

    LRM score formula:
        Δ̂_LRM = π(path) · Z_first / (S_0 · σ · √Δt)

    where Δt = T_total / n_monitor.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))
    dt = T_total / n_monitor
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T_total)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        sim = _bs_barrier_doc_simulate(
            S0, strike, barrier, vol, r, T_total, n_monitor, rng,
        )
        call_payoff = np.maximum(sim["S_T"] - strike, 0.0)
        payoff = call_payoff * sim["alive"] * discount

        # LRM score: only first-step Brownian increment carries S_0 dependence
        # (Markov property of GBM). Score = ξ_1 / (S_0 · σ · √Δt).
        score = sim["Z_first"] / (S0.flatten() * vol * sqrt_dt)
        lrm_delta = payoff * score

        y_all[:, p] = payoff
        dydx_all[:, p] = lrm_delta

    y = y_all.mean(axis=1, keepdims=True)
    dydx_lrm = dydx_all.mean(axis=1).reshape(n_samples, 1, 1)
    lrm_var = dydx_all.var(axis=1)

    return {
        "x": S0,
        "y": y,
        "dydx_lrm": dydx_lrm,
        "lrm_var": lrm_var,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "black_scholes_gbm",
            "label_method": "lrm_intermediate_monitoring_only",
            "strike": strike,
            "barrier": barrier,
            "vol": vol,
            "r": r,
            "T_total": T_total,
            "n_monitor": n_monitor,
            "dt": dt,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def pathwise_barrier_bs_intermediate_only(
    n_samples: int,
    strike: float = 100.0,
    barrier: float = 85.0,
    vol: float = 0.20,
    r: float = 0.0,
    T_total: float = 2.0 / 3.0,
    n_monitor: int = 2,
    k_paths: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    BS barrier pathwise (biased) labels — intermediate-only monitoring.

    Differentiates THROUGH the alive indicator (treats it as a constant).
    This misses the Dirac at the barrier, hence biased — the standard G&K v2
    "pathwise on a discontinuous payoff" baseline.

    Δ_pw = 1{S_T > K} · 1{path-alive} · S_T / S_0 · discount
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))
    discount = np.exp(-r * T_total)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        sim = _bs_barrier_doc_simulate(
            S0, strike, barrier, vol, r, T_total, n_monitor, rng,
        )
        S_T = sim["S_T"]
        alive = sim["alive"]
        call_payoff = np.maximum(S_T - strike, 0.0)
        call_indicator = (S_T > strike).astype(np.float64)
        payoff = call_payoff * alive * discount
        S0_flat = S0.flatten()
        # Pathwise (biased): differentiate through indicator → dπ/dS_0 = 1{alive} · 1{S_T>K} · S_T/S_0 · discount
        pathwise_delta = alive * call_indicator * (S_T / S0_flat) * discount

        y_all[:, p] = payoff
        dydx_all[:, p] = pathwise_delta

    return {
        "x": S0,
        "y": y_all.mean(axis=1, keepdims=True),
        "dydx_pw": dydx_all.mean(axis=1).reshape(n_samples, 1, 1),
        "config": {
            "payoff": "barrier_doc_call",
            "model": "black_scholes_gbm",
            "label_method": "pathwise_intermediate_monitoring_only",
            "strike": strike,
            "barrier": barrier,
            "vol": vol,
            "r": r,
            "T_total": T_total,
            "n_monitor": n_monitor,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def fuzzy_barrier_bs_intermediate_only(
    n_samples: int,
    strike: float = 100.0,
    barrier: float = 85.0,
    vol: float = 0.20,
    r: float = 0.0,
    T_total: float = 2.0 / 3.0,
    n_monitor: int = 2,
    k_paths: int = 10,
    eps_barrier_mult: float = 0.5,
    eps_barrier_override: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    BS barrier fuzzy (Savine call-spread) labels — intermediate-only monitoring.

    Replaces the discrete alive indicator at each intermediate monitoring date
    with a smooth ramp:

        cSpr(S_T_j - B, ε) = clip((S_T_j - B + ε/2) / ε, 0, 1)

    Path "survival" is the product of cSpr at all monitoring dates. Pathwise
    delta via product rule.
    """
    from dml_benchmark.fuzzy_smoothing import call_spread, call_spread_deriv, calibrate_epsilon

    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))
    dt = T_total / n_monitor
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T_total)

    # Calibrate ε from std(S_T_first - B) on a small pre-simulation
    if eps_barrier_override is not None:
        eps_barrier = eps_barrier_override
    else:
        rng_cal = np.random.RandomState(seed + 7777)
        n_cal = min(n_samples, 1000)
        S0_cal = rng_cal.uniform(strike * 0.5, strike * 1.5, n_cal)
        Z_cal = rng_cal.standard_normal(n_cal)
        S_T1_cal = S0_cal * np.exp((r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z_cal)
        eps_barrier = calibrate_epsilon(S_T1_cal - barrier, eps_mult=eps_barrier_mult)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        # Simulate full path
        Z_all = rng.standard_normal((n_samples, n_monitor))
        log_S0 = np.log(S0.flatten())
        log_S = log_S0.copy()

        # Track survival_dt (product of cSpr at intermediate dates)
        # and d_survival/d_S0 via product rule
        survival_dt = np.ones(n_samples)
        d_survival_dS0 = np.zeros(n_samples)
        sensitivity = np.ones(n_samples)  # ∂S_t/∂S_0 = S_t/S_0 in GBM

        for step in range(n_monitor):
            R_step = np.exp((r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z_all[:, step])
            log_S = log_S + (r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z_all[:, step]
            sensitivity = sensitivity * R_step  # ∂S_t/∂S_0 product

            if step < n_monitor - 1:
                # Apply fuzzy barrier check at intermediate date
                S_at_step = np.exp(log_S)
                barrier_cond = S_at_step - barrier
                dt_step = call_spread(barrier_cond, eps_barrier)
                d_dt_dS = call_spread_deriv(barrier_cond, eps_barrier) * sensitivity

                # Update survival via product rule
                d_survival_dS0 = d_survival_dS0 * dt_step + survival_dt * d_dt_dS
                survival_dt = survival_dt * dt_step

        S_T = np.exp(log_S)
        call_payoff = np.maximum(S_T - strike, 0.0)
        call_indicator = (S_T > strike).astype(np.float64)
        S0_flat = S0.flatten()
        # Pathwise sensitivity: ∂(call_payoff)/∂S_0 = 1{S_T>K} · S_T/S_0
        d_call_dS0 = call_indicator * (S_T / S0_flat)

        # Full payoff and delta via product rule
        fuzzy_payoff = call_payoff * survival_dt * discount
        fuzzy_delta = (d_call_dS0 * survival_dt + call_payoff * d_survival_dS0) * discount

        y_all[:, p] = fuzzy_payoff
        dydx_all[:, p] = fuzzy_delta

    return {
        "x": S0,
        "y": y_all.mean(axis=1, keepdims=True),
        "dydx_fuzzy": dydx_all.mean(axis=1).reshape(n_samples, 1, 1),
        "epsilon_barrier": float(eps_barrier),
        "config": {
            "payoff": "barrier_doc_call",
            "model": "black_scholes_gbm",
            "label_method": "fuzzy_intermediate_monitoring_only",
            "strike": strike,
            "barrier": barrier,
            "vol": vol,
            "r": r,
            "T_total": T_total,
            "n_monitor": n_monitor,
            "k_paths": k_paths,
            "eps_barrier_mult": eps_barrier_mult,
            "eps_barrier_override": eps_barrier_override,
            "epsilon_barrier_used": float(eps_barrier),
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def bs_barrier_doc_mc_reference(
    S0: np.ndarray,
    strike: float = 100.0,
    barrier: float = 85.0,
    vol: float = 0.20,
    r: float = 0.0,
    T_total: float = 2.0 / 3.0,
    n_monitor: int = 2,
    n_paths: int = 100_000,
    finite_diff_bump: float = 0.01,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    MC reference for BS down-and-out call with intermediate-only monitoring,
    matching the training-set discretization. Returns price + delta via central
    FD on S_0 with shared random numbers (CRN).

    This is the apples-to-apples ground truth for a discrete n_monitor=k
    barrier — different from the closed-form Reiner-Rubinstein formula which
    is for continuous monitoring.
    """
    S0 = np.asarray(S0, dtype=np.float64).flatten()
    n = S0.shape[0]
    eps = finite_diff_bump
    dt = T_total / n_monitor
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T_total)

    rng = np.random.RandomState(seed)
    log_S_c = np.broadcast_to(np.log(S0)[:, None], (n, n_paths)).copy()
    log_S_p = np.broadcast_to(np.log(S0 + eps)[:, None], (n, n_paths)).copy()
    log_S_m = np.broadcast_to(np.log(S0 - eps)[:, None], (n, n_paths)).copy()
    alive_c = np.ones((n, n_paths), dtype=np.float64)
    alive_p = np.ones((n, n_paths), dtype=np.float64)
    alive_m = np.ones((n, n_paths), dtype=np.float64)

    for step in range(n_monitor):
        Z = rng.standard_normal((n, n_paths))
        drift = (r - 0.5 * vol ** 2) * dt
        diffuse = vol * sqrt_dt * Z
        log_S_c += drift + diffuse
        log_S_p += drift + diffuse
        log_S_m += drift + diffuse
        if step < n_monitor - 1:
            alive_c *= (np.exp(log_S_c) > barrier).astype(np.float64)
            alive_p *= (np.exp(log_S_p) > barrier).astype(np.float64)
            alive_m *= (np.exp(log_S_m) > barrier).astype(np.float64)

    S_T_c = np.exp(log_S_c)
    S_T_p = np.exp(log_S_p)
    S_T_m = np.exp(log_S_m)
    payoff_c = np.maximum(S_T_c - strike, 0.0) * alive_c * discount
    payoff_p = np.maximum(S_T_p - strike, 0.0) * alive_p * discount
    payoff_m = np.maximum(S_T_m - strike, 0.0) * alive_m * discount

    price = payoff_c.mean(axis=1)
    price_p = payoff_p.mean(axis=1)
    price_m = payoff_m.mean(axis=1)
    se_price = payoff_c.std(axis=1, ddof=1) / np.sqrt(n_paths)
    delta = (price_p - price_m) / (2.0 * eps)
    delta_per_path = (payoff_p - payoff_m) / (2.0 * eps)
    se_delta = delta_per_path.std(axis=1, ddof=1) / np.sqrt(n_paths)

    return {
        "x": S0.reshape(n, 1),
        "y": price.reshape(n, 1),
        "dydx": delta.reshape(n, 1, 1),
        "std_err_price": se_price,
        "std_err_delta": se_delta,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "black_scholes_gbm",
            "label_method": "mc_reference_intermediate_only",
            "strike": strike, "barrier": barrier, "vol": vol, "r": r,
            "T_total": T_total, "n_monitor": n_monitor, "n_paths": n_paths,
            "finite_diff_bump": eps, "seed": seed, "n_samples": n,
        },
    }
