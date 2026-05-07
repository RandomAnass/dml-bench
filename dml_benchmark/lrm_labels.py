"""
Likelihood Ratio Method (LRM) derivative label generators.

Provides unbiased gradient estimators for discontinuous payoffs using the
score function method. Designed as a drop-in replacement for pathwise
derivatives when the payoff function is not differentiable (digital options,
barrier options, etc.).

Reference:
    Glasserman & Karmarkar (2025), "Differential ML with a Difference",
    arXiv:2512.05301. Our implementation follows their formulas but adds
    multi-path averaging and variance tracking.

Integration:
    All generators return numpy arrays compatible with train_single_experiment():
        x:    (n_samples, d)
        y:    (n_samples, 1)
        dydx: (n_samples, 1, d)  — LRM-based gradient labels

No changes to existing dml_benchmark modules are required.
"""

import warnings

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.stats import norm as scipy_norm


# ============================================================================
# BLACK-SCHOLES DIGITAL OPTION — LRM LABELS
# ============================================================================

def lrm_digital_bs(
    n_samples: int,
    strike: float = 100.0,
    vol: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    k_paths: int = 10,
    seed: int = 42,
    return_pathwise: bool = False,
) -> Dict[str, Any]:
    """
    Generate Black-Scholes digital call option data with LRM derivative labels.

    Digital payoff: π(S_T) = 1{S_T > K}
    Pathwise delta: ∂π/∂S_0 = 0 almost everywhere (biased!)
    LRM delta:      π(S_T) · Z / (S_0 · σ · √T)    (unbiased)

    where S_T = S_0 · exp((r - σ²/2)T + σ√T · Z), Z ~ N(0,1).

    Following G&K (2025) §2.1–2.2, Eq. (4)–(7).

    Args:
        n_samples: Number of input spot prices to generate.
        strike: Strike price K.
        vol: Black-Scholes volatility σ.
        r: Risk-free rate.
        T: Time to maturity in years.
        k_paths: Number of MC paths per input for noise reduction (G&K use 10).
        seed: Random seed for reproducibility.
        return_pathwise: If True, also return pathwise (biased) labels.

    Returns:
        Dictionary with keys:
            x:        (n_samples, 1) spot prices
            y:        (n_samples, 1) MC-averaged digital call prices
            dydx_lrm: (n_samples, 1, 1) LRM delta estimates
            dydx_pw:  (n_samples, 1, 1) pathwise delta (≡ 0, only if return_pathwise)
            y_exact:  (n_samples, 1) analytical digital call price
            dydx_exact: (n_samples, 1, 1) analytical delta (from BS formula)
            lrm_var:  (n_samples,) per-sample LRM estimator variance
            config:   dict of parameters
    """
    rng = np.random.RandomState(seed)

    # Spot prices S_0 ∈ [0.5K, 1.5K]
    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))  # (n, 1)

    # Simulate k_paths per input: Z ~ N(0,1) shape (n_samples, k_paths)
    Z = rng.standard_normal((n_samples, k_paths))

    # Terminal prices: S_T = S_0 · exp((r - σ²/2)T + σ√T Z)
    drift = (r - 0.5 * vol ** 2) * T
    diffuse = vol * np.sqrt(T)
    S_T = S0 * np.exp(drift + diffuse * Z)  # (n, k)

    # Digital payoff (discounted)
    discount = np.exp(-r * T)
    payoff = (S_T > strike).astype(np.float64) * discount  # (n, k)

    # LRM delta per path: payoff · Z / (S_0 · σ · √T)
    lrm_delta_per_path = payoff * Z / (S0 * vol * np.sqrt(T))  # (n, k)

    # Average over k paths
    y = payoff.mean(axis=1, keepdims=True)                     # (n, 1)
    dydx_lrm = lrm_delta_per_path.mean(axis=1, keepdims=True)  # (n, 1)
    dydx_lrm = dydx_lrm.reshape(n_samples, 1, 1)

    # LRM variance (per-sample, across k paths)
    lrm_var = lrm_delta_per_path.var(axis=1)  # (n,)

    # Analytical (exact) values for validation
    d2 = (np.log(S0 / strike) + (r - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    y_exact = discount * scipy_norm.cdf(d2)                   # (n, 1)
    # Analytical delta of digital call: e^{-rT} · φ(d2) / (S0 · σ · √T)
    dydx_exact = (discount * scipy_norm.pdf(d2) / (S0 * vol * np.sqrt(T)))
    dydx_exact = dydx_exact.reshape(n_samples, 1, 1)

    result = {
        "x": S0,
        "y": y,
        "dydx_lrm": dydx_lrm,
        "y_exact": y_exact,
        "dydx_exact": dydx_exact,
        "lrm_var": lrm_var,
        "config": {
            "payoff": "digital_call",
            "model": "black_scholes",
            "label_method": "lrm",
            "strike": strike,
            "vol": vol,
            "r": r,
            "T": T,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }

    if return_pathwise:
        # Pathwise: ∂(1{S_T > K})/∂S_0 = 0 a.e. (biased!)
        result["dydx_pw"] = np.zeros((n_samples, 1, 1))

    return result


# ============================================================================
# BLACK-SCHOLES BARRIER OPTION — LRM LABELS
# ============================================================================

def lrm_barrier_bs(
    n_samples: int,
    strike: float = 100.0,
    barrier: float = 80.0,
    vol: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    n_steps: int = 252,
    k_paths: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Black-Scholes knock-out barrier call with LRM derivative labels.

    Payoff: max(S_T - K, 0) · 1{min_{t} S_t > B}  (down-and-out call)

    The payoff is discontinuous at the barrier, so pathwise differentiation
    through the barrier indicator is biased. LRM provides unbiased labels:

        LRM delta = Payoff · Z_total / (S_0 · σ · √T)

    where Z_total is the cumulative score for the first step (initial spot
    sensitivity). For single-step simulation (terminal only), this reduces
    to the standard score Z / (S_0 · σ · √T). For multi-step monitoring,
    we use the initial increment's contribution.

    Following G&K (2025) §2.4 (barrier options).

    Args:
        n_samples: Number of input spot prices.
        strike: Strike price K.
        barrier: Lower barrier B < K (knock-out if S_t ≤ B).
        vol: Volatility σ.
        r: Risk-free rate.
        T: Time to maturity.
        n_steps: Number of monitoring steps for barrier.
        k_paths: MC paths per input for noise reduction.
        seed: Random seed.

    Returns:
        Dictionary with x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)

    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    discount = np.exp(-r * T)
    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        Z_all = rng.standard_normal((n_samples, n_steps))  # (n, n_steps)

        # Simulate paths step by step
        S = S0.copy()  # (n, 1) current price
        alive = np.ones(n_samples, dtype=bool)  # barrier not hit

        for step in range(n_steps):
            Z_step = Z_all[:, step:step + 1]  # (n, 1)
            S = S * np.exp((r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z_step)
            alive &= (S.flatten() > barrier)

        S_T = S.flatten()
        call_payoff = np.maximum(S_T - strike, 0)
        payoff = call_payoff * alive.astype(np.float64) * discount  # (n,)

        # LRM delta: sensitivity to S_0 via first step's score
        # Full path LRM: payoff · Σ_k ∂log p(S_{k+1}|S_k) / ∂S_0
        # For GBM the dependency on S_0 flows through the multiplicative structure:
        # S_T = S_0 · Π exp(...), so ∂log S_T / ∂S_0 = 1/S_0.
        # The LRM score for the full path w.r.t. S_0 is Z_1 / (S_0 · σ · √dt)
        # where Z_1 is the first step's normal draw.
        Z_first = Z_all[:, 0]  # (n,)
        lrm_delta = payoff * Z_first / (S0.flatten() * vol * sqrt_dt)

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
            "payoff": "barrier_call_knock_out",
            "model": "black_scholes",
            "label_method": "lrm",
            "strike": strike,
            "barrier": barrier,
            "vol": vol,
            "r": r,
            "T": T,
            "n_steps": n_steps,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


# ============================================================================
# BACHELIER BASKET DIGITAL OPTION — LRM LABELS (MULTI-DIM)
# ============================================================================

def lrm_basket_bachelier(
    n_samples: int,
    d: int = 1,
    strike: float = 100.0,
    base_vol: float = 20.0,
    T: float = 1.0,
    rho: float = 0.5,
    k_paths: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Bachelier basket digital option data with LRM derivative labels.

    Under Bachelier dynamics: S_{T,i} = x_i + σ_i √T Z_i  (correlated Z).
    Basket = Σ w_i S_{T,i}, weights w_i = 1/d (equal-weighted).
    Payoff = 1{Basket > K} (digital).

    LRM delta for each component i:
        Δ̂_{LRM,i} = 1{Basket > K} · [Σ⁻¹ Z]_i / (σ_i √T)

    where Σ is the correlation matrix and Z is the correlated normal vector.
    For equicorrelation, Σ⁻¹ has closed form.

    This is the key experiment for testing LRM variance scaling in high
    dimensions. G&K test d = 1, 7, 20; we extend to d = 50.

    Following G&K (2025) §3.3 (basket digitals, Bachelier model).

    Args:
        n_samples: Number of input spot vectors.
        d: Number of assets (dimensionality).
        strike: Strike price K.
        base_vol: Base volatility (Bachelier, in price units).
        T: Time to maturity.
        rho: Pairwise equicorrelation.
        k_paths: MC paths per input.
        seed: Random seed.

    Returns:
        Dictionary with x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)

    # Correlation matrix: Σ = (1-ρ)I + ρ 1·1^T
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)

    # Cholesky decomposition for correlated samples
    L = np.linalg.cholesky(corr)  # (d, d)

    # Inverse correlation (equicorrelation has closed form):
    # Σ⁻¹ = (1/(1-ρ)) [I − (ρ/(1+(d-1)ρ)) 1·1^T]
    if d == 1:
        corr_inv = np.array([[1.0]])
    else:
        a = 1.0 / (1.0 - rho)
        b = rho / ((1.0 - rho) * (1.0 + (d - 1) * rho))
        corr_inv = a * np.eye(d) - b * np.ones((d, d))

    # Per-asset volatilities (slight random variation around base)
    rng_vol = np.random.RandomState(seed + 1000)
    sigmas = base_vol * (1.0 + 0.1 * rng_vol.randn(d))
    sigmas = np.abs(sigmas)

    weights = np.ones(d) / d

    # Input spots: x_i ~ N(100, 10²) so basket ≈ ATM
    x = 100.0 + 10.0 * rng.randn(n_samples, d)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths, d))

    for p in range(k_paths):
        # Independent normals → correlated via Cholesky
        Z_indep = rng.standard_normal((n_samples, d))  # (n, d)
        Z_corr = Z_indep @ L.T  # (n, d) — correlated N(0, Σ)

        # Terminal prices: S_{T,i} = x_i + σ_i √T Z_i
        S_T = x + sigmas[None, :] * np.sqrt(T) * Z_corr  # (n, d)

        # Basket value
        basket = (S_T * weights[None, :]).sum(axis=1)  # (n,)

        # Digital payoff
        payoff = (basket > strike).astype(np.float64)  # (n,)

        # LRM score: Σ⁻¹ Z / (σ_i √T)
        # Score vector: [Σ⁻¹ Z]_i / (σ_i √T)
        score = (Z_corr @ corr_inv.T) / (sigmas[None, :] * np.sqrt(T))  # (n, d)

        # LRM delta per component
        lrm_delta = payoff[:, None] * score  # (n, d)

        y_all[:, p] = payoff
        dydx_all[:, p, :] = lrm_delta

    y = y_all.mean(axis=1, keepdims=True)              # (n, 1)
    dydx_lrm = dydx_all.mean(axis=1).reshape(n_samples, 1, d)  # (n, 1, d)
    lrm_var = dydx_all.var(axis=1).mean(axis=1)        # (n,) avg var across dims

    # Analytical values for validation (d=1 case)
    if d == 1:
        sigma_basket = sigmas[0]
        basket_spot = x.flatten()
        d_val = (basket_spot - strike) / (sigma_basket * np.sqrt(T))
        y_exact = scipy_norm.cdf(d_val).reshape(n_samples, 1)
        dydx_exact = (scipy_norm.pdf(d_val) / (sigma_basket * np.sqrt(T))).reshape(n_samples, 1, 1)
    else:
        y_exact = None
        dydx_exact = None

    result = {
        "x": x,
        "y": y,
        "dydx_lrm": dydx_lrm,
        "lrm_var": lrm_var,
        "config": {
            "payoff": "digital_basket",
            "model": "bachelier",
            "label_method": "lrm",
            "d": d,
            "strike": strike,
            "base_vol": base_vol,
            "T": T,
            "rho": rho,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }
    if y_exact is not None:
        result["y_exact"] = y_exact
        result["dydx_exact"] = dydx_exact

    return result


# ============================================================================
# HESTON EULER-SCHEME LRM (NOVEL — G&K DID NOT IMPLEMENT THIS)
# ============================================================================

def lrm_euler_heston(
    n_samples: int,
    strike: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 1.0,
    n_steps: int = 100,
    k_paths: int = 10,
    payoff_type: str = "call",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Heston model option data with Euler-scheme LRM derivative labels.

    Heston SDE:
        dS_t = r S_t dt + √v_t S_t dW_1
        dv_t = κ(θ − v_t) dt + σ_v √v_t dW_2
        Corr(dW_1, dW_2) = ρ

    Euler discretization (log-spot for stability):
        log S_{k+1} = log S_k + (r − v_k/2) Δt + √v_k √Δt Z_{1,k}
        v_{k+1} = v_k + κ(θ − v_k) Δt + σ_v √v_k √Δt Z_{2,k}

    LRM score for ∂/∂S_0 (initial spot):
        The first step's transition density depends on S_0 through log S_0.
        Score of first step: Z_{1,0} / (√v_0 · S_0 · √Δt)
        Subsequent steps are independent of S_0 given S_1.
        Total score: payoff · Z_{1,0} / (√v_0 · S_0 · √Δt)

    For digital payoff (1{S_T > K}), this provides an unbiased delta estimate
    whereas pathwise would give 0 a.e.

    G&K discuss Euler-scheme LRM in §3.5 but never implement it. This is our
    novel contribution — showing where Euler-LRM variance explodes and how
    GradNorm automatically compensates.

    Args:
        n_samples: Number of initial spot prices.
        strike: Strike price K.
        v0: Initial variance.
        kappa: Mean reversion speed.
        theta: Long-term variance.
        sigma_v: Vol-of-vol.
        rho: Spot-vol correlation.
        r: Risk-free rate.
        T: Time to maturity.
        n_steps: Number of Euler steps (more steps → more variance in LRM).
        k_paths: MC paths per input.
        payoff_type: 'call' for max(S_T - K, 0) or 'digital' for 1{S_T > K}.
        seed: Random seed.

    Returns:
        Dictionary with x, y, dydx_lrm, lrm_var, config.

    DEPRECATED: this v1 function uses an incomplete LRM score formula —
    `Z1 / (S_0 √(V_0 Δt))` instead of the correct ρ-corrected
    `(Z_1 − ρ/√(1−ρ²) Z_indep) / (S_0 √(V_0 Δt))` (see derivation at
    `repos/docs/heston_extension/heston_lrm_score_derivation.md`).
    For Heston with ρ ≠ 0 this is a biased-variance estimator (still
    unbiased in mean, but with strictly higher variance than v2 for any
    nonzero leverage). Code-review LOW #1 (2026-05-04).

    PRESERVED unchanged so existing experiments using this function
    (e.g. `results/lrm_comparison/heston_*` rows in our paper) reproduce
    bit-identically. New work should use `lrm_euler_heston_score`.
    """
    warnings.warn(
        "lrm_euler_heston is preserved for reproducibility but uses an "
        "incomplete LRM score formula (missing ρ-correction term). "
        "For new work use lrm_euler_heston_score. See "
        "repos/docs/heston_extension/heston_lrm_score_derivation.md.",
        DeprecationWarning,
        stacklevel=2,
    )

    rng = np.random.RandomState(seed)

    # Input: S_0 values (1D for now — could extend to (S_0, v_0) as in functions.py)
    S0 = rng.uniform(strike * 0.7, strike * 1.3, (n_samples, 1))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        # Initial state
        log_S = np.log(S0.flatten())  # (n,)
        v = np.full(n_samples, v0)     # (n,)

        # Store first step's Z for LRM score
        Z1_first = None

        for step in range(n_steps):
            # Enforce v ≥ 0 (full truncation scheme — Andersen 2008)
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Correlated normals: Z1, Z2 with correlation ρ
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            if step == 0:
                Z1_first = Z1.copy()

            # Euler step (log-spot)
            log_S = log_S + (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1

            # Euler step (variance)
            v = v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * Z2

        S_T = np.exp(log_S)  # (n,)

        # Payoff
        if payoff_type == "call":
            payoff = np.maximum(S_T - strike, 0.0) * discount
        elif payoff_type == "digital":
            payoff = (S_T > strike).astype(np.float64) * discount
        else:
            raise ValueError(f"Unknown payoff_type: {payoff_type}. Use 'call' or 'digital'.")

        # LRM delta w.r.t. S_0:
        # Score of first Euler step: Z_{1,0} / (√v_0 · S_0 · √Δt)
        # (S_0 entered the first step only through log S_0)
        sqrt_v0 = np.sqrt(max(v0, 1e-10))
        lrm_delta = payoff * Z1_first / (S0.flatten() * sqrt_v0 * sqrt_dt)

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
            "payoff": payoff_type,
            "model": "heston_euler",
            "label_method": "lrm",
            "strike": strike,
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "r": r,
            "T": T,
            "n_steps": n_steps,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


# ============================================================================
# CONVENIENCE: PREPARE DATA DICT FOR train_single_experiment()
# ============================================================================

def prepare_for_training(
    data: Dict[str, Any],
    test_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Split LRM data into train/test and format for train_single_experiment().

    Args:
        data: Output from any lrm_* generator (must have 'x', 'y', 'dydx_lrm').
        test_frac: Fraction of data for test set.
        seed: Random seed for split.

    Returns:
        Dictionary with x_train, y_train, dydx_train, x_test, y_test, dydx_test.
    """
    rng = np.random.RandomState(seed)
    n = data["x"].shape[0]
    indices = rng.permutation(n)
    n_test = int(n * test_frac)

    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    return {
        "x_train": data["x"][train_idx],
        "y_train": data["y"][train_idx],
        "dydx_train": data["dydx_lrm"][train_idx],
        "x_test": data["x"][test_idx],
        "y_test": data["y"][test_idx],
        "dydx_test": data["dydx_lrm"][test_idx],
    }


# ============================================================================
# HESTON EULER BARRIER (DOWN-AND-OUT CALL) — LRM LABELS WITH ρ-CORRECTED SCORE
# ============================================================================

def lrm_barrier_heston(
    n_samples: int,
    strike: float = 1.0,
    barrier: float = 0.85,
    v0: float = 0.04,
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
) -> Dict[str, Any]:
    """
    Heston Euler down-and-out call with single intermediate barrier check.

    Generates training data with the **ρ-corrected single-step LRM score**
    for ∂/∂S_0. Derivation in `docs/heston_extension/heston_lrm_score_derivation.md`.

    Payoff: e^{-r T2} * 1{S_{T1} > B} * max(S_{T2} - K, 0).

    By the Markov property of the Heston Euler discretisation, only the FIRST
    Euler step's transition density depends on S_0. The score function is

        score = (Z_{1,0} - (ρ/√(1-ρ²)) Z_{indep,0}) / (S_0 √(V_0 Δt))

    where (Z_{1,0}, Z_{indep,0}) are the first-step normals in our convention
    (Z_1 drives spot directly, Z_2 = ρ Z_1 + √(1-ρ²) Z_indep drives variance).

    NOTE: this fixes a missing `(ρ/√(1-ρ²)) Z_indep` term in the existing
    `lrm_euler_heston` function (which uses Z_1 only — see Bug 4 in
    `docs/heston_extension/verification_synthesis.md`). The existing function
    is preserved unchanged for reproducibility of `results/heston_dig/...`
    runs; this new function is used for fresh Heston barrier experiments.

    Returns shapes:
        x:        (n_samples, 1) spot prices
        y:        (n_samples, 1) MC-averaged barrier-call payoff
        dydx_lrm: (n_samples, 1, 1) ρ-corrected LRM delta labels

    References:
        - Glasserman, P., and S. H. Karmarkar (2025/2026). Differential ML
          with a Difference. arXiv:2512.05301 v2 §3.4 (BS barrier setup),
          eq. \\ref{barrierlrm}; §3.6 (Heston Euler density).
        - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering,
          §7.3.4 (general SDE LRM via Euler density).
        - Heston, S. L. (1993). RFS 6(2), 327-343.
        - Andersen, L. (2008). JCF 11(3), 1-42 (full-truncation Euler).
        - Derivation note: docs/heston_extension/heston_lrm_score_derivation.md.

    Args:
        n_samples: number of input spot prices.
        strike: K.
        barrier: B (knock-out level; B < S_0 expected).
        v0, kappa, theta, sigma_v, rho: Heston parameters.
        r: risk-free rate.
        T1: barrier monitoring time.
        T2: expiry.
        n_substeps_to_T1, n_substeps_T1_to_T2: Euler-step counts.
        k_paths: MC paths per input.
        seed: random seed.

    Returns:
        Dictionary with keys: x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, (n_samples, 1))
    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)

    # ρ correction factor for the LRM score
    if abs(rho) >= 1.0 - 1e-10:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")
    rho_correction = rho / np.sqrt(1.0 - rho ** 2)
    sqrt_v0_dt1 = np.sqrt(max(v0, 1e-12) * dt1)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        # Per-path: simulate from 0 to T1
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)

        Z1_first = None
        Z_indep_first = None

        for step in range(n_substeps_to_T1):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            if step == 0:
                Z1_first = Z1.copy()
                Z_indep_first = Z_indep.copy()

            log_S = log_S + (r - 0.5 * v_pos) * dt1 + sqrt_v * sqrt_dt1 * Z1
            v = v + kappa * (theta - v_pos) * dt1 + sigma_v * sqrt_v * sqrt_dt1 * Z2

        # Barrier check at T1
        S_T1 = np.exp(log_S)
        alive = (S_T1 > barrier).astype(np.float64)

        # Phase 2: simulate from T1 to T2
        for step in range(n_substeps_T1_to_T2):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            log_S = log_S + (r - 0.5 * v_pos) * dt2 + sqrt_v * sqrt_dt2 * Z1
            v = v + kappa * (theta - v_pos) * dt2 + sigma_v * sqrt_v * sqrt_dt2 * Z2

        S_T2 = np.exp(log_S)
        payoff = np.maximum(S_T2 - strike, 0.0) * alive * discount

        # ρ-corrected LRM score for ∂/∂S_0
        # See heston_lrm_score_derivation.md §4
        z_score = Z1_first - rho_correction * Z_indep_first
        lrm_delta = payoff * z_score / (S0.flatten() * sqrt_v0_dt1)

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
            "model": "heston_full_truncation_euler",
            "label_method": "lrm_rho_corrected",
            "strike": strike,
            "barrier": barrier,
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "rho_correction_term": rho_correction,
            "r": r,
            "T1": T1,
            "T2": T2,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def lrm_multistep_heston_barrier(
    n_samples: int,
    strike: float = 1.0,
    barrier: float = 0.85,
    v0: float = 0.04,
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
) -> Dict[str, Any]:
    """
    Multi-step LRM for Heston barrier ∂/∂S_0 — 2D BEL form.

    BUG FIX (2026-05-05): a previous version of this function applied
    Chen-Glasserman 2007 Eq. (10) using the spot-driver Z_{1,i} only,
    which is the correct 1D specialisation but is BIASED by ~6% on Heston
    when ρ ≠ 0. The 1D theorem assumes σ depends on the same coordinate
    as the score variable; in Heston σ = √V depends on V (not log S), and
    V is correlated with Z_1 through Z_2 = ρ Z_1 + √(1-ρ²) Z_indep, so
    the joint-density factorisation gives a per-step conditional Gaussian
    for log S_{i+1} with variance (1-ρ²) V_i⁺ Δt and mean shifted by
    ρ √(V_i⁺ Δt) Z_{2,i+1}. The unbiased per-step score uses the
    orthogonal-to-V noise

        ξ_perp,i = √(1-ρ²) Z_{1,i} - ρ Z_indep,i

    or equivalently in (Z_1, Z_indep) coordinates,

        Z_{1,i} - (ρ/√(1-ρ²)) Z_indep,i.

    The Δ estimator (BEL-2D form, equivalent to single-step ρ-corrected
    score at N=1):

        Δ̂ = π(path) · (1/(S_0 T)) · Σ_{i=1..N} √Δt
                     · (Z_{1,i} - (ρ/√(1-ρ²)) Z_indep,i) / √(V_{i-1}^+)

    Empirical FD comparison (production parameters: S_0=1, V_0=θ=0.04,
    κ=1, σ_v=0.15, ρ=−0.7, T_1=1/3, T_2=2/3, 168 substeps,
    5 seeds × 5×10⁵ paths):
        FD reference     Δ = 0.5685
        BEL-2D (this)    Δ = 0.5668  (−0.3%, within MC noise)
        Z₁-only (old)    Δ = 0.5324  (−6.4% systematic bias)

    The Z₁-only form coincides with this one at ρ=0 (no V-S correlation).

    Variance: discrete BEL-2D variance is bounded in N, in contrast to
    the single-step LRM whose variance scales as N. For Feller-violating
    parameter regimes (2κθ ≤ σ_v²) the non-degeneracy assumption fails
    and a small floor V_i⁺ → max(V_i, v_floor) on the LRM denominator is
    required; we use v_floor = 1e-8.

    Returns shapes:
        x:        (n_samples, 1) spot prices
        y:        (n_samples, 1) MC-averaged barrier-call payoff
        dydx_lrm: (n_samples, 1, 1) multi-step LRM delta labels

    References:
        - paper/sections/E_theory_crossover.tex §E.3
          (\ref{eq:lrm-bel2d}) — paper-ready derivation.
        - paper/agents/HESTON_MATH_VERIFICATION.md §4 — full proof and FD
          empirical unbiasedness check.
        - Chen, N., and P. Glasserman (2007). "Malliavin Greeks without
          Malliavin calculus", SPA 117(11), 1689-1723. (1D theorem; BEL-2D
          is the 2D extension specific to Heston-type V-correlated SDEs.)
        - Fournié, E., J.-M. Lasry, J. Lebuchoux, P.-L. Lions, N. Touzi
          (1999). Finance & Stochastics 3(4), 391-412. (Continuous-time
          Bismut-Elworthy-Li / Malliavin Δ; BEL-2D is its discrete form.)
        - Alòs, E., D. García-Lorite (2021). Malliavin Calculus in Finance,
          CRC Press. Reference impl: Dagalon/PyStochasticVolatility (Apache-2.0).
        - Heston (1993) RFS 6(2); Andersen (2008) JCF 11(3) (full-truncation
          Euler scheme).
        - Glasserman, P., and S. H. Karmarkar (2025). arXiv:2512.05301
          §3.6 (single-step Heston LRM; equivalent to our (*) up to noise
          basis rotation).

    Args:
        n_samples: number of input spot prices.
        strike: K.
        barrier: B (knock-out level; B < S_0 expected).
        v0, kappa, theta, sigma_v, rho: Heston parameters.
        r: risk-free rate.
        T1: barrier monitoring time.
        T2: expiry.
        n_substeps_to_T1, n_substeps_T1_to_T2: Euler-step counts.
        k_paths: MC paths per input.
        v_floor: small-V truncation to prevent 1/sqrt(0) (CG Assumption 3.2(2)).
        seed: random seed.

    Returns:
        Dictionary with keys: x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, (n_samples, 1))
    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)
    T_total = T2  # full integration window (CG Eq. 10 prefactor 1/T)
    N_total = n_substeps_to_T1 + n_substeps_T1_to_T2

    if abs(rho) >= 1.0 - 1e-10:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")
    # ρ-correction for the per-step orthogonal-to-vol noise. The naive
    # CG2007 Eq. (10) specialisation that accumulates only Z_{1,i} is biased
    # when ρ ≠ 0 because Heston is 2D: σ = √V depends on V, and V is
    # correlated with Z_1 through Z_2 = ρ Z_1 + √(1-ρ²) Z_indep. The
    # joint-density factorisation
    #   p(log S_{i+1}, V_{i+1} | log S_i, V_i)
    #     = p(V_{i+1} | log S_i, V_i) · p(log S_{i+1} | log S_i, V_i, V_{i+1})
    # gives a conditional Gaussian for log S_{i+1} with variance
    # (1-ρ²) V_i⁺ Δt and mean shifted by ρ √(V_i⁺ Δt) Z_{2,i+1}. The unbiased
    # per-step score uses the orthogonal-to-V noise
    #   ξ_perp,i = √(1-ρ²) Z_{1,i} - ρ Z_indep,i
    # Equivalent in (Z_1, Z_indep) coordinates: (Z_{1,i} - (ρ/√(1-ρ²)) Z_indep,i).
    # See paper/sections/E_theory_crossover.tex §E.3 (BEL-2D form),
    # paper/agents/HESTON_MATH_VERIFICATION.md §4 for the full derivation
    # and the FD empirical unbiasedness check (mean Δ matches FD to 3 decimals
    # vs −6.4% systematic bias for the Z₁-only form).
    rho_correction = rho / np.sqrt(1.0 - rho ** 2)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)
        # BEL-2D weight accumulator: Σ_i √Δt · (Z_{1,i} - (ρ/√(1-ρ²)) Z_indep,i)
        #                              / √(V_{i-1}^+)
        weight_sum = np.zeros(n_samples)

        # Phase 1: simulate to T1, accumulating weight
        # NOTE: v_floor applies ONLY to the LRM weight denominator (CG 3.2(2)
        # non-degeneracy guard against 1/√0). The Euler simulation update uses
        # max(V, 0) — the standard Andersen 2008 full-truncation scheme, matching
        # the MC reference and our single-step LRM. Decoupling these two avoids
        # a small numerical inconsistency between the simulated SDE and the
        # reference. (Code review 2026-05-04, MED #1.)
        for step in range(n_substeps_to_T1):
            v_pos_sim = np.maximum(v, 0.0)        # for Euler simulation
            v_pos_score = np.maximum(v, v_floor)  # for LRM weight only
            sqrt_v_sim = np.sqrt(v_pos_sim)
            sqrt_v_score = np.sqrt(v_pos_score)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            # BEL-2D per-step increment using V_{i-1}^+ (BEFORE i-th step update)
            weight_sum = weight_sum + (
                sqrt_dt1 * (Z1 - rho_correction * Z_indep) / sqrt_v_score
            )

            # Euler updates use sim variant (matches Andersen 2008 + MC reference)
            log_S = log_S + (r - 0.5 * v_pos_sim) * dt1 + sqrt_v_sim * sqrt_dt1 * Z1
            v = v + kappa * (theta - v_pos_sim) * dt1 + sigma_v * sqrt_v_sim * sqrt_dt1 * Z2

        # Barrier check at T1
        S_T1 = np.exp(log_S)
        alive = (S_T1 > barrier).astype(np.float64)

        # Phase 2: simulate from T1 to T2, continuing weight accumulation
        for step in range(n_substeps_T1_to_T2):
            v_pos_sim = np.maximum(v, 0.0)
            v_pos_score = np.maximum(v, v_floor)
            sqrt_v_sim = np.sqrt(v_pos_sim)
            sqrt_v_score = np.sqrt(v_pos_score)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            weight_sum = weight_sum + (
                sqrt_dt2 * (Z1 - rho_correction * Z_indep) / sqrt_v_score
            )

            log_S = log_S + (r - 0.5 * v_pos_sim) * dt2 + sqrt_v_sim * sqrt_dt2 * Z1
            v = v + kappa * (theta - v_pos_sim) * dt2 + sigma_v * sqrt_v_sim * sqrt_dt2 * Z2

        S_T2 = np.exp(log_S)
        payoff = np.maximum(S_T2 - strike, 0.0) * alive * discount

        # Multi-step LRM delta = (1/S_0) · payoff · (1/T) · weight_sum
        lrm_delta = payoff * weight_sum / (T_total * S0.flatten())

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
            "model": "heston_full_truncation_euler",
            "label_method": "lrm_multistep_bel2d",
            "strike": strike,
            "barrier": barrier,
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "r": r,
            "T1": T1,
            "T2": T2,
            "T_total": T_total,
            "N_total": N_total,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "v_floor": v_floor,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def bel_barrier_heston(
    n_samples: int,
    strike: float = 1.0,
    barrier: float = 0.85,
    v0: float = 0.04,
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
) -> Dict[str, Any]:
    """
    Fournié-localised Malliavin Δ estimator (BEL: Bermudan-Equity-Linked,
    after Alòs–García-Lorite 2021 and the Dagalon QE-weight reference impl)
    for Heston barrier ∂/∂S_0.

    Differs from `lrm_multistep_heston_barrier` by integrating the ρ-corrected
    orthogonal-to-vol noise at every step, instead of only the spot-driver Z_1.

    The BEL Δ-weight in our (Z_1, Z_indep) convention (derivation: orthogonal
    rotation around the (Z_1, Z_2) basis where Z_2 = ρ Z_1 + √(1-ρ²) Z_indep
    drives variance):

        Δ_BEL = π(path) · (1/(T·S_0)) · Σ_i √Δt_i ·
                (Z_{1,i} − (ρ/√(1-ρ²)) Z_{indep,i}) / √(V_{i-1}^+)

    Equivalently, with ξ_i = √(1-ρ²) Z_{1,i} − ρ Z_{indep,i} (orthogonal-to-V
    increment, unit variance), this is the canonical Fournié–Lasry–
    Lebuchoux–Lions–Touzi 1999 (Prop 3.2) discrete weight:

        Δ_BEL = π(path) · (1/(√(1-ρ²)·T·S_0)) · Σ_i √Δt_i · ξ_i / √(V_{i-1}^+)

    Both forms are pointwise equal; we use the first form to share simulation
    code with `lrm_multistep_heston_barrier`.

    Comparison with the existing CG2007 multi-step:
      - CG2007 uses Z_{1,i} only inside the score (no Z_indep correction);
        its derivation conditions on the V-path and treats σ = √V exogenously.
      - BEL uses the orthogonal-to-V combination; its derivation perturbs the
        S_0-dependent measure along the W^S direction localised onto the
        component orthogonal to W^V.
      - At ρ = 0 they coincide. At ρ ≠ 0 the BEL adds noise via Z_indep but
        is the canonical Fournié estimator.

    Variance caveat: same `inf σ(x) > 0` requirement as CG2007. We clip
    V^+ ≤ v_floor in the LRM denominator only; the Euler simulation uses
    max(V, 0) full-truncation (Andersen 2008).

    Returns shapes:
        x:        (n_samples, 1) spot prices
        y:        (n_samples, 1) MC-averaged barrier-call payoff
        dydx_lrm: (n_samples, 1, 1) BEL delta labels

    References:
        - Fournié, E., J.-M. Lasry, J. Lebuchoux, P.-L. Lions, N. Touzi (1999).
          "Applications of Malliavin calculus to Monte Carlo methods in
          finance", Finance and Stochastics 3(4), 391-412. Prop. 3.2 and §4.
        - Alòs, E., D. García-Lorite (2021). Malliavin Calculus in Finance,
          CRC Press. Companion code: github.com/Dagalon/PyStochasticVolatility,
          MC_Engines/MC_Heston/HestonTools.py (Apache-2.0).
        - Chen, N., P. Glasserman (2007). SPA 117(11), 1689-1723.
          (CG2007 multi-step is the special case Z_indep ≡ 0.)
        - Heston, S. L. (1993). RFS 6(2), 327-343.
        - Andersen, L. (2008). JCF 11(3), 1-42.
        - Search log: repos/docs/heston_extension/agents/bel_malliavin_heston_search.md.

    Args:
        n_samples, strike, barrier, v0, kappa, theta, sigma_v, rho, r, T1, T2,
        n_substeps_to_T1, n_substeps_T1_to_T2, k_paths, v_floor, seed,
        spot_low_mult, spot_high_mult: as in `lrm_multistep_heston_barrier`.

    Returns:
        Dictionary with keys: x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, (n_samples, 1))
    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)
    T_total = T2
    N_total = n_substeps_to_T1 + n_substeps_T1_to_T2

    if abs(rho) >= 1.0 - 1e-10:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")
    rho_correction = rho / np.sqrt(1.0 - rho ** 2)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)
        weight_sum = np.zeros(n_samples)

        for step in range(n_substeps_to_T1):
            v_pos_sim = np.maximum(v, 0.0)
            v_pos_score = np.maximum(v, v_floor)
            sqrt_v_sim = np.sqrt(v_pos_sim)
            sqrt_v_score = np.sqrt(v_pos_score)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            # BEL: orthogonal-to-vol perturbation at this step
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

        bel_delta = payoff * weight_sum / (T_total * S0.flatten())

        y_all[:, p] = payoff
        dydx_all[:, p] = bel_delta

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
            "model": "heston_full_truncation_euler",
            "label_method": "bel_fournie_localized_malliavin",
            "strike": strike,
            "barrier": barrier,
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "r": r,
            "T1": T1,
            "T2": T2,
            "T_total": T_total,
            "N_total": N_total,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "v_floor": v_floor,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def lrm_multistep_heston_digital(
    n_samples: int,
    strike: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 1.0,
    n_steps: int = 100,
    k_paths: int = 10,
    payoff_type: str = "digital",
    v_floor: float = 1e-8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Multi-step LRM for Heston digital/call ∂/∂S_0 — 2D BEL form.

    Same formula as `lrm_multistep_heston_barrier` but for terminal-only
    payoffs (no barrier monitoring). Bug-fixed 2026-05-05 to use the BEL-2D
    per-step orthogonal-to-V noise (Z_{1,i} − (ρ/√(1-ρ²)) Z_indep,i) instead
    of Z_{1,i} alone — see `lrm_multistep_heston_barrier` for the full
    derivation, references, and FD empirical unbiasedness check.

        Δ = (1/(S_0 T)) · Φ(S_T) · Σ_{i=1..N} √Δt
                          · (Z_{1,i} − (ρ/√(1-ρ²)) Z_indep,i) / √(V_{i-1}^+)

    Args:
        Same as `lrm_euler_heston` plus `v_floor`.

    Returns:
        Dictionary with keys: x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * 0.7, strike * 1.3, (n_samples, 1))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)

    if abs(rho) >= 1.0 - 1e-10:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")
    rho_correction = rho / np.sqrt(1.0 - rho ** 2)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)
        weight_sum = np.zeros(n_samples)

        # Same v_floor decoupling as lrm_multistep_heston_barrier:
        # floor applies ONLY to LRM weight denominator (1/√V guard); Euler
        # simulation uses max(V, 0) to match Andersen 2008 + MC reference.
        for step in range(n_steps):
            v_pos_sim = np.maximum(v, 0.0)
            v_pos_score = np.maximum(v, v_floor)
            sqrt_v_sim = np.sqrt(v_pos_sim)
            sqrt_v_score = np.sqrt(v_pos_score)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            weight_sum = weight_sum + (
                sqrt_dt * (Z1 - rho_correction * Z_indep) / sqrt_v_score
            )

            log_S = log_S + (r - 0.5 * v_pos_sim) * dt + sqrt_v_sim * sqrt_dt * Z1
            v = v + kappa * (theta - v_pos_sim) * dt + sigma_v * sqrt_v_sim * sqrt_dt * Z2

        S_T = np.exp(log_S)
        if payoff_type == "call":
            payoff = np.maximum(S_T - strike, 0.0) * discount
        elif payoff_type == "digital":
            payoff = (S_T > strike).astype(np.float64) * discount
        else:
            raise ValueError(f"Unknown payoff_type: {payoff_type}")

        lrm_delta = payoff * weight_sum / (T * S0.flatten())

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
            "payoff": payoff_type,
            "model": "heston_full_truncation_euler",
            "label_method": "lrm_multistep_bel2d",
            "strike": strike,
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "r": r,
            "T": T,
            "n_steps": n_steps,
            "v_floor": v_floor,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }


def lrm_euler_heston_score(
    n_samples: int,
    strike: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 1.0,
    n_steps: int = 100,
    k_paths: int = 10,
    payoff_type: str = "call",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Heston Euler digital/call with ρ-corrected single-step LRM score for ∂/∂S_0.

    Replaces the buggy `lrm_euler_heston` (Bug 4 fix). The original function
    used `Z_1 / (S_0 √(V_0 Δt))` — missing the `-(ρ/√(1-ρ²)) Z_indep`
    correction. This version implements the full ρ-corrected score.

    Behaviour identical to `lrm_euler_heston` except the score formula uses
    both Z_{1,0} and Z_{indep,0} from the first Euler step (see derivation in
    docs/heston_extension/heston_lrm_score_derivation.md).

    The original `lrm_euler_heston` is preserved unchanged for reproducibility
    of existing `results/heston_dig/*` runs; this v2 is used for new
    experiments.

    Args:
        Same as `lrm_euler_heston`.

    Returns:
        Dictionary with keys: x, y, dydx_lrm, lrm_var, config.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * 0.7, strike * 1.3, (n_samples, 1))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)

    if abs(rho) >= 1.0 - 1e-10:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")
    rho_correction = rho / np.sqrt(1.0 - rho ** 2)
    sqrt_v0_dt = np.sqrt(max(v0, 1e-12) * dt)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)

        Z1_first = None
        Z_indep_first = None

        for step in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

            if step == 0:
                Z1_first = Z1.copy()
                Z_indep_first = Z_indep.copy()

            log_S = log_S + (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1
            v = v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * Z2

        S_T = np.exp(log_S)

        if payoff_type == "call":
            payoff = np.maximum(S_T - strike, 0.0) * discount
        elif payoff_type == "digital":
            payoff = (S_T > strike).astype(np.float64) * discount
        else:
            raise ValueError(f"Unknown payoff_type: {payoff_type}. Use 'call' or 'digital'.")

        # ρ-corrected score
        z_score = Z1_first - rho_correction * Z_indep_first
        lrm_delta = payoff * z_score / (S0.flatten() * sqrt_v0_dt)

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
            "payoff": payoff_type,
            "model": "heston_full_truncation_euler",
            "label_method": "lrm_rho_corrected_v2",
            "strike": strike,
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "rho_correction_term": rho_correction,
            "r": r,
            "T": T,
            "n_steps": n_steps,
            "k_paths": k_paths,
            "n_samples": n_samples,
            "seed": seed,
        },
    }
