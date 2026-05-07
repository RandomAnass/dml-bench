"""
Fuzzy Logic Payoff Smoothing for Differential Machine Learning.

Implements Savine's call-spread (cSpr) smoothing to replace discontinuous
payoff indicators with smooth approximations, enabling valid pathwise
derivative estimation. This is a direct Python translation of the fuzzy
logic evaluation in Savine's C++ Scripting library:
    https://github.com/asavine/Scripting  →  scriptingFuzzyEval.h

Key operations:
    cSpr(x, ε)   — call spread: smooth step from 0 to 1 around x=0
    bFly(x, ε)   — butterfly:  peaked at x=0, smooth Dirac approximation
    AND(DT1, DT2) = DT1 * DT2
    OR(DT1, DT2)  = DT1 + DT2 - DT1 * DT2
    NOT(DT)        = 1 - DT
    IF C THEN S1 ELSE S2  →  DT * S1 + (1-DT) * S2

ε calibration (following Savine's PDF §11–12):
    ε = eps_mult × σ × √T × S_0   (for GBM-based digital options)
    where eps_mult is a fraction (typical: 0.1 – 1.0) and σ√T S_0 is an
    approximation of the standard deviation of the condition expression.

Integration:
    All generators return numpy arrays compatible with train_single_experiment():
        x:         (n_samples, d)
        y:         (n_samples, 1)      — smoothed payoff
        dydx:      (n_samples, 1, d)   — pathwise delta of smoothed payoff
        y_exact:   (n_samples, 1)      — analytical exact price (not smoothed)
        dydx_exact:(n_samples, 1, d)   — analytical exact delta

    The key insight: we TRAIN on fuzzy-smoothed labels but EVALUATE on exact
    analytical values. This is fair because the smoothing is a training
    technique, and we want to measure how well the network extrapolates the
    true discontinuous function.

Reference:
    Antoine Savine, "Fuzzy Payoff Evaluation for Stable Risk Sensitivities",
    Danske Bank presentation (2024). Available via Brian Huge.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    from scipy.stats import norm as scipy_norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# CORE FUZZY FUNCTIONS (from scriptingFuzzyEval.h)
# ============================================================================

def call_spread(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Call spread smoothing — smooth step from 0 to 1.

    cSpr(x, ε) = clip((x + ε/2) / ε, 0, 1)

    Equivalent to Savine's scriptingFuzzyEval.h::cSpr():
        if x < -eps/2: return 0
        if x >  eps/2: return 1
        return (x + eps/2) / eps

    Args:
        x: Condition expression values (e.g., S_T - K for digital)
        eps: Smoothing bandwidth (> 0)

    Returns:
        Degree of truth (DT) ∈ [0, 1]
    """
    assert eps > 0, f"eps must be positive, got {eps}"
    return np.clip((x + eps / 2.0) / eps, 0.0, 1.0)


def call_spread_deriv(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Derivative of call spread w.r.t. x.

    d/dx cSpr(x, ε) = 1/ε  if |x| < ε/2, else 0.

    This is the smooth approximation to the Dirac delta.

    Args:
        x: Condition expression values
        eps: Smoothing bandwidth

    Returns:
        Derivative values
    """
    assert eps > 0
    return np.where(np.abs(x) < eps / 2.0, 1.0 / eps, 0.0)


def butterfly(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Butterfly smoothing — smooth peaked function at x=0.

    bFly(x, ε) = max(0, (ε/2 - |x|)) / (ε/2)

    Used for equality conditions (x == 0) → degree of truth.
    From scriptingFuzzyEval.h::bFly().

    Args:
        x: Condition expression values
        eps: Smoothing bandwidth

    Returns:
        Degree of truth ∈ [0, 1], peaked at x=0
    """
    assert eps > 0
    return np.maximum(0.0, (eps / 2.0 - np.abs(x))) / (eps / 2.0)


def fuzzy_and(dt1: np.ndarray, dt2: np.ndarray) -> np.ndarray:
    """AND(DT1, DT2) = DT1 * DT2 — product rule for degrees of truth."""
    return dt1 * dt2


def fuzzy_or(dt1: np.ndarray, dt2: np.ndarray) -> np.ndarray:
    """OR(DT1, DT2) = DT1 + DT2 - DT1*DT2 — probabilistic union."""
    return dt1 + dt2 - dt1 * dt2


def fuzzy_not(dt: np.ndarray) -> np.ndarray:
    """NOT(DT) = 1 - DT — complement."""
    return 1.0 - dt


def fuzzy_if(dt: np.ndarray, val_true: np.ndarray, val_false: np.ndarray) -> np.ndarray:
    """
    IF C THEN val_true ELSE val_false → DT * val_true + (1-DT) * val_false.

    From scriptingFuzzyEval.h::visit(NodeIf):
        Save state → evaluate if_true → save S1 → restore → evaluate if_false
        → save S2 → blend: DT * S1 + (1-DT) * S2
    """
    return dt * val_true + (1.0 - dt) * val_false


# ============================================================================
# ε CALIBRATION
# ============================================================================

def calibrate_epsilon(
    condition_values: np.ndarray,
    eps_mult: float = 0.5,
    min_eps: float = 1e-6,
) -> float:
    """
    Calibrate ε from the standard deviation of the condition expression.

    Following Savine's recommendation (PDF §11–12):
        ε = eps_mult × std(condition)

    With a pre-simulation of the condition expression to estimate its spread.

    Args:
        condition_values: Array of condition expression evaluations
        eps_mult: Fraction of std to use as ε (typically 0.1–1.0)
        min_eps: Minimum ε to avoid division by zero

    Returns:
        Calibrated ε value
    """
    std = np.std(condition_values)
    return max(eps_mult * std, min_eps)


# ============================================================================
# BLACK-SCHOLES DIGITAL CALL — FUZZY-SMOOTHED LABELS
# ============================================================================

def fuzzy_digital_bs(
    n_samples: int,
    strike: float = 100.0,
    vol: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    k_paths: int = 10,
    eps_mult: float = 0.5,
    eps_override: float = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Black-Scholes digital call data with fuzzy-smoothed pathwise labels.

    Digital payoff: π(S_T) = 1{S_T > K}
    Fuzzy payoff:   π_ε(S_T) = cSpr(S_T − K, ε)
    Fuzzy delta:    dπ_ε/dS_0 = (1/ε) · 1{|S_T − K| < ε/2} · (S_T / S_0)

    The last factor S_T/S_0 is the standard GBM sensitivity: ∂S_T/∂S_0 = S_T/S_0.

    ε is calibrated as: eps_mult × std(S_T − K), estimated via pre-simulation,
    OR set directly via eps_override.

    Args:
        n_samples: Number of input spot prices to generate.
        strike: Strike price K.
        vol: Black-Scholes volatility σ.
        r: Risk-free rate.
        T: Time to maturity.
        k_paths: MC paths per input for noise reduction.
        eps_mult: Fraction of std(S_T − K) for ε calibration.
        eps_override: If not None, use this ε directly (ignores eps_mult).
        seed: Random seed.

    Returns:
        Dictionary with keys:
            x:            (n_samples, 1) spot prices
            y:            (n_samples, 1) fuzzy-smoothed digital prices
            dydx_fuzzy:   (n_samples, 1, 1) smoothed pathwise delta
            y_exact:      (n_samples, 1) analytical digital call price
            dydx_exact:   (n_samples, 1, 1) analytical exact delta
            epsilon:      float, calibrated ε used
            config:       dict of parameters
    """
    assert SCIPY_AVAILABLE, "scipy required for Black-Scholes analytics"
    rng = np.random.RandomState(seed)

    # Spot prices S_0 ∈ [0.5K, 1.5K]
    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))  # (n, 1)

    # Simulate k_paths per input: Z ~ N(0,1) shape (n_samples, k_paths)
    Z = rng.standard_normal((n_samples, k_paths))

    # Terminal prices: S_T = S_0 · exp((r − σ²/2)T + σ√T Z)
    drift = (r - 0.5 * vol ** 2) * T
    diffuse = vol * np.sqrt(T)
    S_T = S0 * np.exp(drift + diffuse * Z)  # (n, k)

    # Condition expression: S_T − K
    condition = S_T - strike  # (n, k)

    # Calibrate ε
    if eps_override is not None:
        epsilon = eps_override
    else:
        epsilon = calibrate_epsilon(condition.flatten(), eps_mult=eps_mult)

    # Discounting
    discount = np.exp(-r * T)

    # Fuzzy-smoothed payoff: cSpr(S_T − K, ε)
    fuzzy_payoff = call_spread(condition, epsilon) * discount  # (n, k)

    # Fuzzy pathwise delta:
    #   dπ_ε/dS_0 = cSpr'(S_T − K, ε) · (∂S_T/∂S_0) · discount
    #             = (1/ε) · 1{|S_T−K| < ε/2} · (S_T / S_0) · discount
    sensitivity = S_T / S0  # ∂S_T/∂S_0 for GBM = S_T / S_0
    fuzzy_delta_per_path = call_spread_deriv(condition, epsilon) * sensitivity * discount  # (n, k)

    # Average over k paths
    y = fuzzy_payoff.mean(axis=1, keepdims=True)                        # (n, 1)
    dydx_fuzzy = fuzzy_delta_per_path.mean(axis=1, keepdims=True)       # (n, 1)
    dydx_fuzzy = dydx_fuzzy.reshape(n_samples, 1, 1)

    # Analytical (exact) values for validation
    d2 = (np.log(S0 / strike) + (r - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    y_exact = discount * scipy_norm.cdf(d2)
    dydx_exact = (discount * scipy_norm.pdf(d2) / (S0 * vol * np.sqrt(T)))
    dydx_exact = dydx_exact.reshape(n_samples, 1, 1)

    return {
        "x": S0,
        "y": y,
        "dydx_fuzzy": dydx_fuzzy,
        "y_exact": y_exact,
        "dydx_exact": dydx_exact,
        "epsilon": float(epsilon),
        "config": {
            "payoff": "digital_call",
            "model": "black_scholes",
            "label_method": "fuzzy_callspread",
            "strike": strike,
            "vol": vol,
            "r": r,
            "T": T,
            "k_paths": k_paths,
            "eps_mult": eps_mult,
            "eps_override": eps_override,
            "epsilon_used": float(epsilon),
            "n_samples": n_samples,
            "seed": seed,
        },
    }


# ============================================================================
# BLACK-SCHOLES BARRIER OPTION — FUZZY-SMOOTHED LABELS
# ============================================================================

def fuzzy_barrier_bs(
    n_samples: int,
    strike: float = 100.0,
    barrier: float = 80.0,
    vol: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    n_steps: int = 252,
    k_paths: int = 10,
    eps_mult: float = 0.5,
    eps_barrier_mult: float = 0.5,
    eps_strike_override: float = None,
    eps_barrier_override: float = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Black-Scholes knock-out barrier call with fuzzy-smoothed labels.

    Payoff: max(S_T − K, 0) · 1{min_t S_t > B}  (down-and-out call)

    Fuzzy barrier payoff:
        survival_dt = Π_t cSpr(S_t − B, ε_B)         (fuzzy AND over steps)
        call_spread  = cSpr(S_T − K, ε_K) · (S_T − K)  ... or just max(S_T − K, 0)
        fuzzy_payoff = call_payoff · survival_dt

    For the call constituent, we keep the max(S_T − K, 0) as-is (it's already
    differentiable everywhere except at S_T = K, which is measure-zero). The key
    discontinuity to smooth is the barrier indicator.

    Pathwise delta via chain rule through the product:
        d/dS_0 [call_payoff · survival_dt]
        = (d call_payoff/dS_0) · survival_dt + call_payoff · (d survival_dt/dS_0)

    Args:
        n_samples: Number of input spot prices.
        strike: Strike K.
        barrier: Lower barrier B < K.
        vol: Volatility σ.
        r: Risk-free rate.
        T: Time to maturity.
        n_steps: Number of barrier monitoring steps.
        k_paths: MC paths per input.
        eps_mult: ε multiplier for the strike condition.
        eps_barrier_mult: ε multiplier for barrier condition.
        eps_strike_override: Direct ε for strike.
        eps_barrier_override: Direct ε for barrier.
        seed: Random seed.

    Returns:
        Dictionary with x, y, dydx_fuzzy, epsilon_strike, epsilon_barrier, config.
    """
    rng = np.random.RandomState(seed)

    S0 = rng.uniform(strike * 0.5, strike * 1.5, (n_samples, 1))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    # Pre-simulation for ε calibration (use a separate seed to not disturb main sim)
    rng_cal = np.random.RandomState(seed + 7777)
    n_cal = min(n_samples, 1000)
    S0_cal = rng_cal.uniform(strike * 0.5, strike * 1.5, (n_cal, 1))

    # Quick simulation to get condition expression std
    Z_cal = rng_cal.standard_normal((n_cal, n_steps))
    S_cal = S0_cal.copy()
    barrier_conditions = []
    for step in range(n_steps):
        S_cal = S_cal * np.exp((r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z_cal[:, step:step + 1])
        barrier_conditions.append(S_cal.flatten() - barrier)
    terminal_condition = S_cal.flatten() - strike

    # Calibrate ε values
    if eps_strike_override is not None:
        eps_strike = eps_strike_override
    else:
        eps_strike = calibrate_epsilon(terminal_condition, eps_mult=eps_mult)

    if eps_barrier_override is not None:
        eps_barrier = eps_barrier_override
    else:
        all_barrier_cond = np.concatenate(barrier_conditions)
        eps_barrier = calibrate_epsilon(all_barrier_cond, eps_mult=eps_barrier_mult)

    # Main simulation
    for p in range(k_paths):
        Z_all = rng.standard_normal((n_samples, n_steps))

        S = S0.copy()
        # Track smoothed survival: product of cSpr(S_t − B, ε_B)
        survival_dt = np.ones(n_samples)  # starts at 1
        # Track d(survival_dt)/dS_0 via product rule
        d_survival_dS0 = np.zeros(n_samples)

        # Track ∂S_t/∂S_0 = S_t / S_0 for GBM multiplicative paths
        sensitivity = np.ones(n_samples)  # ∂S_t/∂S_0 at step t

        for step in range(n_steps):
            Z_step = Z_all[:, step:step + 1]
            R_step = np.exp((r - 0.5 * vol ** 2) * dt + vol * sqrt_dt * Z_step).flatten()
            S_old = S.flatten()
            S = (S_old * R_step).reshape(-1, 1)
            sensitivity = sensitivity * R_step  # ∂S_t/∂S_0 = Π R_step

            # Fuzzy barrier indicator for this step
            barrier_cond = S.flatten() - barrier
            dt_barrier = call_spread(barrier_cond, eps_barrier)  # DT_t ∈ [0,1]
            d_dt_barrier_dSt = call_spread_deriv(barrier_cond, eps_barrier)  # d DT_t / d S_t
            d_dt_barrier_dS0 = d_dt_barrier_dSt * sensitivity  # chain: d DT_t / d S_0

            # Update survival: survival = Π DT_t  (fuzzy AND)
            # d(Π DT)/dS_0 = Σ_t [ (Π_{j≠t} DT_j) · d DT_t/dS_0 ]
            # Using the identity: d(A·B)/dS_0 = dA/dS_0 · B + A · dB/dS_0
            d_survival_dS0 = d_survival_dS0 * dt_barrier + survival_dt * d_dt_barrier_dS0
            survival_dt = survival_dt * dt_barrier

        S_T = S.flatten()

        # Call payoff (NOT smoothed — already differentiable except at S_T = K)
        call_payoff = np.maximum(S_T - strike, 0.0)
        call_indicator = (S_T > strike).astype(np.float64)
        d_call_dS0 = call_indicator * sensitivity  # ∂max(S-K,0)/∂S_0 = 1{S>K} · ∂S_T/∂S_0

        # Full payoff = call_payoff × survival_dt × discount
        payoff = call_payoff * survival_dt * discount

        # Full delta via product rule:
        # d[call · surv]/dS_0 = d_call/dS_0 · surv + call · d_surv/dS_0
        delta = (d_call_dS0 * survival_dt + call_payoff * d_survival_dS0) * discount

        y_all[:, p] = payoff
        dydx_all[:, p] = delta

    y = y_all.mean(axis=1, keepdims=True)
    dydx_fuzzy = dydx_all.mean(axis=1).reshape(n_samples, 1, 1)

    return {
        "x": S0,
        "y": y,
        "dydx_fuzzy": dydx_fuzzy,
        "epsilon_strike": float(eps_strike),
        "epsilon_barrier": float(eps_barrier),
        "config": {
            "payoff": "barrier_call_knock_out",
            "model": "black_scholes",
            "label_method": "fuzzy_callspread",
            "strike": strike,
            "barrier": barrier,
            "vol": vol,
            "r": r,
            "T": T,
            "n_steps": n_steps,
            "k_paths": k_paths,
            "eps_mult": eps_mult,
            "eps_barrier_mult": eps_barrier_mult,
            "eps_strike": float(eps_strike),
            "eps_barrier": float(eps_barrier),
            "n_samples": n_samples,
            "seed": seed,
        },
    }


# ============================================================================
# BACHELIER BASKET DIGITAL — FUZZY-SMOOTHED LABELS (MULTI-DIM)
# ============================================================================

def fuzzy_basket_bachelier(
    n_samples: int,
    d: int = 1,
    strike: float = 100.0,
    base_vol: float = 20.0,
    T: float = 1.0,
    rho: float = 0.5,
    k_paths: int = 10,
    eps_mult: float = 0.5,
    eps_override: float = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Bachelier basket digital option data with fuzzy-smoothed labels.

    Under Bachelier dynamics: S_{T,i} = x_i + σ_i √T Z_i  (correlated Z).
    Basket = Σ w_i S_{T,i}, weights w_i = 1/d.
    Payoff = 1{Basket > K} → cSpr(Basket − K, ε).

    Fuzzy delta for each component i:
        dπ_ε/dx_i = cSpr'(Basket − K, ε) · w_i

    ε calibrated from std(Basket − K).

    Args:
        n_samples: Number of input spot vectors.
        d: Number of assets.
        strike: Strike K.
        base_vol: Bachelier volatility (price units).
        T: Time to maturity.
        rho: Pairwise equicorrelation.
        k_paths: MC paths per input.
        eps_mult: Fraction of std for ε calibration.
        eps_override: Direct ε value.
        seed: Random seed.

    Returns:
        Dictionary with x, y, dydx_fuzzy, epsilon, config.
    """
    rng = np.random.RandomState(seed)

    # Correlation matrix
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    # Per-asset volatilities
    rng_vol = np.random.RandomState(seed + 1000)
    sigmas = base_vol * (1.0 + 0.1 * rng_vol.randn(d))
    sigmas = np.abs(sigmas)

    weights = np.ones(d) / d

    # Input spots
    x = 100.0 + 10.0 * rng.randn(n_samples, d)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths, d))

    # Pre-simulation for ε calibration
    rng_cal = np.random.RandomState(seed + 5555)
    n_cal = min(n_samples, 1000)
    x_cal = 100.0 + 10.0 * rng_cal.randn(n_cal, d)
    Z_cal = rng_cal.standard_normal((n_cal, d)) @ L.T
    S_T_cal = x_cal + sigmas[None, :] * np.sqrt(T) * Z_cal
    basket_cal = (S_T_cal * weights[None, :]).sum(axis=1)
    condition_cal = basket_cal - strike

    if eps_override is not None:
        epsilon = eps_override
    else:
        epsilon = calibrate_epsilon(condition_cal, eps_mult=eps_mult)

    for p in range(k_paths):
        Z_indep = rng.standard_normal((n_samples, d))
        Z_corr = Z_indep @ L.T  # correlated normals

        S_T = x + sigmas[None, :] * np.sqrt(T) * Z_corr  # (n, d)
        basket = (S_T * weights[None, :]).sum(axis=1)  # (n,)

        # Condition expression
        condition = basket - strike

        # Fuzzy payoff
        fuzzy_payoff = call_spread(condition, epsilon)  # (n,)

        # Fuzzy pathwise delta per component:
        #   dπ_ε/dx_i = cSpr'(basket − K, ε) · ∂basket/∂x_i
        #             = cSpr'(basket − K, ε) · w_i
        # (Bachelier: ∂S_{T,i}/∂x_i = 1, so ∂basket/∂x_i = w_i)
        csp_deriv = call_spread_deriv(condition, epsilon)  # (n,)
        fuzzy_delta = csp_deriv[:, None] * weights[None, :]  # (n, d)

        y_all[:, p] = fuzzy_payoff
        dydx_all[:, p, :] = fuzzy_delta

    y = y_all.mean(axis=1, keepdims=True)
    dydx_fuzzy = dydx_all.mean(axis=1).reshape(n_samples, 1, d)

    # Analytical values for validation (d=1 case)
    if d == 1 and SCIPY_AVAILABLE:
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
        "dydx_fuzzy": dydx_fuzzy,
        "epsilon": float(epsilon),
        "config": {
            "payoff": "digital_basket",
            "model": "bachelier",
            "label_method": "fuzzy_callspread",
            "d": d,
            "strike": strike,
            "base_vol": base_vol,
            "T": T,
            "rho": rho,
            "k_paths": k_paths,
            "eps_mult": eps_mult,
            "eps_override": eps_override,
            "epsilon_used": float(epsilon),
            "n_samples": n_samples,
            "seed": seed,
        },
    }
    if y_exact is not None:
        result["y_exact"] = y_exact
        result["dydx_exact"] = dydx_exact

    return result


# ============================================================================
# HESTON EULER — FUZZY-SMOOTHED LABELS
# ============================================================================

def fuzzy_euler_heston(
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
    eps_mult: float = 0.5,
    eps_override: float = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Heston model option data with fuzzy-smoothed pathwise labels.

    For digital payoff: 1{S_T > K} → cSpr(S_T − K, ε)
    For call payoff: max(S_T − K, 0) is already smooth; fuzzy has no effect.

    Pathwise delta of fuzzy digital:
        dπ_ε/dS_0 = cSpr'(S_T − K, ε) · ∂S_T/∂S_0

    where ∂S_T/∂S_0 is accumulated through the Euler scheme.

    Args:
        n_samples: Number of initial spot prices.
        strike: Strike K.
        v0: Initial variance.
        kappa: Mean reversion speed.
        theta: Long-term variance.
        sigma_v: Vol-of-vol.
        rho: Spot-vol correlation.
        r: Risk-free rate.
        T: Time to maturity.
        n_steps: Euler steps.
        k_paths: MC paths per input.
        payoff_type: 'call' or 'digital'.
        eps_mult: ε calibration multiplier.
        eps_override: Direct ε value.
        seed: Random seed.

    Returns:
        Dictionary with x, y, dydx_fuzzy, epsilon, config.
    """
    rng = np.random.RandomState(seed)

    S0 = rng.uniform(strike * 0.7, strike * 1.3, (n_samples, 1))
    dt_step = T / n_steps
    sqrt_dt = np.sqrt(dt_step)
    discount = np.exp(-r * T)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    # Pre-simulation for ε calibration
    if eps_override is None and payoff_type == "digital":
        rng_cal = np.random.RandomState(seed + 3333)
        n_cal = min(n_samples, 500)
        S0_cal = rng_cal.uniform(strike * 0.7, strike * 1.3, (n_cal, 1))
        log_S_cal = np.log(S0_cal.flatten())
        v_cal = np.full(n_cal, v0)
        for step in range(n_steps):
            v_pos_cal = np.maximum(v_cal, 0.0)
            Z1_cal = rng_cal.standard_normal(n_cal)
            Z_indep_cal = rng_cal.standard_normal(n_cal)
            Z2_cal = rho * Z1_cal + np.sqrt(1.0 - rho ** 2) * Z_indep_cal
            log_S_cal += (r - 0.5 * v_pos_cal) * dt_step + np.sqrt(v_pos_cal) * sqrt_dt * Z1_cal
            v_cal += kappa * (theta - v_pos_cal) * dt_step + sigma_v * np.sqrt(v_pos_cal) * sqrt_dt * Z2_cal
        condition_cal = np.exp(log_S_cal) - strike
        epsilon = calibrate_epsilon(condition_cal, eps_mult=eps_mult)
    elif eps_override is not None:
        epsilon = eps_override
    else:
        epsilon = 1.0  # call payoff doesn't need smoothing

    for p in range(k_paths):
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)

        # Track ∂log(S_T)/∂log(S_0) for pathwise sensitivity
        # For GBM-like multiplicative dynamics, ∂S_T/∂S_0 ≈ S_T/S_0
        # But for Euler-Heston with varying vol, we accumulate properly.
        # d log S_{k+1} / d log S_0 = d log S_k / d log S_0
        # (since the vol v_k doesn't depend on S_0 in our setup)
        # So ∂log S_T/∂log S_0 = 1, i.e., ∂S_T/∂S_0 = S_T / S_0
        # This is exact because v dynamics are independent of S (one-way coupling).

        for step in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S += (r - 0.5 * v_pos) * dt_step + sqrt_v * sqrt_dt * Z1
            v += kappa * (theta - v_pos) * dt_step + sigma_v * sqrt_v * sqrt_dt * Z2

        S_T = np.exp(log_S)
        sensitivity = S_T / S0.flatten()  # ∂S_T/∂S_0 = S_T / S_0

        if payoff_type == "digital":
            condition = S_T - strike
            fuzzy_payoff = call_spread(condition, epsilon) * discount
            fuzzy_delta = call_spread_deriv(condition, epsilon) * sensitivity * discount
        elif payoff_type == "call":
            fuzzy_payoff = np.maximum(S_T - strike, 0.0) * discount
            fuzzy_delta = (S_T > strike).astype(np.float64) * sensitivity * discount
        else:
            raise ValueError(f"Unknown payoff_type: {payoff_type}")

        y_all[:, p] = fuzzy_payoff
        dydx_all[:, p] = fuzzy_delta

    y = y_all.mean(axis=1, keepdims=True)
    dydx_fuzzy = dydx_all.mean(axis=1).reshape(n_samples, 1, 1)

    return {
        "x": S0,
        "y": y,
        "dydx_fuzzy": dydx_fuzzy,
        "epsilon": float(epsilon),
        "config": {
            "payoff": payoff_type,
            "model": "heston_euler",
            "label_method": "fuzzy_callspread",
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
            "eps_mult": eps_mult,
            "eps_override": eps_override,
            "epsilon_used": float(epsilon),
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
    Split fuzzy data into train/test and format for train_single_experiment().

    Uses the fuzzy-smoothed labels (y, dydx_fuzzy) for training.

    Args:
        data: Output from any fuzzy_* generator (must have 'x', 'y', 'dydx_fuzzy').
        test_frac: Fraction of data for test set.
        seed: Random seed for split.

    Returns:
        Dictionary with x_train, y_train, dydx_train, x_test, y_test, dydx_test.
        Also includes y_exact_test and dydx_exact_test if available.
    """
    rng = np.random.RandomState(seed)
    n = data["x"].shape[0]
    indices = rng.permutation(n)
    n_test = int(n * test_frac)

    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    result = {
        "x_train": data["x"][train_idx],
        "y_train": data["y"][train_idx],
        "dydx_train": data["dydx_fuzzy"][train_idx],
        "x_test": data["x"][test_idx],
        "y_test": data["y"][test_idx],
        "dydx_test": data["dydx_fuzzy"][test_idx],
    }

    # Include exact test values if available (for evaluation against ground truth)
    if "y_exact" in data and data["y_exact"] is not None:
        result["y_exact_test"] = data["y_exact"][test_idx]
    if "dydx_exact" in data and data["dydx_exact"] is not None:
        result["dydx_exact_test"] = data["dydx_exact"][test_idx]

    return result


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def compare_smoothing(
    x: np.ndarray,
    eps_values: list = [0.1, 0.5, 1.0, 2.0],
    strike: float = 100.0,
    vol: float = 0.2,
    T: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compare different smoothing levels on digital option for diagnostics.

    Useful for plotting: how does ε affect the approximation quality?

    Args:
        x: Spot prices (n,) or (n,1)
        eps_values: List of ε values to compare
        strike: Strike price
        vol: Volatility
        T: Time to maturity

    Returns:
        Dictionary with 'exact_price', 'exact_delta', and per-ε results.
    """
    assert SCIPY_AVAILABLE
    x = x.flatten()
    d2 = (np.log(x / strike)) / (vol * np.sqrt(T)) - vol * np.sqrt(T) / 2

    results = {
        "spots": x,
        "exact_price": scipy_norm.cdf(d2),
        "exact_delta": scipy_norm.pdf(d2) / (x * vol * np.sqrt(T)),
    }

    for eps in eps_values:
        # Direct fuzzy evaluation (at t=T, S_T = S_0 for ATM test)
        condition = x - strike
        results[f"fuzzy_price_eps{eps}"] = call_spread(condition, eps)
        results[f"fuzzy_delta_eps{eps}"] = call_spread_deriv(condition, eps)

    return results


# ============================================================================
# HESTON EULER BARRIER (DOWN-AND-OUT CALL) — FUZZY-SMOOTHED LABELS
# ============================================================================

def fuzzy_barrier_heston(
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
    eps_barrier_mult: float = 0.5,
    eps_barrier_override: float = None,
    seed: int = 42,
    spot_low_mult: float = 0.7,
    spot_high_mult: float = 1.3,
) -> Dict[str, Any]:
    """
    Heston Euler down-and-out call with fuzzy-smoothed pathwise barrier labels.

    Single intermediate barrier check at T1 (matches our `lrm_barrier_heston`
    setup). The discontinuity at the barrier 1{S_{T1} > B} is replaced by
    a Savine call-spread ramp `cSpr(S_{T1} - B, ε_B)` so that pathwise
    differentiation is well-defined.

    Fuzzy payoff:
        survival_dt = cSpr(S_{T1} - B, ε_B)         (smooth survival in [0,1])
        call_payoff = max(S_{T2} - K, 0)            (already differentiable)
        π_ε = e^{-r T2} * call_payoff * survival_dt

    Pathwise delta via product rule (∂S_t/∂S_0 = S_t/S_0 under Heston Euler
    since V is S_0-independent — see derivation note):
        d(survival_dt)/dS_0 = cSpr'(S_{T1} - B, ε_B) * (S_{T1} / S_0)
        d(call_payoff)/dS_0 = 1{S_{T2} > K} * (S_{T2} / S_0)
        d(π_ε)/dS_0 = e^{-r T2} * [1{S_{T2}>K} * (S_{T2}/S_0) * survival_dt
                                   + call_payoff * cSpr'(S_{T1}-B, ε_B) * (S_{T1}/S_0)]

    ε_B is calibrated as `eps_barrier_mult × std(S_{T1} − B)` from a small
    pre-simulation, OR set directly via `eps_barrier_override`. This follows
    Savine's call-spread bandwidth recipe (eps ≈ fraction of the std of the
    discontinuity condition).

    Note on `eps_barrier_mult` default (0.5): Savine's recommended range is
    0.1–1.0; 0.5 is mid-range. For our pilot (S_0 ~ 1, σ_eff ≈ √V_0 ≈ 0.2,
    √T_1 ≈ 0.58) this gives ε_B ≈ 0.05–0.10, a band ~5–10% of S_0 around the
    barrier. **G&K v2 §3.3 explicitly note** the choice of ε is sensitive
    and "even less obvious when discontinuities are path-dependent". A
    sensitivity sweep over ε_barrier_mult ∈ {0.1, 0.25, 0.5, 0.75, 1.0} is
    recommended before drawing conclusions about fuzzy barrier performance.
    Use `eps_barrier_override` to pass a fixed ε directly. (Code-review LOW #3.)

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
        eps_barrier_mult: fraction of std(S_{T1} − B) for ε calibration.
        eps_barrier_override: direct ε for the barrier ramp.
        seed: random seed.

    Returns:
        Dictionary with keys: x, y, dydx_fuzzy, epsilon_barrier, lrm_var
        (set to None — fuzzy is not LRM), config.

    References:
        - Savine, A. (2018/2024). Modern Computational Finance: Fuzzy
          Payoff Evaluation. (Call-spread / cSpr bandwidth recipe.)
        - Glasserman, P., and S. H. Karmarkar (2025/2026). arXiv:2512.05301
          v2 §3.3 (ramp smoothing on a digital). Note: G&K explicitly flag
          path-dependent fuzzy as "even less obvious" (§3.3 caveat); this
          function is one realisation of that approach for our Heston barrier.
        - Andersen, L. (2008). JCF 11(3), 1-42 (full-truncation Euler).
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(strike * spot_low_mult, strike * spot_high_mult, (n_samples, 1))
    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)

    # ε calibration: small pre-simulation to estimate std(S_T1 - B)
    # NOTE: calibration uses the SAME spot range as the main simulation so ε
    # scale is consistent.
    if eps_barrier_override is not None:
        eps_barrier = eps_barrier_override
    else:
        rng_cal = np.random.RandomState(seed + 7777)
        n_cal = min(n_samples, 1000)
        S0_cal = rng_cal.uniform(strike * spot_low_mult, strike * spot_high_mult, n_cal)
        log_S_cal = np.log(S0_cal)
        v_cal = np.full(n_cal, v0)
        for step in range(n_substeps_to_T1):
            v_pos = np.maximum(v_cal, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng_cal.standard_normal(n_cal)
            Z_indep = rng_cal.standard_normal(n_cal)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S_cal = log_S_cal + (r - 0.5 * v_pos) * dt1 + sqrt_v * sqrt_dt1 * Z1
            v_cal = v_cal + kappa * (theta - v_pos) * dt1 + sigma_v * sqrt_v * sqrt_dt1 * Z2
        S_T1_cal = np.exp(log_S_cal)
        eps_barrier = calibrate_epsilon(S_T1_cal - barrier, eps_mult=eps_barrier_mult)

    y_all = np.zeros((n_samples, k_paths))
    dydx_all = np.zeros((n_samples, k_paths))

    for p in range(k_paths):
        log_S = np.log(S0.flatten())
        v = np.full(n_samples, v0)

        # Phase 1: simulate to T1
        for step in range(n_substeps_to_T1):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            Z1 = rng.standard_normal(n_samples)
            Z_indep = rng.standard_normal(n_samples)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep
            log_S = log_S + (r - 0.5 * v_pos) * dt1 + sqrt_v * sqrt_dt1 * Z1
            v = v + kappa * (theta - v_pos) * dt1 + sigma_v * sqrt_v * sqrt_dt1 * Z2

        S_T1 = np.exp(log_S)
        # Fuzzy survival (in [0,1])
        survival_dt = call_spread(S_T1 - barrier, eps_barrier)
        survival_deriv = call_spread_deriv(S_T1 - barrier, eps_barrier)

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
        call_payoff = np.maximum(S_T2 - strike, 0.0)
        call_indicator = (S_T2 > strike).astype(np.float64)

        # Pathwise sensitivities under Heston Euler: ∂S_t/∂S_0 = S_t / S_0
        # (V doesn't depend on S_0 → log_S_t = log_S_0 + S_0-independent → S_t/S_0)
        S0_flat = S0.flatten()
        d_call_dS0 = call_indicator * (S_T2 / S0_flat)
        d_survival_dS0 = survival_deriv * (S_T1 / S0_flat)

        # Full fuzzy payoff and delta via product rule
        fuzzy_payoff = call_payoff * survival_dt * discount
        fuzzy_delta = (d_call_dS0 * survival_dt + call_payoff * d_survival_dS0) * discount

        y_all[:, p] = fuzzy_payoff
        dydx_all[:, p] = fuzzy_delta

    y = y_all.mean(axis=1, keepdims=True)
    dydx_fuzzy = dydx_all.mean(axis=1).reshape(n_samples, 1, 1)

    return {
        "x": S0,
        "y": y,
        "dydx_fuzzy": dydx_fuzzy,
        "epsilon_barrier": float(eps_barrier),
        "config": {
            "payoff": "barrier_doc_call",
            "model": "heston_full_truncation_euler",
            "label_method": "fuzzy_callspread_barrier",
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
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "k_paths": k_paths,
            "eps_barrier_mult": eps_barrier_mult,
            "eps_barrier_override": eps_barrier_override,
            "epsilon_barrier_used": float(eps_barrier),
            "n_samples": n_samples,
            "seed": seed,
        },
    }
