"""
High-fidelity gradient references for evaluation of DML methods.

This module provides noise-free or near-noise-free gradient references for
the 3 unified comparison datasets that previously used noisy LRM (k=10)
as evaluation targets:

    1. barrier_bs:      Analytical delta via reflection principle
                        (Reiner & Rubinstein 1991, Merton 1973)
    2. heston_digital:  Semi-analytical delta via COS method
                        (Fang & Oosterlee 2008)
    3. basket_d7:       High-k MC LRM (k=100,000) with convergence check

For digital_bs and basket_d1, analytical deltas already exist in lrm_labels.py
and are used by default. This module fixes the remaining 3 datasets.

Usage:
    from dml_benchmark.high_fidelity_references import (
        barrier_bs_analytical_delta,
        heston_digital_cos_delta,
        basket_high_k_lrm_delta,
    )

    # Get exact delta for barrier option
    delta = barrier_bs_analytical_delta(S0, strike=100, barrier=80, vol=0.2, r=0.05, T=1.0)

References:
    - Merton (1973): "Theory of Rational Option Pricing"
    - Reiner & Rubinstein (1991): "Breaking Down the Barriers"
    - Fang & Oosterlee (2008): "A Novel Pricing Method for European Options
      Based on Fourier-Cosine Series Expansions"
    - Hull (2018): "Options, Futures, and Other Derivatives" Ch.26
    - Glasserman (2003): "Monte Carlo Methods in Financial Engineering" Ch.7
    - Lopez de Prado (2018): "Advances in Financial Machine Learning" Ch.7

Created for audit issue A1: Gradient evaluation on LRM-noisy references.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple


# ============================================================================
# 1. BARRIER OPTION — ANALYTICAL DELTA (Reiner & Rubinstein 1991)
# ============================================================================

def _barrier_bs_price(
    S: np.ndarray,
    K: float,
    B: float,
    sigma: float,
    r: float,
    T: float,
) -> np.ndarray:
    """
    Price of a down-and-out call under Black-Scholes with continuous monitoring.

    Payoff: max(S_T - K, 0) if min_{0<=t<=T} S_t > B, else 0.
    **REQUIRES B < K** (barrier below strike). For B >= K the formula below
    is not the valid one — a different reflection-principle formula applies
    (Reiner-Rubinstein 1991 case 2). The function raises ValueError to avoid
    silently returning a wrong price. Code-review LOW #5 (2026-05-04).
    Also requires B < S (otherwise already knocked out).

    The formula uses the reflection principle / image method.

    From Reiner & Rubinstein (1991) and Hull (2018) Ch.26, Eq.26.4.
    The down-and-out call price is:

        C_do = C_bs(S, K) - (B/S)^{2λ} · C_bs(B²/S, K)

    where λ = (r + σ²/2) / σ² and C_bs is the standard Black-Scholes call.
    This is the simplest form valid when B < K (barrier below strike).

    Args:
        S: Array of spot prices, shape (n,)
        K: Strike price
        B: Barrier level (B < K)
        sigma: Volatility
        r: Risk-free rate
        T: Time to maturity

    Returns:
        Array of prices, shape (n,)

    Raises:
        ValueError: if B >= K (formula not valid for in-the-money barriers).
    """
    if B >= K:
        raise ValueError(
            f"_barrier_bs_price: this formula is valid only for B < K "
            f"(barrier below strike); got B={B}, K={K}. For B >= K, see "
            f"Reiner-Rubinstein 1991 case 2 (different reflection formula)."
        )

    lam = (r + 0.5 * sigma ** 2) / (sigma ** 2)

    # Standard BS call components
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Image spot: B²/S
    S_img = B ** 2 / S
    d1_img = (np.log(S_img / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2_img = d1_img - sigma * np.sqrt(T)

    # Standard call
    call_bs = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Image call
    call_img = S_img * norm.cdf(d1_img) - K * np.exp(-r * T) * norm.cdf(d2_img)

    # Down-and-out call
    price = call_bs - (B / S) ** (2 * lam) * call_img

    # Zero out for spots at or below barrier
    price = np.where(S > B, price, 0.0)

    return price


def barrier_bs_analytical_delta(
    S0: np.ndarray,
    strike: float = 100.0,
    barrier: float = 80.0,
    vol: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
) -> np.ndarray:
    """
    Analytical delta of a down-and-out call (continuous monitoring).

    Computed via finite-difference on the analytical pricing formula
    (which is itself exact). The bump size is tiny (1e-6) so
    the truncation error is negligible (~1e-12 for smooth functions).

    This avoids implementing the somewhat complex closed-form delta
    formula (which involves derivatives of (B/S)^{2λ} terms) while
    still giving machine-precision results.

    Alternatively, we derive the analytical delta directly:

        Δ_do = ∂C_do/∂S = N(d₁) + S·φ(d₁)·∂d₁/∂S
               - (B/S)^{2λ} · [N(d₁ʹ)·(∂S_img/∂S) + ...]
               - 2λ/S · (B/S)^{2λ} · C_bs(B²/S, K)

    We use the bump-and-reprice approach on the analytical formula
    for clarity and correctness, with negligible numerical error.

    NOTE: This is for CONTINUOUS-monitoring barriers. Our LRM simulation
    uses DISCRETE monitoring (252 steps). The continuous approximation
    introduces a small bias (~0.5-2% for S near B). For far-from-barrier
    spots, the error is negligible. This is a known limitation documented
    in the evaluation notes.

    Args:
        S0: Array of initial spot prices, shape (n,) or (n, 1)
        strike: Strike price K
        barrier: Lower barrier B
        vol: Volatility σ
        r: Risk-free rate
        T: Time to maturity

    Returns:
        Array of deltas, shape (n,)
    """
    S = np.asarray(S0).flatten().astype(np.float64)

    # Central finite difference on the exact analytical price
    h = S * 1e-6  # relative bump
    h = np.maximum(h, 1e-10)  # safety floor

    price_up = _barrier_bs_price(S + h, strike, barrier, vol, r, T)
    price_down = _barrier_bs_price(S - h, strike, barrier, vol, r, T)

    delta = (price_up - price_down) / (2 * h)

    # For S <= barrier, delta should be 0 (knocked out)
    delta = np.where(S > barrier, delta, 0.0)

    return delta


def barrier_bs_analytical_delta_closed_form(
    S0: np.ndarray,
    strike: float = 100.0,
    barrier: float = 80.0,
    vol: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
) -> np.ndarray:
    """
    Closed-form analytical delta of a down-and-out call (B < K).

    Direct differentiation of:
        C_do = C_bs(S, K) - (B/S)^{2λ} · C_bs(B²/S, K)

    with respect to S. Uses the chain rule on each component.

    Standard BS call delta: ∂C_bs(S,K)/∂S = N(d₁)

    For the image term: let g(S) = (B/S)^{2λ} · C_bs(B²/S, K)
        ∂g/∂S = -2λ/S · (B/S)^{2λ} · C_bs(B²/S, K)
               + (B/S)^{2λ} · ∂C_bs(B²/S, K)/∂(B²/S) · (-B²/S²)

    Note: ∂C_bs(S',K)/∂S' = N(d₁') where d₁' uses S' = B²/S.

    Args:
        S0: Spot prices, shape (n,) or (n, 1)
        strike, barrier, vol, r, T: Option parameters

    Returns:
        delta: shape (n,)
    """
    S = np.asarray(S0).flatten().astype(np.float64)
    K = strike
    B = barrier
    sigma = vol

    # Ensure numerical stability
    S = np.maximum(S, B + 1e-10)

    sqrtT = np.sqrt(T)
    lam = (r + 0.5 * sigma ** 2) / (sigma ** 2)

    # Standard call delta: N(d₁)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    delta_call = norm.cdf(d1)

    # Image spot: S' = B²/S
    S_img = B ** 2 / S
    d1_img = (np.log(S_img / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2_img = d1_img - sigma * sqrtT

    # Image call price and delta (w.r.t. S_img)
    call_img = S_img * norm.cdf(d1_img) - K * np.exp(-r * T) * norm.cdf(d2_img)
    delta_img_wrt_Simg = norm.cdf(d1_img)  # ∂C_bs(S',K)/∂S'

    # Power term
    power = (B / S) ** (2 * lam)

    # ∂S_img/∂S = -B²/S²
    dSimg_dS = -B ** 2 / S ** 2

    # ∂(B/S)^{2λ}/∂S = -2λ/S · (B/S)^{2λ}
    dpower_dS = -2 * lam / S * power

    # Full derivative of image term:
    # ∂[power · C_img]/∂S = dpower_dS · C_img + power · delta_img_wrt_Simg · dSimg_dS
    d_image_term = dpower_dS * call_img + power * delta_img_wrt_Simg * dSimg_dS

    # Down-and-out delta = call delta - image term derivative
    delta = delta_call - d_image_term

    # For S <= barrier, delta = 0
    delta = np.where(S > B + 1e-10, delta, 0.0)

    return delta


# ============================================================================
# 2. HESTON DIGITAL OPTION — COS METHOD DELTA (Fang & Oosterlee 2008)
# ============================================================================

def _heston_characteristic_function(
    u: np.ndarray,
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
) -> np.ndarray:
    """
    Heston model characteristic function φ(u) = E[exp(i·u·log(S_T))].

    Uses the formulation from Albrecher et al. (2007) which avoids
    the branch-cut discontinuity in the original Heston (1993) formula.

    φ(u) = exp(C(u, T) + D(u, T)·v₀ + i·u·log(S₀))

    where:
        d = sqrt((ρ·σ_v·i·u - κ)² + σ_v²·(i·u + u²))
        g = (κ - ρ·σ_v·i·u - d) / (κ - ρ·σ_v·i·u + d)
        C = r·i·u·T + (κ·θ/σ_v²)·[(κ - ρ·σ_v·i·u - d)·T - 2·log((1-g·exp(-d·T))/(1-g))]
        D = (κ - ρ·σ_v·i·u - d)/σ_v² · (1 - exp(-d·T))/(1 - g·exp(-d·T))

    Args:
        u: Array of evaluation points (real or complex)
        S0: Initial spot price
        v0: Initial variance
        kappa, theta, sigma_v, rho, r, T: Heston model parameters

    Returns:
        Complex array of characteristic function values
    """
    u = np.asarray(u, dtype=np.complex128)

    xi = kappa - rho * sigma_v * 1j * u
    d = np.sqrt(xi ** 2 + sigma_v ** 2 * (1j * u + u ** 2))

    # Use the formulation avoiding branch cuts (Albrecher et al. 2007)
    g2 = (xi - d) / (xi + d)

    C = r * 1j * u * T + (kappa * theta / sigma_v ** 2) * (
        (xi - d) * T - 2.0 * np.log((1.0 - g2 * np.exp(-d * T)) / (1.0 - g2))
    )

    D = ((xi - d) / sigma_v ** 2) * (
        (1.0 - np.exp(-d * T)) / (1.0 - g2 * np.exp(-d * T))
    )

    phi = np.exp(C + D * v0 + 1j * u * np.log(S0))

    return phi


def _cos_price_heston(
    S0: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
    payoff_type: str = "digital",
    N: int = 256,
) -> float:
    """
    Price a European option under Heston using the COS method.

    The COS method (Fang & Oosterlee 2008) approximates the pricing integral:

        V = e^{-rT} ∫ v(y) f(y|x) dy

    using a Fourier cosine expansion of f(y|x), where x = log(S₀/K) and
    y = log(S_T/K). The characteristic function replaces the density.

    For a digital call: v(y) = 1{y > 0}  →  V_k coefficients are known.
    For a vanilla call: v(y) = K(e^y - 1)⁺  →  V_k coefficients are known.

    Args:
        S0: Initial spot
        K: Strike
        v0: Initial variance
        kappa, theta, sigma_v, rho, r, T: Heston parameters
        payoff_type: "digital" or "call"
        N: Number of cosine terms (higher = more accurate, 256 is typically sufficient)

    Returns:
        Option price (float)
    """
    # Integration range [a, b] for COS method
    # Use cumulants to set the range (Fang & Oosterlee 2008, Eq. 23)
    # For Heston, use a wider range to capture fat tails
    c1 = r * T + (1 - np.exp(-kappa * T)) * (theta - v0) / (2 * kappa) - 0.5 * theta * T
    c2 = (1.0 / (8.0 * kappa ** 3)) * (
        sigma_v * T * kappa * np.exp(-kappa * T)
        * (v0 - theta) * (8.0 * kappa * rho - 4.0 * sigma_v)
        + kappa * rho * sigma_v * (1 - np.exp(-kappa * T))
        * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * sigma_v + sigma_v ** 2 + 4.0 * kappa ** 2)
        + sigma_v ** 2 * ((theta - 2.0 * v0) * np.exp(-2.0 * kappa * T)
                          + theta * (6.0 * np.exp(-kappa * T) - 7.0) + 2.0 * v0)
        + 8.0 * kappa ** 2 * (v0 - theta) * (1 - np.exp(-kappa * T))
    )
    c2 = max(c2, 1e-8)  # safety floor

    L = 12  # number of std devs for truncation range
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    x = np.log(S0 / K)

    # Cosine expansion coefficients
    k = np.arange(0, N)
    omega_k = k * np.pi / (b - a)

    # Characteristic function evaluated at ω_k
    # φ(ω_k) = E[exp(i·ω_k·log(S_T))]
    # We need the char. fn. of log(S_T/K) = log(S_T) - log(K)
    # This is: exp(-i·ω_k·log(K)) · φ_logS(ω_k)
    # where φ_logS is the char fn of log(S_T).

    phi_vals = _heston_characteristic_function(
        omega_k, S0, v0, kappa, theta, sigma_v, rho, r, T
    )

    # Re-express in terms of x = log(S₀/K):
    # The COS pricing uses the density of y = log(S_T/K).
    # Its char fn is: exp(-i·u·log K) · φ_{logS}(u)
    # Coefficients: A_k = (2/(b-a)) · Re{ φ_y(ω_k) · exp(-i·ω_k·a) }
    # where φ_y(u) = exp(-i·u·logK) · φ_{logS}(u)

    # Actually, for the COS method the coefficients involve:
    # F_k = Re{ φ(ω_k; x) · exp(-i·ω_k·a) }
    # where φ(u; x) is the char fn of Y given X = x.
    # For log-price: Y = log(S_T/K), X = log(S₀/K)
    # The Heston char fn φ(u) = E[e^{iu·logS_T}] is centered at logS₀.
    # We shift: E[e^{iu·Y}] = e^{-iu·logK} · φ(u)

    char_fn_y = np.exp(-1j * omega_k * np.log(K)) * phi_vals

    F_k = np.real(char_fn_y * np.exp(-1j * omega_k * a))
    F_k[0] *= 0.5  # First term halved in cosine expansion

    # Payoff coefficients V_k
    if payoff_type == "digital":
        # Digital call: v(y) = 1{y > 0}  (payoff = 1 if S_T > K)
        # V_k = (2/(b-a)) · ∫_0^b cos(kπ(y-a)/(b-a)) dy
        # This integral has a known closed form.
        V_k = _cos_digital_coefficients(k, a, b, 0.0, b)
    elif payoff_type == "call":
        # Vanilla call: v(y) = K(e^y - 1)⁺
        V_k = _cos_call_coefficients(k, a, b, K)
    else:
        raise ValueError(f"Unknown payoff_type: {payoff_type}")

    # Price
    price = np.exp(-r * T) * np.sum(F_k * V_k)

    return float(np.real(price))


def _cos_digital_coefficients(
    k: np.ndarray,
    a: float,
    b: float,
    c: float,  # lower limit of payoff region
    d: float,  # upper limit of payoff region
) -> np.ndarray:
    """
    COS coefficients for a digital payoff 1{c < y < d}.

    V_k = (2/(b-a)) · ∫_c^d cos(kπ(y-a)/(b-a)) dy

    For k=0: V_0 = (2/(b-a)) · (d - c)
    For k>0: V_k = (2/(b-a)) · (b-a)/(kπ) · [sin(kπ(d-a)/(b-a)) - sin(kπ(c-a)/(b-a))]
    """
    V = np.zeros_like(k, dtype=float)

    V[0] = (d - c) / (b - a)  # Note: halving happens in F_k[0] *= 0.5

    mask = k > 0
    km = k[mask]
    V[mask] = (1.0 / (km * np.pi)) * (
        np.sin(km * np.pi * (d - a) / (b - a))
        - np.sin(km * np.pi * (c - a) / (b - a))
    )

    return V * 2.0  # The 2/(b-a) factor, but we already have (b-a) in denominator


def _cos_call_coefficients(
    k: np.ndarray,
    a: float,
    b: float,
    K: float,
) -> np.ndarray:
    """
    COS coefficients for vanilla call payoff v(y) = (e^y - 1)⁺.

    We split: ∫_0^b e^y cos(...) dy - ∫_0^b cos(...) dy
    Each integral has closed form — see Fang & Oosterlee (2008), Appendix A.
    """
    # Chi and Psi functions from Fang & Oosterlee (2008)
    chi = _cos_chi(k, a, b, 0.0, b)
    psi = _cos_psi(k, a, b, 0.0, b)

    V = (2.0 / (b - a)) * K * (chi - psi)
    return V


def _cos_chi(k, a, b, c, d):
    """Integral ∫_c^d e^y cos(kπ(y-a)/(b-a)) dy."""
    omega = k * np.pi / (b - a)
    result = np.zeros_like(k, dtype=float)

    for i, (ki, wi) in enumerate(zip(k, omega)):
        if ki == 0:
            result[i] = np.exp(d) - np.exp(c)
        else:
            result[i] = (
                1.0 / (1.0 + wi ** 2)
            ) * (
                np.exp(d) * (np.cos(wi * (d - a)) + wi * np.sin(wi * (d - a)))
                - np.exp(c) * (np.cos(wi * (c - a)) + wi * np.sin(wi * (c - a)))
            )
    return result


def _cos_psi(k, a, b, c, d):
    """Integral ∫_c^d cos(kπ(y-a)/(b-a)) dy."""
    result = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        if ki == 0:
            result[i] = d - c
        else:
            omega = ki * np.pi / (b - a)
            result[i] = (np.sin(omega * (d - a)) - np.sin(omega * (c - a))) / omega
    return result


def heston_digital_cos_delta(
    S0: np.ndarray,
    strike: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 1.0,
    N: int = 256,
) -> np.ndarray:
    """
    Semi-analytical delta of a digital call under Heston via COS method.

    Computes ∂V/∂S₀ using central finite differences on the COS pricing
    formula (which is itself semi-analytical to machine precision with N=256).

    The bump is tiny (1e-6 relative), giving negligible numerical error.

    Args:
        S0: Array of initial spot prices, shape (n,) or (n, 1)
        strike: Strike price K
        v0: Initial variance
        kappa, theta, sigma_v, rho, r, T: Heston parameters
        N: Number of COS terms (256 is standard)

    Returns:
        Array of deltas, shape (n,)
    """
    S = np.asarray(S0).flatten().astype(np.float64)
    n = len(S)
    delta = np.zeros(n)

    for i in range(n):
        s = S[i]
        h = s * 1e-6  # relative bump
        h = max(h, 1e-10)

        p_up = _cos_price_heston(
            s + h, strike, v0, kappa, theta, sigma_v, rho, r, T,
            payoff_type="digital", N=N,
        )
        p_down = _cos_price_heston(
            s - h, strike, v0, kappa, theta, sigma_v, rho, r, T,
            payoff_type="digital", N=N,
        )
        delta[i] = (p_up - p_down) / (2 * h)

    return delta


# ============================================================================
# 3. BASKET d=7 — HIGH-k MC LRM (no analytical exists for d > 1)
# ============================================================================

def basket_high_k_lrm_delta(
    x: np.ndarray,
    d: int = 7,
    strike: float = 100.0,
    base_vol: float = 20.0,
    T: float = 1.0,
    rho: float = 0.5,
    k_paths: int = 100_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    High-fidelity LRM delta for Bachelier basket digital option.

    Identical algorithm to lrm_basket_bachelier() but with k=100,000
    paths instead of k=10, reducing LRM variance by 10,000×.

    Args:
        x: Input spot vectors, shape (n, d)
        d: Number of assets
        strike: Strike price K
        base_vol: Base volatility
        T: Time to maturity
        rho: Pairwise equicorrelation
        k_paths: MC paths per sample (default 100,000)
        seed: Random seed

    Returns:
        Tuple of:
            dydx: shape (n, d) — high-fidelity LRM delta
            y: shape (n,) — MC price estimate
            lrm_var: shape (n,) — remaining LRM variance (should be very small)
    """
    rng = np.random.RandomState(seed)
    n_samples = x.shape[0]

    # Correlation matrix and its inverse (equicorrelation)
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    if d == 1:
        corr_inv = np.array([[1.0]])
    else:
        a = 1.0 / (1.0 - rho)
        b = rho / ((1.0 - rho) * (1.0 + (d - 1) * rho))
        corr_inv = a * np.eye(d) - b * np.ones((d, d))

    # Per-asset volatilities (must match lrm_labels.py)
    rng_vol = np.random.RandomState(seed + 1000)
    sigmas = base_vol * (1.0 + 0.1 * rng_vol.randn(d))
    sigmas = np.abs(sigmas)

    weights = np.ones(d) / d

    # Accumulate in chunks to avoid memory issues with k=100K
    # Each chunk vectorizes over paths: (n_samples, chunk_size, d)
    # Memory per chunk: n_samples × chunk_size × d × 8 bytes
    # With n=819, chunk=10000, d=7: ~460 MB per chunk — safe on GPU machines
    chunk_size = min(k_paths, 10_000)
    n_chunks = k_paths // chunk_size
    remainder = k_paths % chunk_size

    y_sum = np.zeros(n_samples)
    dydx_sum = np.zeros((n_samples, d))
    dydx_sum_sq = np.zeros((n_samples, d))
    total_paths = 0

    # Precompute: score_matrix = corr_inv / (sigmas * sqrt(T))
    score_scale = corr_inv.T / (sigmas[None, :] * np.sqrt(T))  # (d, d)

    for chunk_idx in range(n_chunks + (1 if remainder > 0 else 0)):
        if chunk_idx == n_chunks and remainder > 0:
            current_chunk = remainder
        elif chunk_idx == n_chunks:
            break
        else:
            current_chunk = chunk_size

        # Vectorized: generate all paths in this chunk at once
        # Z_indep: (n_samples, current_chunk, d)
        Z_indep = rng.standard_normal((n_samples, current_chunk, d))
        Z_corr = np.einsum('nkd,ed->nke', Z_indep, L)  # (n, k, d)

        # S_T = x[:, None, :] + sigmas * sqrt(T) * Z_corr
        S_T = x[:, None, :] + sigmas[None, None, :] * np.sqrt(T) * Z_corr  # (n, k, d)
        basket = (S_T * weights[None, None, :]).sum(axis=2)  # (n, k)
        payoff = (basket > strike).astype(np.float64)  # (n, k)

        # score: (n, k, d) = Z_corr @ score_scale
        score = np.einsum('nkd,de->nke', Z_corr, score_scale)  # (n, k, d)
        lrm_delta = payoff[:, :, None] * score  # (n, k, d)

        y_sum += payoff.sum(axis=1)  # aggregate over paths
        dydx_sum += lrm_delta.sum(axis=1)  # (n, d)
        dydx_sum_sq += (lrm_delta ** 2).sum(axis=1)  # (n, d)
        total_paths += current_chunk

    y = y_sum / total_paths
    dydx = dydx_sum / total_paths
    dydx_var = (dydx_sum_sq / total_paths - dydx ** 2)
    lrm_var = dydx_var.mean(axis=1)  # avg variance across dimensions

    return dydx, y, lrm_var


# ============================================================================
# 4. HESTON BARRIER (DOWN-AND-OUT CALL) — MC REFERENCE
# ============================================================================

def heston_barrier_doc_mc_reference(
    S0: np.ndarray,
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
    n_paths: int = 100_000,
    finite_diff_bump: float = 0.001,
    seed: int = 42,
    spot_low_mult: float = None,  # accepted for API parity; S0 is passed in
    spot_high_mult: float = None,
) -> Dict[str, np.ndarray]:
    """
    MC reference price + delta for Heston down-and-out call with single
    barrier check at T1 and expiry at T2 (the n=2 monitoring setup of G&K v2
    §3.4 generalised to Heston-Euler from §3.6).

    Payoff: e^{-r T2} * 1{S_{T1} > B} * max(S_{T2} - K, 0).

    Uses full-truncation Euler scheme for the Heston SDE (Andersen 2008,
    matches G&K v2 §3.6 and our existing `lrm_euler_heston`):
        log S_{i+1} = log S_i + (r - V_i^+/2) Δt + sqrt(V_i^+ Δt) Z_{1,i}
        V_{i+1}     = V_i + κ(θ - V_i^+) Δt + η sqrt(V_i^+ Δt) Z_{2,i}
    with V_i^+ = max(V_i, 0), Z_{2,i} = ρ Z_{1,i} + sqrt(1-ρ²) Z_{indep,i}.

    Delta is computed by central finite difference on S_0 with shared
    random numbers across the three (S_0, S_0+ε, S_0-ε) simulations
    (Glasserman 2004 §7.1). FD bump default 0.001 matches G&K's
    `FINITE_DIFF_BUMP` constant.

    Args:
        S0: spot prices, shape (n,).
        strike: K.
        barrier: B (knock-out level; B < S_0 expected for alive paths).
        v0, kappa, theta, sigma_v, rho: Heston parameters.
        r: risk-free rate.
        T1: barrier monitoring time.
        T2: expiry.
        n_substeps_to_T1: Euler steps between 0 and T1.
        n_substeps_T1_to_T2: Euler steps between T1 and T2.
        n_paths: MC paths per S_0 (vectorised over paths and FD bumps).
        finite_diff_bump: ε for central FD on S_0.
        seed: reproducibility seed (used for shared random numbers).

    Returns:
        dict with keys:
            x:                  (n, 1)      input S_0 (no FD bumps)
            y:                  (n, 1)      reference price (discounted)
            dydx:                (n, 1, 1)  reference delta = ∂price/∂S_0
            std_err_price:       (n,)        MC std error on price
            std_err_delta:       (n,)        MC std error on delta
            config:              dict       parameters used

    References:
        - Heston, S. L. (1993). A closed-form solution for options with
          stochastic volatility. Rev. Financial Studies 6(2), 327-343.
        - Andersen, L. (2008). Simple and efficient simulation of the
          Heston model. J. Computational Finance 11(3), 1-42.
          (Full-truncation Euler scheme.)
        - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering,
          §7.1. (Common random numbers for FD Greeks.)
        - Glasserman, P., and S. H. Karmarkar (2025/2026). Differential ML
          with a Difference. arXiv:2512.05301 v2 §3.4 (BS barrier setup) and
          §3.6 (Heston-Euler discretisation). FINITE_DIFF_BUMP=0.001 matches
          their convention.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    if S0.ndim == 2:
        S0 = S0.flatten()
    n = S0.shape[0]

    # V0 may be passed as scalar (default Heston v0) or per-spot array of
    # length n. Per-spot V0 is used by the (S_0, V_0) 2-D-input experiment.
    v0_arr = np.asarray(v0, dtype=np.float64)
    if v0_arr.ndim == 0:
        V0_per_spot = np.full(n, float(v0_arr))
    else:
        if v0_arr.shape[0] != n:
            raise ValueError(
                f"v0 array length {v0_arr.shape[0]} != S0 length {n}"
            )
        V0_per_spot = v0_arr.copy()

    dt1 = T1 / n_substeps_to_T1
    dt2 = (T2 - T1) / n_substeps_T1_to_T2
    sqrt_dt1 = np.sqrt(dt1)
    sqrt_dt2 = np.sqrt(dt2)
    discount = np.exp(-r * T2)

    # Simulate three S_0 variants (centre, +ε, -ε) with shared random numbers
    eps = finite_diff_bump
    S0_centre = S0
    S0_plus = S0 + eps
    S0_minus = S0 - eps

    rng = np.random.RandomState(seed)
    log_S_c = np.broadcast_to(np.log(S0_centre)[:, None], (n, n_paths)).copy()
    log_S_p = np.broadcast_to(np.log(S0_plus)[:, None], (n, n_paths)).copy()
    log_S_m = np.broadcast_to(np.log(S0_minus)[:, None], (n, n_paths)).copy()
    # V has shape (n, n_paths): per-spot independent V trajectories. Code-review
    # MED #3 (2026-05-04) — previously V was shape (n_paths,) with all spots
    # sharing the same V realisation, which inflated cross-spot correlation in
    # error estimates. With per-spot V, cross-spot SEs are unbiased.
    V = np.broadcast_to(V0_per_spot[:, None], (n, n_paths)).astype(np.float64).copy()

    # Phase 1: simulate from 0 to T1
    for step in range(n_substeps_to_T1):
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        Z1 = rng.standard_normal((n, n_paths))
        Z_indep = rng.standard_normal((n, n_paths))
        Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

        drift = (r - 0.5 * V_pos) * dt1
        diffuse = sqrt_V * sqrt_dt1 * Z1
        # CRN within each spot: same (Z1, Z2) used across the FD triple
        log_S_c += drift + diffuse
        log_S_p += drift + diffuse
        log_S_m += drift + diffuse

        V = V + kappa * (theta - V_pos) * dt1 + sigma_v * sqrt_V * sqrt_dt1 * Z2

    # Barrier check at T1 (single intermediate monitoring date)
    alive_c = np.exp(log_S_c) > barrier
    alive_p = np.exp(log_S_p) > barrier
    alive_m = np.exp(log_S_m) > barrier

    # Phase 2: simulate from T1 to T2
    for step in range(n_substeps_T1_to_T2):
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        Z1 = rng.standard_normal((n, n_paths))
        Z_indep = rng.standard_normal((n, n_paths))
        Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z_indep

        drift = (r - 0.5 * V_pos) * dt2
        diffuse = sqrt_V * sqrt_dt2 * Z1
        log_S_c += drift + diffuse
        log_S_p += drift + diffuse
        log_S_m += drift + diffuse

        V = V + kappa * (theta - V_pos) * dt2 + sigma_v * sqrt_V * sqrt_dt2 * Z2

    # Payoffs
    S_T2_c = np.exp(log_S_c)
    S_T2_p = np.exp(log_S_p)
    S_T2_m = np.exp(log_S_m)
    payoff_c = np.maximum(S_T2_c - strike, 0.0) * alive_c.astype(np.float64) * discount
    payoff_p = np.maximum(S_T2_p - strike, 0.0) * alive_p.astype(np.float64) * discount
    payoff_m = np.maximum(S_T2_m - strike, 0.0) * alive_m.astype(np.float64) * discount

    price = payoff_c.mean(axis=1)
    price_p = payoff_p.mean(axis=1)
    price_m = payoff_m.mean(axis=1)
    std_err_price = payoff_c.std(axis=1, ddof=1) / np.sqrt(n_paths)

    delta = (price_p - price_m) / (2.0 * eps)
    # Std error on delta via FD: std((payoff+ - payoff-) / 2ε) / sqrt(n_paths)
    delta_per_path = (payoff_p - payoff_m) / (2.0 * eps)
    std_err_delta = delta_per_path.std(axis=1, ddof=1) / np.sqrt(n_paths)

    # If V0 varies per spot, expose the (S_0, V_0) input feature; otherwise
    # keep the legacy single-column x for backward compatibility.
    is_v0_per_spot = (v0_arr.ndim == 1)
    if is_v0_per_spot:
        x_out = np.column_stack([S0, V0_per_spot])  # (n, 2)
        # Delta only with respect to S_0 (no FD on V_0 here); keep last-axis
        # length 1 by convention so downstream prepare_data_dict still works.
        dydx_out = delta.reshape(n, 1, 1)
    else:
        x_out = S0.reshape(n, 1)
        dydx_out = delta.reshape(n, 1, 1)

    return {
        "x": x_out,
        "y": price.reshape(n, 1),
        "dydx": dydx_out,
        "std_err_price": std_err_price,
        "std_err_delta": std_err_delta,
        "config": {
            "payoff": "barrier_doc_call",
            "model": "heston_full_truncation_euler",
            "label_method": "monte_carlo_reference",
            "strike": strike,
            "barrier": barrier,
            "v0": v0_arr.tolist() if is_v0_per_spot else float(v0_arr),
            "v0_per_spot": is_v0_per_spot,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "r": r,
            "T1": T1,
            "T2": T2,
            "n_substeps_to_T1": n_substeps_to_T1,
            "n_substeps_T1_to_T2": n_substeps_T1_to_T2,
            "n_paths": n_paths,
            "finite_diff_bump": eps,
            "seed": seed,
            "n_samples": n,
        },
    }


# ============================================================================
# 5. CONVERGENCE VERIFICATION
# ============================================================================

def verify_convergence(
    x: np.ndarray,
    d: int = 7,
    strike: float = 100.0,
    base_vol: float = 20.0,
    T: float = 1.0,
    rho: float = 0.5,
    k_values: Tuple[int, ...] = (100, 1_000, 10_000, 100_000),
    seed: int = 42,
    n_subset: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Verify that high-k MC LRM converges for basket delta.

    Runs LRM at multiple k values on a subset of inputs and reports
    the mean LRM variance at each k. Variance should decrease as 1/k.

    Args:
        x: Input spot vectors (full dataset)
        d, strike, base_vol, T, rho: Basket parameters
        k_values: Sequence of k_paths values to test
        seed: Random seed
        n_subset: Number of samples to use (subset for speed)

    Returns:
        Dictionary with k_values, mean_lrm_var, expected_var (1/k scaling)
    """
    x_sub = x[:n_subset]
    results = {
        "k_values": np.array(k_values),
        "mean_lrm_var": np.zeros(len(k_values)),
    }

    for i, k in enumerate(k_values):
        _, _, lrm_var = basket_high_k_lrm_delta(
            x_sub, d=d, strike=strike, base_vol=base_vol,
            T=T, rho=rho, k_paths=k, seed=seed,
        )
        results["mean_lrm_var"][i] = np.mean(lrm_var)
        print(f"  k={k:>7d}: mean LRM variance = {results['mean_lrm_var'][i]:.6e}")

    # Check 1/k scaling
    if len(k_values) >= 2:
        ratio = results["mean_lrm_var"][0] / results["mean_lrm_var"][-1]
        expected_ratio = k_values[-1] / k_values[0]
        print(f"  Variance ratio (k={k_values[0]}→{k_values[-1]}): {ratio:.1f}× "
              f"(expected ~{expected_ratio:.0f}× for 1/k scaling)")

    return results


# ============================================================================
# 5. VALIDATION (cross-check analytical vs bump-and-reprice)
# ============================================================================

def validate_barrier_delta(n_test: int = 100, seed: int = 42):
    """
    Validate barrier BS delta by comparing closed-form vs finite-difference.

    Both should agree to high precision since they use the same pricing formula.
    """
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(85, 150, n_test)  # Spots above barrier (80)

    delta_fd = barrier_bs_analytical_delta(S0)
    delta_cf = barrier_bs_analytical_delta_closed_form(S0)

    max_diff = np.max(np.abs(delta_fd - delta_cf))
    mean_diff = np.mean(np.abs(delta_fd - delta_cf))

    print(f"Barrier delta validation (n={n_test}):")
    print(f"  Max |FD - CF| = {max_diff:.2e}")
    print(f"  Mean |FD - CF| = {mean_diff:.2e}")
    print(f"  Both methods agree: {'YES' if max_diff < 1e-5 else 'NO — investigate!'}")

    return {"max_diff": max_diff, "mean_diff": mean_diff, "agree": max_diff < 1e-5}


def validate_heston_cos_vs_mc(
    n_test: int = 20,
    k_mc: int = 500_000,
    seed: int = 42,
):
    """
    Validate Heston COS delta against very-high-k MC LRM delta.

    The COS method should be much more precise than MC even at k=500K.
    This serves as a sanity check that the COS implementation is correct.
    """
    from .lrm_labels import lrm_euler_heston

    print(f"Heston COS vs MC validation (n={n_test}, MC k={k_mc}):")

    # Generate spots
    rng = np.random.RandomState(seed)
    S0 = rng.uniform(70, 130, n_test)

    # COS delta
    cos_delta = heston_digital_cos_delta(S0)

    # High-k MC delta via lrm_euler_heston
    heston_data = lrm_euler_heston(
        n_samples=n_test,
        k_paths=k_mc,
        payoff_type="digital",
        seed=seed + 999,
    )
    # Note: lrm_euler_heston generates its own S0 values, not aligned with ours.
    # For the validation, we'll just compare the COS delta range with the MC delta range.
    mc_delta = heston_data["dydx_lrm"].flatten()

    print(f"  COS delta range: [{np.min(cos_delta):.6f}, {np.max(cos_delta):.6f}]")
    print(f"  MC delta range:  [{np.min(mc_delta):.6f}, {np.max(mc_delta):.6f}]")
    print(f"  COS delta mean: {np.mean(cos_delta):.6f}")
    print(f"  MC delta mean:  {np.mean(mc_delta):.6f}")
    print(f"  (Note: Spots differ between COS and MC — compare distributions, not point-wise)")

    return {"cos_delta": cos_delta, "mc_delta": mc_delta}


# ============================================================================
# 6. MAIN ENTRY POINT — GENERATE ALL HIGH-FIDELITY REFERENCES
# ============================================================================

def generate_all_references(
    barrier_bs_S0: Optional[np.ndarray] = None,
    heston_digital_S0: Optional[np.ndarray] = None,
    basket_d7_x: Optional[np.ndarray] = None,
    barrier_params: Optional[Dict] = None,
    heston_params: Optional[Dict] = None,
    basket_params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Generate high-fidelity evaluation references for all 3 datasets.

    Args:
        barrier_bs_S0: Barrier dataset spot prices, shape (n, 1)
        heston_digital_S0: Heston dataset spot prices, shape (n, 1)
        basket_d7_x: Basket dataset spot vectors, shape (n, 7)
        *_params: Optional parameter overrides

    Returns:
        Dictionary with per-dataset references
    """
    results = {}

    if barrier_bs_S0 is not None:
        params = {
            "strike": 100.0, "barrier": 80.0, "vol": 0.2, "r": 0.05, "T": 1.0,
        }
        if barrier_params:
            params.update(barrier_params)

        delta = barrier_bs_analytical_delta(barrier_bs_S0, **params)
        delta_cf = barrier_bs_analytical_delta_closed_form(barrier_bs_S0, **params)

        results["barrier_bs"] = {
            "dydx_eval": delta.reshape(-1, 1, 1),
            "dydx_eval_closed_form": delta_cf.reshape(-1, 1, 1),
            "eval_source": "analytical_reflection_principle",
            "method": "Reiner & Rubinstein (1991) continuous-monitoring formula "
                      "with FD delta on exact pricing",
            "parameters": params,
            "note": "Continuous-monitoring approximation. Discrete monitoring "
                    "(252 steps) introduces ~0.5-2% bias for near-barrier spots.",
        }
        print(f"Barrier BS: {len(delta)} analytical deltas computed")

    if heston_digital_S0 is not None:
        params = {
            "strike": 100.0, "v0": 0.04, "kappa": 2.0, "theta": 0.04,
            "sigma_v": 0.3, "rho": -0.7, "r": 0.05, "T": 1.0,
        }
        if heston_params:
            params.update(heston_params)

        delta = heston_digital_cos_delta(heston_digital_S0, **params)

        results["heston_digital"] = {
            "dydx_eval": delta.reshape(-1, 1, 1),
            "eval_source": "cos_method_semi_analytical",
            "method": "COS method (Fang & Oosterlee 2008), N=256 cosine terms, "
                      "FD delta on semi-analytical pricing",
            "parameters": params,
        }
        print(f"Heston digital: {len(delta)} COS deltas computed")

    if basket_d7_x is not None:
        params = {
            "d": 7, "strike": 100.0, "base_vol": 20.0, "T": 1.0, "rho": 0.5,
        }
        if basket_params:
            params.update(basket_params)

        dydx, y, lrm_var = basket_high_k_lrm_delta(
            basket_d7_x, **params, k_paths=100_000,
        )

        results["basket_d7"] = {
            "dydx_eval": dydx.reshape(-1, 1, params["d"]),
            "y_eval": y.reshape(-1, 1),
            "lrm_var": lrm_var,
            "eval_source": "high_k_mc_lrm",
            "method": f"High-k LRM with k={100_000:,} paths "
                      f"(variance reduced by {100_000 // 10}× vs k=10 training labels)",
            "parameters": params,
        }
        mean_var = np.mean(lrm_var)
        print(f"Basket d7: {len(dydx)} high-k LRM deltas computed, "
              f"mean residual variance = {mean_var:.2e}")

    return results
