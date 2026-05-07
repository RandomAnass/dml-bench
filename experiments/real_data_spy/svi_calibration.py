"""
Raw SVI calibration for SPY option implied volatility.

Implements the raw SVI parameterization from Gatheral & Jacquier (2014),
"Arbitrage-free SVI volatility surfaces", *Quantitative Finance* 14:1, 59-71,
Eq. 2.1 (p. 60):

    w(k; a, b, rho, m, sigma) = a + b * [rho * (k - m) + sqrt((k - m)^2 + sigma^2)]
    sigma_BS(k, T) = sqrt(w(k) / T)

where k = log(K/F) is forward log-moneyness, F = S * exp(r * T) the option
forward, and w is total implied variance.

Domain (p. 60): a in R, b >= 0, |rho| < 1, m in R, sigma > 0; auxiliary
constraint a + b * sigma * sqrt(1 - rho^2) >= 0 keeps w >= 0 everywhere.

Roger Lee large-strike bound (p. 61): b * (1 + |rho|) <= 4 / T.

For the SPY DML-Bench Option-C target mode we calibrate one independent
slice per (date, maturity) pair (G&J §2, p. 60), then evaluate the slice
to produce a smile-coherent IV at every quote. Both the regression target
(BS price at SVI-IV) and the gradient labels (analytical BS Greeks at
SVI-IV) come from the same calibrated smile model — the H&S 2020
"ground-truth labels" consistency.

Note: G&K (2025) hand-code Greeks analytically in numpy rather than
autodiff through their pricer (`differential_ml_with_a_difference.py:917-936`);
we mirror that pattern: closed-form BS price + Greeks at sigma_SVI, no tape.

Caveats / paper-text scope statements:

* **No dividend yield in the forward** — F = S * exp(r * T), not
  S * exp((r - q) * T). For SPY (q ~ 1.7%) this shifts log-forward-moneyness
  by q*T. The shift is absorbed by SVI's translation parameter `m_loc`
  (verified empirically: re-fitting with q = 0.018 produces a `m_loc` shift
  of order q*T but identical sigma_SVI at every quote, max diff ~ 1e-12 IV).
  The fitted m_loc therefore does NOT have its standard financial
  interpretation as the smile centre in true forward log-moneyness; it is
  an internal SVI parameter only. sigma_SVI(k_quote) is unaffected.

* **Sticky-strike Greeks** — gradient labels are partial derivatives of
  BS(moneyness, T, r, sigma_SVI(K, T)) with sigma held FIXED at its
  calibrated slice value at this (K, T). We do NOT chain-rule through
  d(sigma_SVI)/d(moneyness) or d(sigma_SVI)/d(T). This is the sticky-strike
  Greek (Bergomi 2016, §2.1; Derman 1999), the empirical SPY equity-index
  convention in non-stress regimes. The smile-aware sticky-delta variant
  is an open extension. Justified for our use because the SVI surface is
  a SUPERVISOR for the network, not a tradable hedge ratio: the choice of
  regime affects what the network learns to predict, not the validity of
  the benchmark.

* **No arbitrage enforcement** — independent per-slice fits do not check
  butterfly arbitrage at the slice level or calendar arbitrage across
  slices. Roger Lee's no-arbitrage bound is enforced loosely (b <= 4/T,
  not the strict b*(1+|rho|) <= 4/T which would require a non-linear
  constraint). For supervisor-only use this is acceptable; we do NOT
  claim the calibrated surface is arbitrage-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


N_PARAMS = 5  # (a, b, rho, m_loc, sigma_loc)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def raw_svi_total_variance(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    """w(k) = a + b * [rho * (k - m) + sqrt((k - m)^2 + sigma^2)] (G&J Eq. 2.1)."""
    a, b, rho, m_loc, sigma_loc = params
    z = k - m_loc
    return a + b * (rho * z + np.sqrt(z * z + sigma_loc * sigma_loc))


def raw_svi_iv(k: np.ndarray, T: float, params: np.ndarray) -> np.ndarray:
    """sigma_BS(k, T) = sqrt(max(w, 0) / T)."""
    w = raw_svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 1e-12) / max(T, 1e-12))


# ---------------------------------------------------------------------------
# Calibration (per slice)
# ---------------------------------------------------------------------------

@dataclass
class SliceFitResult:
    params: np.ndarray
    rmse_total_variance: float
    rmse_iv: float
    n_quotes: int
    converged: bool


def _objective(params: np.ndarray, k: np.ndarray, w_target: np.ndarray,
               weights: np.ndarray) -> float:
    w_pred = raw_svi_total_variance(k, params)
    return float(np.sum(weights * (w_pred - w_target) ** 2))


def _initial_guess(k: np.ndarray, w_target: np.ndarray) -> np.ndarray:
    """Practitioner standard: a0 = min(w), m0 = argmin k, b0 = 0.1, rho0 = -0.5,
    sigma0 = 0.1. (Zeliade Systems white paper convention; G&J 2014 itself
    does not specify an initial guess.)"""
    a0 = float(max(np.min(w_target), 1e-8))
    m0 = float(k[int(np.argmin(w_target))])
    return np.array([a0, 0.1, -0.5, m0, 0.1])


def fit_svi_slice(k: np.ndarray, iv: np.ndarray, T: float,
                  weights: Optional[np.ndarray] = None,
                  max_iter: int = 500) -> SliceFitResult:
    """L-BFGS-B fit of raw SVI to a single maturity slice on total-variance SSE.

    Parameters
    ----------
    k : (n_quotes,) forward log-moneyness.
    iv : (n_quotes,) observed market IV.
    T : float, time to expiry (years).
    weights : optional (n_quotes,) per-quote weights (default: uniform).
    max_iter : optimizer cap.

    Returns
    -------
    SliceFitResult with the fitted params, residual RMSEs, and a converged flag.
    Slices with < 5 quotes return a flat-IV fallback (b = 0).
    """
    n = len(k)
    if n < 5:
        sigma_avg = float(np.mean(iv))
        params = np.array([sigma_avg * sigma_avg * T, 0.0, 0.0, 0.0, 0.1])
        return SliceFitResult(params, 0.0, 0.0, n, converged=False)

    w_target = (iv * iv) * T
    if weights is None:
        weights = np.ones_like(w_target)

    initial = _initial_guess(k, w_target)
    # Roger Lee bound: b(1 + |rho|) <= 4/T  ->  b <= 4/T (when rho = 0).
    b_max = 4.0 / max(T, 1e-3)
    bounds = [
        (1e-8, max(np.max(w_target) * 2.0, 1.0)),
        (0.0, b_max),
        (-0.999, 0.999),
        (float(k.min()) - 1.0, float(k.max()) + 1.0),
        (1e-4, 5.0),
    ]
    res = minimize(
        _objective, initial, args=(k, w_target, weights),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )

    params = res.x
    w_pred = raw_svi_total_variance(k, params)
    rmse_w = float(np.sqrt(np.mean((w_pred - w_target) ** 2)))
    iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / T)
    rmse_iv = float(np.sqrt(np.mean((iv_pred - iv) ** 2)))
    return SliceFitResult(params, rmse_w, rmse_iv, n, bool(res.success))


# ---------------------------------------------------------------------------
# Per-day chain calibration
# ---------------------------------------------------------------------------

def fit_svi_chain(
    dates: np.ndarray,
    moneyness: np.ndarray,
    T_arr: np.ndarray,
    iv: np.ndarray,
    r: np.ndarray,
    *,
    min_quotes_per_slice: int = 5,
    T_round_decimals: int = 4,
    progress: bool = False,
) -> Dict[Tuple[str, float], SliceFitResult]:
    """Calibrate one raw SVI per (date, maturity) slice.

    Inputs are length-N arrays with one entry per quoted option. `dates` may
    be a numpy array of YYYY-MM-DD strings or numpy datetime64.

    Returns a dict keyed by (date_string, T_rounded) -> SliceFitResult.
    Slices with < min_quotes_per_slice quotes are skipped.
    """
    if dates.dtype.kind in ("M", "m"):
        date_str = np.datetime_as_string(dates, unit="D")
    else:
        date_str = np.asarray(dates).astype(str)

    # Forward log-moneyness: k = log(K/F) = -log(moneyness) - r*T
    # (zero dividends; see G&J p. 60 and the reference brief.)
    k_all = (-np.log(np.maximum(moneyness, 1e-8)) - r * T_arr).astype(np.float64)
    T_round = np.round(T_arr.astype(np.float64), T_round_decimals)

    # Group rows by (date, rounded T).
    keys = list(zip(date_str.tolist(), T_round.tolist()))
    unique_keys = sorted(set(keys))

    results: Dict[Tuple[str, float], SliceFitResult] = {}
    n_total = len(unique_keys)
    for i, key in enumerate(unique_keys):
        d, T_r = key
        mask = (date_str == d) & (T_round == T_r)
        if mask.sum() < min_quotes_per_slice:
            continue
        T_slice = float(T_arr[mask].mean())
        results[key] = fit_svi_slice(k_all[mask], iv[mask], T_slice)
        if progress and (i + 1) % 500 == 0:
            print(f"  fitted {i+1}/{n_total} slices "
                  f"({len(results)} accepted)", flush=True)

    return results


def evaluate_svi_iv(
    dates: np.ndarray,
    moneyness: np.ndarray,
    T_arr: np.ndarray,
    r: np.ndarray,
    fit_results: Dict[Tuple[str, float], SliceFitResult],
    *,
    fallback_iv: Optional[np.ndarray] = None,
    T_round_decimals: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Look up sigma_SVI for every row by querying its (date, T_rounded) slice.

    Returns
    -------
    iv_svi : (N,) float32, smile-coherent IV at each quote.
    is_fitted : (N,) bool, True where the row was matched to a calibrated slice.

    For rows without a calibrated slice, `fallback_iv[row]` is used.
    `fallback_iv` is required (no silent default) because hiding a missing
    fit behind a hardcoded constant masks bugs.
    """
    if fallback_iv is None:
        raise ValueError(
            "evaluate_svi_iv: rows whose (date, T) slice has no calibrated "
            "SVI need an explicit fallback_iv array (typically the raw "
            "market IV). No silent default is provided."
        )
    if dates.dtype.kind in ("M", "m"):
        date_str = np.datetime_as_string(dates, unit="D")
    else:
        date_str = np.asarray(dates).astype(str)

    k = (-np.log(np.maximum(moneyness, 1e-8)) - r * T_arr).astype(np.float64)
    T_round = np.round(T_arr.astype(np.float64), T_round_decimals)

    iv_svi = np.full(moneyness.shape, np.nan, dtype=np.float64)

    for key, result in fit_results.items():
        d, T_r = key
        mask = (date_str == d) & (T_round == T_r)
        if mask.sum() == 0:
            continue
        T_slice = float(T_arr[mask].mean())
        iv_svi[mask] = raw_svi_iv(k[mask], T_slice, result.params)

    is_fitted = ~np.isnan(iv_svi)
    if not is_fitted.all():
        iv_svi[~is_fitted] = fallback_iv[~is_fitted]

    iv_svi = np.maximum(iv_svi.astype(np.float32), 1e-3)
    return iv_svi, is_fitted


def fit_summary_table(fit_results: Dict[Tuple[str, float], SliceFitResult]) -> dict:
    """Aggregate fit-quality stats for diagnostics."""
    n = len(fit_results)
    if n == 0:
        return {"n_slices": 0}
    rmse_iv = np.array([r.rmse_iv for r in fit_results.values()])
    rmse_w = np.array([r.rmse_total_variance for r in fit_results.values()])
    n_quotes = np.array([r.n_quotes for r in fit_results.values()])
    converged = np.array([r.converged for r in fit_results.values()])
    return {
        "n_slices": n,
        "n_quotes_total": int(n_quotes.sum()),
        "convergence_rate": float(converged.mean()),
        "rmse_iv": {
            "mean": float(rmse_iv.mean()),
            "median": float(np.median(rmse_iv)),
            "p95": float(np.percentile(rmse_iv, 95)),
        },
        "rmse_total_variance": {
            "mean": float(rmse_w.mean()),
            "median": float(np.median(rmse_w)),
        },
    }
