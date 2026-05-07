"""Sanity tests for raw SVI calibration (Gatheral & Jacquier 2014).

Synthetic-recovery test: generate IVs from a known SVI parameter set,
add small Gaussian noise, refit. The fitted IVs must match the targets to
within ~1e-3 absolute (well below typical SPY IV bid-ask spreads).
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.real_data_spy.svi_calibration import (
    raw_svi_total_variance,
    raw_svi_iv,
    fit_svi_slice,
    fit_svi_chain,
    evaluate_svi_iv,
)


def test_total_variance_form():
    # At k = m, the bracket = sqrt(sigma^2) = |sigma| -> w = a + b * sigma.
    a, b, rho, m, sigma = 0.04, 0.4, -0.7, 0.0, 0.1
    params = np.array([a, b, rho, m, sigma])
    w_at_m = float(raw_svi_total_variance(np.array([0.0]), params))
    assert np.isclose(w_at_m, a + b * sigma, atol=1e-12)


def test_iv_positive_under_constraint():
    # a + b*sigma*sqrt(1-rho^2) >= 0 keeps w >= 0 everywhere.
    a, b, rho, m, sigma = 0.0, 0.4, -0.7, 0.05, 0.1
    params = np.array([a, b, rho, m, sigma])
    k = np.linspace(-1.0, 1.0, 401)
    w = raw_svi_total_variance(k, params)
    assert (w > -1e-9).all(), f"w_min = {w.min():.3e}"


def test_synthetic_slice_recovery():
    """Generate IVs from a known SVI param set; refit; check IV agreement."""
    rng = np.random.default_rng(0)
    true_params = np.array([0.04, 0.4, -0.7, 0.05, 0.12])  # mild SPY-like smile
    T = 0.25
    k = np.linspace(-0.4, 0.4, 31)
    iv_true = raw_svi_iv(k, T, true_params)
    iv_obs = iv_true + rng.normal(scale=1e-4, size=k.shape)  # tiny IV noise

    result = fit_svi_slice(k, iv_obs, T)

    iv_fit = raw_svi_iv(k, T, result.params)
    abs_err = np.abs(iv_fit - iv_true)
    assert abs_err.max() < 1e-3, (
        f"max IV error {abs_err.max():.4e} exceeds 1e-3; params_fit={result.params}"
    )
    assert result.rmse_iv < 1e-3
    assert result.converged


def test_few_quote_slice_falls_back_to_flat():
    """With < 5 quotes, fit returns a flat-IV fallback (b = 0)."""
    rng = np.random.default_rng(1)
    iv_obs = np.array([0.20, 0.21, 0.22], dtype=float)
    k = rng.normal(size=3)
    T = 0.1
    result = fit_svi_slice(k, iv_obs, T)
    assert result.params[1] == 0.0
    assert not result.converged


def test_chain_fit_and_evaluate():
    """End-to-end: fit a 2-day, 2-maturity chain; evaluate IVs at each row;
    check we cover all rows that match a fitted slice."""
    rng = np.random.default_rng(2)
    N_per_slice = 20
    rows = []
    true_params_by_key = {}
    for date in ["2021-09-01", "2021-09-02"]:
        for T in [0.10, 0.30]:
            true = np.array([0.04, 0.3, -0.6, 0.05, 0.1])
            key = (date, round(T, 4))
            true_params_by_key[key] = true
            k = rng.uniform(-0.3, 0.3, size=N_per_slice)
            # k = log(K/F) = -log(m) - r*T  =>  log(m) = -k - r*T
            r = 0.02
            log_m = -k - r * T
            m_arr = np.exp(log_m)
            iv = raw_svi_iv(k, T, true) + rng.normal(scale=1e-4, size=k.shape)
            for i in range(N_per_slice):
                rows.append((date, m_arr[i], T, r, iv[i]))

    dates = np.array([r[0] for r in rows])
    moneyness = np.array([r[1] for r in rows])
    T_arr = np.array([r[2] for r in rows])
    r_arr = np.array([r[3] for r in rows])
    iv_arr = np.array([r[4] for r in rows])

    fits = fit_svi_chain(dates, moneyness, T_arr, iv_arr, r_arr,
                        min_quotes_per_slice=5, T_round_decimals=4)
    assert len(fits) == 4

    iv_svi, is_fitted = evaluate_svi_iv(
        dates, moneyness, T_arr, r_arr, fits,
        fallback_iv=iv_arr.astype(np.float32), T_round_decimals=4,
    )
    assert is_fitted.all()
    abs_err = np.abs(iv_svi - iv_arr.astype(np.float32))
    assert abs_err.max() < 5e-3, f"max IV error {abs_err.max():.4e}"


def test_fit_is_deterministic():
    """L-BFGS-B with fixed init + bounds + single-thread BLAS gives bit-equal
    fits on rerun. Empirical: matches across 3 reruns on real SPY-like data."""
    rng = np.random.default_rng(7)
    true_params = np.array([0.04, 0.4, -0.7, 0.05, 0.12])
    T = 0.25
    k = np.linspace(-0.4, 0.4, 31)
    iv = raw_svi_iv(k, T, true_params) + rng.normal(scale=1e-4, size=k.shape)

    res1 = fit_svi_slice(k, iv, T)
    res2 = fit_svi_slice(k, iv, T)
    np.testing.assert_array_equal(res1.params, res2.params)
    assert res1.rmse_iv == res2.rmse_iv
    assert res1.converged == res2.converged


def test_evaluate_requires_explicit_fallback():
    """Q1: evaluate_svi_iv used to default to 0.2 for unfitted rows. After the
    review fix, it raises so the caller cannot silently substitute a magic
    constant for a missing fit."""
    fits = {("2021-09-01", 0.10): None}  # no slice
    with pytest.raises(ValueError, match="explicit fallback_iv"):
        evaluate_svi_iv(
            dates=np.array(["2021-09-02"]),
            moneyness=np.array([1.0]),
            T_arr=np.array([0.10]),
            r=np.array([0.02]),
            fit_results={},
            fallback_iv=None,
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
