"""
SPY proxy-Greek stress-test perturbations.

Given a clean SPY data dict produced by `spy_data_loader.load_spy_data()`,
return a new dict whose `dydx_train` (the proxy Black-Scholes Greeks used
as training derivative labels) has been corrupted along one or more axes.
Test data and the value labels (`y_*`) are NEVER perturbed — we evaluate
how well DML methods learn the true derivative under degraded label
quality, not under degraded test conditions.

Perturbation axes (from `papers/plan_and_notes.md` §7 Priority F):

  1. sigma_staleness      — implied-vol staleness; simulates training on
                            σ_{t−k} when the true label requires σ_t.
                            Implemented as multiplicative iv noise with
                            std scaling as √k; the constant 0.02 is the
                            SPY 2020–2022 daily iv vol-of-vol estimate.
                            k = 0 reproduces the clean case.

  2. sigma_misspec        — implied-vol misspecification by a constant
                            offset Δσ; iv ← iv + Δσ before Greek
                            recomputation. Tests how DML degrades when
                            the σ used to label is biased high or low.

  3. greek_additive_noise — independent Gaussian noise on every Greek
                            column with std ε · std(greek_col).
                            Captures noise from finite-difference Greek
                            estimation or coarse iv interpolation.

  4. greek_multiplicative_noise — Greek columns scaled by (1 + ε · randn).
                            Captures heteroscedastic Greek estimation
                            error.

All perturbations are deterministic given (config, seed). Any combination
of axes can be applied; effects compound multiplicatively.

References:
  - Lopez de Prado 2018, "Advances in Financial ML" §7 (proxy-label limits)
  - Glasserman & Karmarkar 2025, "Likelihood Ratio Method" (clean vs noisy
    Greek estimation)
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from .spy_data_loader import compute_bs_greeks


def perturb_spy_data(
    data: Dict[str, Any],
    *,
    sigma_staleness_days: int = 0,
    sigma_misspec_delta: float = 0.0,
    greek_additive_noise: float = 0.0,
    greek_multiplicative_noise: float = 0.0,
    include_volume: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Return a copy of `data` with `dydx_train` perturbed per the config.

    Args:
        data: dict from `spy_data_loader.load_spy_data()`. Required keys:
              x_train (n,5) [moneyness, T, r, iv, log_volume], dydx_train
              (n,1,d), greeks_train (per-Greek dict).
        sigma_staleness_days: k. iv perturbed by ×(1 + 0.02·√k·N(0,1)).
                              Re-derives Greeks from perturbed iv.
        sigma_misspec_delta: Δσ. iv ← iv + Δσ. Re-derives Greeks.
        greek_additive_noise: ε. dydx ← dydx + ε·std(dydx_col)·N(0,1).
        greek_multiplicative_noise: ε. dydx ← dydx · (1 + ε·N(0,1)).
        include_volume: whether the input includes log_volume (Greek dim).
        seed: deterministic RNG seed (independent of train seed).

    Returns:
        New dict with perturbed `dydx_train`. `dydx_test`, `x_*`, `y_*`
        unchanged. Adds `perturbation` metadata key.
    """
    rng = np.random.RandomState(seed)
    out = copy.copy(data)

    x_train = data["x_train"].copy()
    dydx_train = data["dydx_train"].copy()  # (n, 1, d)
    n = x_train.shape[0]
    d = dydx_train.shape[-1]

    perturbation_log: Dict[str, Any] = {
        "sigma_staleness_days": int(sigma_staleness_days),
        "sigma_misspec_delta": float(sigma_misspec_delta),
        "greek_additive_noise": float(greek_additive_noise),
        "greek_multiplicative_noise": float(greek_multiplicative_noise),
        "perturbation_seed": int(seed),
    }

    # ---- Phase 1: perturb iv → re-derive Greeks ----
    needs_regreek = (sigma_staleness_days > 0) or (sigma_misspec_delta != 0.0)
    if needs_regreek:
        moneyness = x_train[:, 0]
        T = x_train[:, 1]
        r = x_train[:, 2]
        iv = x_train[:, 3].copy()

        # Staleness: multiplicative noise with std scaling as √k.
        # 0.02 ≈ SPY 2020–2022 daily iv vol-of-vol (estimated from data).
        if sigma_staleness_days > 0:
            stale_factor = 1.0 + 0.02 * np.sqrt(sigma_staleness_days) * rng.randn(n)
            iv = iv * stale_factor

        # Misspecification: constant offset.
        if sigma_misspec_delta != 0.0:
            iv = iv + sigma_misspec_delta

        # Clip to reasonable range to avoid numerical blow-up at extremes.
        iv = np.clip(iv, 1e-4, 5.0)

        # Re-derive Greeks with perturbed iv.
        greeks = compute_bs_greeks(moneyness, T, r, iv, include_volume=include_volume)
        # Replace the (n, d) Greek block. dydx_train shape is (n, 1, d);
        # write into [:, 0, :].
        dydx_train[:, 0, :] = greeks["stacked"].astype(dydx_train.dtype)
        perturbation_log["regreek_applied"] = True

    # ---- Phase 2: per-Greek-column noise on the (re-)computed labels ----
    if greek_additive_noise > 0.0:
        # Independent additive Gaussian per column, scaled by per-column std.
        flat = dydx_train.reshape(n, d)
        col_std = flat.std(axis=0, keepdims=True)            # (1, d)
        col_std = np.where(col_std > 0, col_std, 1.0)
        noise = greek_additive_noise * col_std * rng.randn(n, d)
        flat = flat + noise.astype(flat.dtype)
        dydx_train = flat.reshape(dydx_train.shape)

    if greek_multiplicative_noise > 0.0:
        # Independent multiplicative Gaussian per element.
        scale = 1.0 + greek_multiplicative_noise * rng.randn(*dydx_train.shape)
        dydx_train = dydx_train * scale.astype(dydx_train.dtype)

    out["dydx_train"] = dydx_train.astype(data["dydx_train"].dtype)
    out["dydx_test"] = data["dydx_test"]  # explicitly unchanged
    out["x_train"] = data["x_train"]      # x is NOT perturbed (only iv used for Greek derivation)
    out["y_train"] = data["y_train"]      # y unchanged

    meta = dict(out.get("metadata", {}))
    meta["perturbation"] = perturbation_log
    out["metadata"] = meta

    return out
