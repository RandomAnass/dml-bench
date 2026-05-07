"""
A1/A2 closed-form Fourier-linear σ* verification (no intercept).

Direct numerical check of the camera-ready theory appendix
(papers/tmp_sonnet/sobolev_fourier_sigma_star_camera_ready_appendix.tex):

  A1a — K=1 in-class:  σ* = σ_y √((2π)² + 2/λ)              (eq. K1_sigma_star, line 516)
  A1b — K=5 in-class:  σ* = σ_y √(S_5(λ)/T_5(λ)) ≈ 11.83 σ_y at λ=1  (eq. K5, line 583)
  A2  — aliasing N=24, K=5, K'=7:
        (1/N) Φ_5^T Φ_7 = -0.5
        (1/N) Ψ_5^T Ψ_7 = 70π²                                  (line 1056-1059)

Feature map matches the appendix exactly: φ_K(x) ∈ R^{2K} (NO INTERCEPT).
The 11.83 constant is computed for p = 2K = 10. With an intercept the prediction
breaks. See `extrapolation_plan_revised.md` §1.2.

Run:
    python experiments/extrapolation/closed_form_a1_a2.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/closed_form_a1_a2")
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Feature map (NO INTERCEPT)
# -----------------------------------------------------------------------------
def phi(x: np.ndarray, K: int) -> np.ndarray:
    """φ_K(x) ∈ R^{2K}: [sin(2πx), cos(2πx), ..., sin(2πKx), cos(2πKx)]."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * x))
        cols.append(np.cos(2 * np.pi * k * x))
    return np.column_stack(cols)


def psi(x: np.ndarray, K: int) -> np.ndarray:
    """ψ_K(x) = φ_K'(x) ∈ R^{2K}."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    cols = []
    for k in range(1, K + 1):
        ω = 2 * np.pi * k
        cols.append(ω * np.cos(ω * x))
        cols.append(-ω * np.sin(ω * x))
    return np.column_stack(cols)


def uniform_grid(N: int) -> np.ndarray:
    """x_i = -1 + 2i/N, i=0..N-1 — the appendix's grid (line 715-720)."""
    return -1.0 + 2.0 * np.arange(N) / N


# -----------------------------------------------------------------------------
# Closed-form σ* (in-class, no misspecification)
# -----------------------------------------------------------------------------
def sigma_star_finite(N: int, K: int, sigma_y: float, lam: float, x_test: float):
    """Finite-sample σ* via the boxed formula in the appendix (line 316-326).

    For exact in-class, δ_van = δ_dml = 0, so:
        σ*² = σ_y² (A_van - A_dml) / B_dml
    where
        A_van = φ*^T H^{-1} φ*
        A_dml = φ*^T M_λ^{-1} H M_λ^{-1} φ*
        B_dml = λ² φ*^T M_λ^{-1} G M_λ^{-1} φ*
    """
    x = uniform_grid(N)
    Φ = phi(x, K)
    Ψ = psi(x, K)
    H = Φ.T @ Φ
    G = Ψ.T @ Ψ
    M = H + lam * G
    φstar = phi(np.array([x_test]), K).reshape(-1)
    H_inv = np.linalg.inv(H)
    M_inv = np.linalg.inv(M)
    A_van = float(φstar @ H_inv @ φstar)
    A_dml = float(φstar @ M_inv @ H @ M_inv @ φstar)
    B_dml = float(lam ** 2 * (φstar @ M_inv @ G @ M_inv @ φstar))
    if sigma_y == 0:
        return 0.0, dict(A_van=A_van, A_dml=A_dml, B_dml=B_dml)
    num = sigma_y ** 2 * (A_van - A_dml)
    if num <= 0 or B_dml <= 0:
        return 0.0, dict(A_van=A_van, A_dml=A_dml, B_dml=B_dml)
    return float(np.sqrt(num / B_dml)), dict(A_van=A_van, A_dml=A_dml, B_dml=B_dml)


def sigma_star_largeN_K1(sigma_y: float, lam: float) -> float:
    """K=1 large-N orthogonal: σ* = σ_y √((2π)² + 2/λ).  Line 516."""
    return float(sigma_y * np.sqrt((2 * np.pi) ** 2 + 2 / lam))


def sigma_star_largeN_K5(sigma_y: float, lam: float) -> float:
    """K=5 large-N orthogonal: σ* = σ_y √(S_5/T_5).  Line 537-583."""
    ks = np.arange(1, 6)
    ω = 2 * np.pi * ks
    one_plus = 1 + lam * ω ** 2
    S = float(np.sum(1 - 1 / one_plus ** 2))
    T = float(lam ** 2 * np.sum(ω ** 2 / one_plus ** 2))
    return float(sigma_y * np.sqrt(S / T))


# -----------------------------------------------------------------------------
# A1a — K=1 verification
# -----------------------------------------------------------------------------
def a1a_table() -> pd.DataFrame:
    rows = []
    Ns = [24, 50, 100, 500]
    sigma_ys = [0.00, 0.05, 0.10, 0.20]
    for sigma_y in sigma_ys:
        sstar_largeN = sigma_star_largeN_K1(sigma_y, lam=1.0)
        for N in Ns:
            sstar_finite, diag = sigma_star_finite(
                N=N, K=1, sigma_y=sigma_y, lam=1.0, x_test=1.0
            )
            rows.append({
                "K": 1, "lambda": 1.0, "sigma_y": sigma_y, "N": N,
                "sigma_star_finite": sstar_finite,
                "sigma_star_largeN_theory": sstar_largeN,
                "rel_err_pct": (
                    100 * (sstar_finite - sstar_largeN) / sstar_largeN
                    if sstar_largeN > 0 else 0.0
                ),
                **{k: diag[k] for k in ("A_van", "A_dml", "B_dml")},
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# A1b — K=5 verification, with x_test variation to check region invariance
# -----------------------------------------------------------------------------
def a1b_table() -> pd.DataFrame:
    rows = []
    Ns = [24, 50, 100, 500]
    sigma_ys = [0.00, 0.05, 0.10, 0.20]
    x_tests = [1.0, 1.5, 2.0]      # d = 0 (boundary), 0.5, 1.0
    for sigma_y in sigma_ys:
        sstar_largeN = sigma_star_largeN_K5(sigma_y, lam=1.0)
        for N in Ns:
            for x_test in x_tests:
                sstar_finite, diag = sigma_star_finite(
                    N=N, K=5, sigma_y=sigma_y, lam=1.0, x_test=x_test
                )
                rows.append({
                    "K": 5, "lambda": 1.0, "sigma_y": sigma_y, "N": N, "x_test": x_test,
                    "sigma_star_finite": sstar_finite,
                    "sigma_star_largeN_theory": sstar_largeN,
                    "rel_err_pct": (
                        100 * (sstar_finite - sstar_largeN) / sstar_largeN
                        if sstar_largeN > 0 else 0.0
                    ),
                    **{k: diag[k] for k in ("A_van", "A_dml", "B_dml")},
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# A2 — aliasing diagnostics
# -----------------------------------------------------------------------------
def a2_aliasing(N: int = 24) -> dict:
    """Verify line 1056-1059:
        (1/N) Φ_5^T Φ_7 = -0.5
        (1/N) Ψ_5^T Ψ_7 = 70π²
    where Φ_5 = sin(2π·5x) column and Φ_7 = sin(2π·7x) column.
    """
    x = uniform_grid(N)
    sin5, cos5 = np.sin(2 * np.pi * 5 * x), np.cos(2 * np.pi * 5 * x)
    sin7, cos7 = np.sin(2 * np.pi * 7 * x), np.cos(2 * np.pi * 7 * x)
    dsin5, dcos5 = (2 * np.pi * 5) * cos5, -(2 * np.pi * 5) * sin5
    dsin7, dcos7 = (2 * np.pi * 7) * cos7, -(2 * np.pi * 7) * sin7

    return {
        "N": N,
        "phi5_phi7_per_N": {
            "sin5_sin7": float(sin5 @ sin7) / N,
            "cos5_cos7": float(cos5 @ cos7) / N,
            "sin5_cos7": float(sin5 @ cos7) / N,
            "cos5_sin7": float(cos5 @ sin7) / N,
        },
        "psi5_psi7_per_N": {
            "dsin5_dsin7": float(dsin5 @ dsin7) / N,
            "dcos5_dcos7": float(dcos5 @ dcos7) / N,
            "dsin5_dcos7": float(dsin5 @ dcos7) / N,
            "dcos5_dsin7": float(dcos5 @ dsin7) / N,
        },
        "expected": {
            "sin5_sin7_or_cos5_cos7": -0.5,
            "dsin5_dsin7_or_dcos5_dcos7": 70 * np.pi ** 2,
            "70_pi_sq_value": 70 * np.pi ** 2,
        },
    }


def a2_no_aliasing(N: int = 24, K: int = 5, K_prime: int = 6) -> dict:
    """No-aliasing case: K'=6 at N=24 (24 > 2·(5+6)=22). Leakage should be ≈ 0."""
    x = uniform_grid(N)
    Φ_in = phi(x, K)               # 2K columns
    Φ_out_cols = []
    Ψ_out_cols = []
    for k in range(K + 1, K_prime + 1):
        Φ_out_cols.append(np.sin(2 * np.pi * k * x))
        Φ_out_cols.append(np.cos(2 * np.pi * k * x))
        ω = 2 * np.pi * k
        Ψ_out_cols.append(ω * np.cos(ω * x))
        Ψ_out_cols.append(-ω * np.sin(ω * x))
    Φ_out = np.column_stack(Φ_out_cols)
    Ψ_out = np.column_stack(Ψ_out_cols)
    Ψ_in = psi(x, K)
    return {
        "N": N, "K": K, "K_prime": K_prime,
        "no_alias_condition_holds": N > 2 * (K + K_prime),
        "phi_in_phi_out_op_norm_per_N": float(
            np.linalg.norm(Φ_in.T @ Φ_out, ord=2)
        ) / N,
        "psi_in_psi_out_op_norm_per_N": float(
            np.linalg.norm(Ψ_in.T @ Ψ_out, ord=2)
        ) / N,
    }


# -----------------------------------------------------------------------------
# Sanity smoke checks
# -----------------------------------------------------------------------------
def smoke_in_class_zero_y() -> dict:
    """At σ_y=0 with K=5 in-class, σ* must be exactly 0 (vanilla recovers θ*).
    Same for K=1."""
    out = {}
    for K in (1, 5):
        sstar, _ = sigma_star_finite(N=500, K=K, sigma_y=0.0, lam=1.0, x_test=1.0)
        out[f"K={K}_sigmaY=0_finite_sigmaStar"] = sstar
    return out


# -----------------------------------------------------------------------------
# Monte Carlo cross-check (one cell)
# -----------------------------------------------------------------------------
def monte_carlo_check(K=5, N=500, sigma_y=0.10, lam=1.0,
                     x_test=1.0, n_trials=4000, sigma_grid=None):
    """Generate noise, fit closed-form, compute MSE_van and MSE_dml at x_test
    over many trials, find empirical σ* by bisection. Compare to formula.
    Random θ* drawn once, fixed across trials; only ν, ε are resampled."""
    rng = np.random.default_rng(0)
    x = uniform_grid(N)
    Φ = phi(x, K)
    Ψ = psi(x, K)
    H = Φ.T @ Φ
    G = Ψ.T @ Ψ
    M = H + lam * G
    H_inv = np.linalg.inv(H)
    M_inv = np.linalg.inv(M)
    φstar = phi(np.array([x_test]), K).reshape(-1)
    θstar = rng.standard_normal(2 * K)
    f_star = float(φstar @ θstar)
    if sigma_grid is None:
        sigma_grid = np.linspace(0, 6, 49)

    # Closed-form per-trial computation
    mse_van = []
    mse_dml = []
    for sigma in sigma_grid:
        sq_van, sq_dml = 0.0, 0.0
        for _ in range(n_trials):
            ν = rng.standard_normal(N)
            ε = rng.standard_normal(N)
            y = Φ @ θstar + sigma_y * ν
            g = Ψ @ θstar + sigma * ε
            θ_van = H_inv @ (Φ.T @ y)
            θ_dml = M_inv @ (Φ.T @ y + lam * (Ψ.T @ g))
            sq_van += (φstar @ θ_van - f_star) ** 2
            sq_dml += (φstar @ θ_dml - f_star) ** 2
        mse_van.append(sq_van / n_trials)
        mse_dml.append(sq_dml / n_trials)
    mse_van = np.asarray(mse_van)
    mse_dml = np.asarray(mse_dml)
    diff = mse_dml - mse_van
    # Find first σ where diff crosses zero
    cross = None
    for i in range(1, len(sigma_grid)):
        if diff[i - 1] < 0 <= diff[i]:
            x0, x1 = sigma_grid[i - 1], sigma_grid[i]
            y0, y1 = diff[i - 1], diff[i]
            cross = float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12))
            break
    formula, _ = sigma_star_finite(N=N, K=K, sigma_y=sigma_y, lam=lam, x_test=x_test)
    return {
        "K": K, "N": N, "sigma_y": sigma_y, "x_test": x_test,
        "n_trials_per_sigma": n_trials,
        "empirical_sigma_star": cross,
        "formula_sigma_star": formula,
        "rel_err_pct": (
            100 * (cross - formula) / formula
            if (cross is not None and formula > 0) else None
        ),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)

    # Smoke
    smoke = smoke_in_class_zero_y()
    print("===== SMOKE: σ_y=0 in-class → σ*=0 =====")
    print(json.dumps(smoke, indent=2))
    for k, v in smoke.items():
        assert abs(v) < 1e-9, f"FAIL: {k} = {v}, expected 0"
    print("PASS\n")

    # A1a
    df_a1a = a1a_table()
    df_a1a.to_csv(OUT / "a1a_K1.csv", index=False)
    print("===== A1a — K=1, λ=1 =====")
    print(df_a1a[["sigma_y", "N", "sigma_star_finite",
                  "sigma_star_largeN_theory", "rel_err_pct"]].to_string(index=False))
    print()

    # A1b
    df_a1b = a1b_table()
    df_a1b.to_csv(OUT / "a1b_K5.csv", index=False)
    print("===== A1b — K=5, λ=1 (large-N target ≈ 11.83 σ_y) =====")
    print(df_a1b[["sigma_y", "N", "x_test", "sigma_star_finite",
                  "sigma_star_largeN_theory", "rel_err_pct"]].to_string(index=False))
    print()

    # A2 — aliasing
    a2 = a2_aliasing(N=24)
    print("===== A2 — aliasing N=24, ℓ=7 vs k=5 =====")
    print(json.dumps(a2, indent=2))
    sin5_sin7 = a2["phi5_phi7_per_N"]["sin5_sin7"]
    dsin5_dsin7 = a2["psi5_psi7_per_N"]["dsin5_dsin7"]
    expected_70pisq = 70 * math.pi ** 2
    print()
    print(f"  sin5_sin7/N    = {sin5_sin7:+.10f}   expected -0.5 (sign may flip)")
    print(f"  dsin5_dsin7/N  = {dsin5_dsin7:+.6f}   expected ±{expected_70pisq:.6f} (sign may flip)")
    print()

    # A2 — no-aliasing K'=6
    a2_no = a2_no_aliasing(N=24, K=5, K_prime=6)
    print("===== A2 — no-aliasing N=24, K=5, K'=6 =====")
    print(json.dumps(a2_no, indent=2))
    print()

    # Monte Carlo cross-check (one cell)
    mc = monte_carlo_check(K=5, N=500, sigma_y=0.10, lam=1.0,
                           x_test=1.0, n_trials=4000)
    print("===== MC cross-check (K=5, N=500, σ_y=0.10) =====")
    print(json.dumps(mc, indent=2))
    print()

    (OUT / "a2_aliasing.json").write_text(json.dumps(
        {"a2_aliasing": a2, "a2_no_aliasing": a2_no, "mc_check": mc, "smoke": smoke},
        indent=2))
    print(f"Saved: {OUT}/{{a1a_K1,a1b_K5}}.csv  and  {OUT}/a2_aliasing.json")


if __name__ == "__main__":
    main()
