#!/usr/bin/env python3
"""
Compute σ* (noise crossover threshold) per smooth function family with a
percentile bootstrap 95% CI (1000 resamples).

σ* is the noise level at which DML's mean win-rate over (config, seed)
crosses 0.5 (i.e., DML is no longer better than vanilla on the majority of
configurations). For each function family ∈ {poly_trig, trig, bachelier},
we:
  1. Aggregate dml_fixed (λ=1.0 canonical only — lambda ablation excluded)
     vs vanilla on tier1+2+3 result JSONs.
  2. Compute paired-seed win-rate at every observed noise level.
  3. Fit a monotone-in-σ smoother (linear interpolation between observed
     noise levels) to the win-rate curve.
  4. σ* = first σ where the smoothed curve crosses 0.5.
  5. Percentile bootstrap CI (1000 resamples) over per-sigma config-pair
     resampling. Note: this is plain percentile bootstrap, not BCa
     (bias-corrected accelerated) — the BCa correction would require
     computing bias and acceleration constants per per-sigma block, which
     is not implemented here. Percentile bootstrap is consistent and
     reasonable for monotone summary statistics like σ*.

Inputs:  results/tier{1,2,3}_benchmark/*.json
Output:  papers/neurips_DB/evidence/sigma_star_bca.json
         papers/neurips_DB/evidence/sigma_star_bca.txt   (human-readable)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json
TIER_DIRS = [
    ROOT / "results/tier1_benchmark",
    ROOT / "results/tier2_benchmark",
    ROOT / "results/tier3_benchmark",
]
OUT_JSON = ROOT / "papers/neurips_DB/evidence/sigma_star_bca.json"
OUT_TXT = ROOT / "papers/neurips_DB/evidence/sigma_star_bca.txt"

SMOOTH_FUNCS = ["poly_trig", "trig", "bachelier"]
TARGET_METHOD = "dml_fixed"
BASELINE_METHOD = "vanilla"
N_BOOT = 1000
RNG_SEED = 42


def load_pairs():
    """
    For every (function, dim, n_samples, noise, seed), collect
    (test_value_mse_target, test_value_mse_baseline) pairs.

    Field names confirmed from a tier-3 sample (2026-04-29):
      func_type      (NOT "dataset")
      noise_level    (NOT "noise")
      n_samples      (NOT "n_train" — that key only appears in molecular tiers)
      lambda         (lambda-ablation files have lambda != 1.0; we filter
                      target_method to lambda == 1.0 to avoid double-counting
                      the same (func, dim, n, noise, seed) cell)

    Returns: dict[func][noise] -> list of (target_mse, baseline_mse).
    """
    by_config = defaultdict(lambda: {})
    for tdir in TIER_DIRS:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            method = r.get("method")
            if method not in (TARGET_METHOD, BASELINE_METHOD):
                continue
            # Filter dml_fixed lambda-ablation files to canonical λ=1.0 only.
            # Vanilla has no λ; allow any.
            if method == TARGET_METHOD:
                lam = r.get("lambda")
                if lam is not None and float(lam) != 1.0:
                    continue
            func = r.get("func_type") or r.get("dataset")
            if func not in SMOOTH_FUNCS:
                continue
            dim = r.get("dim")
            n = r.get("n_samples") or r.get("n_train")
            sigma = r.get("noise_level")
            if sigma is None:
                sigma = r.get("noise")
            seed = r.get("seed")
            val = r.get("test_value_mse")
            if None in (dim, n, sigma, seed, val):
                continue
            key = (func, dim, n, float(sigma), seed)
            by_config.setdefault(key, {})[method] = float(val)
    paired = defaultdict(lambda: defaultdict(list))   # func -> sigma -> [(t,b)]
    for (func, dim, n, sigma, seed), m in by_config.items():
        if TARGET_METHOD in m and BASELINE_METHOD in m:
            paired[func][float(sigma)].append((m[TARGET_METHOD], m[BASELINE_METHOD]))
    return paired


def winrate_curve(pairs_by_sigma):
    """Mean paired win-rate per sigma. Returns sorted (sigma_arr, wr_arr, n_arr)."""
    sigmas = sorted(pairs_by_sigma.keys())
    wr = []
    n = []
    for s in sigmas:
        pairs = pairs_by_sigma[s]
        if not pairs:
            wr.append(np.nan); n.append(0); continue
        wins = sum(1 for t, b in pairs if t < b)   # lower MSE = win
        wr.append(wins / len(pairs))
        n.append(len(pairs))
    return np.array(sigmas), np.array(wr), np.array(n)


def sigma_star(sigmas, wr, threshold=0.5):
    """First sigma where smoothed curve crosses threshold from above to below.

    Linear interpolation between observed sigmas. Returns NaN if curve never
    crosses (e.g., always above 0.5 ⇒ DML helps everywhere; or always below
    ⇒ DML never helps)."""
    if len(sigmas) < 2:
        return float("nan")
    # Look for the crossing: find adjacent points where wr[i] >= 0.5 > wr[i+1].
    for i in range(len(sigmas) - 1):
        a, b = wr[i], wr[i + 1]
        sa, sb = sigmas[i], sigmas[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a >= threshold > b:
            # linear interp: σ* = sa + (threshold - a) * (sb - sa) / (b - a)
            denom = (b - a)
            if denom == 0:
                return sa
            return float(sa + (threshold - a) * (sb - sa) / denom)
    return float("nan")


def bootstrap_sigma_star(pairs_by_sigma, n_boot=N_BOOT, seed=RNG_SEED):
    """BCa bootstrap on σ* via seed-stratified resampling of pairs.

    Returns dict with point estimate, 2.5/97.5 BCa percentiles, n_boot,
    and "n_undefined" (resamples that produced NaN σ*).
    """
    rng = np.random.default_rng(seed)
    sigmas_obs = sorted(pairs_by_sigma.keys())
    if not sigmas_obs:
        return None
    # Point estimate
    s_arr, wr_arr, _ = winrate_curve(pairs_by_sigma)
    sstar_point = sigma_star(s_arr, wr_arr)

    boot_estimates = np.empty(n_boot)
    n_undefined = 0
    for b in range(n_boot):
        resampled = {}
        for s, pairs in pairs_by_sigma.items():
            if not pairs:
                resampled[s] = []
                continue
            idx = rng.integers(0, len(pairs), size=len(pairs))
            resampled[s] = [pairs[i] for i in idx]
        sa, wa, _ = winrate_curve(resampled)
        ss = sigma_star(sa, wa)
        if np.isnan(ss):
            n_undefined += 1
            boot_estimates[b] = np.nan
        else:
            boot_estimates[b] = ss

    # BCa via scipy when feasible; otherwise percentile fallback.
    valid = boot_estimates[~np.isnan(boot_estimates)]
    if len(valid) < 50:
        return {
            "point_estimate": float(sstar_point) if not np.isnan(sstar_point) else None,
            "ci_method": "insufficient bootstrap samples (less than 50 valid resamples)",
            "ci_lower": None, "ci_upper": None,
            "n_boot": n_boot, "n_undefined": n_undefined,
            "sigmas_observed": s_arr.tolist(),
            "winrate_observed": wr_arr.tolist(),
        }
    # Bias-correction (BCa simplification: percentile-of-z0)
    # z0 = Phi^{-1}( fraction(boot < point) ); CI = Phi(z0 + z_{alpha/2} +- z0).
    # For small samples we use plain percentile to keep this stable; report method.
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "point_estimate": float(sstar_point) if not np.isnan(sstar_point) else None,
        "ci_method": "percentile_2.5_97.5_over_valid_bootstrap_resamples",
        "ci_lower": float(lo), "ci_upper": float(hi),
        "n_boot": n_boot, "n_valid": int(len(valid)), "n_undefined": n_undefined,
        "sigmas_observed": s_arr.tolist(),
        "winrate_observed": wr_arr.tolist(),
    }


def main():
    paired = load_pairs()
    if not paired:
        print("No paired (target, baseline) configs found across tier 1-3.", file=sys.stderr)
        sys.exit(1)
    out = {
        "target_method": TARGET_METHOD,
        "baseline_method": BASELINE_METHOD,
        "smooth_functions": SMOOTH_FUNCS,
        "n_boot": N_BOOT,
        "results": {},
    }
    for func in SMOOTH_FUNCS:
        if func not in paired:
            print(f"  no data for {func}")
            continue
        pairs_by_sigma = paired[func]
        n_pairs_total = sum(len(v) for v in pairs_by_sigma.values())
        print(f"  {func}: n_sigmas={len(pairs_by_sigma)} n_pairs={n_pairs_total}")
        boot = bootstrap_sigma_star(pairs_by_sigma)
        out["results"][func] = boot

    OUT_JSON.write_text(json.dumps(out, indent=2))
    # Human readable
    lines = ["sigma* (noise crossover) — DML vs vanilla, smooth families", ""]
    for func, boot in out["results"].items():
        if boot is None:
            lines.append(f"  {func}: no data"); continue
        sstar = boot["point_estimate"]
        lo = boot["ci_lower"]; hi = boot["ci_upper"]
        lines.append(
            f"  {func:<12s} sigma* = {sstar if sstar is not None else 'undef'}"
            + (f"   95% CI [{lo:.4f}, {hi:.4f}]" if lo is not None else "   CI undefined")
        )
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
