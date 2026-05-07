#!/usr/bin/env python3
"""
σ* (noise crossover threshold) per smooth function family — LOWESS version.

Replaces sigma_star_bca.py's win-rate + linear-interp method with LOWESS-on-log-ratio:
  1. Aggregate dml_fixed (λ=1.0) vs vanilla on tier1+2+3.
  2. For each (config, seed) pair: log_ratio = log10(MSE_DML / MSE_vanilla).
  3. For each σ: aggregate per-config-seed log_ratios.
  4. Fit LOWESS smoother on (σ, log_ratio); σ* = first σ where the smoothed curve
     crosses 0 from below to above.
  5. Percentile bootstrap CI: 1000 resamples over (config, seed) pairs at each σ,
     refit LOWESS, recompute σ*.

LOWESS is strictly better than win-rate + linear-interp when:
  - σ-grid is coarse (linear-interp resolution = grid spacing).
  - Crossings are not monotone (LOWESS is robust to local non-monotonicity).
  - The signal is the magnitude of MSE change, not just its sign (log-ratio is
    continuous; win-rate clips to 0/1 per pair).

Inputs:  results/tier{1,2,3}_benchmark/*.json   (same as sigma_star_bca.py)
Output:  papers/neurips_DB/evidence/sigma_star_lowess.json
         papers/neurips_DB/evidence/sigma_star_lowess.txt
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json   # noqa: E402

TIER_DIRS = [
    ROOT / "results/tier1_benchmark",
    ROOT / "results/tier2_benchmark",
    ROOT / "results/tier3_benchmark",
]
OUT_JSON = ROOT / "papers/neurips_DB/evidence/sigma_star_lowess.json"
OUT_TXT = ROOT / "papers/neurips_DB/evidence/sigma_star_lowess.txt"

SMOOTH_FUNCS = ["poly_trig", "trig", "bachelier"]
TARGET_METHOD = "dml_fixed"
BASELINE_METHOD = "vanilla"
N_BOOT = 1000
RNG_SEED = 42
LOWESS_FRAC = 0.40         # rolling window covers ~40% of σ-grid points


def load_pairs():
    """Mirror of sigma_star_bca.load_pairs — same field discovery + filter logic."""
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
    paired = defaultdict(lambda: defaultdict(list))
    for (func, dim, n, sigma, seed), m in by_config.items():
        if TARGET_METHOD in m and BASELINE_METHOD in m:
            paired[func][float(sigma)].append((m[TARGET_METHOD], m[BASELINE_METHOD]))
    return paired


def log_ratios_per_sigma(pairs_by_sigma):
    """Returns sigmas_arr, list_of_log_ratios_per_sigma."""
    sigmas = sorted(pairs_by_sigma.keys())
    out = []
    for s in sigmas:
        pairs = pairs_by_sigma[s]
        # log10(MSE_target / MSE_baseline). Tiny floor to avoid log(0).
        eps = 1e-30
        out.append(np.array(
            [np.log10((t + eps) / (b + eps)) for t, b in pairs]
        ))
    return np.array(sigmas), out


def sigma_star_lowess(sigmas_arr, log_ratio_lists, frac=LOWESS_FRAC) -> dict:
    """LOWESS-smoothed crossing of 0 (from below to above) on mean log-ratio.

    Returns: {sigma_star, status, mean_log_ratio_per_sigma, lowess_curve}.
    """
    means = np.array([float(np.mean(lr)) if len(lr) > 0 else np.nan
                      for lr in log_ratio_lists])
    valid = ~np.isnan(means)
    sx, sy = sigmas_arr[valid], means[valid]
    if len(sx) < 4:
        return {"sigma_star": None, "status": "insufficient_sigmas",
                "mean_log_ratio_per_sigma": means.tolist(),
                "sigmas": sigmas_arr.tolist()}

    sm = lowess(sy, sx, frac=frac, return_sorted=True)
    xs, ys = sm[:, 0], sm[:, 1]

    # Status logic identical to the pilot's estimate_sigma_star, but on log-ratio:
    #   - left-censored (DML already worse at smallest σ): ys[0] >= 0
    #   - right-censored (no crossing within grid): all ys < 0
    #   - observed: find first crossing, linear-interp on the LOWESS output
    if ys[0] >= 0:
        return {
            "sigma_star": float(sx[0]),
            "status": "left_censored",
            "mean_log_ratio_per_sigma": means.tolist(),
            "sigmas": sigmas_arr.tolist(),
            "lowess_x": xs.tolist(),
            "lowess_y": ys.tolist(),
        }
    if np.all(ys < 0):
        return {
            "sigma_star": float(sx[-1]),
            "status": "right_censored",
            "mean_log_ratio_per_sigma": means.tolist(),
            "sigmas": sigmas_arr.tolist(),
            "lowess_x": xs.tolist(),
            "lowess_y": ys.tolist(),
        }
    for i in range(1, len(xs)):
        if ys[i - 1] < 0 <= ys[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            sstar = float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0 + 1e-12))
            return {
                "sigma_star": sstar,
                "status": "observed",
                "mean_log_ratio_per_sigma": means.tolist(),
                "sigmas": sigmas_arr.tolist(),
                "lowess_x": xs.tolist(),
                "lowess_y": ys.tolist(),
            }
    return {"sigma_star": None, "status": "unknown",
            "mean_log_ratio_per_sigma": means.tolist(),
            "sigmas": sigmas_arr.tolist()}


def bootstrap_sigma_star(pairs_by_sigma, n_boot=N_BOOT, seed=RNG_SEED) -> dict:
    """Percentile bootstrap on σ* via per-σ resampling of (config, seed) pairs.

    Same RNG logic as sigma_star_bca.bootstrap_sigma_star. Lower-coverage CI
    method (percentile, not BCa) — load-bearing claim is the point estimate.
    """
    rng = np.random.default_rng(seed)
    sigmas_obs = sorted(pairs_by_sigma.keys())
    if not sigmas_obs:
        return None

    # Point estimate
    sx, lr_lists = log_ratios_per_sigma(pairs_by_sigma)
    point = sigma_star_lowess(sx, lr_lists)

    # Bootstrap
    boot = np.empty(n_boot)
    n_undef = 0
    for b in range(n_boot):
        resampled = {}
        for s, pairs in pairs_by_sigma.items():
            if not pairs:
                resampled[s] = []
                continue
            idx = rng.integers(0, len(pairs), size=len(pairs))
            resampled[s] = [pairs[i] for i in idx]
        rx, rlr = log_ratios_per_sigma(resampled)
        rs = sigma_star_lowess(rx, rlr)
        if rs["sigma_star"] is None or rs["status"] in ("insufficient_sigmas", "unknown"):
            n_undef += 1
            boot[b] = np.nan
        else:
            boot[b] = rs["sigma_star"]

    valid = boot[~np.isnan(boot)]
    if len(valid) < 50:
        return {
            "point_estimate": point["sigma_star"],
            "status": point["status"],
            "ci_method": "insufficient_bootstrap_samples",
            "ci_lower": None, "ci_upper": None,
            "n_boot": n_boot, "n_undefined": n_undef,
            "mean_log_ratio_per_sigma": point["mean_log_ratio_per_sigma"],
            "sigmas": point["sigmas"],
        }
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "point_estimate": point["sigma_star"],
        "status": point["status"],
        "ci_method": "percentile_2.5_97.5_lowess",
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot": n_boot,
        "n_valid": int(len(valid)),
        "n_undefined": n_undef,
        "mean_log_ratio_per_sigma": point["mean_log_ratio_per_sigma"],
        "sigmas": point["sigmas"],
        "lowess_x": point.get("lowess_x"),
        "lowess_y": point.get("lowess_y"),
        "lowess_frac": LOWESS_FRAC,
    }


def main():
    paired = load_pairs()
    if not paired:
        print("No paired (target, baseline) configs found across tier 1-3.",
              file=sys.stderr)
        sys.exit(1)
    out = {
        "method": "lowess_on_log_ratio",
        "lowess_frac": LOWESS_FRAC,
        "target_method": TARGET_METHOD,
        "baseline_method": BASELINE_METHOD,
        "smooth_functions": SMOOTH_FUNCS,
        "n_boot": N_BOOT,
        "ratio": "log10(MSE_dml / MSE_vanilla)",
        "results": {},
    }
    for func in SMOOTH_FUNCS:
        if func not in paired:
            print(f"  no data for {func}")
            continue
        pairs_by_sigma = paired[func]
        n_pairs_total = sum(len(v) for v in pairs_by_sigma.values())
        print(f"  {func}: n_sigmas={len(pairs_by_sigma)} n_pairs={n_pairs_total}")
        out["results"][func] = bootstrap_sigma_star(pairs_by_sigma)

    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = [
        "σ* (noise crossover, LOWESS on log10(MSE_dml / MSE_vanilla))",
        f"  LOWESS frac: {LOWESS_FRAC}    bootstrap n: {N_BOOT}",
        "",
    ]
    for func, b in out["results"].items():
        if b is None:
            lines.append(f"  {func}: no data"); continue
        sstar = b["point_estimate"]
        lo = b["ci_lower"]; hi = b["ci_upper"]
        status = b["status"]
        s_str = f"{sstar:.4f}" if sstar is not None else "undef"
        ci_str = (f"95% CI [{lo:.4f}, {hi:.4f}]"
                  if lo is not None else "CI undefined")
        lines.append(f"  {func:<12s} σ* = {s_str:<10s}  {ci_str}  status={status}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
