#!/usr/bin/env python3
"""
Calibrate raw SVI on the SPY EOD options panel and cache the smile-coherent IV.

Usage:
  python experiments/real_data_spy/calibrate_svi.py

Output:
  data/spy_options/svi_iv.npy             — sigma_SVI for every row of spy_processed.npz
  data/spy_options/svi_params.npz         — fitted (a,b,rho,m,sigma) per (date, T_rounded)
  data/spy_options/svi_calibration_summary.json — fit-quality stats

The cache is consumed by `spy_data_loader.py` when invoked with target_mode="svi".
Calibration time: ~5 minutes (23k slices). The script is idempotent: if the
output files already exist and the upstream data file's hash matches, it
exits without recomputing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.real_data_spy.svi_calibration import (  # noqa: E402
    fit_svi_chain, evaluate_svi_iv, fit_summary_table,
)


SPY_DATA = ROOT / "data" / "spy_options" / "spy_processed.npz"
OUT_IV   = ROOT / "data" / "spy_options" / "svi_iv.npy"
OUT_PARAMS = ROOT / "data" / "spy_options" / "svi_params.npz"
OUT_SUMMARY = ROOT / "data" / "spy_options" / "svi_calibration_summary.json"
OUT_HASH = ROOT / "data" / "spy_options" / "svi_cache_hash.txt"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Recalibrate even if cache is already up-to-date.")
    ap.add_argument("--min-quotes-per-slice", type=int, default=5)
    ap.add_argument("--T-round-decimals", type=int, default=4)
    args = ap.parse_args()

    if not SPY_DATA.exists():
        raise FileNotFoundError(f"SPY data not found: {SPY_DATA}")
    data_hash = _file_hash(SPY_DATA)

    if (not args.force) and OUT_IV.exists() and OUT_HASH.exists():
        cached_hash = OUT_HASH.read_text().strip()
        if cached_hash == data_hash:
            print(f"SVI cache up-to-date (data sha16={data_hash}); skipping.")
            print(f"  iv_svi: {OUT_IV}")
            print(f"  params: {OUT_PARAMS}")
            return

    print(f"Loading {SPY_DATA}...")
    d = np.load(SPY_DATA)
    X = d["X"]
    dates = d["dates"]
    moneyness = X[:, 0].astype(np.float64)
    T_arr     = X[:, 1].astype(np.float64)
    r_arr     = X[:, 2].astype(np.float64)
    iv        = X[:, 3].astype(np.float64)
    n = len(moneyness)
    print(f"  rows={n}, n_unique_dates={len(np.unique(dates))}")

    print("Calibrating raw SVI per (date, maturity) slice...")
    t0 = time.time()
    fit_results = fit_svi_chain(
        dates=dates, moneyness=moneyness, T_arr=T_arr,
        iv=iv, r=r_arr,
        min_quotes_per_slice=args.min_quotes_per_slice,
        T_round_decimals=args.T_round_decimals,
        progress=True,
    )
    t_fit = time.time() - t0
    print(f"  done in {t_fit:.1f}s; {len(fit_results)} slices accepted.")

    print("Evaluating sigma_SVI for every quote (with raw-IV fallback)...")
    iv_svi, is_fitted = evaluate_svi_iv(
        dates=dates, moneyness=moneyness, T_arr=T_arr, r=r_arr,
        fit_results=fit_results, fallback_iv=iv,
        T_round_decimals=args.T_round_decimals,
    )
    print(f"  fitted: {int(is_fitted.sum())}/{n} ({100*is_fitted.mean():.1f}%)")
    print(f"  sigma_SVI range: [{iv_svi.min():.4f}, {iv_svi.max():.4f}]")
    diff = np.abs(iv_svi - iv.astype(np.float32))
    print(f"  |sigma_SVI - sigma_mkt|: mean={diff.mean():.4f}, "
          f"median={np.median(diff):.4f}, p95={np.percentile(diff, 95):.4f}")

    print(f"Saving cache files...")
    np.save(OUT_IV, iv_svi)

    keys = sorted(fit_results.keys())
    n_keys = len(keys)
    params_arr = np.zeros((n_keys, 5), dtype=np.float32)
    rmse_iv_arr = np.zeros(n_keys, dtype=np.float32)
    rmse_w_arr = np.zeros(n_keys, dtype=np.float32)
    n_quotes_arr = np.zeros(n_keys, dtype=np.int32)
    converged_arr = np.zeros(n_keys, dtype=bool)
    date_arr = np.array([k[0] for k in keys])
    T_arr_arr = np.array([k[1] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        result = fit_results[k]
        params_arr[i] = result.params.astype(np.float32)
        rmse_iv_arr[i] = result.rmse_iv
        rmse_w_arr[i] = result.rmse_total_variance
        n_quotes_arr[i] = result.n_quotes
        converged_arr[i] = result.converged

    np.savez(
        OUT_PARAMS,
        date=date_arr, T_rounded=T_arr_arr,
        params=params_arr,
        rmse_iv=rmse_iv_arr, rmse_total_variance=rmse_w_arr,
        n_quotes=n_quotes_arr, converged=converged_arr,
        param_names=np.array(["a", "b", "rho", "m_loc", "sigma_loc"]),
    )

    summary = fit_summary_table(fit_results)
    summary["data_sha16"] = data_hash
    summary["calibration_time_s"] = float(t_fit)
    summary["coverage"] = {
        "n_total": int(n),
        "n_fitted": int(is_fitted.sum()),
        "fraction_fitted": float(is_fitted.mean()),
    }
    summary["sigma_SVI_vs_market"] = {
        "abs_diff_mean": float(diff.mean()),
        "abs_diff_median": float(np.median(diff)),
        "abs_diff_p95": float(np.percentile(diff, 95)),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    OUT_HASH.write_text(data_hash)
    print(f"\nDone.")
    print(f"  {OUT_IV}")
    print(f"  {OUT_PARAMS}")
    print(f"  {OUT_SUMMARY}")
    print(f"  data_sha16={data_hash}")


if __name__ == "__main__":
    main()
