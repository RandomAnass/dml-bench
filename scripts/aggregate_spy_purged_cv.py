#!/usr/bin/env python3
"""
Aggregate SPY purged-CV results from `results/spy_options_purged_cv_option{A,C}/`
into a per-(target, fold) CSV that F10 (`fig_spy_purged_cv_per_fold.py`) reads.

Schema matches the prior `results/spy_regional/spy_regional_per_fold.csv` exactly so
the figure script needs no change. Carries over the IV-drift columns from the prior
CSV (those are properties of the splits, not the runs, so they don't change).

Aggregation rule per (target, n_train, fold, method):
  - val_mse_mean  = mean of test_value_mse over seeds
  - grad_mse_mean = mean of test_grad_mse over seeds
  - val_pct       = 100 * (method - vanilla) / vanilla   (paired by seed mean)
  - grad_pct      = 100 * (method - vanilla) / vanilla

We pair at the cell level (per-fold seeds collapsed to mean) which matches the
prior CSV's convention and the F10 figure that reads it.

Usage:
    python scripts/aggregate_spy_purged_cv.py
    # writes results/spy_regional/spy_regional_per_fold.csv
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "spy_regional" / "spy_regional_per_fold.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OPTIONS = [
    ("BS-formula",  ROOT / "results" / "spy_options_purged_cv_optionA"),
    ("SVI-coherent", ROOT / "results" / "spy_options_purged_cv_optionC"),
]
METHODS = ["dml_fixed", "dml_fixed_half", "dml_gradnorm", "dml_relobralo", "dml_warmup"]


def load_cells(d: Path) -> dict:
    """{(n_train, fold, method): list of (seed, val_mse, grad_mse)}"""
    out = defaultdict(list)
    for f in glob.glob(str(d / "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        m = j.get("method")
        n_train = j.get("n_train")
        fold = j.get("fold_idx")
        if fold is None:
            fold = (j.get("metadata") or {}).get("fold_idx")
        seed = j.get("seed")
        v = j.get("test_value_mse")
        g = j.get("test_grad_mse")
        if None in (m, n_train, fold, seed, v, g):
            continue
        out[(int(n_train), int(fold), m)].append((int(seed), float(v), float(g)))
    return out


def cell_mean(cells: dict, key: tuple) -> tuple:
    """Returns (n_seeds, mean_val_mse, mean_grad_mse) or (0, nan, nan)."""
    rows = cells.get(key, [])
    if not rows:
        return 0, float("nan"), float("nan")
    vals = np.array([r[1] for r in rows])
    grads = np.array([r[2] for r in rows])
    return len(rows), float(vals.mean()), float(grads.mean())


def load_iv_drift_existing() -> dict:
    """Lift the IV-drift + n_test_outside_train_moneyness columns from the existing
    CSV — those don't change with re-runs (they're properties of the splits)."""
    out = {}
    if not OUT_CSV.exists():
        return out
    with open(OUT_CSV) as fh:
        for r in csv.DictReader(fh):
            key = (r["target"], int(r["fold"]))
            out[key] = {
                "iv_train_mean": r.get("iv_train_mean", ""),
                "iv_test_mean":  r.get("iv_test_mean", ""),
                "iv_drift_abs":  r.get("iv_drift_abs", ""),
                "n_test_outside_train_moneyness": r.get("n_test_outside_train_moneyness", ""),
            }
    return out


def main() -> None:
    iv_drift = load_iv_drift_existing()

    fieldnames = (
        ["target", "fold", "n_seeds", "vanilla_val_mse", "vanilla_grad_mse"] +
        sum([[f"{m}_val_mse", f"{m}_val_pct", f"{m}_grad_pct"] for m in METHODS], []) +
        ["iv_train_mean", "iv_test_mean", "iv_drift_abs", "n_test_outside_train_moneyness"]
    )

    rows_out = []
    for target, d in OPTIONS:
        cells = load_cells(d)
        # F10 expects n=10000 rows (the larger n_train cell ranges in the existing CSV).
        # Verify by inspecting which (n_train, fold) tuples we have.
        n_trains = sorted({k[0] for k in cells.keys()})
        folds = sorted({k[1] for k in cells.keys()})
        if not n_trains or not folds:
            print(f"  [skip] {target}: no cells found in {d}")
            continue
        # Use the LARGEST n_train (matches prior CSV behavior; F10 plots one row per fold).
        n_train_used = max(n_trains)
        for fold in folds:
            van_n, van_v, van_g = cell_mean(cells, (n_train_used, fold, "vanilla"))
            row = {
                "target": target,
                "fold": fold,
                "n_seeds": van_n,
                "vanilla_val_mse": van_v,
                "vanilla_grad_mse": van_g,
            }
            for m in METHODS:
                _, mv, mg = cell_mean(cells, (n_train_used, fold, m))
                pct_v = 100.0 * (mv - van_v) / van_v if van_v > 0 and not np.isnan(mv) else float("nan")
                pct_g = 100.0 * (mg - van_g) / van_g if van_g > 0 and not np.isnan(mg) else float("nan")
                row[f"{m}_val_mse"] = mv
                row[f"{m}_val_pct"] = pct_v
                row[f"{m}_grad_pct"] = pct_g
            ivd = iv_drift.get((target, fold), {})
            row.update({
                "iv_train_mean": ivd.get("iv_train_mean", ""),
                "iv_test_mean":  ivd.get("iv_test_mean", ""),
                "iv_drift_abs":  ivd.get("iv_drift_abs", ""),
                "n_test_outside_train_moneyness": ivd.get("n_test_outside_train_moneyness", ""),
            })
            rows_out.append(row)
        print(f"  [{target}] n_train={n_train_used}, folds={folds}, "
              f"methods present per cell={[len(cells.get((n_train_used, fold, m), [])) for m in ['vanilla'] + METHODS]}")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nwrote {OUT_CSV}: {len(rows_out)} rows")


if __name__ == "__main__":
    main()
