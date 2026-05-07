#!/usr/bin/env python3
"""
Single-cell-drop sensitivity check on the synthetic-pillar R1 statistic.

Identify the synthetic cell with the largest paired log-ratio in either
direction. Remove it from the corpus and recompute R1. Report the shift.

Output (internal, not paper-cited):
  papers/neurips_DB/evidence/single_cell_drop.json
  papers/neurips_DB/evidence/single_cell_drop.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json

TIER_DIRS = [ROOT / f"results/tier{i}_benchmark" for i in (1, 2, 3, 4)]
OUT_JSON = ROOT / "papers/neurips_DB/evidence/single_cell_drop.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/single_cell_drop.md"

DML = "dml_fixed"
VANILLA = "vanilla"
EPS = 1e-12


def load_paired_log_ratios():
    """Load per-cell paired log-ratios at sigma=0."""
    by_cell = defaultdict(dict)
    for tdir in TIER_DIRS:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            method = r.get("method")
            if method not in (DML, VANILLA):
                continue
            if method == DML:
                lam = r.get("lambda")
                if lam is not None and float(lam) != 1.0:
                    continue
            func = r.get("func_type") or r.get("dataset")
            dim = r.get("dim")
            n = r.get("n_samples") or r.get("n_train")
            sigma = r.get("noise_level")
            if sigma is None:
                sigma = r.get("noise")
            seed = r.get("seed")
            v = r.get("test_value_mse")
            g = r.get("test_grad_mse")
            if None in (func, dim, n, sigma, seed, v, g):
                continue
            if float(sigma) != 0.0:
                continue
            cell = (func, int(dim), int(n), int(seed))
            by_cell[cell][method] = (max(EPS, float(v)), max(EPS, float(g)))

    rows = []
    for (func, dim, n, seed), m in by_cell.items():
        if DML not in m or VANILLA not in m:
            continue
        v_dml, g_dml = m[DML]
        v_van, g_van = m[VANILLA]
        rows.append({
            "func": func, "dim": dim, "n": n, "seed": seed,
            "log_ratio_value": float(np.log10(v_dml / v_van)),
            "log_ratio_grad": float(np.log10(g_dml / g_van)),
        })
    return rows


def median_with_drop(rows, drop_index=None, key="log_ratio_value"):
    vals = [r[key] for i, r in enumerate(rows) if i != drop_index]
    return float(np.median(vals)) if vals else float("nan")


def main():
    rows = load_paired_log_ratios()
    print(f"loaded {len(rows)} paired (cell, seed) rows at σ=0")
    if not rows:
        return

    # Index rows by max-impact: largest |log_ratio|
    full_v = median_with_drop(rows, key="log_ratio_value")
    full_g = median_with_drop(rows, key="log_ratio_grad")
    print(f"\nFull corpus medians: log10 ratio value={full_v:+.3f} grad={full_g:+.3f}")

    abs_v = sorted(enumerate(rows), key=lambda kv: -abs(kv[1]["log_ratio_value"]))
    abs_g = sorted(enumerate(rows), key=lambda kv: -abs(kv[1]["log_ratio_grad"]))

    out = {"full": {"n": len(rows),
                    "median_log_ratio_value": full_v,
                    "median_log_ratio_grad": full_g}}

    out["top_value_outliers"] = []
    print("\nTop-5 value-MSE outlier cells (largest |log10 ratio|):")
    for i, r in abs_v[:5]:
        new_v = median_with_drop(rows, drop_index=i, key="log_ratio_value")
        delta = new_v - full_v
        out["top_value_outliers"].append({
            **r, "log_ratio_value": r["log_ratio_value"],
            "new_median_after_drop": new_v, "delta_R1": delta,
        })
        print(f"  ({r['func']}, d={r['dim']}, n={r['n']}, s={r['seed']}): "
              f"log_ratio={r['log_ratio_value']:+.3f} → "
              f"drop → R1 shifts by {delta:+.4f}")

    out["top_grad_outliers"] = []
    print("\nTop-5 grad-MSE outlier cells:")
    for i, r in abs_g[:5]:
        new_g = median_with_drop(rows, drop_index=i, key="log_ratio_grad")
        delta = new_g - full_g
        out["top_grad_outliers"].append({
            **r, "log_ratio_grad": r["log_ratio_grad"],
            "new_median_after_drop": new_g, "delta_R1": delta,
        })
        print(f"  ({r['func']}, d={r['dim']}, n={r['n']}, s={r['seed']}): "
              f"log_ratio={r['log_ratio_grad']:+.3f} → "
              f"drop → R1 shifts by {delta:+.4f}")

    # Maximum shift seen by any single-cell drop
    max_shift_v = max(abs(o["delta_R1"]) for o in out["top_value_outliers"])
    max_shift_g = max(abs(o["delta_R1"]) for o in out["top_grad_outliers"])
    out["max_single_cell_shift_R1_value"] = max_shift_v
    out["max_single_cell_shift_R1_grad"] = max_shift_g
    print(f"\nMax |ΔR1| from any single-cell drop:")
    print(f"  value: {max_shift_v:.4f}")
    print(f"  grad : {max_shift_g:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    with open(OUT_MD, "w") as f:
        f.write("# Single-cell drop sensitivity\n\n")
        f.write(f"Full corpus: n={out['full']['n']}, "
                f"R1_value = {full_v:+.3f}, R1_grad = {full_g:+.3f}.\n\n")
        f.write(f"Maximum shift in R1 from dropping any single cell:\n")
        f.write(f"- value: {max_shift_v:.4f}\n")
        f.write(f"- grad : {max_shift_g:.4f}\n\n")
        f.write("## Top-5 outlier cells (by |log_ratio_value|)\n\n")
        f.write("| func | d | n | seed | log_ratio | ΔR1 if dropped |\n|---|---:|---:|---:|---:|---:|\n")
        for o in out["top_value_outliers"]:
            f.write(f"| {o['func']} | {o['dim']} | {o['n']} | {o['seed']} "
                    f"| {o['log_ratio_value']:+.3f} | {o['delta_R1']:+.4f} |\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
