#!/usr/bin/env python3
"""
Leave-one-family-out (LOFO) robustness check on the synthetic R1 statistic.

For each of the six function families, recompute the per-method R1
(median paired log10(MSE_DML/MSE_vanilla)) on the synthetic σ=0 corpus
with that family removed. Verify whether the dml_fixed-vs-vanilla
ranking flips for any leave-one-out variant.

Output (internal, not paper-cited per user instruction):
  papers/neurips_DB/evidence/lofo_synth.json
  papers/neurips_DB/evidence/lofo_synth.md
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
OUT_JSON = ROOT / "papers/neurips_DB/evidence/lofo_synth.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/lofo_synth.md"

DML = "dml_fixed"
VANILLA = "vanilla"
EPS = 1e-12


def load_synthetic_pairs():
    """Load (func, dim, n, sigma, seed) -> {method: (v_mse, g_mse)} pairs."""
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
            cell = (func, int(dim), int(n), float(sigma), int(seed))
            by_cell[cell][method] = (max(EPS, float(v)), max(EPS, float(g)))
    return by_cell


def compute_r1(by_cell, exclude_func=None, sigma_filter=0.0):
    """Median paired log10 ratio across cells, optionally excluding one family."""
    log_v, log_g = [], []
    for (func, dim, n, sigma, seed), m in by_cell.items():
        if sigma_filter is not None and sigma != sigma_filter:
            continue
        if exclude_func is not None and func == exclude_func:
            continue
        if DML not in m or VANILLA not in m:
            continue
        v_dml, g_dml = m[DML]
        v_van, g_van = m[VANILLA]
        log_v.append(np.log10(v_dml / v_van))
        log_g.append(np.log10(g_dml / g_van))
    return {
        "n": len(log_v),
        "median_log_ratio_value": float(np.median(log_v)) if log_v else float("nan"),
        "median_log_ratio_grad": float(np.median(log_g)) if log_g else float("nan"),
    }


def main():
    by_cell = load_synthetic_pairs()
    funcs = sorted({k[0] for k in by_cell})
    print(f"Loaded {len(by_cell)} cells; families = {funcs}")

    out = {}
    out["full"] = compute_r1(by_cell, exclude_func=None)
    print(f"  full: n={out['full']['n']} R1_value={out['full']['median_log_ratio_value']:+.3f} "
          f"R1_grad={out['full']['median_log_ratio_grad']:+.3f}")

    for f in funcs:
        out[f"drop_{f}"] = compute_r1(by_cell, exclude_func=f)
        r = out[f"drop_{f}"]
        print(f"  drop {f}: n={r['n']} R1_value={r['median_log_ratio_value']:+.3f} "
              f"R1_grad={r['median_log_ratio_grad']:+.3f}")

    # Compare sign of R1 across leave-one-out variants
    full_v = out["full"]["median_log_ratio_value"]
    full_g = out["full"]["median_log_ratio_grad"]
    flips_v = [f for f in funcs
               if np.sign(out[f"drop_{f}"]["median_log_ratio_value"]) != np.sign(full_v)]
    flips_g = [f for f in funcs
               if np.sign(out[f"drop_{f}"]["median_log_ratio_grad"]) != np.sign(full_g)]
    print(f"\nLeave-one-out flips: value={flips_v} grad={flips_g}")
    out["sign_flips_value"] = flips_v
    out["sign_flips_grad"] = flips_g

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    with open(OUT_MD, "w") as f:
        f.write("# LOFO synthetic robustness — leave-one-function-family-out\n\n")
        f.write("Median paired log10(MSE_DML / MSE_vanilla) at σ=0 across synthetic cells, "
                "with each function family removed in turn.\n\n")
        f.write("| Variant | n | R1 (value) | R1 (grad) |\n|---|---:|---:|---:|\n")
        f.write(f"| full corpus | {out['full']['n']} "
                f"| {out['full']['median_log_ratio_value']:+.3f} "
                f"| {out['full']['median_log_ratio_grad']:+.3f} |\n")
        for func in funcs:
            r = out[f"drop_{func}"]
            f.write(f"| drop {func} | {r['n']} "
                    f"| {r['median_log_ratio_value']:+.3f} "
                    f"| {r['median_log_ratio_grad']:+.3f} |\n")
        f.write(f"\nSign flips relative to full corpus: value = {flips_v or 'none'}; "
                f"grad = {flips_g or 'none'}.\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
