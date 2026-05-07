#!/usr/bin/env python3
"""Aggregate polynomial-baseline JSONs vs. existing best-DML results.

For each domain (poly_trig d=1, trig d=1, black_scholes d=1, SPY 4-D),
compute mean ± std across seeds for:
  - best-degree polynomial: (price MSE, grad MSE, n_params at the
    minimum-price-MSE degree)
  - best DML method: scan tier3_benchmark / spy_options_temporal
    (test_value_mse, test_grad_mse) across all dml_* methods at the same
    (n_samples, seed) and pick the method with the smallest MEAN-ACROSS-
    SEEDS test_value_mse.

Print a summary table and a JSON-serialised comparison for the
deliverable.

Usage:
    python experiments/polynomial_baselines/analyze.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

POLY_DIR = Path("results/polynomial_baselines")
TIER3_DIR = Path("results/tier3_benchmark")
SPY_DIR = Path("results/spy_options_temporal")


def load_polynomial_synthetic():
    """Load polynomial results for synthetic d=1 cells.

    Returns: {(domain, n): {seed: {degree: {price_mse, grad_mse, ...}}}}
    """
    out = defaultdict(lambda: defaultdict(dict))
    for f in POLY_DIR.glob("*_d1_*_polynomial.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        domain = d["domain"]
        n = int(d["n_samples"])
        seed = int(d["seed"])
        # convert the str-keyed dict back to int-keyed
        per_deg = {int(k): v for k, v in d["polynomial_results"].items()}
        out[(domain, n)][seed] = per_deg
    return out


def load_polynomial_spy():
    """Load polynomial results for SPY 4-D cells.

    Returns: {n_train: {seed: {degree: {price_mse, grad_mse, ...}}}}
    """
    out = defaultdict(dict)
    for f in POLY_DIR.glob("spy_n*_polynomial.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n = int(d["n_train"])
        seed = int(d["seed"])
        per_deg = {int(k): v for k, v in d["polynomial_results"].items()}
        out[n][seed] = per_deg
    return out


def load_dml_synthetic_tier3(domain: str, n: int) -> dict:
    """Load all DML results for synthetic d=1 cells from tier3_benchmark.

    Returns: {method: {seed: {test_value_mse, test_grad_mse}}}
    """
    out = defaultdict(dict)
    pattern = f"{domain}_d1_n{n}_noise0.0_s*_*.json"
    for f in TIER3_DIR.glob(pattern):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        method = d.get("method", "")
        # focus on DML / vanilla — exclude classical baselines (gp/krr/rf)
        if method.startswith("baseline_") or method == "":
            continue
        seed = int(d["seed"])
        out[method][seed] = {
            "test_value_mse": d["test_value_mse"],
            "test_grad_mse": d["test_grad_mse"],
        }
    return out


def load_dml_spy(n_train: int) -> dict:
    """Load all DML results for SPY temporal split with the given n_train.

    Returns: {method: {seed: {test_value_mse, test_grad_mse}}}
    """
    out = defaultdict(dict)
    pattern = f"spy_n{n_train}_s*_*.json"
    for f in SPY_DIR.glob(pattern):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        method = d.get("method", "")
        if method == "" or method.startswith("baseline_"):
            continue
        seed = int(d["seed"])
        out[method][seed] = {
            "test_value_mse": d["test_value_mse"],
            "test_grad_mse": d["test_grad_mse"],
        }
    return out


def best_dml_method(per_method: dict, metric: str = "test_value_mse"):
    """Return (method_name, mean, std, seeds_list) for the method with the
    smallest mean-across-seeds metric. Skips methods with <2 seeds."""
    candidates = []
    for method, seed_dict in per_method.items():
        vals = [v[metric] for v in seed_dict.values() if metric in v]
        if len(vals) >= 2:
            candidates.append((np.mean(vals), method, np.std(vals), sorted(seed_dict.keys()), vals))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    mean, method, std, seeds, vals = candidates[0]
    return {
        "method": method,
        "mean": mean,
        "std": std,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "values": vals,
    }


def best_polynomial_summary(seed_to_per_deg: dict, metric: str = "price_mse"):
    """For each seed, choose the degree that minimises the metric, then
    aggregate across seeds.  Mirrors the Heston polynomial-baseline
    convention (\"best-degree polynomial\")."""
    if not seed_to_per_deg:
        return None
    per_seed_best = {}
    for seed, per_deg in seed_to_per_deg.items():
        # filter degrees with valid (no error) results
        valid = {deg: r for deg, r in per_deg.items() if "error" not in r}
        if not valid:
            continue
        best_deg = min(valid, key=lambda d: valid[d][metric])
        per_seed_best[seed] = (best_deg, valid[best_deg])
    if not per_seed_best:
        return None
    price_vals = [v[1]["price_mse"] for v in per_seed_best.values()]
    grad_vals  = [v[1]["grad_mse"]  for v in per_seed_best.values()]
    n_params_vals = [v[1]["n_params"] for v in per_seed_best.values()]
    best_degs = [v[0] for v in per_seed_best.values()]
    return {
        "n_seeds": len(per_seed_best),
        "seeds": sorted(per_seed_best),
        "best_degree_per_seed": {s: per_seed_best[s][0] for s in sorted(per_seed_best)},
        "best_degree_modal": int(max(set(best_degs), key=best_degs.count)),
        "n_params_modal": int(np.median(n_params_vals)),
        "price_mse_mean": float(np.mean(price_vals)),
        "price_mse_std": float(np.std(price_vals)),
        "grad_mse_mean": float(np.mean(grad_vals)),
        "grad_mse_std": float(np.std(grad_vals)),
    }


def fixed_degree_summary(seed_to_per_deg: dict, degree: int):
    """Aggregate across seeds at a fixed degree."""
    rows = []
    for seed, per_deg in seed_to_per_deg.items():
        if degree not in per_deg:
            continue
        r = per_deg[degree]
        if "error" in r:
            continue
        rows.append(r)
    if not rows:
        return None
    return {
        "degree": degree,
        "n_seeds": len(rows),
        "n_params": rows[0]["n_params"],
        "price_mse_mean": float(np.mean([r["price_mse"] for r in rows])),
        "price_mse_std": float(np.std([r["price_mse"] for r in rows])),
        "grad_mse_mean": float(np.mean([r["grad_mse"] for r in rows])),
        "grad_mse_std": float(np.std([r["grad_mse"] for r in rows])),
    }


def fmt_pm(mean, std):
    """Format mean ± std in scientific notation."""
    if mean == 0 or not np.isfinite(mean):
        return f"{mean:.3e}"
    return f"{mean:.3e} ± {std:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", nargs="+", type=int, default=[1024],
                    help="Synthetic n_samples to summarise (default: 1024).")
    ap.add_argument("--spy-n", type=int, default=10000)
    ap.add_argument("--out-json", default=None,
                    help="Optional path to dump the comparison JSON.")
    args = ap.parse_args()

    poly_synth = load_polynomial_synthetic()
    poly_spy = load_polynomial_spy()

    summary = {}

    # --- Synthetic d=1 ---
    for domain in ["poly_trig", "trig", "black_scholes"]:
        for n in args.n_samples:
            key = f"{domain}_d1_n{n}"
            seed_results = poly_synth.get((domain, n), {})
            if not seed_results:
                summary[key] = {"status": "no polynomial results found"}
                continue
            poly_summary = best_polynomial_summary(seed_results, metric="price_mse")
            poly_grad_summary = best_polynomial_summary(seed_results, metric="grad_mse")
            dml = load_dml_synthetic_tier3(domain, n)
            best_dml_v = best_dml_method(dml, "test_value_mse")
            best_dml_g = best_dml_method(dml, "test_grad_mse")

            summary[key] = {
                "domain": domain,
                "dim": 1,
                "n_samples": n,
                "polynomial_best_price": poly_summary,
                "polynomial_best_grad": poly_grad_summary,
                # also report fixed-degree-5 for reproducibility
                "polynomial_quintic": fixed_degree_summary(seed_results, 5),
                "best_dml_value_mse": best_dml_v,
                "best_dml_grad_mse": best_dml_g,
            }

            print(f"\n=== {key} ===")
            if poly_summary:
                print(f"  Best-degree polynomial (per-seed): "
                      f"deg-modal={poly_summary['best_degree_modal']} "
                      f"n_params={poly_summary['n_params_modal']}")
                print(f"    price MSE: {fmt_pm(poly_summary['price_mse_mean'], poly_summary['price_mse_std'])}")
                print(f"    grad  MSE: {fmt_pm(poly_summary['grad_mse_mean'], poly_summary['grad_mse_std'])} "
                      f" [grad chosen at price-min degree]")
                if poly_grad_summary:
                    print(f"  Best-degree-on-grad polynomial: "
                          f"deg-modal={poly_grad_summary['best_degree_modal']} "
                          f"grad MSE: {fmt_pm(poly_grad_summary['grad_mse_mean'], poly_grad_summary['grad_mse_std'])}")
            if best_dml_v:
                print(f"  Best DML on price: {best_dml_v['method']} "
                      f"({best_dml_v['n_seeds']} seeds): "
                      f"{fmt_pm(best_dml_v['mean'], best_dml_v['std'])}")
            if best_dml_g:
                print(f"  Best DML on grad : {best_dml_g['method']} "
                      f"({best_dml_g['n_seeds']} seeds): "
                      f"{fmt_pm(best_dml_g['mean'], best_dml_g['std'])}")
            if poly_summary and best_dml_v:
                ratio_v = best_dml_v["mean"] / poly_summary["price_mse_mean"]
                print(f"  Ratio (best-DML / poly) on price: {ratio_v:.4g}")
            if poly_grad_summary and best_dml_g:
                ratio_g = best_dml_g["mean"] / poly_grad_summary["grad_mse_mean"]
                print(f"  Ratio (best-DML / poly) on grad : {ratio_g:.4g}")

    # --- SPY 4-D ---
    n = args.spy_n
    key = f"spy_n{n}"
    seed_results = poly_spy.get(n, {})
    if not seed_results:
        summary[key] = {"status": "no polynomial results found"}
    else:
        poly_summary = best_polynomial_summary(seed_results, metric="price_mse")
        poly_grad_summary = best_polynomial_summary(seed_results, metric="grad_mse")
        dml = load_dml_spy(n)
        best_dml_v = best_dml_method(dml, "test_value_mse")
        best_dml_g = best_dml_method(dml, "test_grad_mse")
        summary[key] = {
            "domain": "spy_bs_target",
            "dim": 4,
            "n_train": n,
            "polynomial_best_price": poly_summary,
            "polynomial_best_grad": poly_grad_summary,
            "polynomial_quintic": fixed_degree_summary(seed_results, 5),
            "best_dml_value_mse": best_dml_v,
            "best_dml_grad_mse": best_dml_g,
        }

        print(f"\n=== {key}  (SPY 4-D temporal split, BS target) ===")
        if poly_summary:
            print(f"  Best-degree polynomial: deg-modal={poly_summary['best_degree_modal']}  "
                  f"n_params={poly_summary['n_params_modal']}")
            print(f"    price MSE: {fmt_pm(poly_summary['price_mse_mean'], poly_summary['price_mse_std'])}")
            print(f"    grad  MSE: {fmt_pm(poly_summary['grad_mse_mean'], poly_summary['grad_mse_std'])} "
                  f" [grad chosen at price-min degree]")
            if poly_grad_summary:
                print(f"  Best-degree-on-grad polynomial: deg-modal={poly_grad_summary['best_degree_modal']}  "
                      f"grad MSE: {fmt_pm(poly_grad_summary['grad_mse_mean'], poly_grad_summary['grad_mse_std'])}")
        if best_dml_v:
            print(f"  Best DML on price: {best_dml_v['method']} "
                  f"({best_dml_v['n_seeds']} seeds): "
                  f"{fmt_pm(best_dml_v['mean'], best_dml_v['std'])}")
        if best_dml_g:
            print(f"  Best DML on grad : {best_dml_g['method']} "
                  f"({best_dml_g['n_seeds']} seeds): "
                  f"{fmt_pm(best_dml_g['mean'], best_dml_g['std'])}")
        if poly_summary and best_dml_v:
            ratio_v = best_dml_v["mean"] / poly_summary["price_mse_mean"]
            print(f"  Ratio (best-DML / poly) on price: {ratio_v:.4g}")
        if poly_grad_summary and best_dml_g:
            ratio_g = best_dml_g["mean"] / poly_grad_summary["grad_mse_mean"]
            print(f"  Ratio (best-DML / poly) on grad : {ratio_g:.4g}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote comparison summary → {args.out_json}")


if __name__ == "__main__":
    main()
