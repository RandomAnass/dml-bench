#!/usr/bin/env python3
"""
R1 paired log-MSE-ratio aggregator for the PDE pillar (Burgers + Darcy,
both `bare` and `ic` regimes).

For each (dataset, regime, method, seed) we compute:
    r = log10(MSE_method / MSE_vanilla)
on test_value_mse and test_grad_mse, then aggregate by:
  * per (dataset, regime) -> reports per-cell median + CI
  * per regime ('bare', 'ic'), pooled across (dataset)
    cluster bootstrap on (dataset, regime) cells.

Reference (canonical R1 definition): see
    papers/neurips_DB/research_notes/ratio_definitions.md

Output:
    papers/neurips_DB/evidence/median_log_ratio_pde.json
    papers/neurips_DB/evidence/median_log_ratio_pde.md

Run: python papers/neurips_DB/evidence/median_log_ratio_pde.py
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

PDE_DIRS = {
    ("burgers", "bare"): ROOT / "results" / "burgers" / "bare",
    ("burgers", "ic"):   ROOT / "results" / "burgers" / "ic",
    ("darcy", "bare"):   ROOT / "results" / "darcy" / "bare",
    ("darcy", "ic"):     ROOT / "results" / "darcy" / "ic",
}
OUT_JSON = ROOT / "papers" / "neurips_DB" / "evidence" / "median_log_ratio_pde.json"
OUT_MD = ROOT / "papers" / "neurips_DB" / "evidence" / "median_log_ratio_pde.md"

REFERENCE = "vanilla"
METHODS = [
    "dml_fixed",
    "dml_fixed_half",
    "dml_gradnorm",
    "dml_relobralo",
    "dml_warmup",
]
N_BOOT = 1000
RNG = np.random.default_rng(42)
EPS = 1e-12


def _load_dir(src: Path):
    """Return {(seed): {method: (val_mse, grad_mse)}} restricted to
    the canonical 4x256 architecture and the directory's regime."""
    by_cell = defaultdict(dict)
    if not src.exists():
        return by_cell, 0, 0
    n_seen = n_skipped = 0
    for p in src.glob("*.json"):
        if p.name.startswith("_"):
            continue
        n_seen += 1
        r = load_result_json(p)
        if r is None:
            n_skipped += 1
            continue
        method = r.get("method")
        seed = r.get("seed")
        v = r.get("test_value_mse")
        g = r.get("test_grad_mse")
        arch = r.get("arch", "4x256")
        if arch != "4x256":
            continue  # restrict to the canonical architecture used in §5.5
        if None in (method, seed, v, g):
            continue
        v = max(EPS, float(v))
        g = max(EPS, float(g))
        by_cell[int(seed)][str(method)] = (v, g)
    return by_cell, n_seen, n_skipped


def _build_rows(by_cell, methods, reference, dataset, regime):
    rows = []
    for seed, m in by_cell.items():
        if reference not in m:
            continue
        v_ref, g_ref = m[reference]
        for method in methods:
            if method not in m:
                continue
            v, g = m[method]
            rows.append({
                "method": method,
                "dataset": dataset,
                "regime": regime,
                "cluster": (dataset, regime),
                "seed": seed,
                "log_ratio_value": float(np.log10(v / v_ref)),
                "log_ratio_grad":  float(np.log10(g / g_ref)),
            })
    return rows


def cluster_bootstrap_median(rows, key, n_boot=N_BOOT, rng=None):
    rng = rng or RNG
    by_cluster = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster"]].append(float(r[key]))
    clusters = list(by_cluster.keys())
    if not clusters:
        return float("nan"), float("nan"), float("nan"), 0, 0
    full = []
    for c in clusters:
        full.extend(by_cluster[c])
    point = float(np.median(full))
    n_clu = len(clusters)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_clu, size=n_clu)
        sample = []
        for j in idx:
            sample.extend(by_cluster[clusters[j]])
        boots[b] = np.median(sample)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi, n_clu, len(full)


def percentile_bootstrap_median(values, n_boot=N_BOOT, rng=None):
    rng = rng or RNG
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.median(arr))
    boots = np.empty(n_boot)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.median(arr[idx])
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi, n


def _fmt_factor(log_ratio: float) -> str:
    if not np.isfinite(log_ratio):
        return "n/a"
    r = 10 ** log_ratio
    if log_ratio < 0:
        return f"{r:.3g}× ({1/r:.0f}× reduction vs vanilla)"
    return f"{r:.3g}× ({(r-1)*100:+.1f}% vs vanilla)"


def main():
    out = {"meta": {
        "n_boot": N_BOOT, "rng_seed": 42, "eps_floor": EPS,
        "reference_method": REFERENCE,
        "cluster_unit": "(dataset, regime)",
    }}

    # Per (dataset, regime) blocks
    all_rows = []
    out["cells"] = {}
    for (ds, regime), src in PDE_DIRS.items():
        by_cell, n_seen, n_skipped = _load_dir(src)
        rows = _build_rows(by_cell, METHODS, REFERENCE, ds, regime)
        all_rows.extend(rows)
        cell_summary = {
            "meta": {"n_seen": n_seen, "n_skipped": n_skipped, "n_pairs": len(rows)},
            "per_method": {},
        }
        if not rows:
            cell_summary["error"] = f"no paired rows in {src}"
            out["cells"][f"{ds}_{regime}"] = cell_summary
            continue
        for method in METHODS:
            mrows = [r for r in rows if r["method"] == method]
            if not mrows:
                continue
            d = {}
            for metric in ("log_ratio_value", "log_ratio_grad"):
                vals = [float(r[metric]) for r in mrows]
                point, lo, hi, n_pairs = percentile_bootstrap_median(vals)
                d[metric] = {
                    "median_log10": point,
                    "factor": float(10 ** point),
                    "factor_human": _fmt_factor(point),
                    "percentile_ci_95": [lo, hi],
                    "n_pairs": n_pairs,
                }
            cell_summary["per_method"][method] = d
        out["cells"][f"{ds}_{regime}"] = cell_summary

    # Pooled across (dataset) per regime, cluster bootstrap on (dataset, regime)
    out["pooled"] = {}
    for regime in ("bare", "ic"):
        rows = [r for r in all_rows if r["regime"] == regime]
        block = {"per_method": {}}
        if not rows:
            block["error"] = f"no rows for regime {regime}"
            out["pooled"][regime] = block
            continue
        for method in METHODS:
            mrows = [r for r in rows if r["method"] == method]
            if not mrows:
                continue
            d = {}
            for metric in ("log_ratio_value", "log_ratio_grad"):
                point, lo, hi, n_clu, n_pairs = cluster_bootstrap_median(mrows, metric)
                d[metric] = {
                    "median_log10": point,
                    "factor": float(10 ** point),
                    "factor_human": _fmt_factor(point),
                    "cluster_ci_95": [lo, hi],
                    "n_clusters": n_clu,
                    "n_pairs": n_pairs,
                }
            block["per_method"][method] = d
        out["pooled"][regime] = block

    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Markdown
    md = ["# PDE paired log10(MSE_method / MSE_vanilla) — R1",
          "",
          f"Cluster = (dataset, regime) for pooled rows; per-cell uses",
          f"percentile bootstrap (no clustering needed, single cell).",
          f"Bootstrap = {N_BOOT}. Negative log-ratio = method beats vanilla.",
          ""]
    for cell_name, cell in out["cells"].items():
        md += [f"## {cell_name}", ""]
        m = cell["meta"]
        md += [f"_n_seen={m['n_seen']}, n_skipped={m['n_skipped']}, n_pairs={m['n_pairs']}_", ""]
        if "per_method" not in cell or not cell.get("per_method"):
            md += [f"_{cell.get('error', 'no per-method data')}_", ""]
            continue
        md += ["| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n pairs |",
               "|---|---:|---:|---:|---:|---:|"]
        for method, d in cell["per_method"].items():
            v = d["log_ratio_value"]
            g = d["log_ratio_grad"]
            md += [f"| {method} | "
                   f"{v['median_log10']:+.3f} [{v['percentile_ci_95'][0]:+.2f}, {v['percentile_ci_95'][1]:+.2f}] | "
                   f"{v['factor']:.3g} | "
                   f"{g['median_log10']:+.3f} [{g['percentile_ci_95'][0]:+.2f}, {g['percentile_ci_95'][1]:+.2f}] | "
                   f"{g['factor']:.3g} | "
                   f"{v['n_pairs']} |"]
        md.append("")
    md += ["## Pooled by regime (cluster bootstrap on (dataset, regime))", ""]
    for regime in ("bare", "ic"):
        block = out["pooled"].get(regime, {})
        md += [f"### regime = {regime}", ""]
        if "per_method" not in block:
            md += [f"_{block.get('error', 'no data')}_", ""]
            continue
        md += ["| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n_clusters | n_pairs |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for method, d in block["per_method"].items():
            v = d["log_ratio_value"]
            g = d["log_ratio_grad"]
            md += [f"| {method} | "
                   f"{v['median_log10']:+.3f} [{v['cluster_ci_95'][0]:+.2f}, {v['cluster_ci_95'][1]:+.2f}] | "
                   f"{v['factor']:.3g} | "
                   f"{g['median_log10']:+.3f} [{g['cluster_ci_95'][0]:+.2f}, {g['cluster_ci_95'][1]:+.2f}] | "
                   f"{g['factor']:.3g} | "
                   f"{v['n_clusters']} | {v['n_pairs']} |"]
        md.append("")

    OUT_MD.write_text("\n".join(md) + "\n")

    print(f"wrote {OUT_JSON}, {OUT_MD}")
    # Console summary
    for cn, cell in out["cells"].items():
        if "per_method" in cell and "dml_fixed" in cell["per_method"]:
            d = cell["per_method"]["dml_fixed"]
            print(f"  {cn} / dml_fixed: "
                  f"value {d['log_ratio_value']['factor_human']}, "
                  f"grad {d['log_ratio_grad']['factor_human']}")


if __name__ == "__main__":
    main()
