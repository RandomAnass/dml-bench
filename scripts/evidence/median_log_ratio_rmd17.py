#!/usr/bin/env python3
"""
R1 paired log-MSE-ratio aggregator for the rMD17 PaiNN block.

For each (molecule, method) pair we compute, per (molecule, split):
    r = log10(MSE_method / MSE_native_EF)
on test_value_mse and test_grad_mse independently. We then aggregate
the medians and 95% percentile cluster bootstrap CI by molecule
(cluster = molecule, all splits within a molecule resampled together).

Reference (canonical R1 definition): see
    papers/neurips_DB/research_notes/ratio_definitions.md

Output:
    papers/neurips_DB/evidence/median_log_ratio_rmd17.json
    papers/neurips_DB/evidence/median_log_ratio_rmd17.md

Run: python papers/neurips_DB/evidence/median_log_ratio_rmd17.py
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

PAINN_DIR = ROOT / "results" / "molecular_painn"
MLP_DIR = ROOT / "results" / "molecular_mlp"
GATV2_DIR = ROOT / "results" / "molecular_gatv2"
OUT_JSON = ROOT / "papers" / "neurips_DB" / "evidence" / "median_log_ratio_rmd17.json"
OUT_MD = ROOT / "papers" / "neurips_DB" / "evidence" / "median_log_ratio_rmd17.md"

REFERENCE = "native_EF"
METHODS = [
    "vanilla",
    "dml_fixed",
    "dml_fixed_half",
    "dml_gradnorm",
    "dml_warmup",
]
N_BOOT = 1000
RNG = np.random.default_rng(42)
EPS = 1e-12


def _load_dir(src: Path):
    """Return {(mol, split): {method: (val_mse, grad_mse)}}."""
    by_cell = defaultdict(dict)
    if not src.exists():
        return by_cell, 0, 0
    n_seen = n_skipped = 0
    for p in src.glob("*.json"):
        n_seen += 1
        r = load_result_json(p)
        if r is None:
            n_skipped += 1
            continue
        mol = r.get("molecule")
        method = r.get("method")
        split = r.get("split_id") or r.get("split")
        v = r.get("test_value_mse")
        g = r.get("test_grad_mse")
        if None in (mol, method, split, v, g):
            continue
        v = max(EPS, float(v))
        g = max(EPS, float(g))
        by_cell[(str(mol), int(split))][str(method)] = (v, g)
    return by_cell, n_seen, n_skipped


def _build_rows(by_cell, methods, reference):
    """For each (mol, split) cell with both reference and method, emit a row."""
    rows = []
    for (mol, split), m in by_cell.items():
        if reference not in m:
            continue
        v_ref, g_ref = m[reference]
        for method in methods:
            if method not in m:
                continue
            v, g = m[method]
            rows.append({
                "method": method,
                "molecule": mol,
                "cluster": mol,
                "split": split,
                "log_ratio_value": float(np.log10(v / v_ref)),
                "log_ratio_grad":  float(np.log10(g / g_ref)),
            })
    return rows


def cluster_bootstrap_median(rows, key, n_boot=N_BOOT, rng=None):
    """Cluster bootstrap on row['cluster'] (molecule)."""
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


def per_molecule_medians(rows, key):
    """Per-molecule medians (no bootstrap, just the descriptive median)."""
    out = {}
    by_mol = defaultdict(list)
    for r in rows:
        by_mol[r["molecule"]].append(float(r[key]))
    for mol, vals in by_mol.items():
        out[mol] = {"median": float(np.median(vals)),
                    "n": int(len(vals))}
    return out


def _fmt_factor(log_ratio: float) -> str:
    if not np.isfinite(log_ratio):
        return "n/a"
    r = 10 ** log_ratio
    if log_ratio < 0:
        return f"{r:.3g}× ({1/r:.0f}× reduction vs native_EF)"
    return f"{r:.3g}× ({(r-1)*100:+.1f}% increase vs native_EF)"


def main():
    out = {"meta": {
        "n_boot": N_BOOT, "rng_seed": 42, "eps_floor": EPS,
        "reference_method": REFERENCE, "cluster_unit": "molecule",
    }}

    for label, src in [("painn", PAINN_DIR), ("mlp", MLP_DIR), ("gatv2", GATV2_DIR)]:
        by_cell, n_seen, n_skipped = _load_dir(src)
        rows = _build_rows(by_cell, METHODS, REFERENCE)
        out[label] = {"meta": {"n_seen": n_seen, "n_skipped": n_skipped,
                               "n_cells_with_ref": int(sum(REFERENCE in m for m in by_cell.values())),
                               "n_pairs": len(rows)}}
        if not rows:
            out[label]["error"] = f"no paired rows under {src}"
            continue
        # Per-method aggregates with cluster bootstrap on molecule
        per_method = {}
        for method in METHODS:
            method_rows = [r for r in rows if r["method"] == method]
            if not method_rows:
                continue
            d = {}
            for metric in ("log_ratio_value", "log_ratio_grad"):
                point, lo, hi, n_clu, n_pairs = cluster_bootstrap_median(method_rows, metric)
                d[metric] = {
                    "median_log10": point,
                    "factor": float(10 ** point),
                    "factor_human": _fmt_factor(point),
                    "cluster_ci_95": [lo, hi],
                    "n_clusters": n_clu,
                    "n_pairs": n_pairs,
                }
            d["per_molecule"] = {
                "log_ratio_value": per_molecule_medians(method_rows, "log_ratio_value"),
                "log_ratio_grad":  per_molecule_medians(method_rows, "log_ratio_grad"),
            }
            per_method[method] = d
        out[label]["per_method"] = per_method

    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Markdown
    md = ["# rMD17 paired log10(MSE_method / MSE_native_EF) — R1",
          "",
          f"Cluster = molecule. Bootstrap = {N_BOOT}. Negative log-ratio means",
          "method beats native_EF; positive means native_EF beats method.",
          ""]
    for label in ("painn", "mlp", "gatv2"):
        block = out.get(label, {})
        md += [f"## {label.upper()}", ""]
        meta = block.get("meta", {})
        md += [f"_n_seen={meta.get('n_seen', '?')}, "
               f"n_skipped={meta.get('n_skipped', '?')}, "
               f"n_pairs={meta.get('n_pairs', '?')}_", ""]
        if "per_method" not in block:
            md += [f"_{block.get('error', 'no data')}_", ""]
            continue
        md += ["| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n clusters | n pairs |",
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
    # Console summary for PaiNN headlines
    if "per_method" in out["painn"]:
        for m in METHODS:
            if m in out["painn"]["per_method"]:
                d = out["painn"]["per_method"][m]
                print(f"  PaiNN {m}: "
                      f"value {d['log_ratio_value']['factor_human']}, "
                      f"grad {d['log_ratio_grad']['factor_human']}")


if __name__ == "__main__":
    main()
