#!/usr/bin/env python3
"""
Paired comparison: dml_softmax_balance (EMA-based ablation of Bischof &
Kraus 2022 Eq. 11) vs dml_relobralo (canonical Eq. 11 implementation)
on the synthetic and rMD17-MLP/GATv2 corpora where both methods ran.

Per-cell paired log-ratio. Output:
  papers/neurips_DB/evidence/softmax_balance_comparison.json
  papers/neurips_DB/evidence/softmax_balance_comparison.md
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

OUT_JSON = ROOT / "papers/neurips_DB/evidence/softmax_balance_comparison.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/softmax_balance_comparison.md"

EPS = 1e-12


def load_paired_synthetic():
    """Load tier1+3 cells where both relobralo and softmax_balance ran."""
    by_cell = defaultdict(dict)
    for tdir in [ROOT / f"results/tier{i}_benchmark" for i in (1, 2, 3, 4)]:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            m = r.get("method")
            if m not in ("dml_relobralo", "dml_softmax_balance"):
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
            by_cell[cell][m] = (max(EPS, float(v)), max(EPS, float(g)))
    pairs = {c: m for c, m in by_cell.items()
             if "dml_relobralo" in m and "dml_softmax_balance" in m}
    return pairs


def load_paired_molecular(arch: str):
    """Load molecular_{mlp, gatv2} cells where both methods ran."""
    by_cell = defaultdict(dict)
    d = ROOT / f"results/molecular_{arch}"
    if not d.exists():
        return {}
    for p in d.glob("*.json"):
        r = load_result_json(p)
        if r is None:
            continue
        m = r.get("method")
        if m not in ("dml_relobralo", "dml_softmax_balance"):
            continue
        mol = r.get("molecule")
        split = r.get("split_id") or r.get("split")
        v = r.get("test_value_mse")
        # use cartesian-force MSE for MLP (gauge-invariant); native for GATv2
        g = r.get("test_grad_mse_cartesian") or r.get("test_grad_mse")
        if None in (mol, split, v, g):
            continue
        cell = (mol, int(split))
        by_cell[cell][m] = (max(EPS, float(v)), max(EPS, float(g)))
    pairs = {c: m for c, m in by_cell.items()
             if "dml_relobralo" in m and "dml_softmax_balance" in m}
    return pairs


def paired_summary(pairs: dict, label: str) -> dict:
    if not pairs:
        return {"label": label, "n": 0}
    log_v, log_g = [], []
    for c, m in pairs.items():
        v_relo, g_relo = m["dml_relobralo"]
        v_smb, g_smb = m["dml_softmax_balance"]
        log_v.append(np.log10(v_smb / v_relo))
        log_g.append(np.log10(g_smb / g_relo))
    return {
        "label": label,
        "n": len(pairs),
        "median_log_ratio_value_smb_vs_relobralo": float(np.median(log_v)),
        "median_log_ratio_grad_smb_vs_relobralo": float(np.median(log_g)),
        "mean_log_ratio_value_smb_vs_relobralo": float(np.mean(log_v)),
        "mean_log_ratio_grad_smb_vs_relobralo": float(np.mean(log_g)),
        "n_value_smb_better": int(np.sum(np.array(log_v) < 0)),
        "n_grad_smb_better": int(np.sum(np.array(log_g) < 0)),
    }


def main():
    syn = load_paired_synthetic()
    mlp = load_paired_molecular("mlp")
    gat = load_paired_molecular("gatv2")
    print(f"loaded paired cells: synth={len(syn)}, MLP={len(mlp)}, GATv2={len(gat)}")

    out = {
        "synth": paired_summary(syn, "synth"),
        "mlp":   paired_summary(mlp, "rMD17 MLP-pairwise"),
        "gatv2": paired_summary(gat, "rMD17 GATv2"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_JSON}")

    with open(OUT_MD, "w") as f:
        f.write("# `dml_softmax_balance` vs `dml_relobralo` paired comparison\n\n")
        f.write("Median paired $\\log_{10}(\\mathrm{MSE}_{\\mathrm{softmax\\_balance}} / "
                "\\mathrm{MSE}_{\\mathrm{relobralo}})$ across paired cells where both "
                "methods ran. Negative means softmax_balance wins.\n\n")
        f.write("| corpus | n | log-ratio (value) | log-ratio (grad) | n value-wins | n grad-wins |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for k in ("synth", "mlp", "gatv2"):
            s = out[k]
            if s["n"] == 0:
                continue
            f.write(f"| {s['label']} | {s['n']} "
                    f"| {s['median_log_ratio_value_smb_vs_relobralo']:+.3f} "
                    f"| {s['median_log_ratio_grad_smb_vs_relobralo']:+.3f} "
                    f"| {s['n_value_smb_better']}/{s['n']} "
                    f"| {s['n_grad_smb_better']}/{s['n']} |\n")
    print(f"wrote {OUT_MD}")
    print()
    for k in ("synth", "mlp", "gatv2"):
        s = out[k]
        if s["n"] == 0:
            continue
        print(f"  {s['label']:25s} n={s['n']:4d}  "
              f"median log10(smb/relo) value={s['median_log_ratio_value_smb_vs_relobralo']:+.3f} "
              f"grad={s['median_log_ratio_grad_smb_vs_relobralo']:+.3f}  "
              f"smb wins value: {s['n_value_smb_better']}/{s['n']}, "
              f"grad: {s['n_grad_smb_better']}/{s['n']}")


if __name__ == "__main__":
    main()
