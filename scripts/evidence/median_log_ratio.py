#!/usr/bin/env python3
"""
S1+S5 from the statistical-methodology proposal:
median paired log10(MSE_DML / MSE_vanilla) + config-level cluster
bootstrap 95 % CI, regenerated for the abstract and §5.2 headline.

Synthetic (sigma=0): cluster on (func, dim, n_train, sigma); each
cluster contributes all its seeds intact.
SPY purged-CV: cluster on fold_idx; each cluster contributes 10 seeds.

Outputs:
  papers/neurips_DB/evidence/median_log_ratio.json
  papers/neurips_DB/evidence/median_log_ratio.md
  papers/neurips_DB/evidence/median_log_ratio.tex   (sentences ready for paper)
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
SPY_PURGED = ROOT / "results/spy_options_purged_cv"
SPY_TEMPORAL = ROOT / "results/spy_options_temporal"
OUT_JSON = ROOT / "papers/neurips_DB/evidence/median_log_ratio.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/median_log_ratio.md"
OUT_TEX = ROOT / "papers/neurips_DB/evidence/median_log_ratio.tex"

DML = "dml_fixed"
VANILLA = "vanilla"
N_BOOT = 1000
RNG = np.random.default_rng(42)
EPS = 1e-12  # MSE floor to keep log10 finite

SMOOTH_FAMILIES = {"bachelier", "black_scholes", "poly_trig"}
HIGH_FREQ = {"trig"}
DISCONT = {"step"}
MC_NOISY = {"heston"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_synthetic_pairs(sigma_filter=None):
    """Return list of (cluster_id, ratio_value, ratio_grad, func) tuples."""
    by_cell = defaultdict(dict)
    n_seen = n_skipped = 0
    for tdir in TIER_DIRS:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            n_seen += 1
            r = load_result_json(p)
            if r is None:
                n_skipped += 1
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
            sigma = float(sigma)
            if sigma_filter is not None and sigma != sigma_filter:
                continue
            cell = (func, int(dim), int(n), sigma, int(seed))
            by_cell[cell][method] = (max(EPS, float(v)), max(EPS, float(g)))

    if n_skipped > 0:
        print(f"WARN: {n_skipped}/{n_seen} synthetic JSONs failed to parse "
              f"and were dropped.", file=sys.stderr)

    rows = []
    for (func, dim, n, sigma, seed), m in by_cell.items():
        if DML not in m or VANILLA not in m:
            continue
        v_dml, g_dml = m[DML]
        v_van, g_van = m[VANILLA]
        cluster_id = (func, dim, n, sigma)
        rows.append({
            "cluster": cluster_id, "func": func, "seed": seed,
            "log_ratio_value": float(np.log10(v_dml / v_van)),
            "log_ratio_grad":  float(np.log10(g_dml / g_van)),
            "ratio_value": v_dml / v_van,
            "ratio_grad": g_dml / g_van,
        })
    return rows


def _load_spy_pairs(src_dir: Path):
    """Return list of (cluster_id=fold_idx, ratio_value, ratio_grad)
    over paired (dml_fixed, vanilla) at the same (fold, seed)."""
    by_cell = defaultdict(dict)
    if not src_dir.exists():
        return []
    n_seen = n_skipped = 0
    for p in src_dir.glob("*.json"):
        n_seen += 1
        r = load_result_json(p)
        if r is None:
            n_skipped += 1
            continue
        method = r.get("method")
        if method not in (DML, VANILLA):
            continue
        seed = r.get("seed")
        fold = r.get("fold_idx") if "fold_idx" in r else r.get("fold")
        v = r.get("test_value_mse")
        g = r.get("test_grad_mse")
        if None in (seed, v, g):
            continue
        # Temporal split has no fold; treat as one cluster (fold=0).
        if fold is None:
            fold = 0
        cell = (int(fold), int(seed))
        by_cell[cell][method] = (max(EPS, float(v)), max(EPS, float(g)))
    if n_skipped > 0:
        print(f"WARN: {n_skipped}/{n_seen} SPY JSONs in {src_dir.name} "
              f"failed to parse and were dropped.", file=sys.stderr)
    rows = []
    for (fold, seed), m in by_cell.items():
        if DML not in m or VANILLA not in m:
            continue
        v_dml, g_dml = m[DML]
        v_van, g_van = m[VANILLA]
        rows.append({
            "cluster": fold, "fold": fold, "seed": seed,
            "log_ratio_value": float(np.log10(v_dml / v_van)),
            "log_ratio_grad":  float(np.log10(g_dml / g_van)),
            "ratio_value": v_dml / v_van,
            "ratio_grad": g_dml / g_van,
        })
    return rows


# ---------------------------------------------------------------------------
# Cluster bootstrap on the median
# ---------------------------------------------------------------------------

def cluster_bootstrap_median(rows, key, n_boot=N_BOOT, rng=None):
    """Resample at the cluster level (with replacement); within each
    sampled cluster, take ALL within-cluster rows. Return point estimate
    + percentile 95% CI on the median of `key`."""
    rng = rng or RNG
    by_cluster = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster"]].append(float(r[key]))
    clusters = list(by_cluster.keys())
    if not clusters:
        return float("nan"), float("nan"), float("nan"), 0
    full = []
    for c in clusters:
        full.extend(by_cluster[c])
    point = float(np.median(full))
    boots = np.empty(n_boot)
    n_clu = len(clusters)
    for b in range(n_boot):
        idx = rng.integers(0, n_clu, size=n_clu)
        sample = []
        for j in idx:
            sample.extend(by_cluster[clusters[j]])
        boots[b] = np.median(sample)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi, n_clu


def percentile_bootstrap_median(rows, key, n_boot=N_BOOT, rng=None):
    """Anti-conservative comparison: flat (row-level) bootstrap on the
    median of `key`. Same return shape as cluster_bootstrap_median."""
    rng = rng or RNG
    arr = np.array([float(r[key]) for r in rows])
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ratio(log_ratio: float) -> str:
    """Convert log10 ratio to a human-readable factor.
    log10 r = -2.42 => DML 263× better than vanilla; print '0.0038x' or '263x reduction'."""
    if not np.isfinite(log_ratio):
        return "n/a"
    r = 10 ** log_ratio
    if log_ratio < 0:
        return f"{r:.3g}× ({1/r:.0f}× reduction)"
    return f"{r:.3g}× ({(r-1)*100:+.1f}% increase)"


def main():
    out = {"meta": {"n_boot": N_BOOT, "rng_seed": 42, "eps_floor": EPS}}

    # --- Synthetic σ=0 (smooth families primary; full-grid secondary)
    sig0_rows = _load_synthetic_pairs(sigma_filter=0.0)
    smooth_rows = [r for r in sig0_rows if r["func"] in SMOOTH_FAMILIES]
    out["synthetic_sigma0_smooth"] = {}
    for label, rows in [("smooth", smooth_rows), ("all_families", sig0_rows)]:
        d = {}
        for metric in ("log_ratio_value", "log_ratio_grad"):
            point, lo, hi, n_clu = cluster_bootstrap_median(rows, metric)
            point_p, lo_p, hi_p, n_rows = percentile_bootstrap_median(rows, metric)
            d[metric] = {
                "median": point,
                "cluster_ci_95": [lo, hi],
                "n_clusters": n_clu,
                "n_rows": n_rows,
                "factor_human": _fmt_ratio(point),
                "percentile_ci_95": [lo_p, hi_p],
            }
        out[f"synthetic_sigma0_{label}"] = d

    # --- Full-grid (all σ) — for the abstract's "across full corpus" line
    full_rows = _load_synthetic_pairs()
    d = {}
    for metric in ("log_ratio_value", "log_ratio_grad"):
        point, lo, hi, n_clu = cluster_bootstrap_median(full_rows, metric)
        d[metric] = {
            "median": point,
            "cluster_ci_95": [lo, hi],
            "n_clusters": n_clu,
            "n_rows": len(full_rows),
            "factor_human": _fmt_ratio(point),
        }
    out["synthetic_full_grid"] = d

    # --- SPY purged-CV (cluster on fold)
    spy_pcv = _load_spy_pairs(SPY_PURGED)
    if spy_pcv:
        d = {}
        for metric in ("log_ratio_value", "log_ratio_grad"):
            point, lo, hi, n_clu = cluster_bootstrap_median(spy_pcv, metric)
            point_p, lo_p, hi_p, n_rows = percentile_bootstrap_median(spy_pcv, metric)
            d[metric] = {
                "median": point,
                "cluster_ci_95": [lo, hi],
                "n_clusters": n_clu,
                "n_rows": n_rows,
                "factor_human": _fmt_ratio(point),
                "percentile_ci_95": [lo_p, hi_p],
            }
        out["spy_purged_cv"] = d
    else:
        out["spy_purged_cv"] = {"error": f"no result files at {SPY_PURGED}"}

    # --- SPY temporal split (single fold, n_seeds clusters not really clusters
    #     so we report only percentile bootstrap)
    spy_temp = _load_spy_pairs(SPY_TEMPORAL)
    if spy_temp:
        d = {}
        for metric in ("log_ratio_value", "log_ratio_grad"):
            point_p, lo_p, hi_p, n_rows = percentile_bootstrap_median(spy_temp, metric)
            d[metric] = {
                "median": point_p,
                "percentile_ci_95": [lo_p, hi_p],
                "n_rows": n_rows,
                "factor_human": _fmt_ratio(point_p),
                "note": "single-fold temporal split; no cluster bootstrap",
            }
        out["spy_temporal"] = d
    else:
        out["spy_temporal"] = {"error": f"no result files at {SPY_TEMPORAL}"}

    OUT_JSON.write_text(json.dumps(out, indent=2))

    # --- Markdown report
    md = ["# Median paired log-MSE-ratio + cluster bootstrap CI (S1+S5)",
          "",
          f"DML method: `{DML}` (λ=1) vs `{VANILLA}`. Bootstrap: {N_BOOT} resamples.",
          "Cluster definition: synthetic = (function, dim, n_train, σ); SPY = fold_idx.",
          "Negative log-ratio means DML beats vanilla; |log_ratio| = orders of magnitude.",
          ""]
    for section, label in [
        ("synthetic_sigma0_smooth", "Synthetic σ=0, smooth families only"),
        ("synthetic_sigma0_all_families", "Synthetic σ=0, all six families"),
        ("synthetic_full_grid",  "Synthetic full grid (all σ)"),
        ("spy_purged_cv",        "SPY purged walk-forward CV"),
        ("spy_temporal",         "SPY temporal split"),
    ]:
        d = out.get(section, {})
        md += [f"## {label}", ""]
        if "error" in d:
            md += [f"_{d['error']}_", ""]
            continue
        for metric in ("log_ratio_value", "log_ratio_grad"):
            m = d[metric]
            ci_text = ""
            if "cluster_ci_95" in m:
                ci_text = (f"cluster CI [10^{m['cluster_ci_95'][0]:+.2f}, "
                           f"10^{m['cluster_ci_95'][1]:+.2f}]")
            elif "percentile_ci_95" in m:
                ci_text = (f"percentile CI [10^{m['percentile_ci_95'][0]:+.2f}, "
                           f"10^{m['percentile_ci_95'][1]:+.2f}]")
            short = "value" if metric == "log_ratio_value" else "gradient"
            md += [f"- **{short} MSE**: median log10 = {m['median']:+.3f} "
                   f"= {m['factor_human']}; {ci_text}; "
                   f"n_clusters={m.get('n_clusters', 'n/a')}, "
                   f"n_rows={m.get('n_rows', 'n/a')}."]
        md.append("")

    OUT_MD.write_text("\n".join(md) + "\n")

    # --- LaTeX-ready sentence(s)
    s = []
    if "spy_purged_cv" in out and "log_ratio_grad" in out["spy_purged_cv"]:
        d = out["spy_purged_cv"]
        v = d["log_ratio_value"]; g = d["log_ratio_grad"]
        v_factor = 10 ** v["median"]; g_factor = 1.0 / (10 ** g["median"])
        s.append(
            f"% Abstract / §5.2 SPY purged-CV headline (S1+S5):\n"
            f"On SPY options under purged walk-forward CV "
            f"(\\(n=5\\) folds × 10 seeds), the median paired ratio of "
            f"\\texttt{{dml\\_fixed}} to \\texttt{{vanilla}} is "
            f"\\(\\mathbf{{{v_factor:.2f}\\times}}\\) on price MSE and "
            f"\\(\\mathbf{{1/{g_factor:.0f}\\times}}\\) on BS-implied "
            f"Greek MSE (cluster bootstrap 95\\% CI on the log-ratio "
            f"medians: "
            f"value \\([10^{{{v['cluster_ci_95'][0]:+.2f}}}, 10^{{{v['cluster_ci_95'][1]:+.2f}}}]\\), "
            f"grad \\([10^{{{g['cluster_ci_95'][0]:+.2f}}}, 10^{{{g['cluster_ci_95'][1]:+.2f}}}]\\); "
            f"clusters = fold)."
        )
    if "synthetic_sigma0_smooth" in out:
        d = out["synthetic_sigma0_smooth"]
        v = d["log_ratio_value"]; g = d["log_ratio_grad"]
        s.append(
            f"% §5.1 synthetic-smooth headline (S1+S5):\n"
            f"On the smooth, low-frequency function families with exact "
            f"gradients (1{{,}}050 paired configurations), the median "
            f"\\texttt{{dml\\_fixed}}-vs-\\texttt{{vanilla}} log-ratio is "
            f"\\(\\mathbf{{{v['median']:+.2f}}}\\) on value MSE "
            f"({_fmt_ratio(v['median'])}) and "
            f"\\(\\mathbf{{{g['median']:+.2f}}}\\) on gradient MSE "
            f"({_fmt_ratio(g['median'])}); cluster bootstrap 95\\% CI "
            f"value \\([10^{{{v['cluster_ci_95'][0]:+.2f}}}, 10^{{{v['cluster_ci_95'][1]:+.2f}}}]\\), "
            f"grad \\([10^{{{g['cluster_ci_95'][0]:+.2f}}}, 10^{{{g['cluster_ci_95'][1]:+.2f}}}]\\); "
            f"{v['n_clusters']} clusters."
        )

    OUT_TEX.write_text("\n\n".join(s) + "\n")
    print(f"wrote {OUT_JSON}, {OUT_MD}, {OUT_TEX}")
    print("Top-line numbers:")
    if "spy_purged_cv" in out and "log_ratio_value" in out["spy_purged_cv"]:
        print(f"  SPY purged-CV value: {out['spy_purged_cv']['log_ratio_value']['factor_human']}")
        print(f"  SPY purged-CV grad:  {out['spy_purged_cv']['log_ratio_grad']['factor_human']}")
    print(f"  Synthetic σ=0 smooth value: {out['synthetic_sigma0_smooth']['log_ratio_value']['factor_human']}")
    print(f"  Synthetic σ=0 smooth grad:  {out['synthetic_sigma0_smooth']['log_ratio_grad']['factor_human']}")


if __name__ == "__main__":
    main()
