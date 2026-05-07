#!/usr/bin/env python3
"""
Two-metric (gradient + value) per-function win-rate aggregator for Table 1.

Reads results from results/tier{1,2,3,4}_benchmark/ and computes paired
(per (func, dim, n, sigma, seed) cell) win-rates of dml_fixed (lambda=1)
vs vanilla on BOTH gradient MSE AND value MSE, with Wilson 95% CIs.

Output:
  papers/neurips_DB/evidence/winrate_two_metric.json
  papers/neurips_DB/evidence/winrate_two_metric.tex   (LaTeX table body)
  papers/neurips_DB/evidence/winrate_two_metric.md    (human-readable)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json
TIER_DIRS = [
    ROOT / "results/tier1_benchmark",
    ROOT / "results/tier2_benchmark",
    ROOT / "results/tier3_benchmark",
    ROOT / "results/tier4_benchmark",
]
OUT_JSON = ROOT / "papers/neurips_DB/evidence/winrate_two_metric.json"
OUT_TEX = ROOT / "papers/neurips_DB/evidence/winrate_two_metric.tex"
OUT_MD = ROOT / "papers/neurips_DB/evidence/winrate_two_metric.md"

TARGET = "dml_fixed"
BASELINE = "vanilla"
FUNCS = ["bachelier", "black_scholes", "poly_trig", "trig", "step", "heston"]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion (Wilson 1927)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def collect():
    """Index by config key, then keep dml_fixed (lambda=1) and vanilla."""
    by_cfg = defaultdict(dict)
    for tdir in TIER_DIRS:
        if not tdir.exists():
            continue
        for p in tdir.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            method = r.get("method")
            if method not in (TARGET, BASELINE):
                continue
            if method == TARGET and r.get("lambda") not in (None, 1.0):
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
            cell = by_cfg[(func, int(dim), int(n), float(sigma), int(seed))]
            cell[method] = {"value": float(v), "grad": float(g)}
    return by_cfg


def winrates(by_cfg, sigma_filter=None):
    """Return per-function counters of paired wins on grad/value."""
    per_func = defaultdict(lambda: {"n": 0, "value_wins": 0, "grad_wins": 0,
                                    "value_ties": 0, "grad_ties": 0})
    for (func, dim, n, sigma, seed), cell in by_cfg.items():
        if TARGET not in cell or BASELINE not in cell:
            continue
        if sigma_filter is not None and not sigma_filter(sigma):
            continue
        t = cell[TARGET]
        b = cell[BASELINE]
        f = per_func[func]
        f["n"] += 1
        if t["value"] < b["value"]:
            f["value_wins"] += 1
        elif t["value"] == b["value"]:
            f["value_ties"] += 1
        if t["grad"] < b["grad"]:
            f["grad_wins"] += 1
        elif t["grad"] == b["grad"]:
            f["grad_ties"] += 1
    return per_func


def render_md(sigma0, all_sig):
    lines = ["# Two-metric per-function win-rate (paired dml_fixed vs vanilla)",
             "",
             "Each row: paired configs at the given σ regime; "
             "Wilson 95% CI in brackets.",
             ""]
    for label, src in [("σ=0 (exact gradients)", sigma0),
                       ("All σ (full grid)",      all_sig)]:
        lines += [f"## {label}", ""]
        lines += ["| Function | n | grad-MSE win | grad-MSE 95% CI | value-MSE win | value-MSE 95% CI |",
                  "|---|---:|---:|---:|---:|---:|"]
        total_n = total_gw = total_vw = 0
        for func in FUNCS:
            d = src.get(func, {"n": 0, "value_wins": 0, "grad_wins": 0})
            n = d["n"]
            if n == 0:
                lines.append(f"| {func} | 0 | -- | -- | -- | -- |")
                continue
            gw, vw = d["grad_wins"], d["value_wins"]
            g_lo, g_hi = wilson_ci(gw, n)
            v_lo, v_hi = wilson_ci(vw, n)
            lines.append(
                f"| {func} | {n} | {gw}/{n} ({100*gw/n:.1f}%) | "
                f"[{100*g_lo:.1f}%, {100*g_hi:.1f}%] | "
                f"{vw}/{n} ({100*vw/n:.1f}%) | "
                f"[{100*v_lo:.1f}%, {100*v_hi:.1f}%] |"
            )
            total_n += n; total_gw += gw; total_vw += vw
        if total_n > 0:
            g_lo, g_hi = wilson_ci(total_gw, total_n)
            v_lo, v_hi = wilson_ci(total_vw, total_n)
            lines.append(
                f"| **total** | **{total_n}** | "
                f"**{total_gw}/{total_n} ({100*total_gw/total_n:.1f}%)** | "
                f"[{100*g_lo:.1f}%, {100*g_hi:.1f}%] | "
                f"**{total_vw}/{total_n} ({100*total_vw/total_n:.1f}%)** | "
                f"[{100*v_lo:.1f}%, {100*v_hi:.1f}%] |"
            )
        lines.append("")
    return "\n".join(lines)


def render_tex(sigma0):
    """LaTeX table body (just sigma=0 cell, with both grad and value cols + Wilson CIs)."""
    lines = [
        "% auto-generated by winrate_two_metric.py",
        "\\begin{tabular}{lrcccc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Gradient MSE} & \\multicolumn{2}{c}{Value MSE} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Function & $n$ & win-rate & 95\\% CI & win-rate & 95\\% CI \\\\",
        "\\midrule",
    ]
    total_n = total_gw = total_vw = 0
    for func in FUNCS:
        d = sigma0.get(func, {"n": 0, "value_wins": 0, "grad_wins": 0})
        n = d["n"]
        if n == 0:
            lines.append(f"\\texttt{{{func}}} & 0 & -- & -- & -- & -- \\\\")
            continue
        gw, vw = d["grad_wins"], d["value_wins"]
        g_lo, g_hi = wilson_ci(gw, n)
        v_lo, v_hi = wilson_ci(vw, n)
        func_tex = func.replace("_", "\\_")
        lines.append(
            f"\\texttt{{{func_tex}}} & {n} & "
            f"{100*gw/n:.1f}\\% & [{100*g_lo:.1f}, {100*g_hi:.1f}] & "
            f"{100*vw/n:.1f}\\% & [{100*v_lo:.1f}, {100*v_hi:.1f}] \\\\"
        )
        total_n += n; total_gw += gw; total_vw += vw
    if total_n > 0:
        g_lo, g_hi = wilson_ci(total_gw, total_n)
        v_lo, v_hi = wilson_ci(total_vw, total_n)
        lines += [
            "\\midrule",
            f"Total & {total_n} & "
            f"\\textbf{{{100*total_gw/total_n:.1f}\\%}} & [{100*g_lo:.1f}, {100*g_hi:.1f}] & "
            f"\\textbf{{{100*total_vw/total_n:.1f}\\%}} & [{100*v_lo:.1f}, {100*v_hi:.1f}] \\\\",
        ]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    by_cfg = collect()
    sigma0 = winrates(by_cfg, sigma_filter=lambda s: s == 0.0)
    all_sig = winrates(by_cfg)
    summary = {
        "sigma_zero": {f: dict(v) for f, v in sigma0.items()},
        "all_sigma": {f: dict(v) for f, v in all_sig.items()},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    OUT_MD.write_text(render_md(sigma0, all_sig))
    OUT_TEX.write_text(render_tex(sigma0))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_TEX}")
    for label, src in [("sigma=0", sigma0), ("all sigma", all_sig)]:
        total = sum(d["n"] for d in src.values())
        gw = sum(d["grad_wins"] for d in src.values())
        vw = sum(d["value_wins"] for d in src.values())
        print(f"  {label}: n={total}, grad-win={gw}/{total} ({100*gw/total:.1f}%), "
              f"value-win={vw}/{total} ({100*vw/total:.1f}%)" if total else f"  {label}: empty")


if __name__ == "__main__":
    main()
