#!/usr/bin/env python3
"""
SPY temporal-split + purged-CV statistics — auditable regenerator.

Emits the value/gradient MSE means, std, Δ% vs vanilla, and
gradient-improvement factor for every cell in the SPY temporal and
purged-CV grids, for both supervisor settings (BS-target = Option A,
SVI-coherent = Option C). The caption fragment for tab:spy-bs-temporal
and tab:spy-svi-temporal is emitted alongside.

Usage:
    python papers/neurips_DB/evidence/spy_stats.py
    python papers/neurips_DB/evidence/spy_stats.py --csv-out results/spy_stats.csv

Provenance:
    All numbers in §5.3 (Tables {tab:spy-bs-temporal, tab:spy-svi-temporal})
    are reproducible from this script + the on-disk JSON corpus at
    results/spy_options_{temporal,purged_cv}_option{A,C}/.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
METHODS_ORDER = ["vanilla", "dml_fixed", "dml_fixed_half",
                 "dml_gradnorm", "dml_relobralo", "dml_warmup"]
METHOD_LABEL = {
    "vanilla":         "Vanilla",
    "dml_fixed":       "DML fixed-λ",
    "dml_fixed_half":  "DML fixed-1/2",
    "dml_gradnorm":    "DML GradNorm",
    "dml_relobralo":   "DML ReLoBRaLo",
    "dml_warmup":      "DML Warmup",
}


def collect(split: str, option: str):
    """Group cells by (n_train, method) → [(seed, value_mse, grad_mse), ...]."""
    pattern = ROOT / "results" / f"spy_options_{split}_option{option}" / "*.json"
    g = defaultdict(list)
    for f in glob.glob(str(pattern)):
        try:
            r = json.load(open(f))
            method = r.get("method")
            n_train = r.get("n_train") or r.get("hparams", {}).get("n_train")
            v = r.get("test_value_mse") or r.get("MSEvalue")
            grad = r.get("test_grad_mse") or r.get("MSEgradient")
            seed = r.get("seed")
            if v is not None and grad is not None:
                g[(n_train, method)].append((seed, float(v), float(grad)))
        except Exception:
            pass
    return g


def summarise(g):
    """Return per-(n_train, method) dict with mean+std+vs-vanilla deltas."""
    out = {}
    for (n_train, method), rows in g.items():
        vs = [r[1] for r in rows]
        gs = [r[2] for r in rows]
        out[(n_train, method)] = {
            "n_seeds": len(rows),
            "value_mse_mean": statistics.mean(vs),
            "value_mse_std":  statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "grad_mse_mean":  statistics.mean(gs),
            "grad_mse_std":   statistics.stdev(gs) if len(gs) > 1 else 0.0,
        }
    # Compute deltas vs vanilla per n_train
    n_trains = sorted({k[0] for k in out.keys() if k[0] is not None})
    for n in n_trains:
        van = out.get((n, "vanilla"))
        if not van:
            continue
        for method in METHODS_ORDER:
            row = out.get((n, method))
            if not row:
                continue
            row["delta_value_pct"] = 100 * (row["value_mse_mean"] - van["value_mse_mean"]) / van["value_mse_mean"]
            row["grad_improvement"] = (van["grad_mse_mean"] / row["grad_mse_mean"]) if row["grad_mse_mean"] > 0 else float("inf")
    return out


def emit_table(summary, supervisor_label):
    """Print a markdown table reproducing the §5.3 LaTeX rows."""
    print(f"\n=== {supervisor_label} ===")
    n_trains = sorted({k[0] for k in summary.keys() if k[0] is not None})
    for n in n_trains:
        print(f"\n  n_train = {n}")
        print(f"  {'method':18s} {'val_mean':>11} {'val_std':>11} {'grad_mean':>11} {'Δval%':>8} {'g_improve':>10}")
        for method in METHODS_ORDER:
            row = summary.get((n, method))
            if not row:
                continue
            dv = row.get("delta_value_pct", 0.0)
            gi = row.get("grad_improvement", 1.0)
            print(f"  {METHOD_LABEL[method]:18s} "
                  f"{row['value_mse_mean']:>11.3e} "
                  f"{row['value_mse_std']:>11.3e} "
                  f"{row['grad_mse_mean']:>11.3e} "
                  f"{dv:>+7.1f}% "
                  f"{gi:>9.0f}×")


def write_csv(summary_dict, out_path):
    rows = []
    for (split, supervisor, n_train, method), s in summary_dict.items():
        rows.append({
            "split": split, "supervisor": supervisor,
            "n_train": n_train, "method": method,
            **s,
        })
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-out", default=None)
    args = p.parse_args()

    print(f"# SPY stats — generated by spy_stats.py at git {git_hash()}")
    big = {}
    for split in ("temporal", "purged_cv"):
        for option, label in [("A", "BS-target"), ("C", "SVI-coherent")]:
            g = collect(split, option)
            if not g:
                continue
            summary = summarise(g)
            emit_table(summary, f"{split} / {label}")
            for (n, method), s in summary.items():
                big[(split, label, n, method)] = s

    if args.csv_out:
        write_csv(big, args.csv_out)
        print(f"\n[wrote {args.csv_out}]")


if __name__ == "__main__":
    main()
