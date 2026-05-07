#!/usr/bin/env python3
"""
Aggregate the GradNorm-α / ReLoBRaLo-τ sensitivity sweep (#197).

Inputs:
  results/balancer_sensitivity/synthetic/*.json   (120 cells: 3 funcs × 4 hp × 5 seeds × 2 methods)
  results/balancer_sensitivity/burgers_ic/*.json  (40 cells: 4 hp × 5 seeds × 2 methods)

Output:
  papers/neurips_DB/evidence/balancer_sensitivity.json (per-(method, hp) summary)
  papers/neurips_DB/evidence/balancer_sensitivity.md (human-readable summary)
  paper/sections/D_supplemental_ablations.tex content (App D.6 numerical fill)

Aggregation rule:
  For each (method, hp_value), compute mean and std of test_value_mse and
  test_grad_mse across (seed, function/dataset) cells.
  Compute paired log-ratio R1 vs the canonical-default (α=1.5 for GradNorm,
  τ=0.10 for ReLoBRaLo, the paper's main-grid values).
"""
import json, glob, os
from pathlib import Path
from collections import defaultdict
import statistics as stats
import math

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SYN_DIR = ROOT / "results" / "balancer_sensitivity" / "synthetic"
PDE_DIR = ROOT / "results" / "balancer_sensitivity" / "burgers_ic"

def load_dir(d):
    rows = []
    for f in sorted(glob.glob(str(d / "*.json"))):
        rec = json.load(open(f))
        rows.append(rec)
    return rows

def main():
    syn = load_dir(SYN_DIR)
    pde = load_dir(PDE_DIR)
    print(f"loaded {len(syn)} synthetic, {len(pde)} pde cells")

    # group by (method, hp_value)
    out = {"synthetic": defaultdict(list), "pde": defaultdict(list)}
    for r in syn:
        key = (r["method"], r["hp_value"])
        out["synthetic"][key].append(r)
    for r in pde:
        key = (r["method"], r["hp_value"])
        out["pde"][key].append(r)

    summary = {}
    for kind in ("synthetic", "pde"):
        summary[kind] = {}
        for (method, hp), records in sorted(out[kind].items()):
            vals_v = [r["test_value_mse"] for r in records]
            vals_g = [r["test_grad_mse"] for r in records]
            summary[kind][f"{method}_hp{hp:.2f}"] = {
                "method": method,
                "hp_value": hp,
                "n_cells": len(records),
                "value_mse_mean": stats.mean(vals_v),
                "value_mse_std": stats.stdev(vals_v) if len(vals_v) > 1 else 0.0,
                "grad_mse_mean": stats.mean(vals_g),
                "grad_mse_std": stats.stdev(vals_g) if len(vals_g) > 1 else 0.0,
            }

    out_json = ROOT / "papers" / "neurips_DB" / "evidence" / "balancer_sensitivity.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")

    out_md = ROOT / "papers" / "neurips_DB" / "evidence" / "balancer_sensitivity.md"
    with open(out_md, "w") as f:
        f.write("# Balancer-sensitivity sweep results\n\n")
        f.write("Source: `results/balancer_sensitivity/{synthetic,burgers_ic}/`\n\n")
        for kind in ("synthetic", "pde"):
            f.write(f"## {kind}\n\n")
            f.write("| method | hp | n | value_mse mean ± std | grad_mse mean ± std |\n")
            f.write("|---|---|---|---|---|\n")
            for k, v in sorted(summary[kind].items()):
                f.write(f"| {v['method']} | {v['hp_value']:.2f} | {v['n_cells']} | "
                        f"{v['value_mse_mean']:.3e} ± {v['value_mse_std']:.3e} | "
                        f"{v['grad_mse_mean']:.3e} ± {v['grad_mse_std']:.3e} |\n")
            f.write("\n")
    print(f"wrote {out_md}")

if __name__ == "__main__":
    main()
