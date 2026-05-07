#!/usr/bin/env python3
"""
ERA5 Z500 sub-pillar aggregator.

Reads results/era5/{bare,state}/*.json (60 cells: 30 each, 4x256 architecture,
3 methods x 10 seeds x 1 year of multi-year data per regime), emits:
  - per-(regime, method) mean + std of value MSE, gradient MSE,
    geostrophic-wind MAE
  - bare->state delta% per method (input-completeness ablation)
  - JSON output at papers/neurips_DB/evidence/era5_summary.json
  - markdown table at papers/neurips_DB/evidence/era5_summary.md

The two regimes share architecture, optimiser, and seeds; only the
input differs (bare = lat/lon/doy with sin/cos periodic encoding;
state = bare + 16 EOF components of the day's geopotential anomaly
field). The deltas isolate the input-completeness contribution.

Usage:
    python papers/neurips_DB/evidence/era5_aggregator.py
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "papers/neurips_DB/evidence/era5_summary.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/era5_summary.md"

METHODS_ORDER = ["vanilla", "dml_fixed", "dml_fixed_half"]
METHOD_LABEL = {
    "vanilla":        "Vanilla",
    "dml_fixed":      "DML fixed-$\\lambda$",
    "dml_fixed_half": "DML fixed-1/2",
}


def collect_regime(regime: str, arch_filter: str | None = "4x256") -> dict:
    """Group cells by method -> [(seed, value, grad, ug, arch)].

    `arch_filter`: when set (default "4x256"), only includes cells whose
    `arch` field matches; pass None to pool across architectures.
    """
    g = defaultdict(list)
    for f in glob.glob(str(ROOT / f"results/era5/{regime}/*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        method = r.get("method")
        v = r.get("MSEvalue") or r.get("test_value_mse")
        grad = r.get("MSEgradient") or r.get("test_grad_mse")
        ug = r.get("geostrophic_wind_MAE_ms")
        seed = r.get("seed")
        arch = r.get("arch")
        if None in (method, v, grad, ug):
            continue
        if arch_filter is not None and arch != arch_filter:
            continue
        g[method].append({
            "seed": seed,
            "arch": arch,
            "value_mse": float(v),
            "grad_mse": float(grad),
            "u_g_mae_ms": float(ug),
        })
    return g


def summarise(cells_by_method: dict) -> dict:
    out = {}
    for method, rows in cells_by_method.items():
        if not rows:
            continue
        vs = [r["value_mse"] for r in rows]
        gs = [r["grad_mse"] for r in rows]
        us = [r["u_g_mae_ms"] for r in rows]
        out[method] = {
            "n": len(rows),
            "value_mse_mean": statistics.mean(vs),
            "value_mse_std":  statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "grad_mse_mean":  statistics.mean(gs),
            "grad_mse_std":   statistics.stdev(gs) if len(gs) > 1 else 0.0,
            "u_g_mae_mean":   statistics.mean(us),
            "u_g_mae_std":    statistics.stdev(us) if len(us) > 1 else 0.0,
        }
    return out


def main():
    # MAIN result: 4x256 architecture only (n=5 seeds per cell).
    bare = collect_regime("bare", arch_filter="4x256")
    state = collect_regime("state", arch_filter="4x256")
    sum_bare = summarise(bare)
    sum_state = summarise(state)
    # Architecture sensitivity: also collect 6x512 separately.
    bare_6x = collect_regime("bare", arch_filter="6x512")
    state_6x = collect_regime("state", arch_filter="6x512")
    sum_bare_6x = summarise(bare_6x)
    sum_state_6x = summarise(state_6x)

    # Compute bare->state delta-% per method
    deltas = {}
    for m in METHODS_ORDER:
        if m in sum_bare and m in sum_state:
            bv = sum_bare[m]["value_mse_mean"]
            sv = sum_state[m]["value_mse_mean"]
            bg = sum_bare[m]["grad_mse_mean"]
            sg = sum_state[m]["grad_mse_mean"]
            deltas[m] = {
                "value_mse_delta_pct": 100 * (sv - bv) / bv,
                "grad_mse_delta_pct":  100 * (sg - bg) / bg,
            }

    summary = {
        "main_4x256": {
            "bare":  sum_bare,
            "state": sum_state,
            "bare_to_state_delta": deltas,
            "n_bare_cells":  sum(s["n"] for s in sum_bare.values()),
            "n_state_cells": sum(s["n"] for s in sum_state.values()),
        },
        "arch_sensitivity_6x512": {
            "bare":  sum_bare_6x,
            "state": sum_state_6x,
            "n_bare_cells":  sum(s["n"] for s in sum_bare_6x.values()),
            "n_state_cells": sum(s["n"] for s in sum_state_6x.values()),
        },
        # Backwards-compat aliases (legacy callers expect top-level)
        "bare":  sum_bare,
        "state": sum_state,
        "bare_to_state_delta": deltas,
        "n_bare_cells":  sum(s["n"] for s in sum_bare.values()),
        "n_state_cells": sum(s["n"] for s in sum_state.values()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Markdown
    lines = [
        "# ERA5 Z500 sub-pillar — input-completeness ablation",
        "",
        "Reanalysis Z500 at 12Z (2014--2020, 1.0 deg, 51 lat x 360 lon, 2557 snapshots; train/val/test 1731/383/383 chronological with 30-day embargos). Main architecture: 4x256 softplus MLP, 100k points sampled per epoch, 5 seeds per (regime, method). 6x512 architecture sensitivity: separate 5-seed sweep.",
        "",
        "## Per-method, per-regime",
        "",
        "| Regime | Method | n | Value MSE (norm.) | Gradient MSE (norm.) | u_g MAE (m/s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for regime, sd in [("bare", sum_bare), ("state", sum_state)]:
        for m in METHODS_ORDER:
            s = sd.get(m)
            if not s:
                continue
            lines.append(
                f"| {regime} | {METHOD_LABEL[m]} | {s['n']} | "
                f"{s['value_mse_mean']:.3f} +- {s['value_mse_std']:.3f} | "
                f"{s['grad_mse_mean']:.3f} +- {s['grad_mse_std']:.3f} | "
                f"{s['u_g_mae_mean']:.2f} +- {s['u_g_mae_std']:.2f} |"
            )
    lines += ["", "## Input-completeness delta (state vs bare)", "",
              "| Method | Value MSE delta-pct | Gradient MSE delta-pct |",
              "|---|---:|---:|"]
    for m in METHODS_ORDER:
        if m in deltas:
            d = deltas[m]
            lines.append(
                f"| {METHOD_LABEL[m]} | "
                f"{d['value_mse_delta_pct']:+.1f}% | "
                f"{d['grad_mse_delta_pct']:+.1f}% |"
            )

    # Append architecture-sensitivity table (6x512 results)
    lines += ["", "## Architecture sensitivity (6x512, 5 seeds)", "",
              "| Regime | Method | n | Value MSE (norm.) | Gradient MSE (norm.) | u_g MAE (m/s) |",
              "|---|---|---:|---:|---:|---:|"]
    for regime, sd in [("bare", sum_bare_6x), ("state", sum_state_6x)]:
        for m in METHODS_ORDER:
            s = sd.get(m)
            if not s:
                continue
            lines.append(
                f"| {regime} | {METHOD_LABEL[m]} | {s['n']} | "
                f"{s['value_mse_mean']:.3f} +- {s['value_mse_std']:.3f} | "
                f"{s['grad_mse_mean']:.3f} +- {s['grad_mse_std']:.3f} | "
                f"{s['u_g_mae_mean']:.2f} +- {s['u_g_mae_std']:.2f} |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
