#!/usr/bin/env python3
"""
S3: almost-stochastic-dominance (Dror, Shlomov, Reichart, ACL 2019)
on SPY purged-CV gradient MSE. Replaces / complements Cohen's d.

Output:
  papers/neurips_DB/evidence/aso_spy.json
  papers/neurips_DB/evidence/aso_spy.md   (caption-ready)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from deepsig import aso

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json
SRC = ROOT / "results/spy_options_purged_cv"
OUT_JSON = ROOT / "papers/neurips_DB/evidence/aso_spy.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/aso_spy.md"

DML_VARIANTS = ["dml_fixed", "dml_gradnorm", "dml_relobralo", "dml_warmup"]
VANILLA = "vanilla"
CONFIDENCE = 0.95
N_BOOT = 1000
RNG = np.random.default_rng(42)


def collect():
    """Per method -> list of test_grad_mse values across (fold, seed)."""
    by_method = defaultdict(list)
    for p in SRC.glob("*.json"):
        r = load_result_json(p)
        if r is None:
            continue
        m = r.get("method")
        g = r.get("test_grad_mse")
        if m is None or g is None:
            continue
        by_method[m].append(float(g))
    return {m: np.array(v) for m, v in by_method.items() if v}


def main():
    data = collect()
    if VANILLA not in data:
        raise SystemExit(f"no vanilla data in {SRC}")
    vanilla = data[VANILLA]
    summary = {
        "n_vanilla": len(vanilla),
        "method_n": {m: len(v) for m, v in data.items()},
        "results": {},
    }
    for m in DML_VARIANTS:
        if m not in data:
            continue
        dml = data[m]
        # aso(scores_a, scores_b) returns the upper bound of epsilon
        # such that A almost-stochastically-dominates B (here: A = DML
        # *being smaller* on a "lower is better" metric = stochastic
        # dominance of the negated). deepsig uses the convention "higher
        # is better"; we negate so "DML better" = higher.
        eps_dml_better = aso(-dml, -vanilla, confidence_level=CONFIDENCE,
                              num_jobs=4, num_bootstrap_iterations=N_BOOT,
                              seed=42)
        summary["results"][m] = {
            "epsilon_dml_dominates_vanilla": float(eps_dml_better),
            "n_dml": int(len(dml)),
            "interpretation": (
                "epsilon < 0.5 ⇒ DML almost-stochastically dominates vanilla "
                "on grad MSE; epsilon close to 0 ⇒ near-complete dominance; "
                "epsilon ≥ 0.5 ⇒ no dominance claim (Dror et al. 2019)."
            ),
        }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    md = ["# S3: Almost-Stochastic Dominance for SPY purged-CV grad MSE",
          "",
          f"Method: deepsig.aso (Dror, Shlomov, Reichart, ACL 2019), "
          f"confidence {CONFIDENCE}, {N_BOOT} bootstrap iterations.",
          "",
          f"Comparison against `vanilla` (n = {summary['n_vanilla']} purged-CV "
          f"runs = 5 folds × 10 seeds).",
          "",
          "| DML variant | n | ε (DML dominates vanilla on grad MSE) |",
          "|---|---:|---:|"]
    for m in DML_VARIANTS:
        if m in summary["results"]:
            r = summary["results"][m]
            md.append(f"| `{m}` | {r['n_dml']} | "
                       f"**{r['epsilon_dml_dominates_vanilla']:.4f}** |")
    md += ["",
           "Interpretation: ε < 0.5 indicates almost-stochastic dominance;",
           "ε close to 0 indicates near-complete dominance (DML's grad-MSE",
           "distribution lies almost entirely below vanilla's). Replaces",
           "Cohen's d as the headline effect size for §5.2 — where d is",
           "in the [318, 464] range its interpretability is poor; ε is",
           "scale-free and bounded.",
           ""]
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_JSON}, {OUT_MD}")
    for m, r in summary["results"].items():
        print(f"  {m}: ε = {r['epsilon_dml_dominates_vanilla']:.4f} (n={r['n_dml']})")


if __name__ == "__main__":
    main()
