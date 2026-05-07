#!/usr/bin/env python3
"""
Aggregate the classical-baseline (GP / KRR / RF) win-rate vs neural DML.

Sources:
  - results/tier3_benchmark/        (has baseline_gp / _krr / _rf at d ≤ 100)
  - results/tier5_extended_baselines/  (has baseline_krr / _rf at d ∈ {50,100})

Produces a one-paragraph factual summary that the §5 paragraph can quote.

Output:
  papers/neurips_DB/evidence/classical_baselines_summary.json
  papers/neurips_DB/evidence/classical_baselines_summary.md   (paragraph)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from dml_benchmark.io import load_result_json
SOURCE_DIRS = [
    # 2026-05-03: switched to the matched-train (64% train fraction) corpus to
    # fix the data-split asymmetry. Pre-fix tier3 / tier5_extended_baselines
    # gave classical methods 80% train while neural methods got 64%; the
    # 990-cell rerun in `classical_matched_split` corrects this.
    ROOT / "results/classical_matched_split",
]
OUT_JSON = ROOT / "papers/neurips_DB/evidence/classical_baselines_summary.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/classical_baselines_summary.md"

# Method names confirmed from tier-3 sample (2026-04-29): baseline_gp,
# baseline_krr, baseline_rf. (NOT "GP", "KRR", "RF".)
CLASSICAL_METHODS = ["baseline_gp", "baseline_krr", "baseline_rf"]
NEURAL_TARGET = "dml_fixed"
NEURAL_VANILLA = "vanilla"


def main():
    sources_present = [d for d in SOURCE_DIRS if d.exists()]
    if not sources_present:
        print("no source dirs present", file=sys.stderr)
        OUT_JSON.write_text(json.dumps({"error": "no source dirs found"}, indent=2))
        OUT_MD.write_text(
            "# Classical baselines summary\n\n"
            "No source directories present. Cite the existing classical-baseline "
            "appendix instead.\n"
        )
        return

    by_config = defaultdict(dict)
    n_files_seen = 0
    for src in sources_present:
        for p in src.glob("*.json"):
            r = load_result_json(p)
            if r is None:
                continue
            n_files_seen += 1
            method = r.get("method")
            # Filter dml_fixed lambda-ablation files: keep only λ=1.0 canonical.
            if method == NEURAL_TARGET:
                lam = r.get("lambda")
                if lam is not None and float(lam) != 1.0:
                    continue
            func = r.get("func_type") or r.get("dataset")
            dim = r.get("dim")
            n = r.get("n_samples") or r.get("n_train")
            seed = r.get("seed")
            val = r.get("test_value_mse")
            if None in (func, dim, n, seed, val):
                continue
            key = (func, dim, n, seed)
            by_config[key][method] = float(val)

    # Tally win-rate of NEURAL_TARGET vs each classical method (paired by config-seed)
    summary = {"comparisons": {}}
    for classical in CLASSICAL_METHODS:
        wins = ties = losses = 0
        margins = []
        for cfg, methods_seen in by_config.items():
            if NEURAL_TARGET in methods_seen and classical in methods_seen:
                t = methods_seen[NEURAL_TARGET]
                c = methods_seen[classical]
                if t < c:
                    wins += 1; margins.append(c - t)
                elif t > c:
                    losses += 1; margins.append(c - t)
                else:
                    ties += 1
        n = wins + ties + losses
        if n > 0:
            summary["comparisons"][classical] = {
                "n_paired": n, "wins": wins, "ties": ties, "losses": losses,
                "winrate": wins / n,
                "median_margin": float(np.median(margins)) if margins else 0.0,
            }

    summary["n_classical_methods_present"] = len(summary["comparisons"])
    summary["target"] = NEURAL_TARGET

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Markdown paragraph
    if not summary["comparisons"]:
        OUT_MD.write_text(
            "# Classical baselines summary\n\n"
            f"No paired (classical, neural-DML) configurations found in `{EXTENDED}`. "
            "Cite the existing classical-baseline appendix instead.\n"
        )
        return

    md_lines = [
        "# Classical baselines (one-paragraph summary for §5)",
        "",
        f"From `results/tier5_extended_baselines/`, paired-config win-rates of "
        f"`{NEURAL_TARGET}` vs each classical baseline:",
        "",
        "| Classical method | n paired configs | wins | ties | losses | win-rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, c in summary["comparisons"].items():
        md_lines.append(
            f"| {m} | {c['n_paired']} | {c['wins']} | {c['ties']} | {c['losses']} | "
            f"{100*c['winrate']:.1f}% |"
        )
    # Compose paragraph
    sentences = []
    for m, c in summary["comparisons"].items():
        sentences.append(f"{int(100*c['winrate'])}% vs {m} ({c['n_paired']} paired configs)")
    md_lines += [
        "",
        "## Suggested paragraph for §5",
        "",
        f"Across the high-dimensional configurations where classical baselines remain "
        f"tractable (d ≤ 50), neural DML wins {', '.join(sentences)}. The compute "
        f"argument for excluding classical baselines from the SPY benchmark is "
        f"orthogonal: GP scaling is O(n^3) and the SPY corpus has 1.57M records.",
    ]
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
