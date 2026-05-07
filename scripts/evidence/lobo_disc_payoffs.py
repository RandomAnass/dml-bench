#!/usr/bin/env python3
"""
Leave-one-balancer-out (LOBO) robustness check on the discontinuous-payoff
CD diagram ranking.

For each of the five DML balancers (fixed, fixed_half, gradnorm, relobralo,
warmup), recompute the balancer-paradigm ranking on the discontinuous-payoff
sub-corpus with that balancer removed. Verify that the
fuzzy-ahead-of-pathwise ordering on every dataset is invariant.

Output (internal, not paper-cited):
  papers/neurips_DB/evidence/lobo_disc_payoffs.json
  papers/neurips_DB/evidence/lobo_disc_payoffs.md
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

UNIFIED_DIR = ROOT / "results/unified_comparison/multi_seed"
OUT_JSON = ROOT / "papers/neurips_DB/evidence/lobo_disc_payoffs.json"
OUT_MD = ROOT / "papers/neurips_DB/evidence/lobo_disc_payoffs.md"

DISC_DATASETS = ["digital_bs", "barrier_bs", "heston_digital", "basket_d1", "basket_d7"]
PARADIGMS = ["pathwise", "fuzzy", "lrm"]
BALANCERS = ["dml_fixed", "dml_fixed_half", "dml_gradnorm", "dml_relobralo", "dml_warmup"]


def _norm_method(method: str) -> tuple[str, str]:
    """Split a method+paradigm token (e.g. dml_warmup_fuzzy) into balancer + paradigm."""
    if method == "vanilla":
        return ("vanilla", "")
    for p in PARADIGMS:
        if method.endswith("_" + p):
            return (method[:-(len(p) + 1)], p)
    return (method, "pathwise")


def load_disc_payoff_results():
    """Build per-(dataset, method) list of test_value_mse across seeds."""
    by = defaultdict(list)
    if not UNIFIED_DIR.exists():
        print(f"WARN: {UNIFIED_DIR} does not exist", file=sys.stderr)
        return by
    for p in UNIFIED_DIR.glob("*.json"):
        r = load_result_json(p)
        if r is None:
            continue
        ds = r.get("dataset") or r.get("func_type")
        method = r.get("method")
        v = r.get("test_value_mse")
        if ds and method and v is not None:
            by[(ds, method)].append(float(v))
    return by


def rank_methods_per_dataset(by, exclude_balancer=None):
    """Return per-dataset ranking. Lower mean test_value_mse = better rank.
    If exclude_balancer given, drop all method tokens whose balancer matches."""
    rankings = {}
    for ds in DISC_DATASETS:
        method_means = {}
        for (d, method), vals in by.items():
            if d != ds:
                continue
            balancer, paradigm = _norm_method(method)
            if exclude_balancer is not None and balancer == exclude_balancer:
                continue
            if not vals:
                continue
            method_means[method] = float(np.mean(vals))
        if not method_means:
            rankings[ds] = []
            continue
        ranked = sorted(method_means.items(), key=lambda kv: kv[1])
        rankings[ds] = [m for m, _ in ranked]
    return rankings


def fuzzy_ahead_of_pathwise(rankings):
    """Boolean per dataset: do all fuzzy methods rank ahead of all pathwise?"""
    out = {}
    for ds, ordered in rankings.items():
        fuzzy_idx = [i for i, m in enumerate(ordered) if "_fuzzy" in m]
        pathwise_idx = [i for i, m in enumerate(ordered)
                        if not any(m.endswith(suf) for suf in ("_fuzzy", "_lrm", "_dirac"))
                        and m != "vanilla"
                        and m.startswith("dml_")]
        if not fuzzy_idx or not pathwise_idx:
            out[ds] = None
        else:
            out[ds] = max(fuzzy_idx) < min(pathwise_idx)
    return out


def main():
    by = load_disc_payoff_results()
    print(f"loaded {sum(len(v) for v in by.values())} (dataset, method, seed) records "
          f"across {len(set(k[0] for k in by))} datasets, {len(set(k[1] for k in by))} methods")

    out = {"full": rank_methods_per_dataset(by, exclude_balancer=None),
           "full_fuzzy_ahead": fuzzy_ahead_of_pathwise(rank_methods_per_dataset(by))}
    print("\nFull-corpus ranking:")
    for ds, ranked in out["full"].items():
        ahead = out["full_fuzzy_ahead"][ds]
        print(f"  {ds:18s} fuzzy_ahead={ahead}; top-3: {ranked[:3]}")

    out["lobo"] = {}
    out["lobo_fuzzy_ahead"] = {}
    for b in BALANCERS:
        rankings = rank_methods_per_dataset(by, exclude_balancer=b)
        ahead = fuzzy_ahead_of_pathwise(rankings)
        out["lobo"][b] = rankings
        out["lobo_fuzzy_ahead"][b] = ahead
        n_invariant = sum(1 for v in ahead.values() if v)
        print(f"\n  drop {b}: fuzzy_ahead invariant on {n_invariant}/{len(ahead)} datasets")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    with open(OUT_MD, "w") as f:
        f.write("# LOBO disc-payoff robustness — leave-one-balancer-out\n\n")
        f.write("Per-dataset balancer-paradigm rankings on the discontinuous-payoff sub-corpus, "
                "with each of the five DML balancers removed in turn.\n\n")
        f.write("| Variant | digital_bs | barrier_bs | heston_dig | basket_d1 | basket_d7 |\n")
        f.write("|---|:-:|:-:|:-:|:-:|:-:|\n")
        full_ahead = out["full_fuzzy_ahead"]
        f.write("| full corpus | " + " | ".join(
            "✓" if full_ahead.get(ds) else ("✗" if full_ahead.get(ds) is False else "—")
            for ds in DISC_DATASETS) + " |\n")
        for b in BALANCERS:
            ahead = out["lobo_fuzzy_ahead"][b]
            f.write(f"| drop {b} | " + " | ".join(
                "✓" if ahead.get(ds) else ("✗" if ahead.get(ds) is False else "—")
                for ds in DISC_DATASETS) + " |\n")
        f.write("\n✓ = fuzzy-paradigm methods all rank ahead of pathwise methods. "
                "— = no fuzzy or no pathwise rows in this variant.\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
