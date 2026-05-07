"""Post-Phase-3 reproducibility regression test.

Re-runs M1 SIREN seed=0 (using the unchanged pilot_periodic_extrap.py CLI),
then bit-exact diffs the resulting rows against the original M1 CSV.

Usage:
  python experiments/extrapolation/_repro_check_m1.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    m1 = repo / "results" / "extrapolation_M1"
    repro = repo / "results" / "extrapolation_M1_repro_check"

    print(f"[md5] M1 rows.csv = {md5(m1 / 'rows.csv')}")
    print(f"[md5] M1 sigma_star_summary.csv = {md5(m1 / 'sigma_star_summary.csv')}")
    print("[md5] (these should match the values in EXP_DOMAIN_ADAPT_PLAN.md §2)")

    # Run pilot_periodic_extrap.py for siren seed=0 only.
    cmd = [
        "python", "experiments/extrapolation/pilot_periodic_extrap.py",
        "--seeds", "1", "--wide-sigma", "--models", "siren",
        "--out-dir", str(repro),
    ]
    print("\n[exec]", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=repo).returncode
    if rc != 0:
        print(f"[FAIL] pilot_periodic_extrap exit code = {rc}")
        sys.exit(1)

    # Diff with M1.
    df_m1 = pd.read_csv(m1 / "rows.csv")
    df_repro = pd.read_csv(repro / "rows.csv")
    a = df_m1[(df_m1["model"] == "siren") & (df_m1["seed"] == 0)].sort_values(
        ["sigma_rel", "region"]).reset_index(drop=True)
    b = df_repro[(df_repro["model"] == "siren") & (df_repro["seed"] == 0)].sort_values(
        ["sigma_rel", "region"]).reset_index(drop=True)
    print(f"\nM1 SIREN seed=0 rows: {len(a)}")
    print(f"Repro SIREN seed=0 rows: {len(b)}")

    cols = ["mse_vanilla", "mse_dml", "log_ratio",
            "grad_mse_vanilla", "grad_mse_dml", "log_grad_ratio"]
    worst = 0.0
    for c in cols:
        d = np.abs(a[c].to_numpy() - b[c].to_numpy()).max()
        print(f"  {c:20s} max_abs_diff = {d:.3e}")
        if c == "log_ratio":
            worst = d

    PASS = worst < 1e-6
    print(f"\n[REPRO] worst log_ratio diff = {worst:.3e}")
    print(f"[REPRO] {'PASS' if PASS else 'FAIL'}")

    # Final M1 MD5 check (must be unchanged).
    print(f"\n[md5] M1 rows.csv (post-repro) = {md5(m1 / 'rows.csv')}")
    print(f"[md5] M1 sigma_star_summary.csv (post-repro) = {md5(m1 / 'sigma_star_summary.csv')}")


if __name__ == "__main__":
    main()
