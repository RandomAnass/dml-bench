"""Merge shard rows.csv files into a single rows.csv at the parent dir,
then run analyze_da to produce sigma_star_summary.csv.

Usage:
  python experiments/extrapolation/merge_da_shards.py \
      --out-dir results/extrapolation_M1_DA --n-shards 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from domain_adapt import analyze_da
from pilot_periodic_extrap import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--n-shards", type=int, required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    dfs = []
    for s in range(args.n_shards):
        shard_csv = out / f"shard{s}" / "rows.csv"
        if not shard_csv.exists():
            print(f"[warn] missing {shard_csv}")
            continue
        dfs.append(pd.read_csv(shard_csv))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(["model", "seed", "sigma_rel", "region", "arm"],
                            keep="last")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    print(f"[merged] {out / 'rows.csv'}  n_rows={len(df)}")

    cfg = Config()
    cfg.out_root = out
    analyze_da(cfg)


if __name__ == "__main__":
    main()
