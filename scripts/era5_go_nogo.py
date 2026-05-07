#!/usr/bin/env python3
"""ERA5 pilot go/no-go check. Reads all JSONs in a regime dir; exit 0 = pass, 1 = fail."""
from __future__ import annotations

import json
import sys
from pathlib import Path

CRITERIA = {
    "MSEvalue":  (0.0, 2.0),         # normalised value MSE upper-bound
    "MSEgradient": (0.0, 50.0),      # normalised grad MSE upper-bound (lenient)
    "geostrophic_wind_MAE_ms": (1.0, 30.0),  # physical synoptic range
}


def main():
    if len(sys.argv) != 2:
        print("usage: era5_go_nogo.py <results_dir>")
        sys.exit(2)
    d = Path(sys.argv[1])
    files = sorted(d.glob("*.json"))
    if not files:
        print(f"FAIL: no result JSONs in {d}")
        sys.exit(1)
    print(f"checking {len(files)} cells in {d}")
    fails = 0
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        for k, (lo, hi) in CRITERIA.items():
            v = r.get(k)
            if v is None or not (lo <= v <= hi) or v != v:
                print(f"  FAIL {f.name}: {k}={v} outside [{lo}, {hi}]")
                fails += 1
    if fails:
        print(f"FAIL total cells with violations: {fails}/{len(files)}")
        sys.exit(1)
    print(f"PASS all {len(files)} cells in range")
    sys.exit(0)


if __name__ == "__main__":
    main()
