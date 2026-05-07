"""
Download ERA5 500-hPa geopotential at 12Z snapshots via the CDS API.

Used for the §5.5 sub-pillar real-reanalysis validation of the
input-completeness ablation. Geopotential `z` is in m²/s² and must be
divided by g = 9.80665 to get geopotential height in gpm — see
paper/ERA5_VALIDATION_PLAN.md §3.

Usage:
    # Pilot: 1 year (2019), 1.0° resolution → ~50 MB
    python scripts/download_era5.py --years 2019 --resolution 1.0 \\
        --out-dir data/era5/pilot

    # Full Stage 2: 7 years (2014-2020), 0.5° resolution → ~3-4 GB
    python scripts/download_era5.py \\
        --years 2014 2015 2016 2017 2018 2019 2020 \\
        --resolution 0.5 --out-dir data/era5/raw

Output: NetCDF per year at <out_dir>/z500_12z_<year>_<res>deg.nc
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cdsapi


def download_year(year: int, out_dir: Path, resolution: float = 0.25,
                   max_retries: int = 5) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res_str = str(resolution).replace(".", "p")
    out_path = out_dir / f"z500_12z_{year}_{res_str}deg.nc"
    if out_path.exists():
        print(f"[skip] {out_path}")
        return out_path

    c = cdsapi.Client()
    request = {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "pressure_level": ["500"],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day":   [f"{d:02d}" for d in range(1, 32)],
        "time":  ["12:00"],                          # 12Z snapshot
        "area":  [70, -180, 20, 180],                # N W S E
        "grid":  [resolution, resolution],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    for attempt in range(max_retries):
        try:
            print(f"[try {attempt+1}/{max_retries}] {year} → {out_path}")
            c.retrieve("reanalysis-era5-pressure-levels", request).download(str(out_path))
            print(f"[done] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
            return out_path
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"[retry in {wait}s] {e}")
            time.sleep(wait)

    raise RuntimeError(f"Failed {max_retries} times for {year}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--resolution", type=float, default=0.25)
    p.add_argument("--out-dir", default="data/era5/raw")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    for y in args.years:
        download_year(y, out_dir, args.resolution)


if __name__ == "__main__":
    main()
