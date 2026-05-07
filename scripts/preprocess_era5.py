#!/usr/bin/env python3
"""
Convert downloaded ERA5 NetCDF (z500, 12Z snapshots) into the per-snapshot
.npz cache consumed by dml_benchmark/era5_dataset.py.

Steps (plan §3-§6):
  1) Φ (m²/s²) → Z (gpm) via Z = Φ / g (g = 9.80665).
  2) Periodic central differences in lon, central + edges in lat.
  3) Chronological train/val/test split with embargo gap.
  4) Climatology + anomalies + PCA(K=16) — all fit on TRAIN only.
  5) Z normalisation (z-score on train) + lat_rad normalisation.

Usage:
  python scripts/preprocess_era5.py \
      --in-glob "data/era5/pilot/z500_12z_*_1p0deg.nc" \
      --out data/era5/pilot/era5_pilot_cache.npz \
      --eof-K 16 --val-frac 0.15 --test-frac 0.15 --embargo-days 30
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

G_GRAVITY = 9.80665


def load_concat(in_glob: str) -> xr.Dataset:
    paths = sorted(glob.glob(in_glob))
    if not paths:
        raise FileNotFoundError(f"No ERA5 files matched: {in_glob}")
    if len(paths) == 1:
        ds = xr.open_dataset(paths[0])
    else:
        # Manual concat avoids dask dependency of open_mfdataset.
        parts = [xr.open_dataset(p) for p in paths]
        time_dim = "time" if "time" in parts[0].dims else "valid_time"
        ds = xr.concat(parts, dim=time_dim, data_vars="minimal", coords="minimal")
    time_dim = "time" if "time" in ds.dims else "valid_time"
    return ds.sortby(time_dim)


def physical_gradients(Z, lat_1d, lon_1d):
    """Z: (n_snap, n_lat, n_lon) gpm. Returns gpm-per-radian gradients."""
    n_snap, n_lat, n_lon = Z.shape
    if lat_1d[0] > lat_1d[-1]:
        Z = Z[:, ::-1, :].copy()
        lat_1d = lat_1d[::-1].copy()

    dlat_rad = np.radians(np.abs(lat_1d[1] - lat_1d[0]))
    dlon_rad = np.radians(360.0 / n_lon)

    dZdlat = np.empty_like(Z)
    dZdlat[:, 1:-1, :] = (Z[:, 2:, :] - Z[:, :-2, :]) / (2.0 * dlat_rad)
    dZdlat[:, 0, :] = (Z[:, 1, :] - Z[:, 0, :]) / dlat_rad
    dZdlat[:, -1, :] = (Z[:, -1, :] - Z[:, -2, :]) / dlat_rad

    dZdlon = np.empty_like(Z)
    dZdlon[:, :, 1:-1] = (Z[:, :, 2:] - Z[:, :, :-2]) / (2.0 * dlon_rad)
    dZdlon[:, :, 0] = (Z[:, :, 1] - Z[:, :, -1]) / (2.0 * dlon_rad)
    dZdlon[:, :, -1] = (Z[:, :, 0] - Z[:, :, -2]) / (2.0 * dlon_rad)

    return Z, lat_1d, dZdlat, dZdlon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eof-K", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--embargo-days", type=int, default=30)
    args = ap.parse_args()

    print(f"[era5] loading: {args.in_glob}")
    ds = load_concat(args.in_glob)
    var = "z" if "z" in ds.data_vars else "Z"
    Z_phi = ds[var].values.astype(np.float32)
    if Z_phi.ndim == 4:
        Z_phi = Z_phi.squeeze(1)
    Z = Z_phi / G_GRAVITY
    lat_1d = ds["latitude"].values.astype(np.float32)
    lon_1d = ds["longitude"].values.astype(np.float32)
    lon_1d = np.where(lon_1d > 180, lon_1d - 360, lon_1d).astype(np.float32)
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    times = ds[time_dim].values
    doy = np.asarray(
        [int(np.datetime64(t, "D").astype(object).timetuple().tm_yday) for t in times],
        dtype=np.int32,
    )
    n_snap, n_lat, n_lon = Z.shape
    print(f"[era5] Z shape={Z.shape}; gpm range=[{Z.min():.1f}, {Z.max():.1f}]")
    assert 4400 < Z.mean() < 6200, f"Z gpm out of plausible range: mean={Z.mean()}"

    Z, lat_1d, dZdlat, dZdlon = physical_gradients(Z, lat_1d, lon_1d)

    n_val = int(args.val_frac * n_snap)
    n_test = int(args.test_frac * n_snap)
    n_train = n_snap - n_val - n_test - 2 * args.embargo_days
    if n_train <= 0:
        raise ValueError("split too small; lower embargo or add more data")
    i_train_end = n_train
    i_val_start = i_train_end + args.embargo_days
    i_val_end = i_val_start + n_val
    i_test_start = i_val_end + args.embargo_days
    i_test_end = i_test_start + n_test
    split_train_idx = np.arange(0, i_train_end, dtype=np.int64)
    split_val_idx = np.arange(i_val_start, i_val_end, dtype=np.int64)
    split_test_idx = np.arange(i_test_start, i_test_end, dtype=np.int64)
    print(f"[era5] split: train={len(split_train_idx)}, val={len(split_val_idx)}, test={len(split_test_idx)}")

    Z_train = Z[split_train_idx]
    doy_train = doy[split_train_idx]
    n_unique_doy = len(np.unique(doy_train))
    if n_unique_doy < 0.5 * len(doy_train):
        # Multi-year: per-doy climatology
        clim = np.zeros((366, n_lat, n_lon), dtype=np.float32)
        counts = np.zeros(366, dtype=np.int32)
        for k in range(len(doy_train)):
            d = doy_train[k] - 1
            clim[d] += Z_train[k]
            counts[d] += 1
        counts = np.maximum(counts, 1)
        clim = clim / counts[:, None, None]
        Z_anom = Z - clim[doy - 1]
    else:
        # Single-year-or-less: use temporal-mean climatology (constant in doy)
        clim_field = Z_train.mean(axis=0)
        Z_anom = Z - clim_field[None, :, :]

    Z_anom_train_flat = Z_anom[split_train_idx].reshape(len(split_train_idx), -1)
    print(f"[era5] fitting PCA(K={args.eof_K}) on {Z_anom_train_flat.shape[0]} fields")
    pca = PCA(n_components=args.eof_K).fit(Z_anom_train_flat)
    eof = pca.transform(Z_anom.reshape(n_snap, -1)).astype(np.float32)
    eof_std = eof[split_train_idx].std(axis=0).clip(min=1e-6)
    eof = eof / eof_std

    z_mean = float(Z_train.mean())
    z_std = float(Z_train.std()) or 1.0
    Z_norm = ((Z - z_mean) / z_std).astype(np.float32)
    s_Z = z_std

    lat_rad = np.radians(lat_1d).astype(np.float32)
    lat_rad_mean = float(lat_rad.mean())
    lat_rad_std = float(lat_rad.std()) or 1.0
    lat_rad_norm = ((lat_rad - lat_rad_mean) / lat_rad_std).astype(np.float32)
    s_lat_rad = lat_rad_std

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        Z_norm=Z_norm, dZdlat=dZdlat.astype(np.float32),
        dZdlon=dZdlon.astype(np.float32),
        doy=doy, eof=eof,
        lat_1d=lat_1d.astype(np.float32),
        lon_1d=lon_1d.astype(np.float32),
        lat_rad_norm=lat_rad_norm,
        s_Z=np.array([s_Z], dtype=np.float32),
        s_lat_rad=np.array([s_lat_rad], dtype=np.float32),
        z_mean=np.array([z_mean], dtype=np.float32),
        evr_top1=np.array([float(pca.explained_variance_ratio_[0])], dtype=np.float32),
        split_train_idx=split_train_idx,
        split_val_idx=split_val_idx,
        split_test_idx=split_test_idx,
    )
    print(f"[era5] wrote {out}  (PC1 EVR = {pca.explained_variance_ratio_[0]:.3f})")


if __name__ == "__main__":
    main()
