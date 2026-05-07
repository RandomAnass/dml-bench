"""
ERA5 Z500 pointwise dataset for the §5.5 sub-pillar.

Loads a per-snapshot .npz cache produced by scripts/preprocess_era5.py
into a torch Dataset of (lat, lon, doy) query points with:

  - sin/cos periodic encoding for longitude and day-of-year (plan §4)
  - lat normalised to radians (gradient label uses radians)
  - optional EOF context (16 components) for the state-augmented regime
  - directional tangent gradient labels: lat-tangent + longitude-tangent
    (d_g = 2; see plan §5)

The full input vector is:
  bare:  [lat_rad_norm, sin_lon, cos_lon, sin_doy, cos_doy]            (d=5)
  state: [lat_rad_norm, sin_lon, cos_lon, sin_doy, cos_doy, eof_1..16] (d=21)

The autograd output of the model is d-dimensional, but DmlLoss only
supervises 2 directional derivatives via the gradient_projection_fn hook
make_era5_directional_projection() defined below.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

R_EARTH_M = 6_371_000.0
G_GRAVITY = 9.80665


def make_era5_directional_projection(
    idx_lat: int = 0,
    idx_sin_lon: int = 1,
    idx_cos_lon: int = 2,
):
    """
    Build the projection callable for DmlLoss.gradient_projection_fn.

    Maps full autograd gradient (batch, 1, d) to (batch, 1, 2):
        ∂f/∂λ = ∂f/∂sin λ · cos λ  −  ∂f/∂cos λ · sin λ

    Per-coord sin/cos gradients are not individually identifiable; only the
    tangent derivative is. d_g = 2.
    """
    def project(grad_pred, grad_target, x_query, dydx_mask):
        if x_query is None:
            raise ValueError(
                "ERA5 directional projection requires x_query. Use Era5Trainer."
            )
        sin_lon = x_query[:, idx_sin_lon].unsqueeze(-1)
        cos_lon = x_query[:, idx_cos_lon].unsqueeze(-1)

        df_dlat = grad_pred[:, :, idx_lat]
        df_dsin = grad_pred[:, :, idx_sin_lon]
        df_dcos = grad_pred[:, :, idx_cos_lon]
        df_dlambda = df_dsin * cos_lon - df_dcos * sin_lon

        proj_pred = torch.stack([df_dlat, df_dlambda], dim=-1)
        proj_target = grad_target[:, :, :2].contiguous()
        proj_mask = (
            dydx_mask[:, :, :2].contiguous()
            if dydx_mask is not None else torch.ones_like(proj_pred)
        )
        return proj_pred, proj_target, proj_mask

    return project


class ERA5Dataset(Dataset):
    """
    Pointwise ERA5 Z500 dataset.

    Each __getitem__ resamples a fixed number of grid cells per epoch. To keep
    epoch-level idempotence the trainer re-seeds the dataset.rng each epoch.
    """

    def __init__(
        self,
        cache_path,
        regime: str = "bare",
        split: str = "train",
        n_points_per_epoch: int = 100_000,
        seed: int = 0,
        eof_K: int = 16,
    ):
        cache_path = Path(cache_path)
        assert cache_path.exists(), f"ERA5 cache missing: {cache_path}"
        assert regime in {"bare", "state"}, regime
        assert split in {"train", "val", "test"}, split
        self.regime = regime
        self.split = split
        self.n_points_per_epoch = int(n_points_per_epoch)
        self.eof_K = eof_K

        with np.load(cache_path, allow_pickle=False) as f:
            split_idx = f[f"split_{split}_idx"]
            self.Z_norm = f["Z_norm"][split_idx].astype(np.float32)
            self.dZdlat = f["dZdlat"][split_idx].astype(np.float32)
            self.dZdlon = f["dZdlon"][split_idx].astype(np.float32)
            self.doy = f["doy"][split_idx].astype(np.int32)
            self.eof = (
                f["eof"][split_idx].astype(np.float32)
                if regime == "state" else None
            )
            self.lat_1d = f["lat_1d"].astype(np.float32)
            self.lon_1d = f["lon_1d"].astype(np.float32)
            self.lat_rad_norm = f["lat_rad_norm"].astype(np.float32)
            self.s_Z = float(f["s_Z"][0]) if f["s_Z"].ndim else float(f["s_Z"])
            self.s_lat_rad = (
                float(f["s_lat_rad"][0]) if f["s_lat_rad"].ndim else float(f["s_lat_rad"])
            )

        if regime == "state":
            assert self.eof is not None and self.eof.shape[1] == eof_K, (
                f"EOF cache mismatch: got {self.eof.shape}, expected (*, {eof_K})"
            )

        self.n_snap, self.n_lat, self.n_lon = self.Z_norm.shape
        self.rng = np.random.default_rng(seed)

        lon_rad = np.radians(self.lon_1d)
        self.sin_lon = np.sin(lon_rad).astype(np.float32)
        self.cos_lon = np.cos(lon_rad).astype(np.float32)

    def __len__(self) -> int:
        return self.n_points_per_epoch

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        s = int(self.rng.integers(0, self.n_snap))
        i_lat = int(self.rng.integers(0, self.n_lat))
        i_lon = int(self.rng.integers(0, self.n_lon))

        doy_v = int(self.doy[s])
        sin_doy = math.sin(2.0 * math.pi * doy_v / 365.0)
        cos_doy = math.cos(2.0 * math.pi * doy_v / 365.0)

        x = [
            float(self.lat_rad_norm[i_lat]),
            float(self.sin_lon[i_lon]),
            float(self.cos_lon[i_lon]),
            float(sin_doy),
            float(cos_doy),
        ]
        if self.regime == "state":
            x.extend(self.eof[s].tolist())
        x = np.asarray(x, dtype=np.float32)
        d_full = x.shape[0]

        y = np.asarray([self.Z_norm[s, i_lat, i_lon]], dtype=np.float32)

        # Directional gradient labels (NORMALISED units; see plan §5)
        target_dlat_norm = (self.s_lat_rad / self.s_Z) * self.dZdlat[s, i_lat, i_lon]
        target_dlambda = (1.0 / self.s_Z) * self.dZdlon[s, i_lat, i_lon]

        dydx = np.zeros((1, d_full), dtype=np.float32)
        dydx[0, 0] = target_dlat_norm
        dydx[0, 1] = target_dlambda

        mask = np.zeros((1, d_full), dtype=np.float32)
        mask[0, 0] = 1.0
        mask[0, 1] = 1.0

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "dydx": torch.from_numpy(dydx),
            "dydx_mask": torch.from_numpy(mask),
        }
