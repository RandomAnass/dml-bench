# `data/` — External datasets

This directory holds the four real-world data sources used by the
benchmark. Synthetic functions are generated on the fly by
`dml_benchmark.functions.generate_data` and need no on-disk material.

**The full operational recipe (download URLs, preprocessing commands,
on-disk layout, licence notes) lives in [`../DATA.md`](../DATA.md).**
Do read that before downloading anything.

## At-a-glance layout

```
data/
├── rmd17/
│   ├── rmd17.tar.bz2                 — original archive
│   └── rmd17/                         — extracted contents
│       ├── npz_data/rmd17_<molecule>.npz
│       └── splits/index_{train,test}_<NN>.csv
├── pdebench/
│   ├── 1D_Burgers_Sols_Nu0.01.hdf5
│   └── 2D_DarcyFlow_beta1.0_Train.hdf5
├── spy_options/
│   ├── spy_processed.npz             — 1,576,419 tuples / 758 days
│   ├── svi_iv.npy                    — SVI-calibrated IV
│   ├── svi_params.npz                — per-(date, maturity) SVI parameters
│   ├── svi_calibration_summary.json  — 23,335 / 23,336 slices accepted
│   └── svi_cache_hash.txt            — content hash for cache invalidation
├── era5/
│   ├── full_1deg/                    — daily Z500 fields, 2014-2020
│   └── pilot/                         — 2019 pilot subset (deprecated)
├── schnetpack_rmd17/                  — SchNetPack DB cache (auto-built)
└── fred_yields.csv                    — FRED Treasury yields (legacy yield-curve experiment)
```

## Source mirrors

| Source | Mirror | Licence | Redistribution |
|---|---|---|---|
| rMD17                  | Figshare (Christensen \& von Lilienfeld 2020)      | CC BY 4.0           | permitted with attribution |
| PDEBench (Burgers, Darcy) | DaRUS (Takamoto et al. 2022)                       | Apache 2.0          | permitted with attribution |
| SPY EOD options        | Kaggle (CC0 dataset, accessed 2026-04)              | CC0                 | permitted (public domain) |
| ERA5 Z500              | Copernicus Climate Data Store                       | Copernicus licence  | **forbidden** |

## Why we do not redistribute

Each user obtains the four real-world datasets from their original
mirror. ERA5 redistribution is forbidden by the Copernicus licence;
the other three permit redistribution but pointing at the canonical
DOI / mirror keeps provenance crisp.

## Smoke-time data prerequisites

The smoke test (`python run_smoke_test.py`) and the synthetic block of
the test suite use only `dml_benchmark.functions.generate_data`. They
do not depend on anything under `data/`.

The full test suite auto-skips the dataset-dependent tests if the
relevant on-disk files are missing:

| Test file | Needs |
|---|---|
| `tests/test_era5_grad_projection_idempotent.py`  | `data/era5/full_1deg/` |
| `tests/test_svi_calibration.py`                  | `data/spy_options/spy_processed.npz` |
| `tests/test_rmd17_cross_arch_splits.py`          | `data/rmd17/rmd17/` |

## Croissant metadata

The repository ships `metadata/croissant.json` describing the four
external sources and the synthetic generator. Validation:

```bash
mlcroissant validate --jsonld metadata/croissant.json
```
