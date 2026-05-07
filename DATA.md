# DATA.md — External datasets used by DML-Bench

Only the synthetic function families are generated on the fly by the
benchmark. The four real-world data sources have to be downloaded from
their original mirrors. This file documents what to download, where to
put it, what licence applies, and what preprocessing the benchmark
expects.

The full datasheet (motivation, composition, collection process,
biases) lives in `paper/sections/A_datasheet.tex`. This file is the
operational counterpart for someone reproducing the results.

---

## Layout overview

After all four sources have been downloaded and preprocessed, `data/`
should look like:

```
data/
├── rmd17/
│   ├── rmd17.tar.bz2                            # original archive
│   └── rmd17/                                    # extracted contents
│       ├── npz_data/rmd17_aspirin.npz
│       ├── npz_data/rmd17_ethanol.npz
│       ├── ...
│       └── splits/index_train_01.csv
├── pdebench/
│   ├── 1D_Burgers_Sols_Nu0.01.hdf5
│   └── 2D_DarcyFlow_beta1.0_Train.hdf5
├── spy_options/
│   ├── spy_processed.npz                        # SPY tuples after filtering
│   ├── svi_iv.npy
│   ├── svi_params.npz
│   └── svi_calibration_summary.json
└── era5/
    ├── full_1deg/                                # cached daily Z500 fields
    └── pilot/                                    # 2019 pilot subset (deprecated)
```

The synthetic functions live in `dml_benchmark/functions.py`; nothing to
download.

---

## 1. Synthetic functions (no download)

Six families, all generated on demand by the benchmark:

| Family | Smoothness | Gradient source |
|---|---|---|
| `poly_trig`     | $C^\infty$         | analytical (JAX autodiff) |
| `trig`          | $C^\infty$         | analytical |
| `bachelier`     | $C^\infty$         | analytical |
| `black_scholes` | $C^\infty$         | analytical |
| `step`          | $C^0$ kinks / digital | pathwise / fuzzy / LRM |
| `heston`        | smooth in expectation | Monte-Carlo (noisy) |

Each call to `dml_benchmark.functions.generate_data(...)` returns a
`FunctionData` triple `(x, y, dydx)`. No raw dataset is shipped; the
dataset is determined by the seed and parameters.

---

## 2. rMD17 (revised MD17)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
Redistribution requires the original Figshare DOI.

**Download:**

The canonical Figshare archive is hosted by Christensen and von Lilienfeld
(2020). The archive name in the wild is `rmd17.tar.bz2` (~6.0 GB
extracted; ~1.8 GB compressed). Drop it into `data/rmd17/` and extract:

```bash
mkdir -p data/rmd17
# Download `rmd17.tar.bz2` from the Figshare archive of Christensen & von Lilienfeld 2020
# (search "Revised MD17 dataset" on figshare; deposit DOI: 10.6084/m9.figshare.12672038)
mv ~/Downloads/rmd17.tar.bz2 data/rmd17/
cd data/rmd17 && tar -xjf rmd17.tar.bz2 && cd ../..
```

After extraction, the per-molecule `.npz` files are at
`data/rmd17/rmd17/npz_data/rmd17_<molecule>.npz` and the canonical splits
are at `data/rmd17/rmd17/splits/index_{train,test}_NN.csv`.

**Used for:**

- `experiments/molecular/run_painn.py` (PaiNN, 10 molecules × 5 splits × 7 methods)
- `experiments/molecular/run_mlp_molecular.py` (MLP-pairwise)
- `experiments/gnn_md17.py` (GATv2)
- `scripts/run_rmd17_tau_sweep.py` (PaiNN τ-sensitivity on aspirin)

**SchNetPack cache:**

PaiNN runs through SchNetPack 2.2.0; the SchNetPack-format database is
built once and cached at `data/schnetpack_rmd17/`. The build script is
`experiments/molecular/_build_painn_db.py`; the runner calls it
automatically the first time.

**Salicylic-acid filename convention:** the rMD17 archive stores the file
as `rmd17_salicylic.npz` while ASE expects `salicylic_acid`. The runner
applies a one-line key translation; no user action required.

---

## 3. PDEBench (Burgers and Darcy)

**License:** Apache 2.0. Redistribution permitted with attribution.

**Download:**

PDEBench is hosted on the University of Stuttgart DaRUS data repository
(deposit by Takamoto et al. 2022). Two HDF5 files are needed:

```bash
mkdir -p data/pdebench
# 1D Burgers, Nu=0.01:
#   filename: 1D_Burgers_Sols_Nu0.01.hdf5     (~2.8 GB)
#   DaRUS deposit: search "PDEBench" on DaRUS, choose the 1D Burgers archive
# 2D Darcy, beta=1.0 train:
#   filename: 2D_DarcyFlow_beta1.0_Train.hdf5 (~1.9 GB)
#   DaRUS deposit: same parent collection
mv ~/Downloads/1D_Burgers_Sols_Nu0.01.hdf5     data/pdebench/
mv ~/Downloads/2D_DarcyFlow_beta1.0_Train.hdf5 data/pdebench/
```

The PDEBench GitHub release (`https://github.com/pdebench/PDEBench`)
provides a download helper. We ship neither the helper nor the data
because the DaRUS mirror is the canonical source.

**Used for:**

- `scripts/run_burgers.py` — Burgers grid (bare and IC regimes, 4×256 +
  6×512 archs, 6 methods, 20 seeds = 480 cells per regime)
- `scripts/run_darcy.py`  — Darcy grid (same shape)
- `scripts/run_burgers_experiment.py` — pre-revision Burgers smoke that
  emits `results/burgers_1d_results.json`

**Sampling:**

Per-simulation random samples of $(x, t)$ for Burgers and $(x, y)$ for
Darcy. We draw 300 query points per training simulation and 100 per
test simulation; the test corpus has ~30,000 query points per PDE.

---

## 4. SPY EOD options

**License:** Creative Commons CC0 1.0 (public domain).

**Download:**

The processed feature panel ships with this release at
`data/spy_options/spy_processed.npz`; reviewers running the SPY pillar do
not need to download the raw archive. The upstream URL of the public
SPY EOD options archive (CC0) is documented in App A of the paper
(datasheet). Equivalent panels for the same 2020-2022 window are
available from commercial market-data APIs such as Polygon, ORATS, or
CBOE DataShop.

```bash
mkdir -p data/spy_options
# Place the raw archive (typically a CSV or zipped CSV per year) at:
#   data/spy_options/raw/spy_eod_2020.csv
#   data/spy_options/raw/spy_eod_2021.csv
#   data/spy_options/raw/spy_eod_2022.csv
```

**Preprocessing:**

```bash
# 1. Filter rows, build the (date, strike, T, IV, spot) tuple panel:
python experiments/real_data_spy/spy_data_loader.py \
    --raw-dir data/spy_options/raw \
    --out      data/spy_options/spy_processed.npz

# 2. SVI calibration per (date, maturity) — required for the SVI-supervised
#    training mode (Option C in the paper):
python experiments/real_data_spy/calibrate_svi.py \
    --in       data/spy_options/spy_processed.npz \
    --out-dir  data/spy_options/

# Output:
#   data/spy_options/spy_processed.npz             1,576,419 tuples (758 days)
#   data/spy_options/svi_iv.npy                    SVI-calibrated IV
#   data/spy_options/svi_params.npz                per-(date, maturity) parameters
#   data/spy_options/svi_calibration_summary.json  acceptance: 23,335/23,336 slices
```

**Used for:**

- `experiments/real_data_spy/run_spy_experiment.py` — temporal split (Option A and Option C)
- `experiments/real_data_spy/run_spy_purged_cv.py` — 5-fold purged walk-forward CV
- `scripts/run_spy_robustness.py`                  — robustness summary
- `scripts/p8_spy_proxy_stress.py`                 — proxy-label stress sweep

**Splits:**

- *Temporal split:* train ≤ 2021-06-30, test ≥ 2021-07-01, 5-day embargo.
- *Purged 5-fold CV:* expanding windows with embargo per López de Prado (2018).

---

## 5. ERA5 Z500 reanalysis

**License:** Copernicus Climate Change Service licence
(`https://apps.ecmwf.int/datasets/licences/copernicus/`). **Free re-use
including for commercial purposes, but redistribution is forbidden.**
Each user must obtain their own access.

### Obtaining access

1. Register at the Copernicus Climate Data Store
   (`https://cds.climate.copernicus.eu`).
2. Accept the ERA5 licence terms in the user dashboard.
3. Place your CDS API key at `~/.cdsapirc` per the CDS docs.

### Download recipe

```bash
mkdir -p data/era5
# Pulls Z500 (500 hPa geopotential), 12Z snapshots, 2014-2020,
# Northern hemisphere mid-latitudes (20-70 deg N), 1.0 deg resolution.
python scripts/download_era5.py \
    --years 2014 2015 2016 2017 2018 2019 2020 \
    --out-dir data/era5/full_1deg

# Build the daily snapshot cache and the 16-EOF state-augmentation table:
python scripts/preprocess_era5.py \
    --in-dir  data/era5/full_1deg \
    --out-dir data/era5/full_1deg
```

The resulting cache stores 2,557 daily fields, each a 51 × 360 grid.

**Used for:**

- `scripts/run_era5.py`        — production runs (`bare` and `state` regimes,
  4×256 and 6×512 architectures, 5 seeds × 3 methods × 2 archs × 2 regimes = 60 cells)
- `scripts/era5_go_nogo.py`    — pre-flight sanity check before launching the grid

**Splits:**

Chronological split 1,731 train / 383 validation / 383 test, with
30-day embargos.

---

## Hugging Face dataset layout (camera-ready)

At camera-ready we publish a Hugging Face dataset that wraps the
released JSON corpus into a Parquet-friendly layout for low-bandwidth
reproduction:

```
hf://datasets/[anonymous]/dml-bench    # populated at camera-ready
├── synthetic/
│   └── per_cell.parquet           # union of tier1/2/3/4 with tier label
├── unified_comparison/
│   └── per_cell.parquet
├── spy/
│   ├── temporal_optionA.parquet
│   ├── temporal_optionC.parquet
│   ├── purged_cv_optionA.parquet
│   └── purged_cv_optionC.parquet
├── molecular/
│   ├── painn_per_cell.parquet
│   ├── gatv2_per_cell.parquet
│   └── mlp_per_cell.parquet
├── pde/
│   ├── burgers_per_cell.parquet
│   └── darcy_per_cell.parquet
├── era5/
│   └── per_cell.parquet
└── README.md
```

The exact URL is appended to this file at camera-ready. The Parquet
layout matches the per-cell JSON schema described in
`paper/sections/G_reproducibility_checklist.tex`; the conversion is a
one-shot pass over the JSONs.

---

## Croissant metadata

`metadata/croissant.json` validates against schema version 1.1 and
includes the NeurIPS 2026 RAI block. Validation:

```bash
mlcroissant validate --jsonld metadata/croissant.json
```

Expected output (verified 2026-05-03):

```
W rdf.py:89 WARNING: The JSON-LD `@context` is not standard.
    (non-fatal; non-standard keys: equivalentProperty, samplingRate)
I validate.py:53 Done.
```

---

## What this repository does not redistribute

- ERA5 reanalysis (Copernicus licence forbids redistribution).
- The raw upstream SPY archive (we ship only the processed feature
  panel; the upstream URL is documented in App A of the paper).
- The Figshare rMD17 archive (CC BY 4.0 permits it, but the canonical
  DOI is Figshare and we point there).
- The DaRUS PDEBench archive (Apache 2.0 permits it, but the canonical
  DOI is DaRUS and we point there).

In every case the download script in `scripts/` or the recipe above
puts the data under `data/` such that the experiment runners find it
without further configuration.
