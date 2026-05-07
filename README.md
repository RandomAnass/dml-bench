# DML-Bench

A reproducibility-first benchmark for derivative-enhanced machine learning: it
trains the same neural network on $(x, y)$ pairs versus on $(x, y, \nabla_x y)$
triples across six synthetic function families, four real-world domains
(rMD17, PDEBench, SPY options, ERA5 Z500), and eleven training methods.

This repository ships the code, configuration, and per-cell JSON corpus
(~19,500 runs) that produced every table and figure in the accompanying
paper. Re-running the aggregator scripts on the released JSONs reproduces
every numerical value without retraining.

Anonymous code mirror: `https://anonymous.4open.science/r/dml-bench-EB7F`
(URL added at submission). Hugging Face datasets and Zenodo DOIs at
camera-ready (placeholders below).

---

## Contents

- [Quick start](#quick-start) — clone, env, smoke test
- [Reproducer table](#reproducer-table) — claim → JSON → aggregator → command
- [Code organisation](#code-organisation)
- [Data access](#data-access) — see `DATA.md` for the long form
- [Hugging Face dataset and Zenodo](#hugging-face-and-zenodo)
- [Citation](#citation)
- [License](#license)

---

## Quick start

### 1. Clone

```bash
git clone https://anonymous.4open.science/r/dml-bench-EB7F dml-bench
cd dml-bench
```

### 2. Create the conda environment

The reproduction environment is named `dml-bench-env`. The exact pin set
that produced every released JSON is in `environment.yml`; a flat
`requirements.txt` is provided for `pip` users.

```bash
conda env create -f environment.yml      # creates env "dml-bench-env"
conda activate dml-bench-env
pip install -e .                          # installs the dml_benchmark package
```

CPU-only reproduction is supported for the smoke test and the closed-form
solver. The full benchmark grid expects two NVIDIA GPUs with at least 16 GB
each.

### 3. Run the smoke test

```bash
python run_smoke_test.py
```

Expected output (CPU, < 2 min): trig at $d=2, n=256, \mathrm{seed}=42$
trained for 30 epochs with each of `vanilla`, `dml_fixed`,
`dml_gradnorm`, `dml_relobralo` plus the three classical baselines
(GP / KRR / RF). Results land in `results/smoke_test/`. The script
prints per-method `test_value_mse` and `test_grad_mse` and a final
status line; if you see `LOSS DID NOT DECREASE`, the install is
broken.

### 4. Run the test suite

```bash
python -m pytest tests/ -v
```

Expected: 158 `def test_` functions are defined in `tests/`. On a clean
install with all data accessible, ~150 collect successfully; the
`era5_*`, `svi_*`, and a few `heston_*` tests are skipped or error if the
relevant dataset is not yet downloaded (they auto-skip when
`pytest.importorskip` succeeds — see `DATA.md` for the data prerequisites).

### 5. Regenerate paper tables from released JSONs

The aggregator scripts live in `scripts/evidence/` (the prior
draft tree); the camera-ready reorganises them under a top-level
`evidence/` directory. The provenance map in `EVIDENCE/claims_registry.csv`
records the script that produces each table or figure.

```bash
# Examples (paths as of 2026-05-06 inventory):
python scripts/evidence/winrate_two_metric.py
python scripts/evidence/classical_baselines_summary.py
```

Both run on a laptop without GPUs in roughly a minute.

---

## Reproducer table

Mapping each main-paper claim to the result directory, the aggregator script,
and the command that produced the underlying JSONs.

| Claim (short) | Result dir | Aggregator | Reproducer command |
|---|---|---|---|
| Synthetic smooth: 85.7 % value-MSE win, 96.8 % grad-MSE win at $\sigma=0$ | `results/tier{1,2,3,4}_benchmark/` | `scripts/evidence/winrate_two_metric.py` | `python run_full_benchmark.py --tier 3` then `--tier 4` |
| Per-family $\sigma^*$ (poly_trig 0.195, trig 0.098, bachelier undefined) | `results/tier{3,4}_benchmark/` | `scripts/evidence/sigma_star_lowess.py`, `sigma_star_bca.py` | `python run_full_benchmark.py --tier 3` then `--tier 4` |
| Synthetic smooth median ~13× value-MSE reduction | `results/tier{3,4}_benchmark/` | `scripts/evidence/median_log_ratio.py` | (same as above) |
| 16× data-efficiency on poly_trig $d=20$ at $\sigma=0$ | `results/tier3_benchmark/` | (inline in `winrate_two_metric.py`) | `python run_full_benchmark.py --tier 3` |
| Disc-payoff CD diagram: warmup-fuzzy 1.8/12, Friedman $p=7.2\times10^{-5}$ | `results/unified_comparison/multi_seed/` | `scripts/evidence/cd_diagram.py` | `python experiments/unified_comparison/run_unified_experiment.py --mode multi_seed` |
| SPY temporal: 1.7×–2.5× value-MSE cut both supervisors | `results/spy_options_temporal_option{A,C}/` | `experiments/real_data_spy/analyze_spy_temporal.py` | `python experiments/real_data_spy/run_spy_experiment.py --target-mode bs_price` then `--target-mode svi` |
| SPY purged-CV per-fold deltas (both supervisors) | `results/spy_options_purged_cv_option{A,C}/` | `experiments/real_data_spy/analyze_spy_purged_cv.py`, `scripts/evidence/spy_stats.py` | `python experiments/real_data_spy/run_spy_purged_cv.py --target-mode bs_price` then `--target-mode svi` |
| rMD17 PaiNN: 25–40× force-MAE reduction, native_EF rank 1/6 | `results/molecular_painn/` | `scripts/evidence/median_log_ratio_rmd17.py` | `python experiments/molecular/run_painn.py` |
| rMD17 cross-arch (PaiNN + GATv2 + MLP, 3×350 cells) | `results/molecular_{painn,gatv2,mlp}/` | (same; plus `softmax_balance_comparison.py`) | `python experiments/molecular/run_painn.py`; `python experiments/gnn_md17.py`; `python experiments/molecular/run_mlp_molecular.py` |
| rMD17 $\tau$-sweep (aspirin × 3 τ × 5 seeds) | `results/rmd17_tau_sweep/` | (inline in `run_painn.py`'s analyser) | `python scripts/run_rmd17_tau_sweep.py` |
| Burgers IC: ~6× value-MSE cut with `dml_fixed_half` | `results/burgers/ic/` | `scripts/evidence/median_log_ratio_pde.py` | `python scripts/run_burgers.py --input-mode ic --archs 4x256` |
| Darcy IC: ~30 % value-MSE cut | `results/darcy/ic/` | (same) | `python scripts/run_darcy.py --input-mode ic --archs 4x256` |
| PDE bare regime: TOST-equivalent across six methods | `results/burgers/bare/`, `results/darcy/bare/` | `scripts/evidence/tost_equivalence.py` | `python scripts/run_burgers.py --input-mode bare --archs 4x256`; `python scripts/run_darcy.py --input-mode bare --archs 4x256` |
| ERA5 bare: −10 % grad-MSE at value parity, geostrophic-wind 5.5–6.8 m/s | `results/era5/bare/` | `scripts/evidence/era5_aggregator.py` | `python scripts/run_era5.py --regime bare` |
| ERA5 state-augmented: +10–19 % value pays for −30 % grad | `results/era5/state/` | (same) | `python scripts/run_era5.py --regime state` |
| Heston barrier (narrow): 9× price-MSE cut with `dml_fuzzy_warmup` | `results/heston_barrier_4way/multi_seed/` | `experiments/heston_barrier_4way/analyze_polynomial_baselines.py` | `python experiments/heston_barrier_4way/run_multi_seed.py` |
| Heston barrier (wide): polynomial baseline beats every DML on price | `results/heston_barrier_4way/multi_seed_widerange/`, `results/heston_barrier_4way/v0_extension/polynomial_2d_baselines.json` | (same) | `python experiments/heston_barrier_4way/run_multi_seed_widerange.py`; `python experiments/heston_barrier_4way/run_v0_extension.py` |
| BS barrier multistep n-sweep (n ∈ {2,4,8,16}) | `results/heston_barrier_4way/bs_n_sweep/` | (same) | `python experiments/heston_barrier_4way/run_bs_n_sweep.py` |
| Trig extrapolation: SIREN only architecture with $\sigma^*>0$ near-extrap | `results/extrapolation_M1/`, `results/extrapolation_M1_DA/` | `scripts/evidence/fig_sigma_star_curve.py` | `python experiments/extrapolation/pilot_periodic_extrap.py`; `python experiments/extrapolation/phase_folded_siren.py` |
| Trig extrapolation geometry-axis (360 cells) | `results/extrapolation_M2/` | `experiments/extrapolation/aggregate_m2.py` | `bash scripts/launch_m2.sh`; `python experiments/extrapolation/aggregate_m2.py` |
| Closed-form Fourier-linear: $\sigma^* = 11.83 \sigma_y$ at $K=5,\lambda=1$ | `results/closed_form_a1_a2/` | (inline in script) | `python experiments/extrapolation/closed_form_a1_a2.py` |
| Classical baselines (matched 64 % train): DML wins 50/83.6/74.7 % vs GP/KRR/RF | `results/classical_matched_split/` | `scripts/evidence/classical_baselines_summary.py` | `python scripts/run_classical_matched_split.py` |
| Compute-matched controls: vanilla_no_es ≡ vanilla; warmup ≠ extra-compute | `results/compute_matched_controls/` | (inline in `run_revision_experiments.py`) | `python scripts/run_revision_experiments.py --section compute_matched` |
| Balancer-sensitivity grid (synth + Burgers IC) | `results/balancer_sensitivity/{synthetic,burgers_ic}/` | `scripts/evidence/balancer_sensitivity_aggregator.py` | `python scripts/run_balancer_sensitivity.py` |
| Higher-order DML (orders 0/1/2 on poly_trig + trig) | `results/higher_order_dml/` (regenerate; flagged orphan) | (no aggregator on disk) | `python experiments/higher_order_dml.py` |

> Aggregator scripts that emit each table or figure live under
> `scripts/evidence/` (the prior draft tree, kept as the
> single canonical aggregator location), with a few domain-specific
> aggregators colocated under `experiments/<domain>/analyze_*.py`.
> The per-claim provenance map in `EVIDENCE/claims_registry.csv`
> records the exact `(claim_id, result_dir, producing_script)`
> tuple for every numerical claim in the paper.

---

## Code organisation

```
dml-bench/
├── dml_benchmark/        # Installable Python package (the benchmark library)
├── experiments/          # Domain-specific experiment runners (rMD17, SPY, …)
├── scripts/              # Top-level launchers for benchmark grids and ablations
├── tests/                # pytest suite (158 test definitions across 18 files)
├── results/              # Per-cell JSON corpus, organised by experiment family
├── data/                 # External datasets (SPY, rMD17, PDEBench, ERA5, …)
├── paper/                # LaTeX sources of the accompanying paper
├── EVIDENCE/             # Claim-level provenance registry (CSV + manifest)
├── papers/               # Aggregator scripts and prior-draft material
├── metadata/             # Croissant 1.1 JSON for dataset description
├── environment.yml       # Conda env pin set (creates env "dml-bench-env")
├── requirements.txt      # Flat pip pin set
├── pyproject.toml        # Package metadata
└── Makefile              # Common targets (test / smoke / analysis / figures)
```

Each top-level subdirectory has its own `README.md` documenting layout and
usage. Skim those when navigating the repository.

---

## Data access

Only the synthetic functions are generated on the fly by the benchmark. The
four real-world data sources have to be downloaded from their original
mirrors. **The detailed download and preprocessing recipe lives in
[`DATA.md`](DATA.md).** The summary:

| Source | Licence | Mirror | Local path |
|---|---|---|---|
| Synthetic functions | MIT (this repo) | generated on call | n/a |
| rMD17 | CC BY 4.0 | Figshare archive of Christensen \& von Lilienfeld (2020) | `data/rmd17/` |
| PDEBench (Burgers + Darcy) | Apache 2.0 | DaRUS deposit of Takamoto et al. (2022) | `data/pdebench/` |
| SPY EOD options | CC0 | upstream URL in App A | `data/spy_options/` |
| ERA5 Z500 | Copernicus licence (no redistribution) | Copernicus Climate Data Store | `data/era5/` |

> ERA5 cannot be redistributed under the Copernicus licence. Each user
> obtains their own access through the Copernicus CDS API; the recipe is in
> `DATA.md`.

---

## Hugging Face and Zenodo

Camera-ready will publish:

- **Hugging Face dataset** — pre-processed JSON corpus + Parquet aggregates,
  validates against the Croissant 1.1 metadata in `metadata/croissant.json`.
  Placeholder URL: `https://huggingface.co/datasets/dml-bench/XXXX` (added at
  camera-ready).
- **Zenodo deposit** — frozen tarball of the code mirror at submission tag
  with a permanent DOI. Placeholder DOI:
  `https://doi.org/10.5281/zenodo.XXXX` (added at camera-ready).

The anonymous review-period URL is
`https://anonymous.4open.science/r/dml-bench-EB7F` (URL added at submission).

---

## Citation

```bibtex
@inproceedings{anonymous2026dmlbench,
    title  = {When Does Derivative Training Help?
              A Controlled-Grid Benchmark for Derivative-Enhanced Machine Learning},
    author = {Anonymous},
    year   = {2026},
    booktitle = {NeurIPS Evaluations and Datasets Track (under review)},
    note   = {Identity withheld during double-blind review.}
}
```

---

## License

MIT. See `LICENSE` for the full text. The code in this repository is the
authors' own work; external dataset licences are listed in
[`DATA.md`](DATA.md) and in the per-source datasheet shipped in
`paper/sections/A_datasheet.tex`.
