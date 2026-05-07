# `experiments/` — Domain-specific experiment runners

Each subdirectory is a self-contained runner for one experiment family
(or one paper section). Runners produce per-cell JSONs under
`results/<benchmark>/`; the JSONs are the source-of-truth for the
paper's tables and figures. Aggregator scripts that turn the JSONs
into tables live under `evidence/` (or `papers/neurips_DB/evidence/`
in the current draft tree).

## Layout

| Subdirectory | Domain | Output corpus | Headline result |
|---|---|---|---|
| `extrapolation/`        | Trigonometric extrapolation (M1 architecture-axis, M2 geometry-axis, M1-DA OOS-grad augmentation, closed-form Fourier-linear) | `results/extrapolation_M1/`, `results/extrapolation_M2/`, `results/extrapolation_M1_DA/`, `results/closed_form_a1_a2/` | SIREN is the only architecture with $\sigma^* > 0$ near-extrap |
| `heston_barrier_4way/`  | Heston path-dependent barrier (narrow / wide / V0 ext / BS-barrier n-sweep / fuzzy ε / CG variance / GK replication / BS reproduction / noise curriculum) | `results/heston_barrier_4way/*` | narrow: every DML beats vanilla; wide: polynomial beats every DML on price |
| `hs_comparison/`        | Huge & Savine (H&S) reproduction: compute-matched controls and per-coordinate $\lambda_j$ ablation | `results/compute_matched_controls/`, `results/lambda_j_ablation/` | warmup gain is mechanism, not extra-compute |
| `lrm_comparison/`       | LRM label paradigm: heston-LRM and LRM-vs-adaptive-balancer | `results/lrm_comparison/` | LRM noise-floor scales with $1/\sigma$ |
| `molecular/`            | rMD17 force learning: PaiNN (canonical) and MLP-pairwise (Cartesian-force chain rule) | `results/molecular_painn/`, `results/molecular_mlp/` | native_EF rank 1/6 across all backbones, Friedman $p=4.8\times10^{-9}$ |
| `polynomial_baselines/` | Quintic-polynomial-in-spot baselines on synthetic and SPY (the missing-from-literature comparator) | `results/polynomial_baselines/` | beats every DML on Heston barrier narrow |
| `proposed_method/`      | Two-phase warmup as a noise-curriculum (vanilla → DML) | `results/warmup_experiments/` | 1.1–72.3× grad improvement over vanilla |
| `real_data_spy/`        | SPY EOD options: temporal split (Option A BS-target, Option C SVI-supervised) and 5-fold purged CV | `results/spy_options_temporal_optionA,C/`, `results/spy_options_purged_cv_optionA,C/` | 1.7–2.5× value-MSE cut on temporal central cells |
| `unified_comparison/`   | Discontinuous payoffs: 11 methods × 5 datasets × 10 seeds | `results/unified_comparison/multi_seed/` | warmup-fuzzy ranks 1.8/12 on disc-payoff CD |
| `gnn_md17.py` (loose)   | rMD17 GATv2 backbone (legacy filename, post-fix corpus at `results/molecular_gatv2/`) | `results/molecular_gatv2/` | baseline GNN comparator for `tab:rmd17-cross-arch` |
| `higher_order_dml.py` (loose) | Higher-order DML orders 0/1/2 on poly_trig + trig | `results/higher_order_dml/` | order 2 cuts MSE 1.2–17× on smooth (target dir is regenerate-on-demand) |
| `learned_derivatives.py` (loose) | Teacher-student protocol: DML with autograd gradients from a vanilla teacher | `results/learned_derivatives/` | teacher-grad captures 54–100 % of true-DML benefit |
| `gradient_noise_sweep.py` (loose) | Crossover-detection sweep on synthetic | `results/gradient_noise_sweep/` | per-family $\sigma^*$ |
| `analyze_new_experiments.py` (loose) | Top-level aggregator entry point pre-revision | (writes summary CSVs) | (legacy) |
| `plot_extensions.py` (loose) | Extension-experiment plotting | `figures/extensions/` | (legacy) |
| `plot_noise_sweep.py` (loose) | Noise-sweep plotting | `figures/noise_*.pdf` | (legacy) |

## How to read a runner

Every runner follows the same shape:

1. Parse a small CLI (`--gpu`, `--methods`, `--seeds`, `--smoke`).
2. Materialise the data (synthetic = on the fly; real-world = read
   from `data/<source>/`).
3. Loop over the cell tuple
   `(family, d, n_train, sigma, method, seed_index)` — order varies by
   experiment.
4. Call `dml_benchmark.trainer.train_single_experiment` (or a domain
   wrapper for non-MLP backbones — PaiNN, GATv2, MLP-pairwise).
5. Write per-cell JSON under `results/<benchmark>/`. The JSON write
   is atomic (`json.dump` to tmp + rename); existence at the expected
   path is treated as completed under `--resume`.

## Resume contract

All runners are idempotent under `--resume` (default on for the
production grids). An existing JSON at the cell's expected path is
treated as completed and not re-trained. Removing the JSON triggers a
re-run that produces a numerically identical JSON on the same hardware
revision.

## Determinism

Every runner calls `dml_benchmark.trainer.set_deterministic(seed)`
before the first network init; the seed is derived from the cell tuple
in a documented way per experiment.

## Domain entry-point cheat sheet

```bash
# Synthetic core grids
python scripts/run_full_benchmark.py --tier 3
python scripts/run_full_benchmark.py --tier 4

# Discontinuous payoffs (Study 1, 11 methods)
python experiments/unified_comparison/run_unified_experiment.py --mode multi_seed

# Heston barrier (Study 2 sub-pillar)
python experiments/heston_barrier_4way/run_multi_seed.py
python experiments/heston_barrier_4way/run_multi_seed_widerange.py

# rMD17 (10 mol × 5 split × 7 method)
python experiments/molecular/run_painn.py
python experiments/molecular/run_mlp_molecular.py
python experiments/gnn_md17.py

# SPY temporal + purged CV (both supervisors)
python experiments/real_data_spy/run_spy_experiment.py --target-mode bs_price
python experiments/real_data_spy/run_spy_experiment.py --target-mode svi
python experiments/real_data_spy/run_spy_purged_cv.py --target-mode bs_price
python experiments/real_data_spy/run_spy_purged_cv.py --target-mode svi

# PDE Burgers + Darcy (bare and IC regimes)
python scripts/run_burgers.py --input-mode bare --archs 4x256
python scripts/run_burgers.py --input-mode ic   --archs 4x256
python scripts/run_darcy.py   --input-mode bare --archs 4x256
python scripts/run_darcy.py   --input-mode ic   --archs 4x256

# ERA5 Z500 (bare and state-augmented)
python scripts/run_era5.py --regime bare
python scripts/run_era5.py --regime state

# Trig extrapolation (M1 + M2 + DA + closed-form)
python experiments/extrapolation/pilot_periodic_extrap.py
bash   scripts/launch_m2.sh
python experiments/extrapolation/aggregate_m2.py
python experiments/extrapolation/closed_form_a1_a2.py
```

## Adding a new domain

1. Create `experiments/<new_domain>/` and a `run_<new_domain>.py` with
   the standard CLI (`--gpu`, `--methods`, `--seeds`, `--resume`).
2. Have it write to `results/<new_domain>/` with the per-cell JSON
   schema (see `dml_benchmark/io.py`).
3. Add the corresponding aggregator under `evidence/`.
4. Add a row to `EVIDENCE/manifest.md` and to `EVIDENCE/claims_registry.csv`.
5. Add a row to the reproducer table in the top-level `README.md`.
