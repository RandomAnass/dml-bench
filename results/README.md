# `results/` — Per-cell JSON corpus

The source-of-truth for every numerical claim in the paper. Every cell
of every experiment writes one JSON file under
`results/<benchmark>/<descriptive_filename>.json` containing:

- the cell tuple `(family, d, n_train, sigma, method, seed_index)` (or
  the domain-specific equivalent);
- the trained model's `value_mse`, `grad_mse`, and any domain metrics
  (`force_mae` for rMD17, `geostrophic_mae` for ERA5, etc.);
- the full `training_log` (epoch, loss components, learning rate);
- training time `time_s`;
- the configuration block (architecture, optimiser, hyperparameters);
- a hash of the input data and the git commit at run time.

Aggregator scripts (`evidence/`, `papers/neurips_DB/evidence/`,
`paper/figures/scripts/`) consume these JSONs and emit the paper's
tables and figures. Re-running the aggregators on a clean corpus
regenerates every value numerically identically without retraining.

## Subdirectory map

| Subdirectory | Cells | Producing script | What it backs |
|---|---:|---|---|
| `tier1_benchmark/` | 1,261 | `run_full_benchmark.py --tier 1` | Synthetic core grid (high-d holdout) |
| `tier2_benchmark/` | 7,426 | `run_full_benchmark.py --tier 2` | Synthetic finance + noise sweep |
| `tier3_benchmark/` | 10,966 | `run_full_benchmark.py --tier 3` | Gap-fill + 1000-epoch + λ-sweep |
| `tier4_benchmark/` | 1,096 | `run_full_benchmark.py --tier 4` | Statistical-power 10-seed coverage |
| `tier5_arch_ablation/` | 30 | `run_architecture_ablation.py` | Architecture ablation (poly_trig d=10) |
| `tier5_extended_baselines/` | 160 | `run_extended_baselines.py` | KRR + RF at d=50, 100 |
| `unified_comparison/` | 550+ | `experiments/unified_comparison/run_unified_experiment.py` | Disc-payoff Study 1 (11 methods × 5 datasets × 10 seeds) |
| `spy_options_temporal/` | 100 | `experiments/real_data_spy/run_spy_temporal.py` | Pre-supervisor-split SPY temporal (legacy) |
| `spy_options_temporal_optionA/` | 120 | `experiments/real_data_spy/run_spy_experiment.py --target-mode bs_price` | SPY temporal, BS-target supervisor |
| `spy_options_temporal_optionC/` | 120 | `experiments/real_data_spy/run_spy_experiment.py --target-mode svi`      | SPY temporal, SVI-supervised |
| `spy_options_purged_cv/` | 500 | `experiments/real_data_spy/run_spy_purged_cv.py` | Pre-supervisor-split SPY purged CV (legacy) |
| `spy_options_purged_cv_optionA/` | 600 | `experiments/real_data_spy/run_spy_purged_cv.py --target-mode bs_price` | SPY 5-fold purged CV, BS-target |
| `spy_options_purged_cv_optionC/` | 600 | `experiments/real_data_spy/run_spy_purged_cv.py --target-mode svi`      | SPY 5-fold purged CV, SVI-supervised |
| `spy_regional/` | (varies) | `experiments/real_data_spy/run_regional_analysis.py` | SPY regional break-down (cited in §5.1) |
| `molecular_painn/` | 350+ | `experiments/molecular/run_painn.py` | rMD17 PaiNN (canonical equivariant baseline) |
| `molecular_mlp/` | 350 | `experiments/molecular/run_mlp_molecular.py` | rMD17 MLP-pairwise (Cartesian-force chain rule) |
| `molecular_gatv2/` | 350 | `experiments/gnn_md17.py` | rMD17 GATv2 backbone |
| `rmd17_tau_sweep/` | 15 | `scripts/run_rmd17_tau_sweep.py` | rMD17 PaiNN τ-sensitivity (3 τ × 5 seeds, aspirin) |
| `burgers/{bare,ic}/` | 190 + 120 | `scripts/run_burgers.py` | PDEBench Burgers value-MSE grid |
| `darcy/{bare,ic}/` | 120 + 120 | `scripts/run_darcy.py` | PDEBench Darcy value-MSE grid |
| `era5/{bare,state}/` | 30 + 30 | `scripts/run_era5.py` | ERA5 Z500 (4×256 + 6×512 archs, 2 regimes) |
| `era5_pilot_2019/` | 15 | `scripts/run_era5.py --pilot 2019` | Deprecated 2019-only pilot |
| `extrapolation_M1/` | CSV-only | `experiments/extrapolation/pilot_periodic_extrap.py` | Trig-extrap architecture-axis (4 archs × 18 σ × 5 seeds) |
| `extrapolation_M1_DA/` | 2 shards | `experiments/extrapolation/domain_adapt.py` | Trig-extrap OOS-grad augmentation |
| `extrapolation_M1_DA_smoke/` | (smoke) | (same with `--smoke`) | Smoke artefacts (do not use) |
| `extrapolation_M1_DA_regression_test/` | (regression) | `experiments/extrapolation/_regression_check_da.py` | Regression-tests for the DA reproducer |
| `extrapolation_M2/` | 360 | `experiments/extrapolation/run_m2.py` (or `bash scripts/launch_m2.sh`) | Trig-extrap geometry-axis |
| `closed_form_a1_a2/` | 1 JSON + 2 CSVs | `experiments/extrapolation/closed_form_a1_a2.py` | Fourier-linear closed-form theory verification |
| `heston_barrier_4way/` | 463 across 9 sub-dirs | `experiments/heston_barrier_4way/run_*.py` | Heston path-dependent barrier (narrow/wide/V0/n-sweep/ε/CG/GK/BS-repro/noise-curr) |
| `polynomial_baselines/` | (varies) | `experiments/polynomial_baselines/run_*.py` | Quintic-polynomial-in-spot comparator |
| `classical_matched_split/` | 990 | `scripts/run_classical_matched_split.py` | GP/KRR/RF on matched 64 % train (Appendix F) |
| `balancer_sensitivity/synthetic/` | 240 | `scripts/run_balancer_sensitivity.py` | GradNorm α + ReLoBRaLo τ on 4 datasets |
| `balancer_sensitivity/burgers_ic/` | 40 | (same) | Burgers-IC balancer sensitivity (App D) |
| `p6_corruption/` | 750 (regenerate) | `scripts/p6_corruption_run.py` | Label-corruption sweep (App D §p6-corruption) |
| `p7_fuzzy_2d/` | 105 | `scripts/p7_fuzzy_2d_run.py` | Fuzzy-2D bandwidth sweep on 2-D barrier-BS |
| `p8_spy_proxy_stress/` | 400 | `scripts/p8_spy_proxy_stress.py` | SPY proxy-label stress (6 σ × methods × seeds) |
| `p9_dimnorm_gradnorm/` | 75 | `scripts/p9_dimnorm_run.py` | Dim-normalised GradNorm null-result |
| `higher_order_dml/` | 100 (regenerate) | `experiments/higher_order_dml.py` | Higher-order DML orders 0/1/2 |

## Per-cell JSON schema (abridged)

```json
{
  "cell": {
    "family": "poly_trig", "d": 10, "n_train": 1024, "sigma": 0.0,
    "method": "dml_fixed", "seed_index": 3
  },
  "config": {
    "architecture": "4x256_softplus", "lr": 0.005, "patience": 50,
    "optimizer": "adam", "lr_schedule": "ReduceLROnPlateau"
  },
  "result": {
    "value_mse": 1.05e-3, "grad_mse": 1.27e-3,
    "test_value_mse": 1.10e-3, "test_grad_mse": 1.32e-3,
    "time_s": 14.3
  },
  "training_log": [ /* per-epoch log, len up to n_epochs */ ],
  "git_commit": "dc27c606...",
  "data_hash": "sha256:..."
}
```

The exact schema is what each runner emits via `json.dump(...)`; the
`dml_benchmark/io.py` helpers (`load_result_json`, `iter_result_jsons`)
read it back.

## Naming convention

`results/<benchmark>/<family>_d<d>_n<n>_sigma<sigma>_<method>_seed<seed>.json`

with the `<family>` and `<sigma>` fields elided where the benchmark
fixes them (e.g. SPY only has one "family", so the SPY filenames omit
it). Every script documents the exact name format inline.

## How to consume

Read with `dml_benchmark.io.load_result_json` (returns a dict) or
iterate with `iter_result_jsons`. Aggregate across cells with the
helper functions in `dml_benchmark/metrics.py:ResultsManager`.

## Determinism contract

The runners do **not** write partial JSON; the final `json.dump`
happens after the cell completes and the temporary file is renamed
atomically. Re-running a cell with the same seed on the same hardware
revision produces a JSON whose `value_mse` and `grad_mse` fields are
bit-identical (verified within the lab's RTX A6000 cluster). The
`time_s` field is hardware-dependent and not numerically reproducible
across machines.

## Backups (local only, NOT committed)

| Directory | Size | Purpose |
|---|---|---|
| `results/_pre_autodiff_full_backup_20260406/` | 147 MB | Snapshot of all results before the autodiff fix |
| `results/_vanilla_zeros_backup/` | 27 MB | Snapshot of just the vanilla-zeros era |

These backups exist to support diff/audit during the H&S autodiff-fix
investigation; they are not part of the main benchmark and are not
shipped in the supplementary tarball.
