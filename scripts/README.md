# `scripts/` — Top-level launchers and pre-/post-processing helpers

This directory holds:

1. The headline experiment-grid launchers (`run_full_benchmark.py` lives
   at the repo root for legacy reasons; everything else lives here).
2. Data download and preprocessing helpers for the four real-world
   sources.
3. Ablation runners (P6 / P7 / P8 / P9, balancer-sensitivity, lr-corroboration).
4. Bash wrappers around the long-running grids
   (`launch_full_rerun.sh`, `launch_post_tier_runs.sh`, `era5_pipeline_chain.sh`).

Each script is reproducer-grade: a `--smoke` flag exists where the
runtime warrants it, and the full grid is launched without arguments
beyond `--gpu` and (where applicable) `--methods`.

## Headline experiment grids

| Script | What it produces | Reproducer call |
|---|---|---|
| `run_burgers.py`                       | `results/burgers/{bare,ic}/`   (PDEBench Burgers, 4×256 + 6×512 archs, 6 methods, 20 seeds) | `python scripts/run_burgers.py --input-mode bare --archs 4x256`; same with `--input-mode ic` |
| `run_darcy.py`                         | `results/darcy/{bare,ic}/`     (PDEBench Darcy, same shape) | `python scripts/run_darcy.py --input-mode bare --archs 4x256`; same with `--input-mode ic` |
| `run_era5.py`                          | `results/era5/{bare,state}/`   (ERA5 Z500, 4×256 + 6×512, 5 seeds × 3 methods × 2 archs × 2 regimes) | `python scripts/run_era5.py --regime bare`; same with `--regime state` |
| `run_burgers_experiment.py`            | `results/burgers_1d_results.json` (single-file 1-D Burgers smoke) | `python scripts/run_burgers_experiment.py` |
| `run_basket_bachelier.py`              | `results/basket_bachelier/`    (4 dims × 7 methods × 5 seeds, H&S-canonical) | `python scripts/run_basket_bachelier.py` |
| `run_classical_matched_split.py`       | `results/classical_matched_split/`  (990 cells, classical baselines on matched 64 % train) | `python scripts/run_classical_matched_split.py` |
| `run_revision_experiments.py`          | `results/compute_matched_controls/`, `results/lambda_j_ablation/`, etc. | `python scripts/run_revision_experiments.py` |
| `run_balancer_sensitivity.py`          | `results/balancer_sensitivity/`  (synth + Burgers IC) | `python scripts/run_balancer_sensitivity.py` |
| `run_rmd17_tau_sweep.py`               | `results/rmd17_tau_sweep/`     (15 cells, 3 τ × 5 seeds on aspirin/PaiNN) | `python scripts/run_rmd17_tau_sweep.py` |
| `run_painn_salicylic.py`               | `results/molecular_painn/painn_salicylic_*.json` (salicylic-acid filename-translation reproduction) | `python scripts/run_painn_salicylic.py` |
| `run_bachelier_noise_extension.py`     | `results/tier3_benchmark/bachelier_noise_*.json` (bachelier σ-extension cells) | `python scripts/run_bachelier_noise_extension.py` |
| `run_spy_robustness.py`                | `results/spy_robustness_analysis_10k_subsample.json` | `python scripts/run_spy_robustness.py` |
| `run_fuzzy_sensitivity.py`             | `results/fuzzy_sensitivity.json` (1-D ε_mult sweep) | `python scripts/run_fuzzy_sensitivity.py` |

## Ablation runners (P6 / P7 / P8 / P9)

| Script | Sweep | Output |
|---|---|---|
| `p6_corruption_run.py`     | Label-corruption sweep (5 corruptions × 5 severities × 3 methods × 5 seeds × 5 cells/severity) | `results/p6_corruption/` |
| `p7_fuzzy_2d_run.py`       | Fuzzy-2D bandwidth sweep on 2-D barrier-BS                                                       | `results/p7_fuzzy_2d/` |
| `p8_spy_proxy_stress.py`   | SPY proxy-label stress (6 σ × methods × seeds)                                                   | `results/p8_spy_proxy_stress/` |
| `p9_dimnorm_run.py`        | Dimension-normalised GradNorm null-result (5 dims × 5 seeds × 3 methods)                         | `results/p9_dimnorm_gradnorm/` |
| `pilot_p4.py` / `pilot_p4.sh` | Phase-4 PaiNN pilot                                                                             | `results/molecular_painn/` |
| `ablation_phase1_es.py`    | Phase-1 early-stopping ablation (3 configs × 5 targets × 5 seeds)                                | `results/ablation_phase1_es/` |
| `ablation_warmup_lr.py`    | Warmup learning-rate strategy (7 strategies × 3 targets × 5 seeds)                               | `results/ablation_warmup_lr/` |
| `ablation_v1_p7_weights.py`| H&S vs 0.5/0.5 weight scheme on aspirin                                                          | `results/ablation_v1_p7/` |
| `jh2_lr_corroboration_run.py` | J-H2 corroboration: aspirin pairwise × {lr/5, lr/10, lr/20}                                    | `results/ablation_lr_corroboration/` |

## Data download and preprocessing

| Script | What it does |
|---|---|
| `download_era5.py`        | Pulls Z500 from the Copernicus CDS (requires `~/.cdsapirc`) |
| `preprocess_era5.py`      | Builds the daily snapshot cache and the 16-EOF state-augmentation table |
| `era5_go_nogo.py`         | Pre-flight sanity check for the ERA5 grid (env, data, cudnn) |
| `era5_pipeline_chain.sh`  | Bash chain that runs `download` → `preprocess` → `run_era5.py --regime bare` then `state` |
| `era5_stage2_chain.sh`    | Re-running stage 2 (preprocessing) without re-downloading |
| `darcy_sanity_check.py`   | Pre-flight sanity check for the Darcy grid |

## Aggregators (post-grid)

| Script | What it consumes / emits |
|---|---|
| `aggregate_spy_purged_cv.py` | per-fold + per-supervisor SPY summary CSV |
| `compute_win_rates.py`       | `results/tier1+2+3+4/` → win-rate tables |
| `relabel_legacy_relobralo.py`| labels legacy ReLoBRaLo runs after the dim-fix |
| `rerun_vanilla_autodiff.py`  | re-evaluates vanilla rows via autodiff (post-fix) |

## Bash launchers

| Script | Purpose |
|---|---|
| `launch_full_rerun.sh`       | Re-launch the full benchmark from a clean checkout |
| `launch_post_tier_runs.sh`   | Run the post-tier ablations (P6 / P7 / P8 / P9) after Tier 3 / 4 finish |
| `launch_continuation.sh`     | Continuation runner that picks up after `--resume` interruption |
| `launch_m2.sh`               | Trig extrapolation M2 geometry-axis launcher |
| `run_everything.sh`          | Convenience: runs every grid in this directory in dependency order |
| `run_gpu0_parallel.sh`       | Parallel-on-GPU0 helper for the smaller grids |
| `rerun_i_h5_contamination.sh`| Re-run the I-H5 GradNorm-contamination cells after the dim-fix |
| `rerun_relobralo_post_relabel.sh` | Re-run ReLoBRaLo rows after the legacy-relabel pass |
| `queue_dml_fixed_half_and_salicylic.sh` | Queue helper: dml_fixed_half + salicylic-acid recovery |

## Determinism contract

Every script calls `dml_benchmark.trainer.set_deterministic(seed)` per
cell; the seed is documented in the script header and re-derived from
the cell tuple. Wall-clock varies with hardware load; numerical fields
do not.

## Resume contract

All grid runners use `--resume` semantics: an existing JSON at the
cell's expected path is treated as completed and not re-trained. To
force re-run, delete the JSON.

## How to add a new launcher

1. Drop a new `run_<name>.py` here with a CLI matching the shape of
   `run_burgers.py` (`--gpu`, `--input-mode`, `--archs`, `--seeds`,
   `--methods`, `--smoke`).
2. Have it produce per-cell JSONs under a fresh
   `results/<name>/` subdirectory.
3. Add the row to `EVIDENCE/manifest.md` and to `EVIDENCE/claims_registry.csv`.
4. Add the corresponding aggregator under `evidence/`.
5. Update the reproducer table in the repo-root `README.md`.
