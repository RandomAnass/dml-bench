# `experiments/real_data_spy/` — SPY EOD options

The real-world finance pillar (§5.1). Tests DML on a panel of
1,576,419 SPY end-of-day option quotes spanning January 2020 to
December 2022 (758 trading days) with **two supervisors**:

- **Option A (`bs_price`):** the row's IV is used both to compute the
  Black-Scholes price target *and* the analytical greeks. Target and
  gradient share a generative model.
- **Option C (`svi`):** the IV is replaced by a per-(date, maturity)
  SVI fit; the resulting smile-coherent target is paired with matched
  analytical greeks.

The earlier "market-mid" Option B (target = mid quote, gradient = BS
greeks at observed IV) is retired because it violates the
target–gradient consistency constraint.

## Output corpus

| Sub-dir | Cells | Producing script | Split |
|---|---:|---|---|
| `results/spy_options_temporal_optionA/`     | 120 | `run_spy_experiment.py --target-mode bs_price` | Temporal (train ≤ 2021-06, test ≥ 2021-07, 5-day embargo) |
| `results/spy_options_temporal_optionC/`     | 120 | `run_spy_experiment.py --target-mode svi`      | Temporal, same dates |
| `results/spy_options_purged_cv_optionA/`    | 600 | `run_spy_purged_cv.py --target-mode bs_price`  | 5-fold purged walk-forward CV with embargo (López de Prado 2018) |
| `results/spy_options_purged_cv_optionC/`    | 600 | `run_spy_purged_cv.py --target-mode svi`       | Same CV scheme |
| `results/spy_options_temporal/`             | 100 | (legacy pre-supervisor-split runner)             | Pre-revision |
| `results/spy_options_purged_cv/`            | 500 | (legacy)                                        | Pre-revision |
| `results/spy_regional/`                     | (varies) | `run_regional_analysis.py`                  | Regional break-down (cited at §5.1) |

## Headline findings

- **Temporal central cells:** DML cuts test value MSE 1.7×–2.5× at both
  training sizes ($n_\mathrm{train} \in \{10\,000, 50\,000\}$) and both
  supervisors. Wilcoxon $p \le 0.002$ across DML methods.
- **Purged-CV BS-target per-fold deltas:** $-69, -44, -48, -39, +2879$ %
  across folds 0–4 (the +2879 % fold-4 outlier is documented in App).
- **Purged-CV SVI-supervised per-fold deltas:** $-70, -37, -53, -22, +47$ %.
- **Per-method spread on temporal:** balancer-specific; 3 of 6 DML
  variants lose to vanilla on at least one (size, supervisor) cell.

## Scripts

| Script | Purpose |
|---|---|
| `run_spy_experiment.py`        | Temporal-split runner. `--target-mode {bs_price, svi}`. |
| `run_spy_purged_cv.py`         | 5-fold purged walk-forward CV runner. |
| `run_regional_analysis.py`     | Regional break-down (per-strike-bucket / per-expiry-bucket). |
| `analyze_spy_temporal.py`      | Aggregator → temporal-split summary CSV. |
| `analyze_spy_purged_cv.py`     | Aggregator → per-fold + per-supervisor summary CSV. |
| `calibrate_svi.py`             | SVI calibration entry point (per-(date, maturity) Bilateral SVI fit). |
| `svi_calibration.py`           | SVI calibration core (Gatheral & Jacquier). |
| `spy_data_loader.py`           | Filtering and tuple-panel construction from the raw CSV archive. |
| `spy_perturbations.py`         | Robustness perturbations (used by `scripts/run_spy_robustness.py`). |

## Running

```bash
# Preprocess once (reads from data/spy_options/raw/, writes data/spy_options/spy_processed.npz)
python experiments/real_data_spy/spy_data_loader.py \
    --raw-dir data/spy_options/raw \
    --out      data/spy_options/spy_processed.npz

# SVI calibration (required for Option C)
python experiments/real_data_spy/calibrate_svi.py \
    --in       data/spy_options/spy_processed.npz \
    --out-dir  data/spy_options/

# Temporal split, both supervisors:
python experiments/real_data_spy/run_spy_experiment.py --gpu 0 --target-mode bs_price
python experiments/real_data_spy/run_spy_experiment.py --gpu 0 --target-mode svi

# Purged 5-fold CV, both supervisors:
python experiments/real_data_spy/run_spy_purged_cv.py --gpu 0 --target-mode bs_price
python experiments/real_data_spy/run_spy_purged_cv.py --gpu 0 --target-mode svi
```

## Data prerequisites

See `../../DATA.md`. Briefly: place the Kaggle SPY EOD archive under
`data/spy_options/raw/` and run the two preprocessing steps above.

## Aggregators

`evidence/spy_summary.py` (and `papers/neurips_DB/evidence/...`)
emit `tab:spy-bs-temporal`, `tab:spy-svi-temporal`, and
`fig:spy-purged-cv-per-fold`.

## Tests

`tests/test_svi_calibration.py` covers the SVI fit (auto-skips if the
preprocessed npz is missing).
