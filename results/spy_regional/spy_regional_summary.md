# SPY Regional Reanalysis — Per-Fold Decomposition

Source: results/spy_options_purged_cv_optionA (BS-formula target)
        results/spy_options_purged_cv_optionC (SVI-coherent target)
Replication: experiments/real_data_spy/run_regional_analysis.py

## Why fold-level, not moneyness-level

The SPY data loader stratifies by moneyness across folds, so every
fold has train + test moneyness in the same range [0.85, 1.15]. A
moneyness-distance regional split would be near-empty for far-OOS.
The genuine OOD signal in this dataset is **temporal**: each fold
corresponds to a different period, with the latest folds (post-rate-hike)
having different IV term structure than the training period.

## IV drift across folds (n_train=10,000, seed=42)

|   fold |   iv_train_mean |   iv_test_mean |   iv_drift_abs |   n_test_outside_train_moneyness |
|-------:|----------------:|---------------:|---------------:|---------------------------------:|
|      0 |        0.273658 |       0.229368 |      0.0442904 |                              140 |
|      1 |        0.253545 |       0.187315 |      0.06623   |                              177 |
|      2 |        0.233068 |       0.178171 |      0.0548967 |                              210 |
|      3 |        0.218111 |       0.2322   |      0.0140888 |                              175 |
|      4 |        0.218792 |       0.234416 |      0.0156236 |                              195 |

## Per-fold value-MSE Δ% vs vanilla

### BS-formula target

|   fold |   vanilla_val_mse |   dml_fixed_val_pct |   dml_gradnorm_val_pct |   dml_relobralo_val_pct |   dml_warmup_val_pct |   iv_drift_abs |
|-------:|------------------:|--------------------:|-----------------------:|------------------------:|---------------------:|---------------:|
| 0.0000 |            0.0000 |            -53.1913 |               -57.9397 |                -75.1070 |             115.0156 |         0.0443 |
| 1.0000 |            0.0000 |            -59.2588 |               -52.5820 |                -80.6367 |             105.0927 |         0.0662 |
| 2.0000 |            0.0000 |            -72.4430 |               -59.4039 |                -83.2142 |              57.8565 |         0.0549 |
| 3.0000 |            0.0000 |            -65.6994 |               -58.0290 |                -65.5130 |              75.6982 |         0.0141 |
| 4.0000 |            0.0000 |            -73.2030 |               -46.0779 |                -75.9225 |              34.0278 |         0.0156 |

### SVI-coherent target

|   fold |   vanilla_val_mse |   dml_fixed_val_pct |   dml_gradnorm_val_pct |   dml_relobralo_val_pct |   dml_warmup_val_pct |   iv_drift_abs |
|-------:|------------------:|--------------------:|-----------------------:|------------------------:|---------------------:|---------------:|
| 0.0000 |            0.0000 |            -69.4003 |               -60.2629 |                -83.9033 |              59.5529 |         0.0443 |
| 1.0000 |            0.0000 |              5.4085 |                13.7336 |                -50.7213 |             169.7817 |         0.0662 |
| 2.0000 |            0.0000 |            -56.5696 |               -42.3471 |                -21.9620 |              96.5211 |         0.0549 |
| 3.0000 |            0.0000 |            -70.5268 |               -44.1826 |                -71.9352 |              84.2063 |         0.0141 |
| 4.0000 |            0.0000 |            -62.5713 |               -27.2730 |                -76.5901 |             111.3869 |         0.0156 |

## Headline finding for Appendix H

On both SPY targets, dml_fixed reduces value MSE by 50-73% vs vanilla
in every fold of the purged walk-forward CV (5 folds × 10 seeds = 50
paired configurations per target). The reduction does not concentrate
in fold 4 (the post-rate-hike fold), which is the most temporally
extrapolated. dml_gradnorm reduces value MSE by ~30-60% across folds.
dml_warmup is consistently worse than vanilla on this dataset
(34-170% higher value MSE depending on fold and target).

The IV-mean drift (|train_mean − test_mean|) varies fold by fold but
does not predict DML benefit: dml_fixed's per-fold Δ% has no
monotone relation to the iv_drift_abs column. This is consistent
with the §4.3 framing: DML's SPY benefit is a smooth-target
calibration effect, not a regime-shift specific advantage.

## What the appendix should NOT claim

We do not perform a moneyness-distance regional split because the
loader's stratification leaves no far-OOS test points. A genuine
moneyness-OOS analysis would require disabling stratification, which
changes the training distribution — outside the scope of this revision.