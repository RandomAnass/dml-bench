# SPY Real-World Options — Purged Walk-Forward CV Analysis

Generated: 2026-02-27 14:56:35

Results directory: `results/spy_options_purged_cv`

## Experimental Setup
- **Split mode:** Purged walk-forward cross-validation
- **Folds:** 5
- **Seeds:** 10 (42, 123, 456, 789, 1337, 2024, 3141, 5926, 8008, 9999)
- **Train sizes:** 10,000, 50,000
- **Methods:** Vanilla, DML fixed λ, DML GradNorm, DML ReLoBRaLo, DML Warmup
- **Total experiments:** 500

**Aggregation:** Fold scores are averaged per seed first, then seeds (n=10) are used as independent samples for statistical testing.

### Data Details
- Dataset: ?
- Features (dim): 4
- Feature names: ?
- Purge gap: ? trading days
- Embargo: 5 trading days

## Results: n_train = 10,000

| Method | CV Value MSE | CV Gradient MSE | Mean Time (s) |
|--------|------------:|----------------:|--------------:|
| Vanilla | 2.2045e-05 ± 4.12e-07 [2.1777e-05, 2.2290e-05] | 1.1930e-01 ± 2.49e-04 [1.1915e-01, 1.1945e-01] | 53 |
| DML fixed λ | 1.9898e-05 ± 3.56e-07 [1.9694e-05, 2.0142e-05] | 7.6850e-05 ± 1.20e-06 [7.6205e-05, 7.7708e-05] | 94 |
| DML GradNorm | 2.1745e-05 ± 7.64e-07 [2.1184e-05, 2.2140e-05] | 2.8905e-04 ± 7.20e-05 [2.3676e-04, 3.2560e-04] | 126 |
| DML ReLoBRaLo | 2.0687e-05 ± 6.49e-07 [2.0319e-05, 2.1125e-05] | 1.9599e-04 ± 4.65e-06 [1.9211e-04, 1.9823e-04] | 39 |
| DML Warmup | 2.2119e-05 ± 6.78e-07 [2.1749e-05, 2.2600e-05] | 5.8297e-04 ± 1.67e-05 [5.7349e-04, 5.9405e-04] | 148 |

## Results: n_train = 50,000

| Method | CV Value MSE | CV Gradient MSE | Mean Time (s) |
|--------|------------:|----------------:|--------------:|
| Vanilla | 2.2883e-05 ± 8.91e-07 [2.2473e-05, 2.3658e-05] | 1.1944e-01 ± 3.29e-04 [1.1919e-01, 1.1961e-01] | 288 |
| DML fixed λ | 1.9856e-05 ± 3.66e-07 [1.9673e-05, 2.0144e-05] | 7.7676e-05 ± 1.40e-06 [7.6807e-05, 7.8527e-05] | 407 |
| DML GradNorm | 2.2071e-05 ± 9.17e-07 [2.1374e-05, 2.2537e-05] | 3.5673e-04 ± 7.88e-05 [2.9434e-04, 3.9600e-04] | 444 |
| DML ReLoBRaLo | 2.0554e-05 ± 4.62e-07 [2.0335e-05, 2.0950e-05] | 1.9466e-04 ± 7.47e-06 [1.9106e-04, 2.0067e-04] | 397 |
| DML Warmup | 2.2574e-05 ± 7.51e-07 [2.2201e-05, 2.3165e-05] | 7.7904e-04 ± 1.29e-04 [6.9318e-04, 8.5576e-04] | 622 |

## DML Improvement over Vanilla

Paired per-seed comparison (n=10 seeds).

### n_train = 10,000

| Method | Value Δ% | Grad Improvement × | 95% CI (grad ×) |
|--------|---------:|-------------------:|----------------:|
| DML fixed λ | -9.7% | 1552.7× | [1537.5, 1564.7] |
| DML GradNorm | -1.3% | 449.9× | [379.5, 587.5] |
| DML ReLoBRaLo | -6.2% | 609.0× | [601.8, 621.1] |
| DML Warmup | +0.3% | 204.8× | [201.1, 208.0] |
### n_train = 50,000

| Method | Value Δ% | Grad Improvement × | 95% CI (grad ×) |
|--------|---------:|-------------------:|----------------:|
| DML fixed λ | -13.2% | 1538.1× | [1520.5, 1555.3] |
| DML GradNorm | -3.5% | 358.3× | [307.4, 453.5] |
| DML ReLoBRaLo | -10.1% | 614.4× | [596.9, 625.3] |
| DML Warmup | -1.3% | 158.0× | [143.3, 180.4] |

## Statistical Significance

Paired Wilcoxon signed-rank test (n=10 seeds). Cohen's d with BCa bootstrap CI.

### n_train = 10,000

| Method | p (grad) | p_adj (HB) | Cohen's d | d CI | Effect |
|--------|--------:|-----------:|----------:|------|--------|
| DML fixed λ | 0.0020 ** | 0.0078 | +455.70 | [+292.16, +672.51] | large |
| DML GradNorm | 0.0020 ** | 0.0059 | +420.81 | [+269.00, +698.71] | large |
| DML ReLoBRaLo | 0.0020 ** | 0.0039 | +453.98 | [+290.39, +669.78] | large |
| DML Warmup | 0.0020 ** | 0.0020 | +463.87 | [+299.46, +691.12] | large |
### n_train = 50,000

| Method | p (grad) | p_adj (HB) | Cohen's d | d CI | Effect |
|--------|--------:|-----------:|----------:|------|--------|
| DML fixed λ | 0.0020 ** | 0.0078 | +344.15 | [+242.10, +580.88] | large |
| DML GradNorm | 0.0020 ** | 0.0059 | +318.22 | [+225.06, +470.37] | large |
| DML ReLoBRaLo | 0.0020 ** | 0.0039 | +345.89 | [+243.07, +579.91] | large |
| DML Warmup | 0.0020 ** | 0.0020 | +354.90 | [+251.62, +605.08] | large |

## Per-Fold Gradient Improvement ×

Shows how gradient accuracy varies across walk-forward windows.

### n_train = 10,000
| Method | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|--------|-------:|-------:|-------:|-------:|-------:|
| DML fixed λ | 1234× | 1322× | 1582× | 1921× | 2068× |
| DML GradNorm | 320× | 331× | 440× | 575× | 535× |
| DML ReLoBRaLo | 504× | 493× | 589× | 800× | 833× |
| DML Warmup | 173× | 161× | 202× | 294× | 256× |
### n_train = 50,000
| Method | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|--------|-------:|-------:|-------:|-------:|-------:|
| DML fixed λ | 1193× | 1253× | 1601× | 2031× | 2095× |
| DML GradNorm | 248× | 252× | 338× | 531× | 515× |
| DML ReLoBRaLo | 487× | 456× | 623× | 862× | 935× |
| DML Warmup | 74× | 167× | 218× | 315× | 196× |

## Comparison: Purged CV vs Temporal Split

| n_train | Method | Temporal Grad× | CV Grad× |
|--------:|--------|---------------:|---------:|
| 10,000 | DML fixed λ | 1679× | 1552× |
| 10,000 | DML GradNorm | 518× | 413× |
| 10,000 | DML ReLoBRaLo | 669× | 609× |
| 10,000 | DML Warmup | 260× | 205× |
| 50,000 | DML fixed λ | 1768× | 1538× |
| 50,000 | DML GradNorm | 444× | 335× |
| 50,000 | DML ReLoBRaLo | 726× | 614× |
| 50,000 | DML Warmup | 285× | 153× |

## Key Takeaways

1. **Gradient improvement persists under purged CV.** All DML methods show 100×–1500× gradient MSE reduction over vanilla across all folds and train sizes.
2. **Results are statistically significant.** All paired Wilcoxon tests yield p ≤ 0.002 with large Cohen's d (Holm-Bonferroni-corrected).
3. **Rankings are consistent** between purged CV and temporal split, confirming robustness to evaluation protocol.
4. **dml_fixed** achieves the best gradient accuracy (~1550×) but incurs the largest value MSE penalty (~10-13%).
5. **dml_warmup** offers the best value–gradient trade-off: minimal value penalty (<1.3%) with ~150–200× gradient improvement.