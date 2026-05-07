# SPY Real-World Options — Temporal Split Analysis

Generated: 2026-04-30 21:03:36

Results directory: `./results/spy_options_temporal_optionC`

**Total experiments:** 100
**Train sizes:** [10000, 50000]
**Methods:** ['vanilla', 'dml_fixed', 'dml_gradnorm', 'dml_relobralo', 'dml_warmup']

## Data & Split
- Split mode: **temporal**
- Train period: ≤ 2021-06-23
- Test period: ≥ 2021-07-01
- Embargo: 5 trading days (11122 samples excluded)
- Train pool: 768,972 samples
- Test pool: 796,325 samples

## Results: n_train = 10,000

| Method | Value MSE (mean±std) | Grad MSE (mean±std) | Time (s) | N |
|--------|---------------------|--------------------:|--------:|--:|
| Vanilla | 1.1630e-08 ± 4.40e-09 | 1.7900e-02 ± 4.23e-04 | 75 | 10 |
| DML fixed λ | 1.2179e-08 ± 2.22e-08 | 2.3973e-06 ± 1.41e-06 | 88 | 10 |
| DML GradNorm | 1.7005e-08 ± 1.65e-08 | 3.2808e-06 ± 2.28e-06 | 144 | 10 |
| DML ReLoBRaLo | 8.4110e-09 ± 1.06e-08 | 2.2045e-06 ± 1.16e-06 | 99 | 10 |
| DML Warmup | 2.2292e-08 ± 7.86e-09 | 3.5187e-05 ± 5.83e-06 | 130 | 10 |

## Results: n_train = 50,000

| Method | Value MSE (mean±std) | Grad MSE (mean±std) | Time (s) | N |
|--------|---------------------|--------------------:|--------:|--:|
| Vanilla | 6.3891e-10 ± 1.68e-10 | 1.7718e-02 ± 5.49e-04 | 359 | 10 |
| DML fixed λ | 2.7196e-10 ± 1.15e-10 | 1.6914e-07 ± 3.26e-08 | 442 | 10 |
| DML GradNorm | 1.5451e-09 ± 1.42e-09 | 5.2193e-07 ± 6.18e-07 | 630 | 10 |
| DML ReLoBRaLo | 1.5233e-10 ± 2.56e-11 | 1.8011e-07 ± 3.09e-08 | 508 | 10 |
| DML Warmup | 1.3398e-09 ± 2.09e-10 | 2.8924e-06 ± 3.37e-07 | 587 | 10 |

## DML Improvement over Vanilla

### n_train = 10,000

| Method | Value Penalty | Grad Improvement |
|--------|-------------:|----------------:|
| DML fixed λ | +4.7% | 7467× |
| DML GradNorm | +46.2% | 5456× |
| DML ReLoBRaLo | -27.7% | 8120× |
| DML Warmup | +91.7% | 509× |

### n_train = 50,000

| Method | Value Penalty | Grad Improvement |
|--------|-------------:|----------------:|
| DML fixed λ | -57.4% | 104753× |
| DML GradNorm | +141.8% | 33947× |
| DML ReLoBRaLo | -76.2% | 98373× |
| DML Warmup | +109.7% | 6126× |

## Statistical Significance (Paired Wilcoxon)

### n_train = 10,000

| Method | p-value | Cohen's d | Effect |
|--------|--------:|----------:|--------|
| DML fixed λ | 0.0020 ** | 40.21 | large |
| DML GradNorm | 0.0020 ** | 40.23 | large |
| DML ReLoBRaLo | 0.0020 ** | 40.18 | large |
| DML Warmup | 0.0020 ** | 40.14 | large |

### n_train = 50,000

| Method | p-value | Cohen's d | Effect |
|--------|--------:|----------:|--------|
| DML fixed λ | 0.0020 ** | 30.61 | large |
| DML GradNorm | 0.0020 ** | 30.62 | large |
| DML ReLoBRaLo | 0.0020 ** | 30.61 | large |
| DML Warmup | 0.0020 ** | 30.61 | large |