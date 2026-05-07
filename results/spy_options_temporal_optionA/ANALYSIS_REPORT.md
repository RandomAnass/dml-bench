# SPY Real-World Options — Temporal Split Analysis

Generated: 2026-04-30 21:03:33

Results directory: `./results/spy_options_temporal_optionA`

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
| Vanilla | 1.7231e-08 ± 7.83e-09 | 1.7755e-02 ± 4.31e-04 | 71 | 10 |
| DML fixed λ | 6.8274e-09 ± 7.57e-09 | 2.3862e-06 ± 8.09e-07 | 86 | 10 |
| DML GradNorm | 2.9844e-08 ± 6.32e-08 | 6.2588e-06 ± 5.99e-06 | 127 | 10 |
| DML ReLoBRaLo | 8.6204e-09 ± 1.77e-08 | 2.5757e-06 ± 2.30e-06 | 96 | 10 |
| DML Warmup | 2.7501e-08 ± 5.75e-09 | 5.0279e-05 ± 7.02e-06 | 122 | 10 |

## Results: n_train = 50,000

| Method | Value MSE (mean±std) | Grad MSE (mean±std) | Time (s) | N |
|--------|---------------------|--------------------:|--------:|--:|
| Vanilla | 9.9267e-10 ± 1.02e-09 | 1.7602e-02 ± 5.53e-04 | 363 | 10 |
| DML fixed λ | 5.8706e-10 ± 4.89e-10 | 2.2085e-07 ± 8.54e-08 | 440 | 10 |
| DML GradNorm | 6.3246e-09 ± 7.55e-09 | 7.7620e-07 ± 7.95e-07 | 631 | 10 |
| DML ReLoBRaLo | 2.1571e-10 ± 9.82e-11 | 2.1746e-07 ± 3.15e-08 | 501 | 10 |
| DML Warmup | 1.7779e-09 ± 1.20e-09 | 3.3410e-06 ± 7.29e-07 | 612 | 10 |

## DML Improvement over Vanilla

### n_train = 10,000

| Method | Value Penalty | Grad Improvement |
|--------|-------------:|----------------:|
| DML fixed λ | -60.4% | 7440× |
| DML GradNorm | +73.2% | 2837× |
| DML ReLoBRaLo | -50.0% | 6893× |
| DML Warmup | +59.6% | 353× |

### n_train = 50,000

| Method | Value Penalty | Grad Improvement |
|--------|-------------:|----------------:|
| DML fixed λ | -40.9% | 79703× |
| DML GradNorm | +537.1% | 22677× |
| DML ReLoBRaLo | -78.3% | 80946× |
| DML Warmup | +79.1% | 5269× |

## Statistical Significance (Paired Wilcoxon)

### n_train = 10,000

| Method | p-value | Cohen's d | Effect |
|--------|--------:|----------:|--------|
| DML fixed λ | 0.0020 ** | 39.07 | large |
| DML GradNorm | 0.0020 ** | 39.30 | large |
| DML ReLoBRaLo | 0.0020 ** | 39.00 | large |
| DML Warmup | 0.0020 ** | 38.72 | large |

### n_train = 50,000

| Method | p-value | Cohen's d | Effect |
|--------|--------:|----------:|--------|
| DML fixed λ | 0.0020 ** | 30.20 | large |
| DML GradNorm | 0.0020 ** | 30.23 | large |
| DML ReLoBRaLo | 0.0020 ** | 30.20 | large |
| DML Warmup | 0.0020 ** | 30.20 | large |