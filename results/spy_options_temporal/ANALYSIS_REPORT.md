# SPY Real-World Options — Temporal Split Analysis

Generated: 2026-04-16 16:12:22

Results directory: `results/spy_options_temporal`

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
| Vanilla | 3.7347e-05 ± 2.12e-06 | 1.8341e-02 ± 3.64e-04 | 44 | 10 |
| DML fixed λ | 2.9599e-05 ± 1.76e-06 | 6.9795e-05 ± 4.27e-06 | 64 | 10 |
| DML GradNorm | 3.5219e-05 ± 2.21e-06 | 2.2597e-04 ± 6.13e-05 | 82 | 10 |
| DML ReLoBRaLo | 3.3422e-05 ± 1.85e-06 | 1.6700e-04 ± 5.43e-06 | 65 | 10 |
| DML Warmup | 3.7548e-05 ± 2.34e-06 | 4.5399e-04 ± 2.39e-05 | 103 | 10 |

## Results: n_train = 50,000

| Method | Value MSE (mean±std) | Grad MSE (mean±std) | Time (s) | N |
|--------|---------------------|--------------------:|--------:|--:|
| Vanilla | 3.7727e-05 ± 1.31e-06 | 1.7946e-02 ± 7.01e-04 | 237 | 10 |
| DML fixed λ | 2.9815e-05 ± 9.06e-07 | 6.6550e-05 ± 2.04e-06 | 306 | 10 |
| DML GradNorm | 3.6635e-05 ± 1.71e-06 | 2.5343e-04 ± 5.09e-05 | 291 | 10 |
| DML ReLoBRaLo | 3.4096e-05 ± 1.06e-06 | 1.6369e-04 ± 3.28e-06 | 273 | 10 |
| DML Warmup | 3.7788e-05 ± 1.30e-06 | 4.0998e-04 ± 2.02e-05 | 469 | 10 |

## DML Improvement over Vanilla

### n_train = 10,000

| Method | Value Penalty | Grad Improvement |
|--------|-------------:|----------------:|
| DML fixed λ | -20.7% | 263× |
| DML GradNorm | -5.7% | 81× |
| DML ReLoBRaLo | -10.5% | 110× |
| DML Warmup | +0.5% | 40× |

### n_train = 50,000

| Method | Value Penalty | Grad Improvement |
|--------|-------------:|----------------:|
| DML fixed λ | -21.0% | 270× |
| DML GradNorm | -2.9% | 71× |
| DML ReLoBRaLo | -9.6% | 110× |
| DML Warmup | +0.2% | 44× |

## Statistical Significance (Paired Wilcoxon)

### n_train = 10,000

| Method | p-value | Cohen's d | Effect |
|--------|--------:|----------:|--------|
| DML fixed λ | 0.0020 ** | 47.63 | large |
| DML GradNorm | 0.0020 ** | 47.35 | large |
| DML ReLoBRaLo | 0.0020 ** | 47.40 | large |
| DML Warmup | 0.0020 ** | 48.84 | large |

### n_train = 50,000

| Method | p-value | Cohen's d | Effect |
|--------|--------:|----------:|--------|
| DML fixed λ | 0.0020 ** | 24.25 | large |
| DML GradNorm | 0.0020 ** | 23.33 | large |
| DML ReLoBRaLo | 0.0020 ** | 24.10 | large |
| DML Warmup | 0.0020 ** | 24.08 | large |