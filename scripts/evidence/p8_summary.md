# P8 SPY proxy-label stress test — summary

Total runs aggregated: **400**.

Each cell shows (value MSE mean ± std) / (grad MSE mean ± std) over 5 seeds.

## Axis: `additive`

### Level: `additive_e0.05`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 2.98 ± 0.13 | 7.17 ± 0.25 |
| dml_gradnorm | 5 | 3.18 ± 0.39 | 13.57 ± 14.53 |
| dml_relobralo | 5 | 3.43 ± 0.27 | 17.61 ± 2.65 |
| dml_warmup | 5 | 3.88 ± 0.23 | 44.44 ± 1.18 |

### Level: `additive_e0.1`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.07 ± 0.27 | 8.58 ± 1.52 |
| dml_gradnorm | 5 | 3.43 ± 0.53 | 21.09 ± 20.83 |
| dml_relobralo | 5 | 3.49 ± 0.31 | 17.49 ± 2.00 |
| dml_warmup | 5 | 3.87 ± 0.21 | 44.79 ± 1.61 |

### Level: `additive_e0.2`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.17 ± 0.76 | 12.18 ± 2.36 |
| dml_gradnorm | 5 | 4.87 ± 2.89 | 120.86 ± 95.79 |
| dml_relobralo | 5 | 3.78 ± 0.43 | 24.90 ± 4.08 |
| dml_warmup | 5 | 3.87 ± 0.20 | 43.70 ± 0.98 |

### Level: `additive_e0.5`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.84 ± 0.77 | 34.72 ± 9.05 |
| dml_gradnorm | 5 | 11.39 ± 4.48 | 341.62 ± 74.88 |
| dml_relobralo | 5 | 6.02 ± 2.14 | 101.51 ± 65.17 |
| dml_warmup | 5 | 3.84 ± 0.22 | 44.16 ± 1.08 |

## Axis: `baseline`

### Level: `baseline`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.03 ± 0.14 | 6.88 ± 0.22 |
| dml_gradnorm | 5 | 3.54 ± 0.49 | 18.07 ± 11.28 |
| dml_relobralo | 5 | 3.51 ± 0.24 | 16.57 ± 0.70 |
| dml_warmup | 5 | 3.85 ± 0.23 | 44.59 ± 1.74 |

## Axis: `combined`

### Level: `combined_d0.05_a0.1`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 4.41 ± 0.09 | 93.70 ± 5.30 |
| dml_gradnorm | 5 | 5.21 ± 1.84 | 89.13 ± 11.72 |
| dml_relobralo | 5 | 3.80 ± 0.33 | 89.81 ± 4.56 |
| dml_warmup | 5 | 3.84 ± 0.25 | 52.06 ± 2.01 |

### Level: `combined_k10_a0.1`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 2.95 ± 0.22 | 7.39 ± 0.66 |
| dml_gradnorm | 5 | 3.81 ± 1.28 | 29.28 ± 24.26 |
| dml_relobralo | 5 | 3.42 ± 0.49 | 19.88 ± 2.50 |
| dml_warmup | 5 | 3.85 ± 0.22 | 43.92 ± 1.42 |

## Axis: `misspec`

### Level: `misspec_d0.01`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.14 ± 0.15 | 10.85 ± 0.50 |
| dml_gradnorm | 5 | 3.53 ± 0.53 | 20.76 ± 9.10 |
| dml_relobralo | 5 | 3.54 ± 0.30 | 19.38 ± 0.71 |
| dml_warmup | 5 | 3.84 ± 0.23 | 44.24 ± 1.41 |

### Level: `misspec_d0.05`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 4.41 ± 0.10 | 91.04 ± 1.78 |
| dml_gradnorm | 5 | 3.77 ± 0.21 | 84.16 ± 6.17 |
| dml_relobralo | 5 | 3.72 ± 0.20 | 85.29 ± 2.39 |
| dml_warmup | 5 | 3.89 ± 0.26 | 52.65 ± 2.76 |

### Level: `misspec_d0.1`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 7.96 ± 0.18 | 301.51 ± 4.27 |
| dml_gradnorm | 5 | 5.36 ± 0.99 | 260.36 ± 48.69 |
| dml_relobralo | 5 | 4.68 ± 0.12 | 266.17 ± 6.76 |
| dml_warmup | 5 | 3.90 ± 0.25 | 79.25 ± 4.08 |

## Axis: `multiplicative`

### Level: `multiplicative_e0.05`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.10 ± 0.48 | 7.53 ± 1.01 |
| dml_gradnorm | 5 | 3.95 ± 1.33 | 20.53 ± 26.20 |
| dml_relobralo | 5 | 3.51 ± 0.19 | 18.03 ± 2.37 |
| dml_warmup | 5 | 3.84 ± 0.26 | 44.24 ± 1.53 |

### Level: `multiplicative_e0.1`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.57 ± 0.51 | 21.13 ± 13.52 |
| dml_gradnorm | 5 | 3.72 ± 1.36 | 47.46 ± 56.10 |
| dml_relobralo | 5 | 3.44 ± 0.31 | 26.71 ± 6.52 |
| dml_warmup | 5 | 3.86 ± 0.24 | 44.30 ± 1.84 |

### Level: `multiplicative_e0.2`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.61 ± 1.35 | 28.76 ± 10.82 |
| dml_gradnorm | 5 | 15.08 ± 18.42 | 184.96 ± 97.51 |
| dml_relobralo | 5 | 4.50 ± 1.52 | 56.06 ± 24.18 |
| dml_warmup | 5 | 3.85 ± 0.23 | 44.54 ± 1.96 |

## Axis: `staleness`

### Level: `staleness_k10`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.10 ± 0.22 | 7.56 ± 0.45 |
| dml_gradnorm | 5 | 3.43 ± 0.69 | 14.86 ± 12.99 |
| dml_relobralo | 5 | 3.46 ± 0.29 | 16.65 ± 1.05 |
| dml_warmup | 5 | 3.86 ± 0.20 | 44.81 ± 1.51 |

### Level: `staleness_k20`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 3.08 ± 0.19 | 7.93 ± 0.68 |
| dml_gradnorm | 5 | 3.31 ± 0.35 | 15.39 ± 14.27 |
| dml_relobralo | 5 | 3.63 ± 0.35 | 17.71 ± 2.86 |
| dml_warmup | 5 | 3.91 ± 0.23 | 44.31 ± 1.48 |

### Level: `staleness_k5`

| Method | n | value MSE (×1e-5) | grad MSE (×1e-5) |
|---|---:|---:|---:|
| vanilla | 5 | 3.87 ± 0.20 | 1809.72 ± 59.42 |
| dml_fixed | 5 | 2.92 ± 0.17 | 7.06 ± 0.46 |
| dml_gradnorm | 5 | 3.53 ± 0.62 | 19.70 ± 17.02 |
| dml_relobralo | 5 | 3.49 ± 0.21 | 16.76 ± 0.62 |
| dml_warmup | 5 | 3.87 ± 0.25 | 43.97 ± 1.59 |
