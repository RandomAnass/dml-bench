# PDE paired log10(MSE_method / MSE_vanilla) — R1

Cluster = (dataset, regime) for pooled rows; per-cell uses
percentile bootstrap (no clustering needed, single cell).
Bootstrap = 1000. Negative log-ratio = method beats vanilla.

## burgers_bare

_n_seen=186, n_skipped=0, n_pairs=96_

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n pairs |
|---|---:|---:|---:|---:|---:|
| dml_fixed | +0.000 [-0.00, +0.00] | 1 | +0.000 [+0.00, +0.00] | 1 | 20 |
| dml_fixed_half | +0.000 [-0.00, +0.00] | 1 | +0.000 [+0.00, +0.00] | 1 | 20 |
| dml_gradnorm | -0.001 [-0.00, -0.00] | 0.997 | -0.000 [-0.00, +0.00] | 1 | 20 |
| dml_relobralo | -0.001 [-0.00, -0.00] | 0.998 | +0.000 [+0.00, +0.00] | 1 | 20 |
| dml_warmup | -0.001 [-0.00, -0.00] | 0.997 | +0.000 [+0.00, +0.00] | 1 | 16 |

## burgers_ic

_n_seen=120, n_skipped=0, n_pairs=100_

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n pairs |
|---|---:|---:|---:|---:|---:|
| dml_fixed | +1.092 [+0.86, +1.15] | 12.4 | +0.023 [-0.08, +0.04] | 1.06 | 20 |
| dml_fixed_half | +0.564 [+0.45, +0.64] | 3.67 | +0.002 [-0.06, +0.02] | 1 | 20 |
| dml_gradnorm | +1.395 [+0.80, +1.52] | 24.9 | +0.117 [+0.04, +0.14] | 1.31 | 20 |
| dml_relobralo | +0.669 [+0.44, +0.78] | 4.67 | -0.014 [-0.07, +0.05] | 0.968 | 20 |
| dml_warmup | -0.008 [-0.13, +0.04] | 0.981 | -0.022 [-0.04, +0.01] | 0.95 | 20 |

## darcy_bare

_n_seen=120, n_skipped=0, n_pairs=100_

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n pairs |
|---|---:|---:|---:|---:|---:|
| dml_fixed | +0.044 [+0.04, +0.05] | 1.11 | +0.001 [-0.01, +0.02] | 1 | 20 |
| dml_fixed_half | +0.032 [+0.03, +0.03] | 1.08 | -0.031 [-0.05, +0.00] | 0.931 | 20 |
| dml_gradnorm | +0.048 [+0.04, +0.06] | 1.12 | +0.015 [-0.05, +0.11] | 1.03 | 20 |
| dml_relobralo | +0.043 [+0.02, +0.05] | 1.1 | -0.027 [-0.05, +0.02] | 0.94 | 20 |
| dml_warmup | +0.011 [+0.01, +0.01] | 1.03 | -0.085 [-0.09, -0.08] | 0.822 | 20 |

## darcy_ic

_n_seen=120, n_skipped=0, n_pairs=100_

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n pairs |
|---|---:|---:|---:|---:|---:|
| dml_fixed | -0.032 [-0.08, +0.08] | 0.929 | +0.476 [+0.39, +0.53] | 2.99 | 20 |
| dml_fixed_half | -0.016 [-0.10, +0.03] | 0.964 | -0.093 [-0.21, -0.02] | 0.807 | 20 |
| dml_gradnorm | -0.010 [-0.08, +0.07] | 0.978 | +0.287 [+0.21, +0.36] | 1.94 | 20 |
| dml_relobralo | -0.014 [-0.06, +0.05] | 0.967 | -0.165 [-0.20, -0.04] | 0.684 | 20 |
| dml_warmup | +0.003 [-0.03, +0.05] | 1.01 | -0.193 [-0.24, -0.07] | 0.641 | 20 |

## Pooled by regime (cluster bootstrap on (dataset, regime))

### regime = bare

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n_clusters | n_pairs |
|---|---:|---:|---:|---:|---:|---:|
| dml_fixed | +0.020 [+0.00, +0.04] | 1.05 | +0.000 [+0.00, +0.00] | 1 | 2 | 40 |
| dml_fixed_half | +0.014 [+0.00, +0.03] | 1.03 | +0.000 [-0.03, +0.00] | 1 | 2 | 40 |
| dml_gradnorm | +0.007 [-0.00, +0.05] | 1.02 | +0.000 [-0.00, +0.01] | 1 | 2 | 40 |
| dml_relobralo | +0.004 [-0.00, +0.04] | 1.01 | +0.000 [-0.03, +0.00] | 1 | 2 | 40 |
| dml_warmup | +0.007 [-0.00, +0.01] | 1.02 | -0.072 [-0.09, +0.00] | 0.848 | 2 | 36 |

### regime = ic

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n_clusters | n_pairs |
|---|---:|---:|---:|---:|---:|---:|
| dml_fixed | +0.108 [-0.03, +1.09] | 1.28 | +0.119 [+0.02, +0.48] | 1.32 | 2 | 40 |
| dml_fixed_half | +0.034 [-0.02, +0.56] | 1.08 | -0.032 [-0.09, +0.00] | 0.929 | 2 | 40 |
| dml_gradnorm | +0.198 [-0.01, +1.40] | 1.58 | +0.148 [+0.12, +0.29] | 1.41 | 2 | 40 |
| dml_relobralo | +0.079 [-0.01, +0.67] | 1.2 | -0.052 [-0.16, -0.01] | 0.887 | 2 | 40 |
| dml_warmup | -0.006 [-0.01, +0.00] | 0.987 | -0.058 [-0.19, -0.02] | 0.875 | 2 | 40 |

