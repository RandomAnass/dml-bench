# rMD17 paired log10(MSE_method / MSE_native_EF) — R1

Cluster = molecule. Bootstrap = 1000. Negative log-ratio means
method beats native_EF; positive means native_EF beats method.

## PAINN

_n_seen=300, n_skipped=0, n_pairs=250_

| Method | log10 ratio (val) | factor (val) | log10 ratio (grad) | factor (grad) | n clusters | n pairs |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | +2.090 [+1.82, +2.50] | 123 | +2.767 [+2.62, +3.04] | 585 | 10 | 50 |
| dml_fixed | +0.045 [-0.20, +0.26] | 1.11 | +0.482 [+0.25, +0.59] | 3.03 | 10 | 50 |
| dml_fixed_half | +0.101 [-0.11, +0.31] | 1.26 | +0.446 [+0.27, +0.60] | 2.79 | 10 | 50 |
| dml_gradnorm | +0.622 [+0.44, +0.85] | 4.19 | +0.835 [+0.66, +0.97] | 6.83 | 10 | 50 |
| dml_warmup | +2.238 [+1.77, +2.54] | 173 | +2.855 [+2.46, +2.92] | 717 | 10 | 50 |

## MLP

_n_seen=350, n_skipped=0, n_pairs=0_

_no paired rows under ./results/molecular_mlp_

## GATV2

_n_seen=350, n_skipped=0, n_pairs=0_

_no paired rows under ./results/molecular_gatv2_

