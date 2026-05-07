# Single-cell drop sensitivity

Full corpus: n=1050, R1_value = -0.556, R1_grad = -1.069.

Maximum shift in R1 from dropping any single cell:
- value: 0.0008
- grad : 0.0004

## Top-5 outlier cells (by |log_ratio_value|)

| func | d | n | seed | log_ratio | ΔR1 if dropped |
|---|---:|---:|---:|---:|---:|
| trig | 1 | 256 | 456 | -5.513 | +0.0008 |
| bachelier | 1 | 512 | 1000 | -4.287 | +0.0008 |
| trig | 1 | 4096 | 1000 | -4.193 | +0.0008 |
| poly_trig | 1 | 8192 | 789 | -3.637 | +0.0008 |
| trig | 1 | 256 | 1000 | -3.608 | +0.0008 |
