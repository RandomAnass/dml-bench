# ERA5 Z500 sub-pillar — input-completeness ablation

Reanalysis Z500 at 12Z (2014--2020, 1.0 deg, 51 lat x 360 lon, 2557 snapshots; train/val/test 1731/383/383 chronological with 30-day embargos). Main architecture: 4x256 softplus MLP, 100k points sampled per epoch, 5 seeds per (regime, method). 6x512 architecture sensitivity: separate 5-seed sweep.

## Per-method, per-regime

| Regime | Method | n | Value MSE (norm.) | Gradient MSE (norm.) | u_g MAE (m/s) |
|---|---|---:|---:|---:|---:|
| bare | Vanilla | 5 | 0.174 +- 0.002 | 2.101 +- 0.047 | 5.55 +- 0.31 |
| bare | DML fixed-$\lambda$ | 5 | 0.177 +- 0.004 | 1.893 +- 0.030 | 5.63 +- 0.24 |
| bare | DML fixed-1/2 | 5 | 0.174 +- 0.003 | 1.889 +- 0.028 | 5.73 +- 0.26 |
| state | Vanilla | 5 | 0.108 +- 0.005 | 2.224 +- 0.092 | 6.67 +- 0.24 |
| state | DML fixed-$\lambda$ | 5 | 0.128 +- 0.009 | 1.577 +- 0.033 | 6.80 +- 0.45 |
| state | DML fixed-1/2 | 5 | 0.119 +- 0.009 | 1.567 +- 0.037 | 6.60 +- 0.16 |

## Input-completeness delta (state vs bare)

| Method | Value MSE delta-pct | Gradient MSE delta-pct |
|---|---:|---:|
| Vanilla | -38.2% | +5.9% |
| DML fixed-$\lambda$ | -27.5% | -16.7% |
| DML fixed-1/2 | -31.6% | -17.0% |

## Architecture sensitivity (6x512, 5 seeds)

| Regime | Method | n | Value MSE (norm.) | Gradient MSE (norm.) | u_g MAE (m/s) |
|---|---|---:|---:|---:|---:|
| bare | Vanilla | 5 | 0.174 +- 0.002 | 2.116 +- 0.011 | 5.67 +- 0.29 |
| bare | DML fixed-$\lambda$ | 5 | 0.177 +- 0.006 | 1.882 +- 0.020 | 5.47 +- 0.24 |
| bare | DML fixed-1/2 | 5 | 0.173 +- 0.001 | 1.890 +- 0.024 | 5.67 +- 0.39 |
| state | Vanilla | 5 | 0.108 +- 0.002 | 2.281 +- 0.070 | 6.87 +- 0.20 |
| state | DML fixed-$\lambda$ | 5 | 0.124 +- 0.009 | 1.554 +- 0.033 | 6.70 +- 0.46 |
| state | DML fixed-1/2 | 5 | 0.116 +- 0.005 | 1.565 +- 0.026 | 6.77 +- 0.22 |
