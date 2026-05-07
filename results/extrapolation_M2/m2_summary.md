# M2 — Extrapolation split aggregate

Source: `results/extrapolation_M2/m2_*.json` (360 cells = 2 funcs × 3 d × 2 N_train × 2 modes × 3 methods × 5 seeds).
Replication: `python experiments/extrapolation/aggregate_m2.py`
Each test set is the OOS half of the cube; train is the other half.
Δ% < 0 means DML reduces value MSE vs vanilla.

## Δ% per (function, d, n_train, mode), 5 seeds

### poly_trig

|   d |   n_train | mode      | method       |   val_pct |   grad_pct |   median_paired_log10_ratio |   dml_wins |   n_seeds |
|----:|----------:|:----------|:-------------|----------:|-----------:|----------------------------:|-----------:|----------:|
|   2 |       512 | halfspace | dml_fixed    |    -33.21 |     -24.91 |                       -0.07 |          3 |         5 |
|   2 |       512 | halfspace | dml_gradnorm |    -25.75 |     -19.45 |                        0.01 |          2 |         5 |
|   2 |       512 | radial    | dml_fixed    |    -88.13 |     -79.02 |                       -0.98 |          5 |         5 |
|   2 |       512 | radial    | dml_gradnorm |    -85.86 |     -76.46 |                       -0.83 |          5 |         5 |
|   2 |      2048 | halfspace | dml_fixed    |     -9.85 |      -5.19 |                        0.02 |          2 |         5 |
|   2 |      2048 | halfspace | dml_gradnorm |    -19.00 |     -12.83 |                       -0.03 |          3 |         5 |
|   2 |      2048 | radial    | dml_fixed    |    -87.72 |     -79.34 |                       -0.96 |          5 |         5 |
|   2 |      2048 | radial    | dml_gradnorm |    -87.22 |     -77.76 |                       -0.90 |          5 |         5 |
|   5 |       512 | halfspace | dml_fixed    |    -36.44 |     -42.56 |                       -0.15 |          3 |         5 |
|   5 |       512 | halfspace | dml_gradnorm |    -24.32 |     -35.62 |                        0.03 |          2 |         5 |
|   5 |       512 | radial    | dml_fixed    |    -96.98 |     -94.54 |                       -1.52 |          5 |         5 |
|   5 |       512 | radial    | dml_gradnorm |    -94.85 |     -92.12 |                       -1.26 |          5 |         5 |
|   5 |      2048 | halfspace | dml_fixed    |    -43.21 |     -34.41 |                       -0.13 |          4 |         5 |
|   5 |      2048 | halfspace | dml_gradnorm |    -35.07 |     -27.09 |                       -0.15 |          3 |         5 |
|   5 |      2048 | radial    | dml_fixed    |    -97.24 |     -95.65 |                       -1.57 |          5 |         5 |
|   5 |      2048 | radial    | dml_gradnorm |    -96.80 |     -94.83 |                       -1.46 |          5 |         5 |
|  10 |       512 | halfspace | dml_fixed    |    -67.98 |     -77.83 |                       -0.54 |          5 |         5 |
|  10 |       512 | halfspace | dml_gradnorm |    -65.77 |     -75.58 |                       -0.55 |          5 |         5 |
|  10 |       512 | radial    | dml_fixed    |    -98.23 |     -97.61 |                       -1.73 |          5 |         5 |
|  10 |       512 | radial    | dml_gradnorm |    -97.44 |     -96.31 |                       -1.61 |          5 |         5 |
|  10 |      2048 | halfspace | dml_fixed    |    -57.88 |     -61.12 |                       -0.41 |          5 |         5 |
|  10 |      2048 | halfspace | dml_gradnorm |    -53.05 |     -57.55 |                       -0.42 |          5 |         5 |
|  10 |      2048 | radial    | dml_fixed    |    -98.39 |     -98.07 |                       -1.80 |          5 |         5 |
|  10 |      2048 | radial    | dml_gradnorm |    -98.62 |     -98.22 |                       -1.84 |          5 |         5 |

### trig

|   d |   n_train | mode      | method       |   val_pct |   grad_pct |   median_paired_log10_ratio |   dml_wins |   n_seeds |
|----:|----------:|:----------|:-------------|----------:|-----------:|----------------------------:|-----------:|----------:|
|   2 |       512 | halfspace | dml_fixed    |     16.89 |      -0.12 |                        0.09 |          1 |         5 |
|   2 |       512 | halfspace | dml_gradnorm |     20.30 |       7.05 |                       -0.00 |          3 |         5 |
|   2 |       512 | radial    | dml_fixed    |    -79.41 |     -43.11 |                       -0.60 |          5 |         5 |
|   2 |       512 | radial    | dml_gradnorm |    -72.65 |     -39.71 |                       -0.36 |          5 |         5 |
|   2 |      2048 | halfspace | dml_fixed    |     14.13 |      -0.48 |                        0.06 |          1 |         5 |
|   2 |      2048 | halfspace | dml_gradnorm |     41.42 |      10.92 |                        0.18 |          1 |         5 |
|   2 |      2048 | radial    | dml_fixed    |    -60.40 |     -12.93 |                       -0.43 |          4 |         5 |
|   2 |      2048 | radial    | dml_gradnorm |    -51.38 |     -10.58 |                       -0.42 |          4 |         5 |
|   5 |       512 | halfspace | dml_fixed    |    541.78 |      84.58 |                        0.76 |          0 |         5 |
|   5 |       512 | halfspace | dml_gradnorm |    245.13 |      32.43 |                       -0.00 |          3 |         5 |
|   5 |       512 | radial    | dml_fixed    |    -50.99 |     -33.66 |                       -0.18 |          5 |         5 |
|   5 |       512 | radial    | dml_gradnorm |    -51.10 |     -34.89 |                       -0.23 |          4 |         5 |
|   5 |      2048 | halfspace | dml_fixed    |    271.20 |     -30.82 |                        0.48 |          0 |         5 |
|   5 |      2048 | halfspace | dml_gradnorm |    193.64 |     -38.39 |                        0.41 |          0 |         5 |
|   5 |      2048 | radial    | dml_fixed    |    -95.02 |     -81.33 |                       -1.40 |          5 |         5 |
|   5 |      2048 | radial    | dml_gradnorm |    -94.39 |     -79.95 |                       -1.31 |          5 |         5 |
|  10 |       512 | halfspace | dml_fixed    |     41.08 |      -0.77 |                        0.15 |          0 |         5 |
|  10 |       512 | halfspace | dml_gradnorm |      6.94 |      -1.24 |                       -0.00 |          3 |         5 |
|  10 |       512 | radial    | dml_fixed    |     12.74 |      -0.32 |                        0.04 |          1 |         5 |
|  10 |       512 | radial    | dml_gradnorm |      0.15 |      -0.60 |                        0.00 |          2 |         5 |
|  10 |      2048 | halfspace | dml_fixed    |    688.36 |      26.69 |                        0.89 |          0 |         5 |
|  10 |      2048 | halfspace | dml_gradnorm |    140.44 |       0.75 |                        0.04 |          2 |         5 |
|  10 |      2048 | radial    | dml_fixed    |     -5.17 |     -12.60 |                        0.05 |          2 |         5 |
|  10 |      2048 | radial    | dml_gradnorm |    -22.27 |     -10.62 |                       -0.02 |          4 |         5 |

## Per-method aggregate value MSE

| func      |   d |   n_train | mode      | method       |   val_mse_mean |   val_mse_std |   val_mse_median |   n_seeds |
|:----------|----:|----------:|:----------|:-------------|---------------:|--------------:|-----------------:|----------:|
| poly_trig |   2 |       512 | halfspace | dml_fixed    |     6.6474e-02 |    9.0624e-02 |       1.8912e-02 |         5 |
| poly_trig |   2 |       512 | halfspace | dml_gradnorm |     7.3896e-02 |    9.1464e-02 |       2.1941e-02 |         5 |
| poly_trig |   2 |       512 | halfspace | vanilla      |     9.9520e-02 |    1.2333e-01 |       1.5701e-02 |         5 |
| poly_trig |   2 |       512 | radial    | dml_fixed    |     1.0160e-03 |    5.7993e-04 |       1.0199e-03 |         5 |
| poly_trig |   2 |       512 | radial    | dml_gradnorm |     1.2111e-03 |    6.1158e-04 |       1.2802e-03 |         5 |
| poly_trig |   2 |       512 | radial    | vanilla      |     8.5628e-03 |    5.2149e-03 |       8.5338e-03 |         5 |
| poly_trig |   2 |      2048 | halfspace | dml_fixed    |     6.8874e-02 |    7.6109e-02 |       3.9249e-02 |         5 |
| poly_trig |   2 |      2048 | halfspace | dml_gradnorm |     6.1883e-02 |    7.7769e-02 |       3.3899e-02 |         5 |
| poly_trig |   2 |      2048 | halfspace | vanilla      |     7.6399e-02 |    1.1344e-01 |       1.0558e-02 |         5 |
| poly_trig |   2 |      2048 | radial    | dml_fixed    |     5.5080e-04 |    4.1263e-04 |       4.7485e-04 |         5 |
| poly_trig |   2 |      2048 | radial    | dml_gradnorm |     5.7313e-04 |    3.0110e-04 |       5.7363e-04 |         5 |
| poly_trig |   2 |      2048 | radial    | vanilla      |     4.4854e-03 |    2.4093e-03 |       4.5537e-03 |         5 |
| poly_trig |   5 |       512 | halfspace | dml_fixed    |     9.1138e-02 |    9.4800e-02 |       4.5787e-02 |         5 |
| poly_trig |   5 |       512 | halfspace | dml_gradnorm |     1.0851e-01 |    9.6231e-02 |       7.3055e-02 |         5 |
| poly_trig |   5 |       512 | halfspace | vanilla      |     1.4339e-01 |    1.7521e-01 |       6.4896e-02 |         5 |
| poly_trig |   5 |       512 | radial    | dml_fixed    |     5.3833e-04 |    2.0515e-04 |       4.4432e-04 |         5 |
| poly_trig |   5 |       512 | radial    | dml_gradnorm |     9.1736e-04 |    2.6066e-04 |       8.6082e-04 |         5 |
| poly_trig |   5 |       512 | radial    | vanilla      |     1.7818e-02 |    9.3658e-03 |       1.3104e-02 |         5 |
| poly_trig |   5 |      2048 | halfspace | dml_fixed    |     7.1242e-02 |    6.3599e-02 |       3.4274e-02 |         5 |
| poly_trig |   5 |      2048 | halfspace | dml_gradnorm |     8.1465e-02 |    8.0944e-02 |       4.6157e-02 |         5 |
| poly_trig |   5 |      2048 | halfspace | vanilla      |     1.2546e-01 |    1.6983e-01 |       4.8005e-02 |         5 |
| poly_trig |   5 |      2048 | radial    | dml_fixed    |     7.8208e-05 |    3.6378e-05 |       5.5784e-05 |         5 |
| poly_trig |   5 |      2048 | radial    | dml_gradnorm |     9.0846e-05 |    3.4486e-05 |       7.5743e-05 |         5 |
| poly_trig |   5 |      2048 | radial    | vanilla      |     2.8375e-03 |    1.7262e-03 |       2.0573e-03 |         5 |
| poly_trig |  10 |       512 | halfspace | dml_fixed    |     1.8056e-01 |    1.3816e-01 |       1.4006e-01 |         5 |
| poly_trig |  10 |       512 | halfspace | dml_gradnorm |     1.9299e-01 |    1.7173e-01 |       1.2112e-01 |         5 |
| poly_trig |  10 |       512 | halfspace | vanilla      |     5.6380e-01 |    3.9931e-01 |       3.5969e-01 |         5 |
| poly_trig |  10 |       512 | radial    | dml_fixed    |     3.7926e-03 |    9.3173e-04 |       3.4966e-03 |         5 |
| poly_trig |  10 |       512 | radial    | dml_gradnorm |     5.4684e-03 |    1.4169e-03 |       4.5961e-03 |         5 |
| poly_trig |  10 |       512 | radial    | vanilla      |     2.1401e-01 |    6.4546e-02 |       2.2796e-01 |         5 |
| poly_trig |  10 |      2048 | halfspace | dml_fixed    |     1.1768e-01 |    7.7029e-02 |       1.1026e-01 |         5 |
| poly_trig |  10 |      2048 | halfspace | dml_gradnorm |     1.3117e-01 |    1.3116e-01 |       7.5095e-02 |         5 |
| poly_trig |  10 |      2048 | halfspace | vanilla      |     2.7936e-01 |    1.8833e-01 |       1.9734e-01 |         5 |
| poly_trig |  10 |      2048 | radial    | dml_fixed    |     5.3636e-04 |    8.9638e-05 |       5.1090e-04 |         5 |
| poly_trig |  10 |      2048 | radial    | dml_gradnorm |     4.5729e-04 |    1.6522e-04 |       3.6498e-04 |         5 |
| poly_trig |  10 |      2048 | radial    | vanilla      |     3.3229e-02 |    1.4273e-02 |       3.3561e-02 |         5 |
| trig      |   2 |       512 | halfspace | dml_fixed    |     3.3592e+01 |    2.1180e+01 |       3.6406e+01 |         5 |
| trig      |   2 |       512 | halfspace | dml_gradnorm |     3.4574e+01 |    2.3904e+01 |       3.4625e+01 |         5 |
| trig      |   2 |       512 | halfspace | vanilla      |     2.8739e+01 |    1.9954e+01 |       4.0066e+01 |         5 |
| trig      |   2 |       512 | radial    | dml_fixed    |     7.3092e-01 |    6.0504e-01 |       8.5833e-01 |         5 |
| trig      |   2 |       512 | radial    | dml_gradnorm |     9.7082e-01 |    1.0115e+00 |       9.8099e-01 |         5 |
| trig      |   2 |       512 | radial    | vanilla      |     3.5499e+00 |    3.7015e+00 |       1.6043e+00 |         5 |
| trig      |   2 |      2048 | halfspace | dml_fixed    |     3.1259e+01 |    2.4049e+01 |       3.5852e+01 |         5 |
| trig      |   2 |      2048 | halfspace | dml_gradnorm |     3.8734e+01 |    2.5637e+01 |       5.1663e+01 |         5 |
| trig      |   2 |      2048 | halfspace | vanilla      |     2.7390e+01 |    2.1896e+01 |       3.0341e+01 |         5 |
| trig      |   2 |      2048 | radial    | dml_fixed    |     6.3589e-01 |    6.1479e-01 |       4.8929e-01 |         5 |
| trig      |   2 |      2048 | radial    | dml_gradnorm |     7.8077e-01 |    8.0618e-01 |       6.9492e-01 |         5 |
| trig      |   2 |      2048 | radial    | vanilla      |     1.6059e+00 |    1.6152e+00 |       8.2076e-01 |         5 |
| trig      |   5 |       512 | halfspace | dml_fixed    |     1.0622e+02 |    6.7363e+01 |       1.1846e+02 |         5 |
| trig      |   5 |       512 | halfspace | dml_gradnorm |     5.7121e+01 |    7.2663e+01 |       2.9602e+01 |         5 |
| trig      |   5 |       512 | halfspace | vanilla      |     1.6551e+01 |    1.6140e+01 |       5.6590e+00 |         5 |
| trig      |   5 |       512 | radial    | dml_fixed    |     2.2628e+00 |    1.4441e+00 |       1.9718e+00 |         5 |
| trig      |   5 |       512 | radial    | dml_gradnorm |     2.2578e+00 |    1.9967e+00 |       1.3256e+00 |         5 |
| trig      |   5 |       512 | radial    | vanilla      |     4.6170e+00 |    2.6899e+00 |       4.2991e+00 |         5 |
| trig      |   5 |      2048 | halfspace | dml_fixed    |     6.6244e+01 |    4.3309e+01 |       6.4243e+01 |         5 |
| trig      |   5 |      2048 | halfspace | dml_gradnorm |     5.2403e+01 |    3.0083e+01 |       6.3434e+01 |         5 |
| trig      |   5 |      2048 | halfspace | vanilla      |     1.7846e+01 |    1.1053e+01 |       1.7479e+01 |         5 |
| trig      |   5 |      2048 | radial    | dml_fixed    |     1.6817e-01 |    1.5023e-01 |       1.1550e-01 |         5 |
| trig      |   5 |      2048 | radial    | dml_gradnorm |     1.8976e-01 |    1.6666e-01 |       1.1458e-01 |         5 |
| trig      |   5 |      2048 | radial    | vanilla      |     3.3802e+00 |    1.5502e+00 |       2.9959e+00 |         5 |
| trig      |  10 |       512 | halfspace | dml_fixed    |     1.3368e+01 |    3.8720e+00 |       1.4653e+01 |         5 |
| trig      |  10 |       512 | halfspace | dml_gradnorm |     1.0133e+01 |    3.5660e+00 |       1.1859e+01 |         5 |
| trig      |  10 |       512 | halfspace | vanilla      |     9.4759e+00 |    3.5622e+00 |       8.3332e+00 |         5 |
| trig      |  10 |       512 | radial    | dml_fixed    |     9.8853e+00 |    2.7966e+00 |       1.1605e+01 |         5 |
| trig      |  10 |       512 | radial    | dml_gradnorm |     8.7821e+00 |    3.1924e+00 |       7.4839e+00 |         5 |
| trig      |  10 |       512 | radial    | vanilla      |     8.7686e+00 |    2.8853e+00 |       7.7626e+00 |         5 |
| trig      |  10 |      2048 | halfspace | dml_fixed    |     9.7320e+01 |    8.2841e+01 |       1.0387e+02 |         5 |
| trig      |  10 |      2048 | halfspace | dml_gradnorm |     2.9681e+01 |    3.8411e+01 |       1.2101e+01 |         5 |
| trig      |  10 |      2048 | halfspace | vanilla      |     1.2345e+01 |    4.0098e+00 |       1.3505e+01 |         5 |
| trig      |  10 |      2048 | radial    | dml_fixed    |     7.9834e+00 |    6.3629e+00 |       5.0181e+00 |         5 |
| trig      |  10 |      2048 | radial    | dml_gradnorm |     6.5439e+00 |    4.7254e+00 |       3.9288e+00 |         5 |
| trig      |  10 |      2048 | radial    | vanilla      |     8.4182e+00 |    3.2282e+00 |       7.5045e+00 |         5 |
