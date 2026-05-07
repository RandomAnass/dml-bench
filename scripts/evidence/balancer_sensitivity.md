# Balancer-sensitivity sweep results

Source: `results/balancer_sensitivity/{synthetic,burgers_ic}/`

## synthetic

| method | hp | n | value_mse mean ± std | grad_mse mean ± std |
|---|---|---|---|---|
| dml_gradnorm | 0.50 | 15 | 2.799e+00 ± 2.251e+00 | 2.617e+00 ± 3.263e+00 |
| dml_gradnorm | 1.00 | 15 | 2.766e+00 ± 2.237e+00 | 2.619e+00 ± 3.251e+00 |
| dml_gradnorm | 1.50 | 15 | 2.768e+00 ± 2.252e+00 | 2.629e+00 ± 3.254e+00 |
| dml_gradnorm | 2.00 | 15 | 2.768e+00 ± 2.251e+00 | 2.620e+00 ± 3.247e+00 |
| dml_relobralo | 0.10 | 15 | 2.971e+00 ± 2.330e+00 | 3.182e+00 ± 4.121e+00 |
| dml_relobralo | 0.25 | 15 | 2.997e+00 ± 2.364e+00 | 3.151e+00 ± 4.056e+00 |
| dml_relobralo | 0.50 | 15 | 3.272e+00 ± 2.661e+00 | 3.585e+00 ± 4.389e+00 |
| dml_relobralo | 1.00 | 15 | 3.295e+00 ± 2.715e+00 | 3.727e+00 ± 4.567e+00 |

## pde

| method | hp | n | value_mse mean ± std | grad_mse mean ± std |
|---|---|---|---|---|
| dml_gradnorm | 0.50 | 5 | 2.409e-01 ± 1.381e-01 | 7.568e-01 ± 5.863e-02 |
| dml_gradnorm | 1.00 | 5 | 2.866e-01 ± 1.246e-01 | 7.785e-01 ± 4.729e-02 |
| dml_gradnorm | 1.50 | 5 | 2.282e-01 ± 1.549e-01 | 7.996e-01 ± 1.253e-05 |
| dml_gradnorm | 2.00 | 5 | 5.178e-01 ± 7.437e-01 | 1.560e+00 ± 1.878e+00 |
| dml_relobralo | 0.10 | 5 | 6.528e-02 ± 1.435e-02 | 6.364e-01 ± 4.542e-02 |
| dml_relobralo | 0.25 | 5 | 2.186e-01 ± 3.611e-01 | 6.300e-01 ± 3.686e-02 |
| dml_relobralo | 0.50 | 5 | 6.742e-02 ± 1.791e-02 | 6.425e-01 ± 8.718e-02 |
| dml_relobralo | 1.00 | 5 | 8.628e-02 ± 7.453e-02 | 6.505e-01 ± 7.306e-02 |

