# LOFO synthetic robustness — leave-one-function-family-out

Median paired log10(MSE_DML / MSE_vanilla) at σ=0 across synthetic cells, with each function family removed in turn.

| Variant | n | R1 (value) | R1 (grad) |
|---|---:|---:|---:|
| full corpus | 1050 | -0.556 | -1.069 |
| drop bachelier | 840 | -0.084 | -1.165 |
| drop black_scholes | 1015 | -0.512 | -1.043 |
| drop heston | 1015 | -0.618 | -1.134 |
| drop poly_trig | 750 | -0.019 | -0.919 |
| drop step | 820 | -0.938 | -1.064 |
| drop trig | 810 | -0.674 | -1.174 |

Sign flips relative to full corpus: value = none; grad = none.
