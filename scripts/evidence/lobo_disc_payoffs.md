# LOBO disc-payoff robustness — leave-one-balancer-out

Per-dataset balancer-paradigm rankings on the discontinuous-payoff sub-corpus, with each of the five DML balancers removed in turn.

| Variant | digital_bs | barrier_bs | heston_dig | basket_d1 | basket_d7 |
|---|:-:|:-:|:-:|:-:|:-:|
| full corpus | ✗ | ✓ | ✗ | ✓ | ✓ |
| drop dml_fixed | ✗ | ✓ | ✗ | ✓ | ✓ |
| drop dml_fixed_half | ✗ | ✓ | ✗ | ✓ | ✓ |
| drop dml_gradnorm | ✗ | ✓ | ✗ | ✓ | ✓ |
| drop dml_relobralo | ✗ | ✓ | ✗ | ✓ | ✓ |
| drop dml_warmup | ✓ | ✓ | ✓ | ✓ | ✓ |

✓ = fuzzy-paradigm methods all rank ahead of pathwise methods. — = no fuzzy or no pathwise rows in this variant.
