# S3: Almost-Stochastic Dominance for SPY purged-CV grad MSE

Method: deepsig.aso (Dror, Shlomov, Reichart, ACL 2019), confidence 0.95, 1000 bootstrap iterations.

Comparison against `vanilla` (n = 100 purged-CV runs = 5 folds × 10 seeds).

| DML variant | n | ε (DML dominates vanilla on grad MSE) |
|---|---:|---:|
| `dml_fixed` | 100 | **0.0000** |
| `dml_gradnorm` | 100 | **0.0000** |
| `dml_relobralo` | 100 | **0.0000** |
| `dml_warmup` | 100 | **0.0000** |

Interpretation: ε < 0.5 indicates almost-stochastic dominance;
ε close to 0 indicates near-complete dominance (DML's grad-MSE
distribution lies almost entirely below vanilla's). Replaces
Cohen's d as the headline effect size for §5.2 — where d is
in the [318, 464] range its interpretability is poor; ε is
scale-free and bounded.
