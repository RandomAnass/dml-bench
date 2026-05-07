# Classical baselines (one-paragraph summary for §5)

From `results/tier5_extended_baselines/`, paired-config win-rates of `dml_fixed` vs each classical baseline:

| Classical method | n paired configs | wins | ties | losses | win-rate |
|---|---:|---:|---:|---:|---:|
| baseline_gp | 90 | 45 | 0 | 45 | 50.0% |
| baseline_krr | 225 | 188 | 0 | 37 | 83.6% |
| baseline_rf | 225 | 168 | 0 | 57 | 74.7% |

## Suggested paragraph for §5

Across the high-dimensional configurations where classical baselines remain tractable (d ≤ 50), neural DML wins 50% vs baseline_gp (90 paired configs), 83% vs baseline_krr (225 paired configs), 74% vs baseline_rf (225 paired configs). The compute argument for excluding classical baselines from the SPY benchmark is orthogonal: GP scaling is O(n^3) and the SPY corpus has 1.57M records.
