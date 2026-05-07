# `dml_softmax_balance` vs `dml_relobralo` paired comparison

Median paired $\log_{10}(\mathrm{MSE}_{\mathrm{softmax\_balance}} / \mathrm{MSE}_{\mathrm{relobralo}})$ across paired cells where both methods ran. Negative means softmax_balance wins.

| corpus | n | log-ratio (value) | log-ratio (grad) | n value-wins | n grad-wins |
|---|---:|---:|---:|---:|---:|
| synth | 2670 | +0.009 | +0.014 | 1119/2670 | 1031/2670 |
| rMD17 MLP-pairwise | 50 | -0.003 | +0.015 | 26/50 | 18/50 |
| rMD17 GATv2 | 50 | -0.605 | -0.542 | 50/50 | 50/50 |
