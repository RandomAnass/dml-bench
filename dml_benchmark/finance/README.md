# `dml_benchmark/finance/` — Hedging back-test scaffold

The finance-appendix back-test that compares delta-hedging quality
across analytical Black-Scholes delta, DML-predicted delta, and
vanilla-NN-predicted delta.

## Status

**Currently NOT integrated into the main benchmark pipeline.** This
module is the back-test scaffold described as a known limitation in
the prior README; it is preserved here for downstream users who want
to extend the benchmark with a pricing-versus-hedging Pareto plot,
but no result corpus under `results/` consumes it.

## Public API

```python
from dml_benchmark.finance.hedging import HedgingResult, ...
```

`HedgingResult` is a dataclass with fields:

- `model_name` — identifier for the hedger (e.g. `"BS analytical"`)
- `mean_pnl`   — mean hedging P&L over the simulated paths
- `std_pnl`    — std of hedging P&L (lower = better hedge)
- `hedging_error_pct` — `std_pnl / option_value * 100`
- `n_paths`, `n_rebalances`

## Metric

`std(hedging P&L)` as a percentage of option notional. Convention
follows Huge & Savine (2020) §4 (their hedging-PNL plots).

## Why it is not on the main path

The synthetic and finance pillars of the benchmark report value MSE
and gradient MSE; downstream hedging quality is an extension that
adds another simulation layer (path generation, rebalancing
schedule). Adding hedging quality to the headline tables would
require committing to a specific path-generation scheme, which is
out of scope for the current benchmark.
