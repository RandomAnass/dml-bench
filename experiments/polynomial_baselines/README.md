# `experiments/polynomial_baselines/` — Polynomial-in-spot comparator

The simplest closed-form alternative to the neural-network DML
methods: a univariate polynomial of degree $1..6$ fit on the input
$x$ (or on the spot $S_0$ for the Heston barrier). Reviewers flagged
that the polynomial-in-spot baseline is missing from prior DML
benchmarks (huge2020, glasserman2025, sakuma2026 0dte). This sub-pillar
extends the baseline from the Heston-barrier sub-pillar to the 1-D
synthetic block and to the SPY temporal split.

## Why $d = 1$ only

At $d \ge 2$ the polynomial parameter count blows up — a quintic at
$d = 20$ has $\binom{25}{5} = 53{,}130$ parameters on $n = 1024$
training points, which is overfit-bound. The $d = 1$ cells are
matched to the Tier-3 grid (see `run_full_benchmark.py`,
`build_tier3_experiments §A`); existing DML / classical-baseline
results live in `results/tier3_benchmark/*_d1_*.json` and the
polynomial cells share the same `(seed, n_samples)` splits.

## Output corpus

| Sub-dir | Domains | Producing script |
|---|---|---|
| `results/polynomial_baselines/synthetic/` | poly_trig, trig, black_scholes (all $d=1$) | `run_synthetic.py` |
| `results/polynomial_baselines/spy/`        | SPY temporal Option A and Option C        | `run_spy.py` |

## Protocol

(mirrors the Heston-barrier polynomial baseline at
`experiments/heston_barrier_4way/analyze_polynomial_baselines.py`):

1. Regenerate $(x, y, dydx)$ deterministically with `generate_data`
   and split via `train_test_split(train_ratio=0.8, seed=seed)` —
   identical to the existing Tier-3 NN / classical-baseline runs.
2. Fit a univariate polynomial of degree $\in \{1, …, 6\}$ on
   $(x_\mathrm{train}, y_\mathrm{train})$.
3. Evaluate price MSE and gradient MSE on
   $(x_\mathrm{test}, y_\mathrm{test}, dydx_\mathrm{test})$. The
   gradient is the analytical derivative of the fitted polynomial
   (`numpy.polyder`).
4. Save one JSON per `(domain, n_samples, seed)` with all degrees.

## Headline findings

- On Heston barrier narrow-range price, a quintic polynomial in $S_0$
  fits price MSE 0.016e-5, beating every DML method (best DML
  0.23e-5).
- Quintic poly fit ratio relative to best DML on BS barrier: 7.3× (n=2),
  37× (n=4), 17× (n=8), 5.9× (n=16).

## Scripts

| Script | Purpose |
|---|---|
| `run_synthetic.py` | Polynomial fit on poly_trig / trig / black_scholes ($d=1$) |
| `run_spy.py`       | Polynomial-in-spot fit on SPY temporal Option A / C |
| `analyze.py`       | Per-(domain, degree) summary CSV |

## Running

```bash
python experiments/polynomial_baselines/run_synthetic.py
python experiments/polynomial_baselines/run_spy.py
python experiments/polynomial_baselines/analyze.py
```
