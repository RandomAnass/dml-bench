# `tests/` — pytest suite

158 `def test_` definitions across 18 files (verified by
`grep -rE "^[[:space:]]*def test_" tests/*.py | wc -l`). On a clean
install with all data accessible, ~150 collect successfully; the
`era5_*`, `svi_*`, and a few `heston_*` tests auto-skip if the
relevant external dataset is not yet on disk.

## Run

```bash
python -m pytest tests/ -v
```

Or via Make:

```bash
make test
```

## File-by-file count

| File | Defs | What it covers |
|---|---:|---|
| `test_balancer_kwargs_idempotent.py`         | 5  | balancer kwarg-passing identity (post-relabel safety) |
| `test_balancing_correctness.py`              | 11 | correctness of GradNorm / ReLoBRaLo / softmax balancers |
| `test_compat.py`                             | 0  | (currently no `def test_`; module-level pytest IDs) |
| `test_dydx_mask.py`                          | 3  | gradient-mask broadcast under partial-supervision |
| `test_era5_grad_projection_idempotent.py`    | 3  | ERA5 lat/lon-tangent gradient projection idempotence |
| `test_functions.py`                          | 7  | function-family API + JAX-autodiff gradient match |
| `test_heston_barrier.py`                     | 17 | Heston path SDE, barrier monitor, LRM + fuzzy labels |
| `test_heston_lrm_unbiased_vs_fd.py`          | 0  | (module-level pytest IDs; no class methods) |
| `test_loss.py`                               | 11 | DmlLoss / VanillaLoss component math |
| `test_model.py`                              | 10 | DmlFeedForward forward + autograd path |
| `test_normalization.py`                      | 7  | data normaliser correctness |
| `test_regression.py`                         | 12 | targeted bug regressions (step-noise, GradNorm dim-fix) |
| `test_regression_and_coverage.py`            | 18 | finance + edge-case coverage |
| `test_rmd17_cross_arch_splits.py`            | 2  | rMD17 split-index alignment across PaiNN / GATv2 / MLP |
| `test_smoke.py`                              | 9  | end-to-end pipeline smoke (5-epoch DML training) |
| `test_stats.py`                              | 35 | bootstrap CI, Wilcoxon, Friedman, TOST, Cohen's d |
| `test_svi_calibration.py`                    | 7  | SVI per-(date, maturity) calibration on SPY |
| `test_task_weights_hook_idempotent.py`       | 1  | GradNorm `task_weights` hook idempotence |

## Subset that needs external data

These tests `pytest.importorskip` or `pytest.skip` automatically when
the relevant dataset is missing under `data/`:

- `test_era5_grad_projection_idempotent.py` — needs `data/era5/full_1deg/`
- `test_svi_calibration.py`                 — needs `data/spy_options/spy_processed.npz`
- `test_rmd17_cross_arch_splits.py`         — needs `data/rmd17/rmd17/`

The other ~135 tests run on a fresh checkout without any external data.

## Conventions

- One file per package module under test; one class per public symbol.
- Class-method tests cover the canonical happy path; module-level
  `def test_*` cover unit-level invariants.
- Numerical tolerances are documented inline; default tolerance is
  `1e-6` for analytical comparisons, `1e-3` for trained-model fits.
- A test that runs > 1 second is marked `@pytest.mark.slow`; on CI we
  collect only `not slow`.

## Adding a new test

1. Place it in the file matching the module under test (or create a
   new `test_<module>.py` if no analogue exists).
2. Use the standard pytest discovery (`def test_*` or
   `class Test*: def test_*`). No custom marker required.
3. If the test depends on an external dataset, `pytest.importorskip`
   the relevant loader and document the missing-data behaviour in the
   docstring.
