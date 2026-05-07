# `experiments/extrapolation/` — Trigonometric extrapolation

The trig-extrapolation pillar (Appendix H + closed-form theory in
Appendix E). Targets a 1-D periodic regression problem and asks
whether DML provides any near-extrapolation or far-extrapolation
benefit, sweeping over (i) architecture, (ii) geometry, and (iii)
out-of-support gradient augmentation.

## Output corpus

| Sub-dir | Cells | Producing script | What it shows |
|---|---:|---|---|
| `results/extrapolation_M1/`           | CSV-only (per-row, see `sigma_star_summary.csv`) | `pilot_periodic_extrap.py`, `phase_folded_siren.py` | Architecture-axis: 4 archs × 18 σ × 5 seeds — SIREN is the only architecture with $\sigma^* > 0$ near-extrap |
| `results/extrapolation_M2/`           | 360 + `m2_delta_pct.csv`         | `run_m2.py` (or `bash scripts/launch_m2.sh`); aggregate via `aggregate_m2.py` | Geometry-axis: 2 funcs × 3 dims × 2 ntrains × 2 modes × 3 methods × 5 seeds |
| `results/extrapolation_M1_DA/`        | 2 shards                          | `domain_adapt.py`, `merge_da_shards.py` | OOS-gradient augmentation: SIREN closes the near- and far-extrap gap |
| `results/closed_form_a1_a2/`          | 1 JSON + 2 CSVs                   | `closed_form_a1_a2.py` | Fourier-linear closed-form theory verification |

## Headline findings

- **Near-extrap σ\*:** SIREN 1.157, softplus left-censored at 0,
  Snake left-censored at 0, Fourier-linear 0.518.
- **Geometry-axis:** trig $d=10, n=2048$ halfspace gives +688 % over
  vanilla; trig $d=5, n=512$ halfspace gives +542 %; poly_trig under
  halfspace retains 9.9–68 % of DML benefit.
- **Radial extrapolation:** reduces value MSE 50–98 % on every
  $(d, n_\mathrm{train})$ pair.
- **OOS-gradient augmentation:** near-extrap $\sigma^*$ moves
  1.16 → 3.00 (right-censored); far-extrap $\sigma^*$ moves
  0.007 → 0.98.
- **Closed-form Fourier-linear:** $\sigma^* = 11.83 \sigma_y$ at
  $K=5, \lambda=1$; $\sigma^* = 6.44 \sigma_y$ at $K=1, \lambda=1$; MC
  cross-check matches the closed-form predictions within −3.1 % at
  $N=500$, 4000 trials.

## Scripts

| Script | Purpose |
|---|---|
| `pilot_periodic_extrap.py` | 4-arch × 18-σ × 5-seed grid (M1) |
| `phase_folded_siren.py`    | SIREN phase-folded diagnostic (App H §siren-phase-folded) |
| `run_m2.py`                | Geometry-axis grid (M2) |
| `aggregate_m2.py`          | Aggregator for M2 → `m2_delta_pct.csv` |
| `domain_adapt.py`          | OOS-grad augmentation (M1-DA) |
| `merge_da_shards.py`       | Merge the two DA shards into the canonical CSV |
| `plot_da_results.py`       | DA-results plotter |
| `closed_form_a1_a2.py`     | Fourier-linear closed-form solver (CPU, ~1 min) |
| `_repro_check_m1.py`       | Regression-test helper for M1 (smoke) |
| `_regression_check_da.py`  | Regression-test helper for DA |

## Running

```bash
# Architecture-axis (M1) — SIREN, softplus, Snake, Fourier-linear:
python experiments/extrapolation/pilot_periodic_extrap.py \
    --seeds 5 --wide-sigma --models siren,softplus,snake,fourier_linear
python experiments/extrapolation/phase_folded_siren.py

# Geometry-axis (M2) — full 360-cell grid:
bash scripts/launch_m2.sh
python experiments/extrapolation/aggregate_m2.py

# OOS-gradient augmentation (M1-DA):
python experiments/extrapolation/domain_adapt.py
python experiments/extrapolation/merge_da_shards.py

# Closed-form theory verification (CPU, ~1 minute):
python experiments/extrapolation/closed_form_a1_a2.py
```

## Determinism

Four separate seeds per cell: `target`, `init`, `noise`, `dml-train`.
The `dml-train` seed is fixed across $\sigma$ within a
`(model, target_seed)` cell so the noise sweep is paired.

## Aggregator

`evidence/extrapolation_aggregate.py` consumes the per-row CSVs and
emits the tables behind `tab:arch-sigma-star`,
`tab:geom-delta-pct`, `tab:oos-grad-augment`, and
`tab:closed-form-verification`.

## Theory link

The closed-form Fourier-linear solver is the empirical anchor for
`thm:sigma-star-fourier` in §E. The MC vs closed-form comparison at
$K=5, \lambda=1$ pins the bound to the ${\sim}3$ % level used in the
paper.
