# Median paired log-MSE-ratio + cluster bootstrap CI (S1+S5)

DML method: `dml_fixed` (λ=1) vs `vanilla`. Bootstrap: 1000 resamples.
Cluster definition: synthetic = (function, dim, n_train, σ); SPY = fold_idx.
Negative log-ratio means DML beats vanilla; |log_ratio| = orders of magnitude.

## Synthetic σ=0, smooth families only

- **value MSE**: median log10 = -1.110 = 0.0776× (13× reduction); cluster CI [10^-1.30, 10^-0.96]; n_clusters=84, n_rows=545.
- **gradient MSE**: median log10 = -1.275 = 0.0531× (19× reduction); cluster CI [10^-1.43, 10^-1.05]; n_clusters=84, n_rows=545.

## Synthetic σ=0, all six families

- **value MSE**: median log10 = -0.556 = 0.278× (4× reduction); cluster CI [10^-0.82, 10^-0.15]; n_clusters=174, n_rows=1050.
- **gradient MSE**: median log10 = -1.069 = 0.0852× (12× reduction); cluster CI [10^-1.27, 10^-0.93]; n_clusters=174, n_rows=1050.

## Synthetic full grid (all σ)

- **value MSE**: median log10 = +0.014 = 1.03× (+3.2% increase); cluster CI [10^-0.01, 10^+0.02]; n_clusters=538, n_rows=2870.
- **gradient MSE**: median log10 = -0.817 = 0.152× (7× reduction); cluster CI [10^-0.91, 10^-0.72]; n_clusters=538, n_rows=2870.

## SPY purged walk-forward CV

- **value MSE**: median log10 = +0.360 = 2.29× (+129.2% increase); cluster CI [10^-0.15, 10^+0.45]; n_clusters=5, n_rows=50.
- **gradient MSE**: median log10 = -2.431 = 0.00371× (270× reduction); cluster CI [10^-2.50, 10^-2.38]; n_clusters=5, n_rows=50.

## SPY temporal split

- **value MSE**: median log10 = -0.101 = 0.793× (1× reduction); percentile CI [10^-0.11, 10^-0.10]; n_clusters=n/a, n_rows=10.
- **gradient MSE**: median log10 = -2.425 = 0.00376× (266× reduction); percentile CI [10^-2.44, 10^-2.41]; n_clusters=n/a, n_rows=10.

