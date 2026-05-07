"""Polynomial-in-inputs baselines for DML-Bench domains.

Extends the polynomial baseline (currently only on the Heston barrier;
see `experiments/heston_barrier_4way/analyze_polynomial_baselines.py`)
to additional domains where it is expected to be informative:

  - 1D synthetic poly_trig    (smooth, bounded; poly should fit well)
  - 1D synthetic trig         (high-freq; poly likely poor at low degree)
  - 1D synthetic black_scholes (smooth call-pricer; poly should fit well)
  - 4D SPY-BS-target          (smooth pricer + market noise)

Domains where polynomial is NOT informative are skipped:
  - rMD17 (high-d molecular features → curse of dimensionality)
  - PDE Burgers/Darcy (spatio-temporal fields)
  - ERA5 (spatio-temporal reanalysis grids)

Per the task specification, this directory creates *new* JSON outputs in
`results/polynomial_baselines/` and DOES NOT modify any existing result
file or experiment script that is already cited in the paper.
"""
