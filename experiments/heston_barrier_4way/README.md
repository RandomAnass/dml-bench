# `experiments/heston_barrier_4way/` — Heston path-dependent barrier

The Heston-barrier sub-pillar of the paper (§5.2 + Appendix). The
target is a continuously-monitored Heston barrier-knockout option
priced under the Heston SDE; the supervisor compares the four label
paradigms (pathwise / fuzzy / LRM / vanilla) across narrow vs wide
spot ranges and a polynomial-in-spot baseline.

## Output corpus

All under `results/heston_barrier_4way/`:

| Sub-dir | Cells | Producing script | What it shows |
|---|---:|---|---|
| `multi_seed/`             | 60  | `run_multi_seed.py`             | Narrow `[0.7K, 1.3K]`, 5 seeds × 12 methods |
| `multi_seed_widerange/`   | 61  | `run_multi_seed_widerange.py`   | Wide   `[0.5K, 1.5K]`, 5 seeds × 12 methods |
| `v0_extension/`           | 22  | `run_v0_extension.py`           | $V_0$-extension grid + polynomial-in-spot baseline |
| `bs_n_sweep/`             | 149 | `run_bs_n_sweep.py`             | BS barrier with $n \in \{2, 4, 8, 16\}$ monitor steps |
| `eps_sweep_fuzzy/`        | 25  | `run_eps_sweep.py`              | Fuzzy-bandwidth $\varepsilon$ sweep |
| `cg_variance_check/`      | 40  | `run_cg_variance.py`            | LRM nstep-vs-variance ablation |
| `noise_curriculum/`       | 25  | `run_noise_curriculum.py`       | Label-noise sweep $\sigma \in \{0, 0.1, 0.5, 1.0\}$ → ratios 1.31/1.28/0.38/0.23 |
| `gk_replication/`         | 21  | `run_gk_replication.py`         | Glasserman–Kou replication |
| `bs_reproduction/`        | 20  | `run_bs_reproduction.py`        | BS-form Heston reproduction |

## Headline findings

- **Narrow spot range:** every DML configuration beats vanilla on price
  MSE; `dml_fuzzy_warmup` cuts the price MSE 9× (2.10e-5 → 0.23e-5),
  but a six-parameter quintic polynomial in $S_0$ fits price MSE
  0.016e-5 — beating every DML method.
- **Wide spot range:** vanilla delta MSE 2.43e-3, quintic 1.67e-3,
  best DML beats quintic delta MSE by a 0.11× ratio.
- **n-sweep:** `dml_fuzzy_warmup` is the best at every $n \in \{2, 4, 8, 16\}$.
  Pathwise+warmup degrades 8× at $n=2$ to 167× at $n=16$ (missing
  Dirac per monitor).

## Running

```bash
# Narrow + wide multi-seed (~5 GPU-h on one A6000):
python experiments/heston_barrier_4way/run_multi_seed.py            --gpu 0 --resume
python experiments/heston_barrier_4way/run_multi_seed_widerange.py  --gpu 0 --resume

# Polynomial-in-spot baseline + V0 extension:
python experiments/heston_barrier_4way/run_v0_extension.py          --gpu 0 --resume

# BS barrier multi-step n-sweep (~2 h):
python experiments/heston_barrier_4way/run_bs_n_sweep.py            --gpu 0 --resume

# Smaller ablations:
python experiments/heston_barrier_4way/run_eps_sweep.py             --gpu 0 --resume
python experiments/heston_barrier_4way/run_cg_variance.py           --gpu 0 --resume
python experiments/heston_barrier_4way/run_noise_curriculum.py      --gpu 0 --resume
python experiments/heston_barrier_4way/run_gk_replication.py        --gpu 0 --resume
python experiments/heston_barrier_4way/run_bs_reproduction.py       --gpu 0 --resume
```

## Aggregator

`evidence/heston_barrier_summary.py` (or
`papers/neurips_DB/evidence/...` in the prior draft tree) consumes the
per-cell JSONs and emits the table behind `tab:heston-barrier-multi-seed`.

## Polynomial-in-spot baseline

`run_v0_extension.py` writes `polynomial_2d_baselines.json` containing
a quintic-in-$S_0$ fit on the held-out test points. This is the
simplest closed-form alternative to the DML neural networks; it is
significantly stronger than any DML method on narrow-range price MSE
and is the comparator missing from prior DML benchmarks (huge2020,
glasserman2025, sakuma2026 0dte).

## Implementation notes

- The pilot script (`run_pilot.py`) is the single-seed prototype; the
  multi-seed runners import it and override `SEEDS` before calling
  through.
- Default early-stopping patience is 50 for backwards compatibility;
  pass `--early-stopping-patience 200` (or 999999 to disable) to avoid
  the cross-seed ES artefact described in `multi_seed_deep_analysis.md`.
- The Heston path SDE uses the Andersen QE scheme; the barrier monitor
  is continuous-via-Brownian-bridge per Glasserman–Kou.

## Tests

`tests/test_heston_barrier.py` covers the path SDE, the barrier
monitor, the LRM label construction, and the fuzzy-mollifier
pre-conditions.
