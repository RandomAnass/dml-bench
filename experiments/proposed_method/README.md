# `experiments/proposed_method/` — Two-phase warmup ("warmup-as-noise-curriculum")

The warmup method: Phase 1 trains vanilla (value-only) for $W$
epochs, then Phase 2 switches to DML (with GradNorm or fixed-λ
balancing) for the remaining $E - W$ epochs. The mechanism is a
noise-curriculum: Phase 1 insulates the value branch from noisy
gradient labels.

## Output corpus

`results/warmup_experiments/` (~120 cells across SPY and Heston).

## Hypothesis

A two-phase schedule that starts vanilla and gradually introduces
derivative supervision should achieve:

1. value MSE ≤ vanilla (because Phase 1 *is* vanilla);
2. gradient MSE $\ll$ vanilla (because Phase 2 uses DML);
3. a better Pareto point than pure GradNorm.

## Validation evidence

The warmup advantage is **mechanism, not extra-compute**:

- The compute-matched controls
  (`experiments/hs_comparison/run_compute_matched_controls.py` →
  `results/compute_matched_controls/`) confirm that
  `vanilla_no_es` ≡ `vanilla` (0 % delta across all 5 datasets);
  warmup still beats both.
- The Heston barrier label-noise sweep
  (`experiments/heston_barrier_4way/run_noise_curriculum.py` →
  `results/heston_barrier_4way/noise_curriculum/`) shows the warmup
  gain scales monotonically with gradient-label noise: ratios
  1.31 / 1.28 / 0.38 / 0.23 at $\sigma \in \{0, 0.1, 0.5, 1.0\}$.

## Scripts

| Script | Purpose |
|---|---|
| `run_warmup_experiment.py` | Two-phase warmup on SPY (analytical Greeks) and Heston Euler-LRM (noisy MC Greeks). |

## Running

```bash
python experiments/proposed_method/run_warmup_experiment.py --gpu 0
python experiments/proposed_method/run_warmup_experiment.py --gpu 0 --spy
python experiments/proposed_method/run_warmup_experiment.py --gpu 0 --heston
```

## Phase-1 / Phase-2 schedule

- Phase 1 (epochs $0..W$): vanilla loss only ($\lambda = 0$). Early
  stopping on value-only val loss.
- Phase 2 (epochs $W..E$): DML with GradNorm (or fixed-λ if
  `--phase2 fixed`). Early stopping on combined DML val loss.
- Best checkpoint is the best COMBINED val loss across Phase 2 (Phase
  1 best-value-loss ckpt is preserved separately for diagnostic).

The default $W = 250$ matches the convention used in
§5.4 / §5.5; `scripts/ablation_warmup_lr.py` ablates the phase-2 LR
strategy across 7 strategies × 3 targets × 5 seeds and selects
`lr/10` as the best (D022).
