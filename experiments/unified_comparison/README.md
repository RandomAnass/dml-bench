# `experiments/unified_comparison/` — Discontinuous-payoff Study 1

Compares the eleven method × label-paradigm combinations on five
discontinuous-payoff datasets (digital BS, barrier BS, Heston-Euler
digital, basket digital $d \in \{1, 7\}$) across ten seeds. The
backbone of §5.3 / §5.4 — every CD-diagram and per-dataset table in
the disc-payoff section sources from this corpus.

## Output corpus

`results/unified_comparison/`

| Sub-dir | Cells | Note |
|---|---:|---|
| `multi_seed/`              | 550 | 11 methods × 5 datasets × 10 seeds — **the canonical corpus** |
| `single_seed/`             |  55 | smoke / pilot artefacts; superseded by `multi_seed/` |
| `multi_seed_v1_lrm_eval/`  | 550 | earlier LRM-eval round; kept for diff |

## Headline findings

- **CD ranking (Friedman $p = 7.2 \times 10^{-5}$):**
  warmup-fuzzy 1.8/12, vanilla 5.8/12, pathwise+GradNorm and
  pathwise+ReLoBRaLo tie at 11.2 mean rank.
- **Pathwise methods rank 7–10** on every dataset (degrade value MSE
  by 117–562 %).
- **Fuzzy methods incur < 2 % value penalty** on every disc-payoff
  dataset.
- **Kendall-τ between value-MSE and gradient-MSE rankings:** 0.67 (the
  Pareto frontier is non-trivial; only vanilla and warmup-fuzzy are
  non-dominated).

## The 11 methods

```
Pathwise (biased ≡ 0 for digital):  1. vanilla
                                     2. dml_fixed (H&S λ=1)
                                     3. dml_gradnorm
                                     4. dml_relobralo
                                     5. dml_warmup (vanilla → GradNorm)
LRM (Glasserman & Karmarkar 2025):   6. dml_lrm
                                     7. dml_gradnorm_lrm
                                     8. dml_warmup_lrm
Fuzzy (Savine 2018 + ours):           9. dml_fuzzy
                                     10. dml_gradnorm_fuzzy
                                     11. dml_warmup_fuzzy (NOVEL)
```

## The 5 datasets

| Dataset | Dim | Notes |
|---|---|---|
| Digital BS         | 1     | G&K's showcase |
| Barrier BS         | 1     | multi-step → noisy LRM labels |
| Heston-Euler digital | 1   | stochastic vol → LRM variance explodes |
| Basket digital     | 1 / 7 | dimension-scaling test |

## Scripts

| Script | Purpose |
|---|---|
| `run_unified_experiment.py`  | The runner. `--mode {smoke_test, single_seed, multi_seed}`. |
| `analyze_unified.py`         | Per-dataset summary CSV. |
| `plot_unified.py`            | Generates the unified-figures family (`figures/unified/*.pdf`). |

## Running

```bash
# Smoke (~5 min):
python experiments/unified_comparison/run_unified_experiment.py --mode smoke_test --gpu 0

# Single-seed pilot (~30 min):
python experiments/unified_comparison/run_unified_experiment.py --mode single_seed --gpu 0

# Full multi-seed (~12 GPU-h):
python experiments/unified_comparison/run_unified_experiment.py --mode multi_seed --gpu 0

# Analyze without running:
python experiments/unified_comparison/run_unified_experiment.py --analyze-only
```

## Aggregators

| Aggregator | Output |
|---|---|
| `paper/figures/scripts/cd_diagram.py`         | CD diagram (`fig:cd-disc`) |
| `paper/figures/scripts/fig_unified_pareto.py` | Pareto frontier (`fig:tradeoff`) |
| `evidence/unified_summary.py`                 | `tab:unified_summary`, `tab:unified_best`, `tab:ranking` |

## Method-paradigm registry

The 11-method registry is constructed in
`run_unified_experiment.py:METHODS` from the cross-product of
`{vanilla, dml_fixed, dml_gradnorm, dml_relobralo, dml_warmup}` and
`{pathwise, lrm, fuzzy}` minus the unsupported combinations
(vanilla × any-label = vanilla; the gradnorm × lrm and relobralo × lrm
combinations are tested only in this experiment).
