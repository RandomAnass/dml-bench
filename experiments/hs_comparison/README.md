# `experiments/hs_comparison/` — Huge & Savine reproduction

Reproduces the Huge & Savine (H&S) 2020 deep-learning protocol with
two specific supplements:

1. **Compute-matched controls** isolate the warmup advantage from a
   plain extra-compute effect.
2. **Per-coordinate $\lambda_j$ ablation** quantifies the impact of
   the per-coordinate gradient-MSE weight from H&S Appendix 2 / 3
   (which our `DmlLoss` does *not* apply by default).

## Output corpus

| Sub-dir | Cells | Producing script |
|---|---:|---|
| `results/compute_matched_controls/` | 176 | `run_compute_matched_controls.py` |
| `results/lambda_j_ablation/`        | 51  | `run_lambda_j_ablation.py` |
| `results/hs_comparison/`            | (see) | `run_hs_comparison.py` |

## Headline findings

- **Compute-matched controls:** `vanilla_no_es` is identical to
  `vanilla` (0 % delta across all 5 unified-comparison datasets).
  Therefore the warmup advantage cannot be explained by Phase 1
  simply running more epochs.
- **$\lambda_j$ ablation:** value-MSE ratio 1.000 across all 5
  datasets — the per-coordinate weight is implementation-level, not
  load-bearing on standardised problems.

## Scripts

| Script | Purpose |
|---|---|
| `run_compute_matched_controls.py` | `vanilla_500_no_es`, `dml_fixed_no_es`, `dml_gradnorm_no_es` at fixed 500-epoch budget |
| `run_lambda_j_ablation.py`        | Per-coordinate weight on/off ablation |
| `run_hs_comparison.py`            | H&S reproduction on the 5 unified-comparison datasets |

## Running

```bash
python experiments/hs_comparison/run_compute_matched_controls.py --gpu 0
python experiments/hs_comparison/run_lambda_j_ablation.py        --gpu 0
python experiments/hs_comparison/run_hs_comparison.py            --gpu 0
```

## Reference

- Huge, B. & Savine, A. (2020). *Differential Machine Learning.*
- Notebooks at
  `https://github.com/differential-machine-learning/notebooks`.

## Implementation note

Our `DmlLoss` matches the H&S **main-text** equation
$\mathcal{L} = \mathrm{MSE}_y + \lambda \cdot \overline{\mathrm{MSE}}_{\nabla y}$
with the dimension-aware split
$w_y = 1/(1 + \lambda d)$, $w_g = \lambda d / (1 + \lambda d)$.
The H&S Appendix 2 / 3 per-coordinate weight $1 / \|\tilde Z_j\|^2$
is not applied; this is documented in
`paper/sections/G_reproducibility_checklist.tex` §App H&S
implementation detail.
