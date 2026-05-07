# Classical-baseline source-of-truth note (2026-05-03)

The canonical classical-baseline (GP/KRR/RF) results for Appendix F of the DML-Bench paper live in:

  `results/classical_matched_split/`  (990 cells, 64%-matched train)

The pre-fix asymmetric (80%-train) classical-baseline results in:
  - `results/tier3_benchmark/baseline_{gp,krr,rf}_*.json`
  - `results/tier5_extended_baselines/baseline_{krr,rf}_*.json`

are **kept on disk for audit only** and are not used by the paper. The aggregator at `papers/neurips_DB/evidence/classical_baselines_summary.py` reads only the matched-split corpus.
