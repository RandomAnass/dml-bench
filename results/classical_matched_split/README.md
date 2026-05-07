# results/classical_matched_split/ — Classical baselines on matched 64% train

## Contents
990 result JSONs (90 baseline_gp + 225 baseline_krr + 225 baseline_rf + 225 dml_fixed + 225 vanilla) on the synthetic-block smooth families (poly_trig, trig, bachelier, black_scholes), trained on the matched 64% train fraction.

## Provenance
- Generated 2026-04-30 by `scripts/run_classical_matched_split.py`.
- Each cell uses `split_train_frac=0.64`, matching the effective training data the neural methods see after their internal 20% validation hold-out.

## Status
- **Canonical** for the §5.1 + Appendix F classical-baseline comparison (post 2026-05-03 fix).
- The pre-fix asymmetric numbers in `results/tier3_benchmark/baseline_*.json` and `results/tier5_extended_baselines/baseline_*.json` are kept on disk as audit copies but **must NOT** be used for the win-rate paragraph in Appendix F. The aggregator at `papers/neurips_DB/evidence/classical_baselines_summary.py` now reads only this directory; re-running it reproduces the matched-train numbers in Appendix F.

## Aggregator
`papers/neurips_DB/evidence/classical_baselines_summary.py` → emits `summary.json` and `summary.md` with the matched-train win-rates (50.0% vs GP, 83.6% vs KRR, 74.7% vs RF).
