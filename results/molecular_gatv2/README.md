# rMD17 GATv2 results — canonical post-fix corpus

## Provenance

- Generated 2026-04-16 to 2026-04-26 with the GATv2 backbone after the
  vanilla-gradient autodiff fix (commit `dc27c606`, 2026-04-11) and the
  ES-on-total-loss fix (D004, 2026-04-13).
- Grid: 10 molecules × 5 splits × 7 methods (vanilla, dml_fixed,
  dml_fixed_half, dml_gradnorm, dml_relobralo, dml_softmax_balance,
  dml_warmup) = 350 cells.
- Filename pattern: `gnn_md17_<molecule>_split<n>_<method>.json` (legacy
  prefix preserved for aggregator compatibility).

## Why the old D007 deprecation no longer applies

`EVIDENCE/DEVIATIONS_FROM_CANONICAL.md:D007` previously flagged this
directory (under its old name `results/gnn_md17/`) as containing 20
pre-fix files generated 2026-02-15/16, scheduled for replacement by a
fresh rerun in `results/molecular_gatv2/`. The replacement ran
in-place: the 20 pre-fix files were overwritten and the directory now
contains the 350-cell post-fix corpus described above. The directory
was renamed `results/gnn_md17/ → results/molecular_gatv2/` on
2026-05-03 to match the canonical naming convention used for
`molecular_mlp/` and `molecular_painn/`.

## Aggregators

- `paper/figures/scripts/fig_rmd17_force_mae.py`
- `paper/figures/scripts/fig_rmd17_cd_diagram.py`
- (any other aggregator that points at `results/gnn_md17/` should be
  updated; grep before relying on numbers)
