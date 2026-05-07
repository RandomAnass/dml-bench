# `EVIDENCE/` — Per-claim provenance registry

The audit trail that links every numerical claim in the paper to a
specific result directory and producing script. Reviewers who want to
verify a claim go: paper → `claims_registry.csv` → `manifest.md` →
`results/<directory>` → producing script.

## Files

| File | Purpose | Rows |
|---|---|---:|
| `claims_registry.csv`             | Every numerical / structural claim in the paper, mapped to source files and producing script. | 276 |
| `numbers_source_map.csv`          | Every numerical *value* (with units) appearing in the paper, mapped to producing script + verification status. | 121 |
| `manifest.md`                     | Top-level inventory of all result-bearing artefacts under `results/`, with status, owning script, and trainer-fix flag. | (text) |
| `EVIDENCE_AUDIT_TABLE.md`         | Audit table used during the 2026-04-13 audit round. | (text) |
| `paper_inclusions.csv`            | Whitelist of result directories included in the headline corpus framing. | — |
| `paper_exclusions.csv`            | Result directories excluded from the headline corpus framing (e.g. legacy pre-fix snapshots). | — |
| `reviewer_coverage.md`            | Round-by-round mapping of reviewer findings to fixed / deferred / moot. | (text) |
| `DEVIATIONS_FROM_CANONICAL.md`    | Documents every deviation from the canonical method specs (H&S, GradNorm, ReLoBRaLo, …). | (text) |
| `INCIDENT_I-H5_GRADNORM_SILENT_BREAK.md` | Incident report for the I-H5 GradNorm dim-fix break. | (text) |
| `AUDIT_dml_relobralo_legacy_split.md` | Audit of the legacy ReLoBRaLo split applied during the relabel pass. | (text) |
| `hs2018_formula_check.md`         | H&S 2018 formula sanity check. | (text) |
| `warmup_definition.md`            | Canonical warmup definition (Phase 1 vanilla, Phase 2 DML). | (text) |
| `external_papers/`                | Drop folder for paper PDFs that reviewer subagents could not fetch directly. | — |

## How a claim is verified

```
paper §X "DML cuts force-MAE 25–40× on rMD17"
  └─→ claims_registry.csv : C178
        ├─ source_files: results/molecular_painn/
        ├─ source_scripts: experiments/molecular/run_painn.py
        ├─ target: tab:rmd17-aspirin
        └─ status: supported (high risk)
  └─→ manifest.md → "rMD17 PaiNN" row → 300 cells, complete
  └─→ results/molecular_painn/painn_aspirin_split0_native_EF.json (etc.)
  └─→ producing script: experiments/molecular/run_painn.py
```

Re-running the producing script with `--resume` writes new JSONs only
where the existing JSON is missing; running the aggregator
(`paper/figures/scripts/fig_rmd17_force_mae.py`) regenerates the
table.

## Status notation

| Status | Meaning |
|---|---|
| `complete`        | result corpus fully on disk; aggregator runs to completion |
| `partial`         | producing script exists but result dir empty / count below claim |
| `running`         | grid currently being filled in |
| `pre-autodiff`    | result was produced before the autodiff fix (commit `dc27c606`); kept for audit only |
| `superseded`      | claim or directory replaced by a newer entry; kept for audit trail |

## Update procedure

1. Run a new experiment grid; the runner writes JSONs under `results/<benchmark>/`.
2. Append a row to `claims_registry.csv` with the claim text, the
   source files, and the producing script.
3. Append the corresponding numerical value(s) to `numbers_source_map.csv`.
4. Update `manifest.md`'s sub-pillar inventory table.
5. If this update changes a paper claim, document the change in a
   dated update note in the appropriate paper agents file under
   `paper/agents/`.

## Orphan claims

Eight claims are flagged as orphan (paper text without a clear backing
artefact) at the bottom of `manifest.md`. They fall into three classes:

- compute-breakdown claims that need an aggregator pass (C068 / C154 / N059–N064);
- textbook claims that need an explicit citation (C133, C134);
- corpus-regeneration items (C217–C220 higher-order, C249 P6 corruption,
  `tab:headline` definition).

The 2026-05-06 provenance update at
`paper/agents/PROVENANCE_UPDATE_2026-05-06.md` summarises the queue.

## Croissant integration

Each claim's `source_files` field maps onto the
`recordSet`/`fileSet` definitions in `metadata/croissant.json`; the
Croissant validator confirms the dataset description is internally
consistent (see DATA.md).
