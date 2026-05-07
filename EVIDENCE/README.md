# `EVIDENCE/` — Per-claim provenance registry

The audit trail that links every numerical claim in the paper to a
specific result directory and producing script. Reviewers who want to
verify a claim go: paper → `claims_registry.csv` → `results/<directory>`
→ producing script.

## Files shipped

| File | Purpose | Rows |
|---|---|---:|
| `claims_registry.csv` | Every numerical / structural claim in the paper, mapped to source files and producing script. | 268 |
| `numbers_source_map.csv` | Every numerical *value* (with units) appearing in the paper, mapped to producing script + verification status. | 120 |
| `README.md` | This file. | — |

## How a claim is verified

```
paper §X "DML cuts force-MAE 25–40× on rMD17"
  └─→ claims_registry.csv : C178
        ├─ source_files: results/molecular_painn/
        ├─ source_scripts: experiments/molecular/run_painn.py
        ├─ target: tab:rmd17-aspirin
        └─ status: supported
  └─→ results/molecular_painn/painn_aspirin_split0_native_EF.json (etc.)
  └─→ producing script: experiments/molecular/run_painn.py
```

Re-running the producing script with `--resume` writes new JSONs only
where the existing JSON is missing; running the aggregator
(`paper/figures/scripts/fig_rmd17_force_mae.py`) regenerates the table.

## Status notation in `claims_registry.csv`

| Status | Count | Meaning |
|---|---:|---|
| `supported` | 235 | claim is backed by JSONs on disk and an aggregator that reproduces the number. |
| `partial`   |  33 | producing script exists but the result corpus is incomplete (cell count below the claim, or aggregator runs but covers a subset). |

The registry tracks experimental claims only. Textbook facts, citation
claims, and compute totals are documented in the paper itself with
inline references; they are not entered as registry rows because there
is no JSON artefact to map onto.

## Croissant integration

Each claim's `source_files` field maps onto the
`recordSet` / `fileSet` definitions in `metadata/croissant.json`; the
Croissant validator confirms the dataset description is internally
consistent (see `DATA.md`).
