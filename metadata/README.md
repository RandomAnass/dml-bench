# `metadata/` — Croissant 1.1 dataset description

This directory ships the Croissant JSON-LD that describes every data
source used by the benchmark. Croissant is the ML-Commons standard
for ML dataset metadata (`https://github.com/mlcommons/croissant`);
NeurIPS 2026 mandates schema version 1.1 with the RAI block.

## Contents

| File | Purpose |
|---|---|
| `croissant.json` | Single Croissant JSON-LD covering the synthetic generators and the four external data sources |

## Validation

```bash
mlcroissant validate --jsonld metadata/croissant.json
```

Expected output (verified 2026-05-03):

```
W rdf.py:89 WARNING: The JSON-LD `@context` is not standard.
    (non-fatal; non-standard keys: equivalentProperty, samplingRate)
I validate.py:53 Done.
```

The non-standard-context warning is non-fatal; the keys flagged are
domain-specific extensions that do not break consumer tooling.

## What is described

| Source | Croissant `recordSet` |
|---|---|
| Synthetic functions | `synthetic_function_corpus` |
| rMD17               | `rmd17_dataset` |
| PDEBench (Burgers + Darcy) | `pdebench_burgers_dataset`, `pdebench_darcy_dataset` |
| SPY EOD options     | `spy_options_dataset` |
| ERA5 Z500           | `era5_z500_dataset` |

Each entry carries:

- `description` — human-readable abstract;
- `license`    — SPDX identifier;
- `cite_as`    — preferred citation;
- `field`-list with type and unit annotations;
- a `samplingRate` extension recording the cell-tuple cardinality of
  the benchmark grid that consumes it.

## Cross-reference

`paper/sections/A_datasheet.tex` is the human-readable datasheet
counterpart; `DATA.md` at the repo root is the operational
download/preprocessing recipe.

## Camera-ready hand-off

At camera-ready the Croissant JSON is uploaded alongside the
Hugging Face dataset deposit and the Zenodo tarball. The HuggingFace
dataset's auto-generated `dataset-infos.json` will be cross-checked
against this file.
