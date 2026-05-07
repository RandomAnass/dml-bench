# Zenodo / Harvard Dataverse upload prerequisites

The NeurIPS 2025 D&B Track call for papers
([CFP](https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks),
[hosting guidelines](https://neurips.cc/Conferences/2025/DataHostingGuidelines))
requires:

- "Datasets and code should be available and accessible to all reviewers,
  ACs and SACs at the time of submission."
- Preferred hosting: **Harvard Dataverse, Kaggle, Hugging Face, OpenML**.
- "If your dataset is accepted, you will be required to make it public by
  the camera ready deadline."
- DOI is **not required at submission**; Zenodo is permitted as a
  non-preferred alternative.

Recommendation for DML-Bench: use **Harvard Dataverse** (in the preferred
list, automatically issues a DOI, integrates with Croissant). Fall back to
Zenodo only if Dataverse is unavailable.

## What we need from you to perform the upload

### Option A — Harvard Dataverse (preferred)

1. **Account** at <https://dataverse.harvard.edu/> (free; sign in with
   institutional, Google, GitHub, ORCID, or email).
2. **Dataverse name** — usually personal or institutional. We will create
   a "DML-Bench" sub-Dataverse under it.
3. **API token** — generate at *Account → API Token → Create Token*.
   We need the token string to upload programmatically (it's a 32-char
   secret; treat as a password).
4. **Dataset metadata fields** — title, authors, description, keywords,
   license. We can pre-populate from the existing `croissant.json`.
5. **Confirm size** — the result corpus is ~21,500 JSONs; total uncompressed
   ~150 MB; well below Dataverse limits.

### Option B — Zenodo (fallback)

1. **Account** at <https://zenodo.org/> (free; sign in with email,
   ORCID, or GitHub). Same author flow.
2. **Personal access token** — *Settings → Applications → Personal
   access tokens → New token* with scopes `deposit:write` and
   `deposit:actions`.
3. **Concept DOI vs version DOI** — choose whether we want a single DOI
   that always points to the latest version (concept DOI, recommended)
   or a per-version DOI.

### Either path

- **License** — current README says CC0 for SPY (Kaggle source); CC-BY-4.0
  for PDEBench Burgers and rMD17; MIT for our code. Confirm before upload.
- **Datasheet** — Gebru et al. 2021 datasheet template (already drafted
  in the independent_writer LaTeX appendix; will be ported to the
  neurips_DB paper).
- **Croissant metadata** — already exists at `croissant.json` in repo root;
  needs a DOI placeholder updated post-upload.

## Programmatic upload (when ready)

The user needs to provide the API token and target Dataverse / Zenodo
collection name. Then a one-shot script (TBD; will be written under
`papers/neurips_DB/scripts/upload_to_dataverse.py`) will:

1. Tar the result corpus and the code snapshot.
2. Create a new dataset in the specified Dataverse / Zenodo collection.
3. Upload the tarball + Croissant + datasheet.
4. Print the resulting DOI.
5. Patch the LaTeX paper's `\href{DOI URL}` placeholder.

## Status (2026-04-29)

- **Awaiting:** user's choice of Harvard Dataverse vs Zenodo, account
  + API token, target collection name.
- **Not blocking:** we can submit the paper with a `\href{TBD-DOI}{Hosted
  at \url{TBD}}` placeholder if the upload is incomplete by submission;
  the camera-ready deadline is the hard one.
