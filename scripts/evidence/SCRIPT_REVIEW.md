# Evidence-script review

Reviewer: independent code-review agent  
Date: 2026-04-28  
Repo root: `.`  
Scripts reviewed (read-only; no execution):

1. `papers/neurips_DB/evidence/fig_1d_pathwise_dirac.py`
2. `papers/neurips_DB/evidence/sigma_star_bca.py`
3. `papers/neurips_DB/evidence/fig_sigma_star_curve.py`
4. `papers/neurips_DB/evidence/fig_spy_distribution_shift.py`
5. `papers/neurips_DB/evidence/classical_baselines_summary.py`

All three tier directories and the tier-5 extended-baseline directory were inspected
directly (not via script execution) to verify field names against script assumptions.

---

## fig_1d_pathwise_dirac.py

- **Correctness:** Correct throughout. The three panels faithfully represent (A) the digital
  payoff `1[x>K]`, (B) a stylised Dirac arrow at K, and (C) the fuzzy-kernel approximation
  `(1/ε)·1[|x−K|<ε/2]`. The kernel integrates to 1 over its support as the docstring claims
  (width ε, height 1/ε → area = 1). The `ROOT = Path(__file__).resolve().parents[3]` resolves
  to the repo root, and `OUT.parent.mkdir(parents=True, exist_ok=True)` ensures the output
  directory is created regardless of CWD. No data dependencies; no field-name issues.

- **Robustness:** The script is self-contained. The only failure mode is a missing LaTeX/PDF
  renderer, which is handled by `matplotlib.use("Agg")`. No external data are read. Fail-safe.

- **Readability:** Clean and compact. One minor note: the suptitle uses `y=1.02`, which can be
  clipped without `bbox_inches="tight"`; `fig.savefig(..., bbox_inches="tight")` is already
  present, so this is fine in practice.

- **Output:** The final `print(f"figure_id=F_1D_dirac panels=3 K={K} eps={EPS} output={OUT}")`
  gives enough information to spot-check the output PDF.

- **Verdict:** SAFE TO RUN — no issues.

---

## sigma_star_bca.py

- **Correctness:** **HARD BUG B1 (script produces zero output).** The JSON result files in all
  three tier directories use the key `"func_type"` for the function family name. The script
  reads `func = r.get("dataset")` (line 66), which returns `None` for every file. The
  subsequent guard `if func not in SMOOTH_FUNCS: continue` (line 67) then skips every record,
  so `paired` is an empty `defaultdict`. `main()` detects the empty dict and calls
  `sys.exit(1)` — the script aborts with no output rather than silently writing bad JSON.
  The script cannot produce any result in its current form.

  **HARD BUG B2 (lambda-ablation contamination, latent until B1 is fixed).** The tier-3
  directory contains 320 lambda-ablation files (e.g.
  `trig_d100_n4096_noise0.0_s1000_dml_fixed_lam0.01.json`) whose `"method"` field is
  `"dml_fixed"` and whose `(func, dim, n_samples, noise_level, seed)` tuple is identical to
  the canonical runs in tier-1/tier-3. The config key used in `by_config` (line 77) does not
  include `lambda`, so the last file read at OS glob order wins. Depending on the glob order,
  any of the five lambda values (0.001, 0.01, 0.1, 1.0, 10.0) may populate the `dml_fixed`
  slot, corrupting the paired comparison. Measured: 320 potentially overwritten entries across
  the smooth-function families once the dataset field is corrected.

  **MEDIUM M1 (false BCa claim).** The module docstring, function docstring, and inline comment
  all claim "BCa bootstrap 95% CI". The actual computation (lines 169–172) is
  `np.percentile(valid, [2.5, 97.5])` — a plain percentile bootstrap, not BCa. The `ci_method`
  field in the output JSON is set to `"percentile_2.5_97.5_over_valid_bootstrap_resamples"`,
  which accurately describes what is computed, but the claim in the module docstring (line 5)
  and function docstring (line 124) contradicts this. `scipy.stats.bootstrap` is imported as
  `scipy_bootstrap` (line 30) but never called anywhere in the file.

  **MEDIUM M2 (misleading docstring — bootstrap stratification).** The docstring for
  `bootstrap_sigma_star` says "seed-stratified resamples of the configs". The actual
  implementation resamples `pairs_by_sigma[s]` independently for each sigma level (lines
  141–146). Because each element of `pairs_by_sigma[s]` is already an aggregated `(target_mse,
  baseline_mse)` tuple (not identified by seed), the resampling is per-sigma-level pair
  resampling, not a seed-stratified resample across all sigma levels simultaneously. The
  statistics produced are coherent (percentile CI on σ*), but the docstring is inaccurate.

  The `sigma_star()` interpolation logic (lines 109–119) correctly finds the first downward
  crossing through 0.5 using linear interpolation. The NaN propagation (skipping NaN edges,
  returning NaN when no crossing found) is correct. The `denom == 0` guard prevents a ZeroDivision.

- **Robustness:** Missing tier directories are handled via `if not tdir.exists(): continue`.
  Malformed JSON is caught by the bare `except Exception`. The `if not paired: sys.exit(1)`
  guard prevents writing an empty JSON silently. One gap: if `paired` is non-empty but no
  SMOOTH_FUNCS are present in it (e.g., only `step` or `black_scholes` data came through),
  `out["results"]` will be `{}` and the TXT will be blank — no error is raised. This is a
  silent-useless-output scenario but unlikely given the `"dataset"` fix.

- **Readability:** The per-function progress print `{func}: n_sigmas=... n_pairs=...` is
  useful for verification. The `ci_method` string stored in the JSON is a good self-describing
  field.

- **Output:** When working, the human-readable `.txt` and the `.json` together are sufficient
  for spot-checking. The `n_undefined` field in the JSON enables sanity-checking bootstrap
  stability.

- **Verdict:** NEEDS FIX — apply patches P1, P2, P3 (see below) before running.

---

## fig_sigma_star_curve.py

- **Correctness:** **HARD BUG B1 (same as sigma_star_bca.py).** `collect_winrate()` also reads
  `func = r.get("dataset")` (line 57), which is always `None`. All records are skipped;
  `paired` is empty; `main()` calls `sys.exit(1)` at line 100.

  **HARD BUG B2 (same lambda contamination, latent).** `collect_winrate()` uses the same key
  structure `(func, dim, n, float(sigma), seed)` without including `lambda`, so the 320
  lambda-ablation files in tier-3 create the same last-write-wins overwrite problem once the
  dataset field is fixed.

  The statistical method in `winrate_with_ci()` (lines 76–93) is correct: a paired win-rate
  computed per sigma, with a percentile bootstrap CI. The label in the plot describes it as
  "Wilson 95% CI via bootstrap" in the docstring (line 77), but a Wilson interval is a
  closed-form analytic formula, not a bootstrap — this is another docstring mismatch. The
  implementation is a bootstrap, not Wilson. The plot's visual output would be correct; only
  the docstring label is wrong.

  The σ* vertical lines are drawn from the `sigma_star_bca.json` file (lines 102–128). If that
  file is absent or has `null` for a function (because `sigma_star_bca.py` failed), the
  `axvline` is simply skipped — graceful degradation.

- **Robustness:** Same missing-tier-dir handling as `sigma_star_bca.py`. `sys.exit(1)` on empty
  `paired`. SIGMA_STAR_JSON absence is silently tolerated (lines 102–107). The `rng` object is
  shared across functions, ensuring bootstrap results are reproducible from the seed set at line
  110.

- **Readability:** The final `print(f"figure_id=F_sigma_star n_funcs=...")` is adequate. A
  per-function print of the win-rate values before plotting would help spot-checking.

- **Output:** The figure is the deliverable; no supplementary text file is written. Recommend
  adding a short TSV or print of (func, sigma, winrate, lo, hi) so the figure can be verified
  without re-running.

- **Verdict:** NEEDS FIX — apply patches P1 and P2 (identical fixes to sigma_star_bca.py, see
  below). Also correct the "Wilson" docstring to "bootstrap percentile".

---

## fig_spy_distribution_shift.py

- **Correctness:** The core split and histogram logic are correct. `dates < CUTOFF` and
  `dates >= CUTOFF` partition the data cleanly because the date strings are ISO-8601 format
  and lexicographic comparison is identical to chronological comparison. The actual date range
  in the data is 2020-01-02 to 2022-12-30, confirming the temporal split is meaningful.

  **MEDIUM M3 (legend label off by one day).** The histogram legend (line 65) reads
  `f"train (≤{CUTOFF}, n={n_train:,})"` but the mask is `dates < CUTOFF` (strictly less
  than). The date 2021-07-01 appears in the dataset (verified: 2,458 records) and belongs to
  the *test* set, not train. The legend should read `< {CUTOFF}` or `≤ 2021-06-30`. This is a
  0.16% data-attribution error in the legend text only; the data split itself is correct.

  **MEDIUM M4 (docstring overclaims).** The module docstring (line 11) says the figure draws
  "the four model-input features (moneyness, T, r, iv) *plus the implied-vol-of-vol summary
  across train/test*". The code produces exactly four panels — no ivol-of-ivol panel is
  implemented. Remove the "plus the implied-vol-of-vol summary" clause from the docstring.

  The `zip(axes, FEATURE_NAMES, FEATURE_LIMITS)` loop correctly maps features to subplots
  (i=0..3 → X columns 0..3, excluding X[:,4]=log_volume). The feature-name list
  `["moneyness", "T (yrs)", "r", "iv"]` matches the npz `feature_names` array order
  (verified). The summary `.txt` file (lines 83–98) uses the same masks as the plot, so it is
  consistent.

- **Robustness:** `sys.exit(1)` if `DATA` is missing. No other external dependencies. The
  trimming (`train_vis`, `test_vis`) for visualization is non-destructive (the `n_train`,
  `n_test` counts used in labels are computed before trimming). However, the `density=True`
  histogram is drawn on the *trimmed* subset, which means the density is computed over the
  visible range only, not the full distribution. For features with long tails this can
  overstate the in-range density. This is acceptable for an illustrative figure but should be
  noted.

- **Readability:** The summary `.txt` printed to stdout doubles as a spot-check tool. The
  figure title uses `y=1.04` which risks clipping without `bbox_inches="tight"` — that
  argument is already present, so fine.

- **Output:** The combined `print(f"figure_id=F_spy_dist n_train=... n_test=...")` plus the
  printed summary table give enough information to verify the split.

- **Verdict:** NEEDS FIX — apply patch P4 (legend label, minor). Docstring fix is P5 (cosmetic,
  safe to defer).

---

## classical_baselines_summary.py

- **Correctness:** **HARD BUG B1 (same dataset field).** Line 50 reads
  `func = r.get("dataset")` which is always `None`; line 55 then skips every record
  (`None in (func, dim, n, seed, val)` is True). `by_config` stays empty.

  **HARD BUG B3 (method-name mismatch).** The `CLASSICAL_METHODS` list (line 25) contains
  `["KRR", "RF", "GP", "krr", "rf", "gp", "vanilla_KRR", "vanilla_RF"]`. The actual method
  values in `results/tier5_extended_baselines/` are `"baseline_krr"` and `"baseline_rf"`
  (verified by reading all 40 files). None of the script's classical-method names match the
  actual data. Even after fixing B1, `by_config[key]` would contain `"baseline_krr"` /
  `"baseline_rf"` keys, and the tally loop (line 64, `if NEURAL_TARGET in methods_seen and
  classical in methods_seen`) would find zero matches for every entry in `CLASSICAL_METHODS`.
  The output would be an empty comparisons dict and the "No paired configs found" markdown.

  The `NEURAL_TARGET = "dml_fixed"` is correct — this method name does appear in tier-5.

  The `margins` accumulation (lines 70–73) mixes positive values (wins: `c > t`, so `c - t > 0`)
  and negative values (losses: `c < t`, so `c - t < 0`) into the same list. The resulting
  `np.median(margins)` is therefore a median over signed deltas, which is ambiguous: a
  `median_margin` near zero could mean the method ties on average or that wins and losses
  cancel. This is a minor design issue; the field is not quoted in the paper paragraph, only
  the win-rate is.

- **Robustness:** Missing `EXTENDED` directory is gracefully handled (lines 32–41): an error
  JSON and a placeholder MD are written and the function returns. This is good defensive
  practice. If `EXTENDED` exists but is empty after filtering, the "No paired configs found"
  branch (lines 88–94) also handles it gracefully.

- **Readability:** The structure is clear. The suggested paragraph at the bottom of the MD
  output is convenient for the writer.

- **Output:** If the bugs were fixed, the MD table and JSON would be self-explanatory.
  Currently the script exits producing a misleading "No paired configs found" message even when
  data exists.

- **Verdict:** NEEDS FIX — apply patches P1 and P6 before running.

---

## Cross-cutting issues

1. **"dataset" vs "func_type" mismatch affects three scripts.** Every result JSON in tier-1,
   tier-2, tier-3, and tier-5 uses `"func_type"` for the function-family label. Three scripts
   (`sigma_star_bca.py`, `fig_sigma_star_curve.py`, `classical_baselines_summary.py`) read
   `r.get("dataset")`. The field `"dataset"` does not exist in any result file. This is the
   single highest-priority fix.

2. **Lambda-ablation contamination in tier-3.** 320 files in `results/tier3_benchmark/`
   carry `"method": "dml_fixed"` with non-canonical lambda values (0.001, 0.01, 0.1, 10.0).
   They share the same `(func_type, dim, n_samples, noise_level, seed)` key as canonical runs.
   The fix is to filter by `r.get("lambda") == 1.0` in both `sigma_star_bca.py` and
   `fig_sigma_star_curve.py` (or equivalently add `lambda` to the config key), to exclude
   ablation runs from the win-rate aggregation.

3. **Unused import.** `scipy.stats.bootstrap` is imported in `sigma_star_bca.py` but never
   called. Remove the import and correct the BCa/percentile wording in the docstring.

4. **Unclosed file handles.** All three aggregator scripts use `json.load(open(p))` without a
   context manager. This is a minor resource-management issue; CPython's reference counting
   closes them quickly but it is not best practice for a script iterating hundreds of files.

5. **No `noise` key exists in any tier.** The fallback chain `r.get("noise") if "noise" in r
   else r.get("noise_level")` works correctly because `"noise"` is never present; the
   `else`-branch always fires. This is safe but the `if`-branch is dead code in the current
   data. Consider simplifying to `r.get("noise_level")` after confirming no future tier will
   introduce `"noise"`.

---

## Patches (specific code changes)

### P1 — Fix `"dataset"` → `"func_type"` (applies to three scripts)

**sigma_star_bca.py, line 66:**
```python
# BEFORE
func = r.get("dataset")
# AFTER
func = r.get("func_type") or r.get("dataset")
```

**fig_sigma_star_curve.py, line 57:**
```python
# BEFORE
func = r.get("dataset")
# AFTER
func = r.get("func_type") or r.get("dataset")
```

**classical_baselines_summary.py, line 50:**
```python
# BEFORE
func = r.get("dataset")
# AFTER
func = r.get("func_type") or r.get("dataset")
```

The `or r.get("dataset")` fallback preserves compatibility with any future JSONs that use
the `"dataset"` field name.

---

### P2 — Filter out lambda ablations (applies to sigma_star_bca.py and fig_sigma_star_curve.py)

In both `load_pairs()` (sigma_star_bca.py, after line 70) and `collect_winrate()`
(fig_sigma_star_curve.py, after line 63), add a lambda guard:

**sigma_star_bca.py — add after line 70 (after reading `val`):**
```python
            # Exclude non-canonical lambda ablations (lambda != 1.0)
            lam = r.get("lambda", 1.0)
            if lam != 1.0:
                continue
```

**fig_sigma_star_curve.py — add after line 64 (after reading `val`):**
```python
            # Exclude non-canonical lambda ablations
            lam = r.get("lambda", 1.0)
            if lam != 1.0:
                continue
```

---

### P3 — Remove false BCa claim and unused scipy import (sigma_star_bca.py)

**Line 5 (module docstring):** change "BCa bootstrap 95% CI" → "percentile bootstrap 95% CI".

**Line 15 (module docstring):** change "BCa bootstrap CI on σ*" → "percentile bootstrap CI on σ*".

**Line 30:** remove `from scipy.stats import bootstrap as scipy_bootstrap`.

**Lines 123–124 (function docstring):** change "BCa bootstrap on σ* via seed-stratified
resampling of pairs" → "percentile bootstrap on σ* via per-sigma-level pair resampling".

---

### P4 — Fix legend label in fig_spy_distribution_shift.py (minor, affects paper text)

**fig_spy_distribution_shift.py, line 65:**
```python
# BEFORE
density=True, label=f"train (≤{CUTOFF}, n={n_train:,})")
# AFTER
density=True, label=f"train (< {CUTOFF}, n={n_train:,})")
```

---

### P5 — Fix docstring overclaim in fig_spy_distribution_shift.py (cosmetic)

**Lines 10–12 (module docstring):** remove the clause  
`"plus the implied-vol-of-vol summary across train/test"`.  
The script produces four panels, not five.

---

### P6 — Fix CLASSICAL_METHODS names in classical_baselines_summary.py

**Line 25:**
```python
# BEFORE
CLASSICAL_METHODS = ["KRR", "RF", "GP", "krr", "rf", "gp", "vanilla_KRR", "vanilla_RF"]
# AFTER
CLASSICAL_METHODS = ["baseline_krr", "baseline_rf", "baseline_gp",
                     "KRR", "RF", "GP", "krr", "rf", "gp",
                     "vanilla_KRR", "vanilla_RF"]
```

Placing the actual tier-5 names first ensures matches are found. The legacy aliases are kept
for forward compatibility. Alternatively, strip the "baseline\_" prefix uniformly when reading.

---

## Summary table

| Script | Status | Blocking bugs |
|---|---|---|
| fig_1d_pathwise_dirac.py | SAFE TO RUN | none |
| sigma_star_bca.py | NEEDS FIX | B1 (P1), B2 (P2), M1/M2 (P3) |
| fig_sigma_star_curve.py | NEEDS FIX | B1 (P1), B2 (P2) |
| fig_spy_distribution_shift.py | NEEDS FIX (minor) | M3 (P4), M4 (P5) |
| classical_baselines_summary.py | NEEDS FIX | B1 (P1), B3 (P6) |
