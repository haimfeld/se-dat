# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2026-10-10

### Added

- **Target analysis** (`sedat.target`): `analyze_target(df, target)` returns a
  `TargetAnalysis` — per-feature effect sizes (Pearson / point-biserial /
  correlation ratio η / Cramér's V) chosen by the target's inferred role,
  binning-based mutual information (with normalised MI), a ranked summary table,
  and leakage detection. Features whose effect size *or* normalised MI meet
  `leakage_threshold` (default 0.98) are flagged and warn by default. High-
  cardinality categorical and near-unique text columns are safely bucketed so
  they never look like perfect predictors. Wired into `EDAReport` as
  `target_summary` and `leakage_features`; the target column gets its own
  `role` in the profile.
- **Data dictionary support** (`sedat.datadict`): `DataDictionary` loads from a
  dict mapping, JSON file or CSV, and drives the whole pipeline — column type
  overrides (with notes in the profile), range constraints for plausibility
  checks, derived-column definitions, and target-column marking. Passed as the
  `dictionary=` argument to `EDAReport.create()` whose new `set_target()` /
  dictionary-driven target resolution keep the target out of the feature list.
- **Plausibility checks** (`sedat.quality`): `check_ranges(df, range_rules,
  non_negative, percentages)` reports out-of-range / negative / non-percentage
  values per column; `duplicate_rows(df)` counts full-row duplicates.
  Surfaced as `EDAReport.plausibility` and `plausibility_summary`, with a new
  "Plausibility checks" section in the HTML report.
- **Transform suggestions** (`sedat.transforms`): `suggest_transforms(df)`
  proposes `log1p` for heavily skewed non-negative columns or a fitted
  `box_cox` (with shift otherwise). `TransformPlan.apply / apply_all / inverse`
  give round-trip transformations; `EDAReport.transform_summary` and
  `apply_all_transforms()` expose them on the facade.
- **Collinearity diagnostics** (`sedat.correlations`): `variance_inflation_factors(df)`
  and `condition_number(df)` to catch multi-way collinearity that pairwise
  correlation misses. Vanished-pair (collapsed) categorical flags now also
  appear as `collapsed_summary` / `flagged_pairs_collapsed`. `EDAReport` exposes
  `vif_summary`, `condition_number` and `collapsed_correlation_summary`; VIF is
  skipped beyond `max_vif_features` (default 50) numeric columns to stay fast.
- **New outlier methods** (`sedat.outliers`): `"mad"` (modified z-score, median-
  based, threshold 3.5) and `"log_iqr"` (IQR on log-scaled positive-skewed
  values) alongside the existing `"iqr"` / `"zscore"`.
- **Constant / duplicate detection**: constant columns are now flagged as
  `is_constant` on their column profile (with a note), and full-row duplicates
  are counted via `duplicate_rows(df)` and `EDAReport.duplicate_row_count`.
- **Scalability controls**: `EDAReport.create()` accepts `sample=` (cap on
  analysis rows, default `DEFAULT_FAST_SAMPLE` = 50 000) and `max_columns=`
  (cap on correlation columns, default 100) with `fast=True` turning both on and
  skipping VIF; the full frame is kept for plan application. Correlations can
  run parallel via `n_jobs=` and show progress with `progress=True`
  (see `SEDAT_PROGRESS` env var).
- **Encoding / persistence hardening**: `EncodingPlan.apply_all(..., warn_target_leakage=True)`
  warns when a target encoding is fit on the data it is applied to (fit on a
  training frame, save, then apply to held-out data instead). Saved plan
  envelopes now record `se_dat_version` alongside `schema_version`; plans from
  a missing-schema older release load as v1, and newer-version plans are
  rejected with a clear error.
- **Pre-commit hooks**: added `.pre-commit-config.yaml` with trailing-whitespace,
  end-of-file-fixer, check-yaml, check-toml, check-added-large-files,
  isort (black profile), ruff linter (`--fix`) and ruff formatter.
  Ruff config added to `pyproject.toml` targeting Python 3.9+ with rules
  E, F, UP, B, SIM, RUF.
- **Testing & CI**: pytest suite covering type-inference edge cases (all-null
  columns, single-unique values, numeric-looking strings, binary-like strings,
  high-cardinality text, datetime formats, mixed-type columns); GitHub Actions
  workflow running pytest on Python 3.9–3.12; tag-triggered PyPI publishing via
  trusted publishing (`pypa/gh-action-pypi-publish`).
- **Missing-value handling** (`sedat.imputation`): `suggest_imputations(df)`
  returns an `ImputationPlan` — median for numeric columns (mean noted as an
  alternative), mode for categorical-like columns, and `drop_column`
  suggestions above a configurable missing-percentage threshold (default 50%).
  Wired into `EDAReport` as `imputation_summary` and
  `apply_all_imputations()`.
- **Outlier detection** (`sedat.outliers`): `detect_outliers(df,
  method="iqr"|"zscore")` returns per-column counts, percentages and flagged
  row indices; one-line notes are surfaced in `profile.summary` and the full
  report stored as `EDAReport.outliers`.
- **HTML report export** (`sedat.report_html`): `EDAReport.to_html(path)`
  writes a static self-contained page with all summary tables and correlation
  heatmaps embedded as base64 PNGs. No JavaScript or external assets.
- **scikit-learn interoperability** (`sedat.sklearn_compat`, optional extra
  `se-dat[sklearn]`): `EncodingPlan.to_column_transformer()` mirrors the plan
  with native `OrdinalEncoder` / `OneHotEncoder` / `TargetEncoder`
  transformers plus passthrough for untouched columns.
- **Datetime feature extraction** (`sedat.datetime_features`):
  `suggest_datetime_features(df)` proposes year/month/day/day-of-week/
  is-weekend (and hour when a time component is present) per detected datetime
  column, applied as new `{col}_year`, `{col}_dow`, ... columns.
- **Plan persistence** (`sedat.persistence`): `.save(path)` / `.load(path)` on
  `EncodingPlan`, `ImputationPlan` and `DatetimeFeaturePlan`; JSON payloads
  store fitted decisions so loaded plans re-apply to new DataFrames with a
  stable output schema.

### Changed

- One-hot encoding now aligns its dummy columns to the categories captured at
  suggest time: unseen categories produce all-zero rows instead of changing
  the output schema. Binary and ordinal encodings freeze their value→code maps
  at suggest time for reproducible train/test transforms.
- Plan envelope now carries `se_dat_version` so consumers can tell which
  package version produced a saved plan (backward-compatible; older files
  without `schema_version` are read as v1).
- The dependency `watson` correlation-ratio path now returns NaN for
  degenerate groups (fewer observations than groups) instead of a false
  "perfect" score.
- `__version__` is read from installed package metadata (`importlib.metadata`)
  so it matches the released version, with a `0.0.0+unknown` fallback when not
  installed.
- **Correlation speed-ups**: Cramér's V now factorises each column once and
  builds contingency tables with `numpy.bincount` instead of one `crosstab`
  per pair, and the eta matrix is computed with a single grouped reduction per
  categorical column instead of one group-by per numeric-categorical pair.
  Together with the existing `sample` / `max_columns` / `fast` caps, these cut
  `EDAReport.create` on 5 000 rows × 265 columns from tens of seconds to a few
  (e.g. 43 s → 4.6 s plain, ~2 s with `fast=True`).
- **Shared profiling**: `suggest_encodings` / `suggest_imputations` /
  `suggest_transforms` accept an optional `profile=` so `EDAReport.create` no
  longer re-profiles the frame once per plan — for sampled runs the profile is
  built from the same analysis frame used by the other statistics.

## [0.1.0] - 2026-08-23

### Added

- Initial release: column type profiling with confidence flags, correlation
  analysis (Pearson/Spearman, Cramér's V, correlation ratio η) with heatmaps
  and multicollinearity flags, encoding suggestions with automatic transforms,
  and the `EDAReport` facade combining them.
