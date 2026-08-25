# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

## [0.1.0] - 2026-08-23

### Added

- Initial release: column type profiling with confidence flags, correlation
  analysis (Pearson/Spearman, Cramér's V, correlation ratio η) with heatmaps
  and multicollinearity flags, encoding suggestions with automatic transforms,
  and the `EDAReport` facade combining them.
