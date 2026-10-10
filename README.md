# se-dat

[![CI](https://github.com/haimfeld/se-dat/actions/workflows/ci.yml/badge.svg)](https://github.com/haimfeld/se-dat/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/se-dat.svg)](https://pypi.org/project/se-dat/)
[![Python versions](https://img.shields.io/pypi/pyversions/se-dat.svg)](https://pypi.org/project/se-dat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple Exploratory Data Analysis tool. Profiles your columns, measures
associations across numeric/categorical data, and suggests encodings so you can
move from raw data to a model-ready frame in one call.

## Why se-dat?

Most EDA tools hand you a dashboard and leave the decisions to you. `se-dat`
is decision-first: it emits inspectable, editable **plans** — encoding plans,
imputation plans, datetime feature plans — whose suggestions you can accept
individually, apply all at once, save to JSON and reuse on a held-out test set.
It is also sklearn-native: any encoding plan converts to a fitted-ready
`ColumnTransformer`. And it stays light: the core depends only on
numpy/pandas/scipy/matplotlib/seaborn, with scikit-learn strictly optional.

## Install

```bash
pip install se-dat
```

With scikit-learn interoperability:

```bash
pip install "se-dat[sklearn]"
```

For local development:

```bash
pip install -e ".[dev]"
```

## Features

1. **Column type profiling** — per column: inferred type (`numeric`, `boolean`,
   `categorical`, `string`, `datetime`, `id`), missing %, unique count,
   cardinality ratio, a confidence flag when the inference is ambiguous
   (numeric-looking strings, `0/1` vs `yes`/`no` booleans, etc.), and flags for
   constant columns and full-row duplicates.
2. **Correlation analysis** — Pearson/Spearman matrices + heatmap
   (numeric-numeric), Cramér's V (categorical-categorical), correlation ratio
   eta (numeric-categorical), pairwise multicollinearity flags, plus
   variance-inflation factors and a condition number to catch multi-way
   collinearity that pairwise checks miss.
3. **Target analysis** — `analyze_target(df, target)` picks an effect size by
   the target's inferred role (Pearson / point-biserial / eta / Cramér's V),
   adds binning-based mutual information, returns a ranked feature table and
   flags features that almost perfectly predict the target (leakage).
4. **Encoding suggestions + auto-transform** — detects binary-like strings
   (`yes`/`no`, `true`/`false`, `0/1`), suggests one-hot for low-cardinality
   categoricals, and target/ordinal encoding for high-cardinality ones (with a
   dimensionality warning). Fit-on-apply target encoding is warned against.
5. **Missing-value handling** — median for numeric columns, mode for
   categorical-like ones, and drop-column suggestions past a configurable
   missing-percentage threshold.
6. **Outlier detection** — IQR, z-score, modified z-score (`mad`) or `log_iqr`
   flags per numeric column with counts, percentages and row indices, surfaced
   directly in the profile summary.
7. **Plausibility checks** — opt-in range rules, `non_negative` / `percentages`
   shorthands and value whitelists, reported per column.
8. **Transform suggestions** — `log1p` or Box-Cox for skewed columns, with
   round-trip inverses so features can be un-transformed after modelling.
9. **Data dictionary** — declarative column types, ranges, derived columns and
   target marking from a dict, JSON or CSV, driving the whole pipeline.
10. **HTML report export** — a static, self-contained page (tables + embedded
    heatmap images) from a single call.
11. **scikit-learn interop** — convert an encoding plan into a native
    `ColumnTransformer`.
12. **Datetime feature extraction** — year/month/day/day-of-week/weekend/hour
    suggestions for detected datetime columns.
13. **Reproducible plans** — every plan saves to and loads from JSON, keeping
    train/test transforms consistent.
14. **Scalability** — `sample=` / `max_columns=` caps, a `fast=True` preset and
    parallel correlation loops (`n_jobs`, `progress`) for large frames.

## Usage

```python
import pandas as pd
import sedat

df = pd.DataFrame({
    "id": range(500),
    "age": ...,
    "income": ["45000", "67000", ...],       # numeric stored as string
    "smoker": ["yes", "no", ...],            # binary-like
    "region": ["north", "south", ...],
    "target": ...,                            # numeric target
})

report = sedat.EDAReport.create(df, target=df["target"])
print(report.profile.summary)                 # column type profiling (+ outliers)
print(report.correlations.flagged_pairs)      # multicollinearity warnings
print(report.encoding_summary)                # suggested encodings
print(report.imputation_summary)              # suggested fills/drops

model_ready = report.apply_all_encodings()    # apply every encoding suggestion
complete = report.apply_all_imputations()     # apply every imputation suggestion
```

### Granular APIs

```python
profile = sedat.profile_dataframe(df)
pearson = sedat.numeric_correlation(df, method="pearson")
cramers = sedat.categorical_correlation(df)       # Cramér's V matrix
eta = sedat.numeric_categorical_correlation(df)   # correlation ratio matrix
fig = sedat.correlation_heatmap(pearson, title="Numeric correlations")

plan = sedat.suggest_encodings(df, target=df["target"], cardinality_threshold=10)
only_smoker = next(s for s in plan.suggestions if s.column == "smoker")
one_hot_step = only_smoker.apply(df)              # accept one suggestion
all_encoded = plan.apply_all(df)                  # accept all suggestions
```

### Missing-value handling

```python
imp_plan = sedat.suggest_imputations(df, missing_threshold=50)
imp_plan.summary            # strategy, n_missing, fill_value, rationale per column
filled = imp_plan.apply(df, column="age")   # median fill for one column
complete = imp_plan.apply_all(df)           # everything (drops 'gone' too)

report.apply_all_imputations()              # same, straight off the report
```

Numeric columns get a robust median (the rationale notes the mean as an
alternative), categorical-like columns get their mode, and columns missing more
than `missing_threshold` percent of values are recommended for dropping rather
than inventing data.

### Outlier detection

```python
outliers = sedat.detect_outliers(df, method="iqr")   # or "zscore" | "mad" | "log_iqr"
outliers.summary                                       # count/%/bounds per column
fare = outliers.get("fare")
df.loc[fare.indices, ["fare"]]                         # flagged rows

# also surfaced automatically inside the profile table:
report.profile.summary["outliers"]                     # "fare: 12 potential outliers (2.4%)"
```

`mad` uses a median-based modified z-score (robust to skew, threshold 3.5);
`log_iqr` detects outliers on log-scaled positive-skewed values first.

### HTML reports

```python
report.to_html("eda_report.html")
```

One static file: profile table, outlier notes, flagged correlation pairs,
heatmap images (embedded base64 PNGs) and both suggestion tables. Inline CSS,
no JavaScript, no external assets.

### scikit-learn interop

```python
ct = plan.to_column_transformer()          # requires pip install "se-dat[sklearn]"
Xt = ct.fit_transform(train_df, train_y)   # TargetEncoder columns receive y
X_test = ct.transform(test_df)             # unseen categories handled gracefully
```

Binary maps become `OrdinalEncoder`s, low-cardinality categoricals become
`OneHotEncoder(handle_unknown="ignore")`, high-cardinality ones become
`TargetEncoder`, and already-numeric columns pass through untouched.

### Datetime features

```python
dt_plan = sedat.suggest_datetime_features(df)
dt_plan.summary                       # proposed features per datetime column
featurized = dt_plan.apply_all(df)    # adds {col}_year/_month/_day/_dow/_is_weekend[/hour]
```

The `hour` feature is only proposed when the data actually contains a
non-midnight time component. Extracted columns use nullable integers, so
missing timestamps stay missing.

### Saving & loading plans

```python
plan.save("encoding_plan.json")               # same API on imputation & datetime plans
loaded = sedat.EncodingPlan.load("encoding_plan.json")

# re-appliable to new data without recomputation:
test_encoded = loaded.apply_all(test_df)      # stable schema; unseen values -> zeros/NaN
```

Plans serialize their decisions (categories, fill values, code mappings), not
function references, so transforms stay identical between training and test
runs — including one-hot schemas when a category is absent from the test data.

### Target analysis & leakage detection

```python
analysis = sedat.analyze_target(df, target=df["target"], warn=True)
analysis.summary                 # ranked effect size + MI per feature
analysis.leakage_features        # e.g. ['customer_id'] if it predicts the target
```

Receive the same analysis on any report that was created with a target:

```python
report = sedat.EDAReport.create(df, target=df["target"])
report.target_summary            # ranked table against the chosen target
report.leakage_features          # potentially leaking features, if any
```

Effect sizes are chosen from the target's inferred role (Pearson for numeric,
point-biserial for binary, η / Cramér's V for categorical). Effect size *or*
normalised mutual information meeting the `leakage_threshold` (default 0.98)
flags a feature as leakage. High-cardinality text and Id-like columns are
bucketed so they never look like perfect predictors.

### Data dictionaries

```python
dictionary = sedat.DataDictionary.load("dictionary.json")
report = sedat.EDAReport.create(df, dictionary=dictionary)
```

A dictionary (dict mapping, JSON file or CSV) can set each column's semantic
`type`, its `min`/`max` (or `allowed`) ranges for plausibility checks, mark
derived columns and designate the `target`. The data dictionary drives type
identification, range checks and target handling in one place, and its entries
show up as notes in the profile.

### Plausibility checks

```python
report = sedat.EDAReport.create(
    df, range_rules={"age": (0, 120), "segment": {"allowed": ["a", "b", "c"]}},
    non_negative=["count"], percentages=["completion_pct"],
)
report.plausibility_summary      # column / range / n_violations per constraint
```

### Transform suggestions

```python
plan = sedat.suggest_transforms(df)
plan.summary                     # column -> log1p / box_cox, with skew before/after
transformed = plan.apply_all(df)
back = plan.inverse(transformed) # round-trips to the original scale
```

### Collinearity diagnostics

```python
sedat.variance_inflation_factors(df)   # VIF per numeric column (multi-way)
sedat.condition_number(df)             # design-matrix condition number
report.vif_summary                     # same, via the report
```

VIF is skipped when there are more than `max_vif_features` (default 50) numeric
columns; correlation-ratio η returns NaN for degenerate groups instead of a
false "perfect" score.

### Scalability

```python
report = sedat.EDAReport.create(df, target=df["target"], fast=True)
#     -> profile on a seeded 50k-row sample, correlations capped at 100 columns,
#        VIF skipped; the full frame is still used to apply encoding plans.

report = sedat.EDAReport.create(df, sample=20_000, max_columns=50,
                                n_jobs=4, progress=True)
```

The analysis frame is sampled (default via `fast=True`), but every plan
produced by the report is applied to the full frame you passed in.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```
