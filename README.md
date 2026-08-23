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
   cardinality ratio, and a confidence flag when the inference is ambiguous
   (numeric-looking strings, `0/1` vs `yes`/`no` booleans, etc.).
2. **Correlation analysis** — Pearson/Spearman matrices + heatmap
   (numeric-numeric), Cramér's V (categorical-categorical), correlation ratio
   eta (numeric-categorical), plus multicollinearity flags above a threshold.
3. **Encoding suggestions + auto-transform** — detects binary-like strings
   (`yes`/`no`, `true`/`false`, `0/1`), suggests one-hot for low-cardinality
   categoricals, and target/ordinal encoding for high-cardinality ones (with a
   dimensionality warning). Apply suggestions individually or all at once.
4. **Missing-value handling** — median for numeric columns, mode for
   categorical-like ones, and drop-column suggestions past a configurable
   missing-percentage threshold.
5. **Outlier detection** — IQR or z-score flags per numeric column with counts,
   percentages and row indices, surfaced directly in the profile summary.
6. **HTML report export** — a static, self-contained page (tables + embedded
   heatmap images) from a single call.
7. **scikit-learn interop** — convert an encoding plan into a native
   `ColumnTransformer`.
8. **Datetime feature extraction** — year/month/day/day-of-week/weekend/hour
   suggestions for detected datetime columns.
9. **Reproducible plans** — every plan saves to and loads from JSON, keeping
   train/test transforms consistent.

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
outliers = sedat.detect_outliers(df, method="iqr")     # or method="zscore"
outliers.summary                                       # count/%/bounds per column
fare = outliers.get("fare")
df.loc[fare.indices, ["fare"]]                         # flagged rows

# also surfaced automatically inside the profile table:
report.profile.summary["outliers"]                     # "fare: 12 potential outliers (2.4%)"
```

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

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```
