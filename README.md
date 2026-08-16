# se-dat

Simple Exploratory Data Analysis tool. Profiles your columns, measures
associations across numeric/categorical data, and suggests encodings so you can
move from raw data to a model-ready frame in one call.

## Install

```bash
pip install -e .
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
print(report.profile.summary)                 # column type profiling
print(report.correlations.flagged_pairs)      # multicollinearity warnings
print(report.encoding_summary)                # suggested encodings

model_ready = report.apply_all_encodings()    # apply every suggestion
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

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```