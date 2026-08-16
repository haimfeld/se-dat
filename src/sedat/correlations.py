"""Correlation analysis.

Numeric-numeric: Pearson/Spearman matrices plus a heatmap.
Categorical-categorical: Cramer's V.
Numeric-categorical: correlation ratio (eta), an ANOVA-style association measure.

Highly correlated pairs above a threshold are flagged as potential
multicollinearity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .types import binary_mapping

CORRELATION_TYPES = ("numeric", "boolean", "categorical")


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])


def _categorical_df(df: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for col in df.columns:
        dtype = df[col].dtype
        if (
            pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or dtype == object
            or isinstance(dtype, pd.StringDtype)
        ):
            picked.append(col)
    return df[picked]


def numeric_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pearson or Spearman correlation matrix over numeric columns."""
    num = _numeric_df(df)
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method)


def cramers_v(a: pd.Series, b: pd.Series, correction: bool = True) -> float:
    """Cramer's V association between two categorical series, in [0, 1]."""
    if a.name is not None and b.name is not None and a.name == b.name:
        return 1.0
    cross = pd.crosstab(a.astype(str), b.astype(str))
    if min(cross.shape) < 2:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(cross, correction=correction)
    n = int(cross.to_numpy().sum())
    phi2 = chi2 / n
    r, k = cross.shape
    if correction:
        phi2 = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r = max(0, r - ((r - 1) ** 2) / (n - 1))
        k = max(0, k - ((k - 1) ** 2) / (n - 1))
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2 / denom))


def categorical_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Cramer's V matrix over categorical/boolean columns."""
    cats = _categorical_df(df)
    if cats.shape[1] < 2:
        return pd.DataFrame()
    mat = pd.DataFrame(index=cats.columns, columns=cats.columns, dtype=float)
    for i, a in enumerate(cats.columns):
        for j, b in enumerate(cats.columns):
            if i <= j:
                continue
            v = cramers_v(cats[a], cats[b])
            mat.loc[a, b] = v
            mat.loc[b, a] = v
    for i in range(mat.shape[0]):
        mat.iat[i, i] = 1.0
    return mat


def correlation_ratio(numeric: pd.Series, categorical: pd.Series) -> float:
    """Eta correlation ratio between one numeric and one categorical series."""
    y = pd.to_numeric(numeric, errors="coerce")
    g = categorical.astype(str)
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    if len(d) < 2 or d["g"].nunique() < 2:
        return float("nan")
    grand_mean = float(d["y"].mean())
    group = d.groupby("g")["y"].agg(["mean", "count"])
    ss_between = float(((group["mean"] - grand_mean) ** 2 * group["count"]).sum())
    ss_total = float(((d["y"] - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    return float(np.sqrt(ss_between / ss_total))


def numeric_categorical_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Eta matrix with numeric columns as rows and categorical columns as cols."""
    num = _numeric_df(df)
    cats = _categorical_df(df)
    if num.shape[1] < 1 or cats.shape[1] < 1:
        return pd.DataFrame()
    rows = []
    for n in num.columns:
        for c in cats.columns:
            rows.append({"numeric": n, "categorical": c, "eta": correlation_ratio(num[n], cats[c])})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="numeric", columns="categorical", values="eta")


def _upper_pairs(mat: pd.DataFrame) -> Iterator[tuple[str, str, float]]:
    cols = list(mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = mat.iat[i, j]
            if np.isnan(v):
                continue
            yield cols[i], cols[j], float(v)


@dataclass
class CorrelationReport:
    numeric: pd.DataFrame
    numeric_spearman: pd.DataFrame
    categorical: pd.DataFrame
    numeric_categorical: pd.DataFrame
    method_numeric: str = "pearson"
    threshold: float = 0.7

    @property
    def summary(self) -> pd.DataFrame:
        rows = []
        for mat, kind, method in (
            (self.numeric, "numeric", self.method_numeric),
            (self.numeric_spearman, "numeric", "spearman"),
            (self.categorical, "categorical", "cramers_v"),
            (self.numeric_categorical, "numeric-categorical", "eta"),
        ):
            if mat.empty:
                continue
            if kind == "numeric-categorical":
                for num_col in mat.index:
                    for cat_col in mat.columns:
                        v = mat.loc[num_col, cat_col]
                        if np.isnan(v):
                            continue
                        rows.append(
                            {
                                "column_a": num_col,
                                "column_b": cat_col,
                                "kind": kind,
                                "method": method,
                                "value": round(float(v), 4),
                                "flagged": abs(float(v)) >= self.threshold,
                            }
                        )
            else:
                for a, b, v in _upper_pairs(mat):
                    rows.append(
                        {
                            "column_a": a,
                            "column_b": b,
                            "kind": kind,
                            "method": method,
                            "value": round(v, 4),
                            "flagged": abs(v) >= self.threshold,
                        }
                    )
        return pd.DataFrame(rows)

    @property
    def flagged_pairs(self) -> pd.DataFrame:
        s = self.summary
        return s[s["flagged"]] if not s.empty else s


def correlation_report(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.7,
) -> CorrelationReport:
    return CorrelationReport(
        numeric=numeric_correlation(df, method=method),
        numeric_spearman=numeric_correlation(df, method="spearman"),
        categorical=categorical_correlation(df),
        numeric_categorical=numeric_categorical_correlation(df),
        method_numeric=method,
        threshold=threshold,
    )


def _binary_numeric_map(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        col: mapping
        for col in df.columns
        if (mapping := binary_mapping(df[col])) is not None
    }


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary-like object columns to 0/1 numeric for correlation use."""
    out = df.copy()
    for col, mapping in _binary_numeric_map(df).items():
        out[col] = out[col].astype(str).str.strip().str.lower().map(mapping)
    return out


def correlation_heatmap(matrix: pd.DataFrame, title: str = "Correlation matrix") -> object:
    """Plot a heatmap for a correlation matrix; returns a matplotlib Figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.7), max(5, matrix.shape[0] * 0.6)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig