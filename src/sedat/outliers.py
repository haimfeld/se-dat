"""Outlier detection for numeric columns.

Two methods are supported:

- ``"iqr"``: values outside ``[Q1 - k * IQR, Q3 + k * IQR]`` (k default 1.5)
- ``"zscore"``: values whose absolute z-score exceeds a threshold
  (default 3.0)

Missing values are ignored: they never count as outliers and are excluded
from the denominators of the reported percentages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np
import pandas as pd

IQR_METHOD = "iqr"
ZSCORE_METHOD = "zscore"

DEFAULT_IQR_SCALE = 1.5
DEFAULT_Z_THRESHOLD = 3.0


def outlier_note(result: "ColumnOutliers") -> str:
    """One-line human-readable summary, e.g. 'age: 12 potential outliers (2.4%)'."""
    if result.count == 0:
        return ""
    return f"{result.column}: {result.count} potential outliers ({result.pct:.1f}%)"


@dataclass
class ColumnOutliers:
    """Outlier statistics for a single numeric column."""

    column: str
    method: str
    count: int = 0
    pct: float = 0.0
    lower: float | None = None
    upper: float | None = None
    indices: list[Any] = field(default_factory=list)


@dataclass
class OutlierReport:
    """Per-column outlier statistics across a frame."""

    method: str
    columns: list[ColumnOutliers] = field(default_factory=list)

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": c.column,
                "method": c.method,
                "outlier_count": c.count,
                "outlier_pct": round(c.pct, 2),
                "lower_bound": c.lower,
                "upper_bound": c.upper,
                "note": outlier_note(c),
            }
            for c in self.columns
        ]
        return pd.DataFrame(rows)

    def get(self, column: Hashable) -> ColumnOutliers | None:
        return next((c for c in self.columns if c.column == column), None)

    def indices(self, column: Hashable) -> list[Any]:
        result = self.get(column)
        return result.indices if result is not None else []

    def __repr__(self) -> str:
        return repr(self.summary)


def _detect_iqr(
    series: pd.Series, scale: float
) -> tuple[np.ndarray, float | None, float | None]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    lower = float(q1) - scale * iqr
    upper = float(q3) + scale * iqr
    mask = ((series < lower) | (series > upper)).to_numpy()
    return mask, lower, upper


def _detect_zscore(
    series: pd.Series, threshold: float
) -> tuple[np.ndarray, float | None, float | None]:
    std = float(series.std())
    if std == 0 or np.isnan(std):
        return np.zeros(len(series), dtype=bool), None, None
    mean = float(series.mean())
    z = (series - mean) / std
    mask = (z.abs() > threshold).to_numpy()
    return mask, None, None


def detect_outliers(
    df: pd.DataFrame,
    method: str = IQR_METHOD,
    iqr_scale: float = DEFAULT_IQR_SCALE,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> OutlierReport:
    """Detect outliers in every numeric column of ``df``.

    Parameters
    ----------
    df:
        Input DataFrame; non-numeric columns are skipped.
    method:
        ``"iqr"`` (default) or ``"zscore"``.
    iqr_scale:
        Multiplier ``k`` on the interquartile range (IQR method).
    z_threshold:
        Absolute z-score cutoff (z-score method).

    Percentages are computed over non-null values.
    """
    if method not in (IQR_METHOD, ZSCORE_METHOD):
        raise ValueError(
            f"Unknown method '{method}'. Expected one of: {IQR_METHOD}, {ZSCORE_METHOD}"
        )
    num = df.select_dtypes(include=[np.number])
    results: list[ColumnOutliers] = []
    for col in num.columns:
        series = num[col].dropna()
        n = len(series)
        if n == 0:
            results.append(ColumnOutliers(column=col, method=method))
            continue
        if method == IQR_METHOD:
            mask, lower, upper = _detect_iqr(series, iqr_scale)
        else:
            mask, lower, upper = _detect_zscore(series, z_threshold)
        count = int(mask.sum())
        results.append(
            ColumnOutliers(
                column=col,
                method=method,
                count=count,
                pct=(count / n * 100) if n else 0.0,
                lower=lower,
                upper=upper,
                indices=list(series.index[mask]),
            )
        )
    return OutlierReport(method=method, columns=results)
