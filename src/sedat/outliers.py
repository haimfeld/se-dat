"""Outlier detection for numeric columns.

Four methods are supported:

- ``"iqr"``: values outside ``[Q1 - k * IQR, Q3 + k * IQR]`` (k default 1.5)
- ``"zscore"``: values whose absolute z-score exceeds a threshold (default 3.0)
- ``"mad"``: modified z-score ``0.6745 * |x - median| / MAD`` above a threshold
  (default 3.5); robust on skewed data where standard deviation is inflated
- ``"log_iqr"``: IQR applied to a ``log1p``-transformed series, for
  right-skewed / heavy-tailed columns (e.g. income or spend)

Missing values are ignored: they never count as outliers and are excluded
from the denominators of the reported percentages.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

IQR_METHOD = "iqr"
ZSCORE_METHOD = "zscore"
MAD_METHOD = "mad"
LOG_IQR_METHOD = "log_iqr"

METHODS = (IQR_METHOD, ZSCORE_METHOD, MAD_METHOD, LOG_IQR_METHOD)

DEFAULT_IQR_SCALE = 1.5
DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_MAD_THRESHOLD = 3.5
_MAD_SCALE = 0.6745


def outlier_note(result: ColumnOutliers) -> str:
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


def _detect_iqr(series: pd.Series, scale: float) -> tuple[np.ndarray, float | None, float | None]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    lower = float(q1) - scale * iqr
    upper = float(q3) + scale * iqr
    mask = ((series < lower) | (series > upper)).to_numpy()
    return mask, lower, upper


def _detect_zscore(series: pd.Series, threshold: float) -> tuple[np.ndarray, float | None, float | None]:
    std = float(series.std())
    if std == 0 or np.isnan(std):
        return np.zeros(len(series), dtype=bool), None, None
    mean = float(series.mean())
    z = (series - mean) / std
    mask = (z.abs() > threshold).to_numpy()
    return mask, None, None


def _detect_mad(series: pd.Series, threshold: float) -> tuple[np.ndarray, float | None, float | None]:
    median = float(series.median())
    mad = float(np.median(np.abs(series - median)))
    if mad == 0 or np.isnan(mad):
        return np.zeros(len(series), dtype=bool), None, None
    modified_z = _MAD_SCALE * (series - median) / mad
    mask = (modified_z.abs() > threshold).to_numpy()
    half_width = threshold * mad / _MAD_SCALE
    return mask, median - half_width, median + half_width


def _detect_log_iqr(series: pd.Series, scale: float) -> tuple[np.ndarray, float | None, float | None]:
    # shift so every value is >= 0 before log1p; bounds are mapped back
    shift = float(-series.min() + 1.0) if series.min() < 0 else 0.0
    logged = np.log1p(series + shift)
    mask, lower_log, upper_log = _detect_iqr(logged, scale)
    lower = None if lower_log is None else float(np.expm1(lower_log) - shift)
    upper = None if upper_log is None else float(np.expm1(upper_log) - shift)
    return mask, lower, upper


def detect_outliers(
    df: pd.DataFrame,
    method: str = IQR_METHOD,
    iqr_scale: float = DEFAULT_IQR_SCALE,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    mad_threshold: float = DEFAULT_MAD_THRESHOLD,
) -> OutlierReport:
    """Detect outliers in every numeric column of ``df``.

    Parameters
    ----------
    df:
        Input DataFrame; non-numeric columns are skipped.
    method:
        ``"iqr"`` (default), ``"zscore"``, ``"mad"`` or ``"log_iqr"``.
    iqr_scale:
        Multiplier ``k`` on the interquartile range (IQR and log-IQR methods).
    z_threshold:
        Absolute z-score cutoff (z-score method).
    mad_threshold:
        Modified z-score cutoff (MAD method).

    Percentages are computed over non-null values.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Expected one of: {', '.join(METHODS)}")
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
        elif method == ZSCORE_METHOD:
            mask, lower, upper = _detect_zscore(series, z_threshold)
        elif method == MAD_METHOD:
            mask, lower, upper = _detect_mad(series, mad_threshold)
        else:
            mask, lower, upper = _detect_log_iqr(series, iqr_scale)
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
