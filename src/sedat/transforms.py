"""Skew-correcting transform suggestions for numeric columns.

Mirrors the plan/suggestion convention used by :mod:`sedat.encoding`: for each
numeric column whose absolute skewness exceeds ``skew_threshold`` we propose a
transform — ``log1p`` for non-negative right-skewed columns, otherwise a fitted
``box_cox`` (the data is shifted positive first when needed). Suggestions carry
the fitted ``lambda``/``shift`` so ``apply`` reproduces the exact transform on
held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profile import DataFrameProfile, profile_dataframe

LOG_STRATEGY = "log1p"
BOX_COX_STRATEGY = "box_cox"

DEFAULT_SKEW_THRESHOLD = 1.0


@dataclass
class TransformSuggestion:
    column: str
    strategy: str
    skew: float
    rationale: str
    lambda_: float | None = None
    shift: float = 0.0

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with this column transformed."""
        out = df.copy()
        col = self.column
        if col not in out.columns:
            return out
        values = pd.to_numeric(out[col], errors="coerce") + self.shift
        if self.strategy == LOG_STRATEGY:
            out[col] = np.log1p(values)
        elif self.strategy == BOX_COX_STRATEGY:
            from scipy import stats

            out[col] = stats.boxcox(values.to_numpy(dtype=float), lmbda=self.lambda_)
        return out

    def inverse(self, series: pd.Series) -> pd.Series:
        """Undo the transform on ``series`` (for interpretation on the original scale)."""
        if self.strategy == LOG_STRATEGY:
            return np.expm1(series) - self.shift
        if self.strategy == BOX_COX_STRATEGY:
            from scipy.special import inv_boxcox

            return pd.Series(inv_boxcox(series.to_numpy(dtype=float), self.lambda_), index=series.index) - self.shift
        return series


@dataclass
class TransformPlan:
    suggestions: list[TransformSuggestion] = field(default_factory=list)
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": s.column,
                "strategy": s.strategy,
                "skew": round(s.skew, 3),
                "lambda": None if s.lambda_ is None else round(s.lambda_, 4),
                "shift": s.shift,
                "rationale": s.rationale,
            }
            for s in self.suggestions
        ]
        return pd.DataFrame(rows)

    def get(self, column: str) -> TransformSuggestion | None:
        return next((s for s in self.suggestions if s.column == column), None)

    def apply(self, df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
        result = df.copy()
        if column is not None:
            suggestion = self.get(column)
            if suggestion is None:
                raise KeyError(f"No transform suggestion for column '{column}'")
            return suggestion.apply(result)
        for suggestion in self.suggestions:
            result = suggestion.apply(result)
        return result

    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)

    def __repr__(self) -> str:
        return repr(self.summary)


def _fit_box_cox(values: pd.Series) -> tuple[float, float]:
    """Return ``(lambda, shift)`` for a positive box-cox fit of ``values``."""
    from scipy import stats

    shift = 0.0
    if values.min() <= 0:
        shift = float(-values.min() + 1.0)
    positive = values + shift
    try:
        lambda_ = float(stats.boxcox_normmax(positive.to_numpy(dtype=float), method="mle"))
    except (ValueError, RuntimeError):  # pragma: no cover - degenerate data
        lambda_ = 0.0
    return lambda_, shift


def suggest_transforms(
    df: pd.DataFrame,
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
    profile: DataFrameProfile | None = None,
) -> TransformPlan:
    """Build a :class:`TransformPlan` for every sufficiently skewed numeric column.

    ``profile`` reuses an existing :class:`DataFrameProfile` of ``df`` when
    available so wide frames are not re-profiled on every plan.
    """
    profile = profile or profile_dataframe(df)
    suggestions: list[TransformSuggestion] = []

    for column in profile.columns:
        if column.inferred_type != "numeric" or column.is_constant:
            continue
        series = pd.to_numeric(df[column.name], errors="coerce").dropna()
        if len(series) < 3 or series.nunique() < 2:
            continue
        skew = float(series.skew())
        if np.isnan(skew) or abs(skew) < skew_threshold:
            continue
        if series.min() >= 0:
            suggestions.append(
                TransformSuggestion(
                    column=column.name,
                    strategy=LOG_STRATEGY,
                    skew=skew,
                    rationale=f"right-skewed (skew={skew:.2f}): log1p compresses the tail",
                )
            )
        else:
            lambda_, shift = _fit_box_cox(series)
            suggestions.append(
                TransformSuggestion(
                    column=column.name,
                    strategy=BOX_COX_STRATEGY,
                    skew=skew,
                    rationale=(f"skewed (skew={skew:.2f}, includes non-positive values): fitted Box-Cox"),
                    lambda_=lambda_,
                    shift=shift,
                )
            )
    return TransformPlan(suggestions=suggestions, skew_threshold=skew_threshold)


def apply_transforms(
    df: pd.DataFrame,
    plan: TransformPlan | None = None,
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
) -> pd.DataFrame:
    if plan is None:
        plan = suggest_transforms(df, skew_threshold=skew_threshold)
    return plan.apply_all(df)
