"""Missing-value handling.

Produces one suggestion per column that contains missing values:

- numeric columns -> median fill (robust to outliers; mean noted as an
  alternative in the rationale)
- categorical/boolean/string columns -> mode fill
- any column whose missing percentage exceeds a configurable threshold
  (default 50%) -> ``drop_column`` instead of imputing

Follows the same conventions as :mod:`sedat.encoding`: an ``ImputationPlan``
holding ``ImputationSuggestion`` objects, a ``.summary`` DataFrame, applying a
single column via ``plan.apply(df, column=...)`` and everything at once via
``plan.apply_all(df)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .profile import profile_dataframe

MEDIAN_STRATEGY = "median"
MEAN_STRATEGY = "mean"
MODE_STRATEGY = "mode"
DROP_COLUMN_STRATEGY = "drop_column"

FILL_STRATEGIES = (MEDIAN_STRATEGY, MEAN_STRATEGY, MODE_STRATEGY)

DEFAULT_MISSING_THRESHOLD = 50.0


@dataclass
class ImputationSuggestion:
    column: str
    current_type: str
    strategy: str
    n_missing: int
    missing_pct: float
    fill_value: Any = None
    rationale: str = ""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with this suggestion applied."""
        out = df.copy()
        col = self.column
        if col not in out.columns:
            return out
        if self.strategy == DROP_COLUMN_STRATEGY:
            return out.drop(columns=[col])
        if self.strategy in FILL_STRATEGIES and self.fill_value is not None:
            out[col] = out[col].fillna(self.fill_value)
        return out


@dataclass
class ImputationPlan:
    suggestions: list[ImputationSuggestion] = field(default_factory=list)
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": s.column,
                "current_type": s.current_type,
                "strategy": s.strategy,
                "n_missing": s.n_missing,
                "missing_pct": s.missing_pct,
                "fill_value": s.fill_value,
                "rationale": s.rationale,
            }
            for s in self.suggestions
        ]
        return pd.DataFrame(rows)

    def get(self, column: str) -> ImputationSuggestion | None:
        """Return the suggestion for ``column``, or None if it has no missings."""
        return next((s for s in self.suggestions if s.column == column), None)

    def apply(
        self, df: pd.DataFrame, column: str | None = None
    ) -> pd.DataFrame:
        """Apply one column's suggestion, or every suggestion when ``column`` is None."""
        result = df.copy()
        if column is not None:
            suggestion = self.get(column)
            if suggestion is None:
                raise KeyError(f"No imputation suggestion for column '{column}'")
            return suggestion.apply(result)
        for suggestion in self.suggestions:
            result = suggestion.apply(result)
        return result

    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)

    def __repr__(self) -> str:
        return repr(self.summary)


def _mode_value(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    return modes.iloc[0] if len(modes) else None


def suggest_imputations(
    df: pd.DataFrame,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> ImputationPlan:
    """Build an :class:`ImputationPlan` with one suggestion per incomplete column.

    Parameters
    ----------
    df:
        Input DataFrame.
    missing_threshold:
        Columns with a missing percentage strictly above this value are
        suggested for dropping instead of imputation.
    """
    profile = profile_dataframe(df)
    suggestions: list[ImputationSuggestion] = []

    for col_profile in profile.columns:
        series = df[col_profile.name]
        n_missing = int(series.isna().sum())
        if n_missing == 0:
            continue
        col = col_profile.name
        col_type = col_profile.inferred_type
        pct = col_profile.missing_pct

        if pct > missing_threshold:
            suggestions.append(
                ImputationSuggestion(
                    column=col,
                    current_type=col_type,
                    strategy=DROP_COLUMN_STRATEGY,
                    n_missing=n_missing,
                    missing_pct=pct,
                    rationale=(
                        f"{pct:.1f}% missing (above {missing_threshold:.0f}% "
                        "threshold): dropping is safer than imputing"
                    ),
                )
            )
        elif col_type == "numeric":
            suggestions.append(
                ImputationSuggestion(
                    column=col,
                    current_type=col_type,
                    strategy=MEDIAN_STRATEGY,
                    n_missing=n_missing,
                    missing_pct=pct,
                    fill_value=float(series.median()),
                    rationale=(
                        "numeric column: median is robust to outliers "
                        "(mean is a reasonable alternative)"
                    ),
                )
            )
        else:
            suggestions.append(
                ImputationSuggestion(
                    column=col,
                    current_type=col_type,
                    strategy=MODE_STRATEGY,
                    n_missing=n_missing,
                    missing_pct=pct,
                    fill_value=_mode_value(series),
                    rationale=(
                        f"{col_type} column: fill with the most frequent value"
                    ),
                )
            )

    return ImputationPlan(
        suggestions=suggestions, missing_threshold=missing_threshold
    )


def apply_imputations(
    df: pd.DataFrame,
    plan: ImputationPlan | None = None,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> pd.DataFrame:
    """Shortcut: build a plan (if not given) and apply every suggestion."""
    if plan is None:
        plan = suggest_imputations(df, missing_threshold=missing_threshold)
    return plan.apply_all(df)
