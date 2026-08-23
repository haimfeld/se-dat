"""Datetime feature extraction suggestions.

For each detected datetime column, proposes calendar-derived features:
year, month, day, day_of_week (Monday=0), is_weekend and — when the data
actually contains a time component — hour.

Follows the plan/suggestion conventions of :mod:`sedat.encoding` and
:mod:`sedat.imputation`: ``suggest_datetime_features(df)`` returns a
``DatetimeFeaturePlan`` whose ``apply``/``apply_all`` produce new columns
named ``"{col}_year"``, ``"{col}_dow"`` etc. The original column is kept.
Extracted values use pandas' nullable ``Int64`` dtype so missing timestamps
stay missing instead of becoming sentinel numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .profile import profile_dataframe

YEAR = "year"
MONTH = "month"
DAY = "day"
DAY_OF_WEEK = "day_of_week"
IS_WEEKEND = "is_weekend"
HOUR = "hour"

BASE_FEATURES: tuple[str, ...] = (YEAR, MONTH, DAY, DAY_OF_WEEK, IS_WEEKEND)
ALL_FEATURES: tuple[str, ...] = BASE_FEATURES + (HOUR,)

FEATURE_SUFFIXES: dict[str, str] = {
    YEAR: "_year",
    MONTH: "_month",
    DAY: "_day",
    DAY_OF_WEEK: "_dow",
    IS_WEEKEND: "_is_weekend",
    HOUR: "_hour",
}


@dataclass
class DatetimeFeatureSuggestion:
    column: str
    features: list[str] = field(default_factory=lambda: list(BASE_FEATURES))
    has_time_component: bool = False

    @property
    def new_columns(self) -> list[str]:
        return [self.column + FEATURE_SUFFIXES[f] for f in self.features]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with the proposed feature columns added."""
        out = df.copy()
        if self.column not in out.columns:
            return out
        parsed = pd.to_datetime(out[self.column], errors="coerce")
        for feat in self.features:
            name = self.column + FEATURE_SUFFIXES[feat]
            if feat == YEAR:
                values = parsed.dt.year.astype("Int64")
            elif feat == MONTH:
                values = parsed.dt.month.astype("Int64")
            elif feat == DAY:
                values = parsed.dt.day.astype("Int64")
            elif feat == DAY_OF_WEEK:
                values = parsed.dt.dayofweek.astype("Int64")
            elif feat == IS_WEEKEND:
                # floor-division maps Mon-Fri (0-4) -> 0, Sat/Sun (5-6) -> 1,
                # and propagates <NA> for missing timestamps
                values = (parsed.dt.dayofweek.astype("Int64") // 5).astype("Int64")
            elif feat == HOUR:
                values = parsed.dt.hour.astype("Int64")
            else:  # pragma: no cover - guarded by ALL_FEATURES
                raise ValueError(f"Unknown datetime feature '{feat}'")
            out[name] = values
        return out


@dataclass
class DatetimeFeaturePlan:
    suggestions: list[DatetimeFeatureSuggestion] = field(default_factory=list)

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": s.column,
                "n_features": len(s.features),
                "features": ", ".join(s.features),
                "has_time_component": s.has_time_component,
                "new_columns": ", ".join(s.new_columns),
            }
            for s in self.suggestions
        ]
        return pd.DataFrame(rows)

    def get(self, column: str) -> DatetimeFeatureSuggestion | None:
        return next((s for s in self.suggestions if s.column == column), None)

    def apply(self, df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
        """Apply one datetime column's features, or every suggestion."""
        result = df.copy()
        if column is not None:
            suggestion = self.get(column)
            if suggestion is None:
                raise KeyError(
                    f"No datetime feature suggestion for column '{column}'"
                )
            return suggestion.apply(result)
        for suggestion in self.suggestions:
            result = suggestion.apply(result)
        return result

    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)

    def __repr__(self) -> str:
        return repr(self.summary)


def _has_time_component(parsed: pd.Series) -> bool:
    valid = parsed.dropna()
    if valid.empty:
        return False
    return bool(
        (valid.dt.hour != 0).any()
        or (valid.dt.minute != 0)
        .any()
        or (valid.dt.second != 0).any()
    )


def suggest_datetime_features(df: pd.DataFrame) -> DatetimeFeaturePlan:
    """Build a :class:`DatetimeFeaturePlan` for every datetime column in ``df``.

    Columns are detected with the same type inference used by
    :func:`sedat.profile_dataframe`, so both native datetime dtypes and
    parseable strings qualify. ``hour`` is only proposed when a non-midnight
    time component is actually present in the data.
    """
    profile = profile_dataframe(df)
    suggestions: list[DatetimeFeatureSuggestion] = []
    for col_profile in profile.columns:
        if col_profile.inferred_type != "datetime":
            continue
        parsed = pd.to_datetime(df[col_profile.name], errors="coerce")
        has_time = _has_time_component(parsed)
        features = list(BASE_FEATURES)
        if has_time:
            features.append(HOUR)
        suggestions.append(
            DatetimeFeatureSuggestion(
                column=col_profile.name,
                features=features,
                has_time_component=has_time,
            )
        )
    return DatetimeFeaturePlan(suggestions=suggestions)


def apply_datetime_features(
    df: pd.DataFrame,
    plan: DatetimeFeaturePlan | None = None,
) -> pd.DataFrame:
    """Shortcut: build a plan (if not given) and apply every suggestion."""
    if plan is None:
        plan = suggest_datetime_features(df)
    return plan.apply_all(df)
