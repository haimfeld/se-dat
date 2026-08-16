"""Encoding suggestions and automatic transforms.

Detects binary-like string columns ("yes"/"no", "true"/"false", 0/1) and offers
an .encode()-style 0/1 map. Suggests one-hot encoding for low-cardinality
categoricals and warns/suggests target or ordinal encoding for high-cardinality
ones to avoid a dimensionality explosion.

Suggestions can be applied individually (``suggestion.apply``) or all at once
(``EncodingPlan.apply_all``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profile import profile_dataframe
from .types import binary_mapping

BINARY_STRATEGY = "binary_encode"
ONE_HOT_STRATEGY = "one_hot"
ORDINAL_STRATEGY = "ordinal_encode"
TARGET_STRATEGY = "target_encode"
NONE_STRATEGY = "none"

DEFAULT_CARDINALITY_THRESHOLD = 10


@dataclass
class EncodingSuggestion:
    column: str
    current_type: str
    suggested_type: str
    strategy: str
    cardinality: int
    rationale: str
    categories: list[str] = field(default_factory=list)
    target_means: dict[str, float] = field(default_factory=dict)

    def apply(self, df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
        out = df.copy()
        col = self.column
        if col not in out.columns:
            return out
        if self.strategy == NONE_STRATEGY:
            return out
        if self.strategy == BINARY_STRATEGY:
            mapping = binary_mapping(out[col])
            if mapping is None:
                ordered = sorted(out[col].dropna().unique())
                mapping = {v: i for i, v in enumerate(ordered)}
            out[col] = out[col].astype(str).str.strip().str.lower().map(mapping)
        elif self.strategy == ONE_HOT_STRATEGY:
            dummies = pd.get_dummies(out[col], prefix=col, dtype=int)
            out = out.drop(columns=[col])
            out = pd.concat([out, dummies], axis=1)
        elif self.strategy == ORDINAL_STRATEGY:
            ordered = sorted(out[col].dropna().astype(str).unique())
            out[col] = out[col].astype(str).map({v: i for i, v in enumerate(ordered)})
        elif self.strategy == TARGET_STRATEGY:
            if target is not None:
                means = target.groupby(out[col].astype(str)).transform("mean")
                out[col] = pd.to_numeric(means, errors="coerce")
            elif self.target_means:
                out[col] = out[col].astype(str).map(self.target_means)
        return out


@dataclass
class EncodingPlan:
    suggestions: list[EncodingSuggestion]
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD
    target: pd.Series | None = None

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": s.column,
                "current_type": s.current_type,
                "suggested_type": s.suggested_type,
                "strategy": s.strategy,
                "cardinality": s.cardinality,
                "rationale": s.rationale,
            }
            for s in self.suggestions
        ]
        return pd.DataFrame(rows)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply_all(df)

    def apply_all(self, df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
        result = df.copy()
        tgt = target if target is not None else self.target
        for suggestion in self.suggestions:
            result = suggestion.apply(result, target=tgt)
        return result

    def __repr__(self) -> str:
        return repr(self.summary)


def _warn(msg: str) -> None:
    import warnings

    warnings.warn(msg, UserWarning, stacklevel=3)


def suggest_encodings(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
) -> EncodingPlan:
    """Build an :class:`EncodingPlan` of encoding suggestions for ``df``.

    Parameters
    ----------
    df:
        Input DataFrame.
    target:
        Optional numeric target Series used for target-encoding suggestions on
        high-cardinality categoricals. The Series index must match ``df``.
    cardinality_threshold:
        Categoricals with at most this many distinct values get a one-hot
        suggestion; above it they get target/ordinal encoding.
    """
    profile = profile_dataframe(df)
    suggestions: list[EncodingSuggestion] = []

    for col_profile in profile.columns:
        col = col_profile.name
        series = df[col]
        cardinality = col_profile.n_unique
        current_type = col_profile.inferred_type

        if current_type == "boolean":
            if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
                suggestions.append(
                    EncodingSuggestion(
                        column=col,
                        current_type=current_type,
                        suggested_type="boolean (0/1)",
                        strategy=BINARY_STRATEGY,
                        cardinality=cardinality,
                        rationale="binary-like strings map cleanly to 0/1",
                        categories=sorted(series.dropna().unique().tolist()),
                    )
                )
            continue

        if current_type not in ("categorical",):
            continue

        if cardinality == 0:
            continue

        if cardinality == 2:
            suggestions.append(
                EncodingSuggestion(
                    column=col,
                    current_type=current_type,
                    suggested_type="boolean (0/1)",
                    strategy=BINARY_STRATEGY,
                    cardinality=cardinality,
                    rationale="two distinct values: binary encode",
                    categories=sorted(series.dropna().unique().tolist()),
                )
            )
        elif cardinality <= cardinality_threshold:
            suggestions.append(
                EncodingSuggestion(
                    column=col,
                    current_type=current_type,
                    suggested_type="one-hot (dummy) columns",
                    strategy=ONE_HOT_STRATEGY,
                    cardinality=cardinality,
                    rationale=f"low-cardinality categorical (k={cardinality})",
                    categories=sorted(series.dropna().unique().tolist()),
                )
            )
        else:
            if target is not None:
                suggestions.append(
                    EncodingSuggestion(
                        column=col,
                        current_type=current_type,
                        suggested_type="target-encoded numeric",
                        strategy=TARGET_STRATEGY,
                        cardinality=cardinality,
                        rationale=(
                            f"high-cardinality categorical (k={cardinality}): "
                            "one-hot would explode dimensionality"
                        ),
                        categories=sorted(series.dropna().unique().tolist()),
                        target_means={
                            str(k): float(v)
                            for k, v in target.groupby(series.astype(str)).mean().items()
                        },
                    )
                )
            else:
                _warn(
                    f"Column '{col}' has high cardinality ({cardinality}). "
                    "Pass a `target` Series to enable target encoding; "
                    "falling back to ordinal encoding."
                )
                suggestions.append(
                    EncodingSuggestion(
                        column=col,
                        current_type=current_type,
                        suggested_type="ordinal-encoded numeric",
                        strategy=ORDINAL_STRATEGY,
                        cardinality=cardinality,
                        rationale=(
                            f"high-cardinality categorical (k={cardinality}): "
                            "one-hot would explode dimensionality"
                        ),
                        categories=sorted(series.dropna().astype(str).unique().tolist()),
                    )
                )

    return EncodingPlan(suggestions=suggestions, cardinality_threshold=cardinality_threshold, target=target)


def apply_encodings(
    df: pd.DataFrame,
    plan: EncodingPlan | None = None,
    target: pd.Series | None = None,
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
) -> pd.DataFrame:
    """Shortcut: build a plan (if not given) and apply every suggestion."""
    if plan is None:
        plan = suggest_encodings(df, target=target, cardinality_threshold=cardinality_threshold)
    return plan.apply_all(df, target=target)