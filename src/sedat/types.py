"""Column type inference helpers.

Infers a semantic type for each column: numeric, boolean, categorical,
string/text, datetime or id (high-cardinality unique values), and reports a
confidence level plus explanatory notes when the inference is ambiguous.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BINARY_VALUE_SETS: dict[str, set[str]] = {
    "yes/no": {"yes", "no", "y", "n"},
    "true/false": {"true", "false", "t", "f"},
    "0/1": {"0", "1"},
}

_BINARY_MAPS: dict[frozenset[str], dict[str, int]] = {
    frozenset({"yes", "no", "y", "n"}): {"yes": 1, "no": 0, "y": 1, "n": 0},
    frozenset({"true", "false", "t", "f"}): {"true": 1, "false": 0, "t": 1, "f": 0},
    frozenset({"0", "1"}): {"0": 1, "1": 0},
}


def binary_mapping(series: pd.Series) -> dict[str, int] | None:
    """Return a {value: 0/1} mapping if ``series`` is binary-like, else None."""
    unique = set(series.dropna().astype(str).str.strip().str.lower().unique())
    if not unique:
        return None
    for allowed, mapping in _BINARY_MAPS.items():
        if unique.issubset(allowed):
            return {v: mapping[v] for v in unique}
    return None


def is_binary_like(series: pd.Series) -> bool:
    return binary_mapping(series) is not None


def _parseable_as_datetime(series: pd.Series, threshold: float = 0.9) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    try:
        with pd.option_context("mode.chained_assignment", None):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        return False
    return float(parsed.notna().mean()) >= threshold


def _numeric_like_string(series: pd.Series, threshold: float = 0.9) -> bool:
    sample = series.dropna().head(500)
    if sample.empty:
        return False
    coerced = pd.to_numeric(sample, errors="coerce")
    return float(coerced.notna().mean()) >= threshold


def _looks_like_identifier(series: pd.Series, threshold: float = 0.95) -> bool:
    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return False
    no_whitespace = ~sample.str.contains(r"\s", regex=True)
    return float(no_whitespace.mean()) >= threshold


def _basic_stats(series: pd.Series) -> dict[str, float | int]:
    n_non_null = int(series.notna().sum())
    n_rows = len(series)
    n_unique = int(series.nunique(dropna=True))
    missing_pct = 0.0 if n_rows == 0 else (n_rows - n_non_null) / n_rows * 100
    cardinality_ratio = n_unique / n_non_null if n_non_null else 0.0
    return {
        "n_non_null": n_non_null,
        "n_unique": n_unique,
        "missing_pct": round(missing_pct, 2),
        "cardinality_ratio": round(cardinality_ratio, 4),
    }


def infer_column_type(series: pd.Series) -> dict[str, Any]:
    """Infer the semantic type of a single ``Series``.

    Returns a dict with keys: inferred_type, confidence, notes, plus the basic
    stats (n_non_null, n_unique, missing_pct, cardinality_ratio).
    """
    stats = _basic_stats(series)
    notes: list[str] = []
    inferred_type: str
    confidence: str = "high"

    if isinstance(series.dtype, pd.CategoricalDtype):
        inferred_type = "categorical"
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        inferred_type = "datetime"
    elif pd.api.types.is_bool_dtype(series.dtype):
        inferred_type = "boolean"
    elif pd.api.types.is_numeric_dtype(series.dtype):
        mapping = binary_mapping(series)
        if mapping is not None and stats["n_unique"] <= 2:
            inferred_type = "boolean"
            confidence = "medium"
            notes.append("numeric 0/1 column: ambiguous boolean vs numeric")
        elif (
            pd.api.types.is_integer_dtype(series.dtype)
            and stats["cardinality_ratio"] >= 0.95
        ):
            inferred_type = "id"
            notes.append("high-cardinality unique integer values")
        else:
            inferred_type = "numeric"
    else:
        if stats["n_non_null"] == 0:
            inferred_type = "string"
            confidence = "low"
            notes.append("empty column")
        else:
            mapping = binary_mapping(series)
            if mapping is not None:
                inferred_type = "boolean"
                confidence = "medium"
                notes.append(f"stored as strings ({', '.join(sorted(mapping))})")
            elif _parseable_as_datetime(series):
                inferred_type = "datetime"
                confidence = "medium"
                notes.append("parsed from object dtype")
            elif _numeric_like_string(series):
                inferred_type = "numeric"
                confidence = "low"
                notes.append("numeric-looking values stored as string")
            elif stats["cardinality_ratio"] >= 0.95:
                if _looks_like_identifier(series):
                    inferred_type = "id"
                    notes.append("high-cardinality unique values")
                else:
                    inferred_type = "string"
                    notes.append("high-cardinality free text")
            elif stats["cardinality_ratio"] >= 0.5:
                inferred_type = "string"
                notes.append("many unique values")
            else:
                inferred_type = "categorical"

    return {**stats, "inferred_type": inferred_type, "confidence": confidence, "notes": notes}