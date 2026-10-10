"""Data-quality checks: range/plausibility rules and duplicate rows.

Range rules are opt-in. Pass them as ``{column: bounds}`` where ``bounds`` is a
``(low, high)`` pair (either side may be ``None``), a ``{"min": .., "max": ..}``
dict, or ``{"allowed": [..]}`` for a value whitelist. Two convenience shorthands
cover the most common cases: ``non_negative=["age", ...]`` and
``percentages=["body_fat_pct", ...]`` (``0..100``). Anything outside a bound is
reported with its count, percentage and example values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class RangeViolation:
    column: str
    rule: str
    n_violations: int
    pct: float = 0.0
    lower: float | None = None
    upper: float | None = None
    examples: list[Any] = field(default_factory=list)

    @property
    def message(self) -> str:
        shown = ", ".join(str(v) for v in self.examples[:3])
        return f"{self.column}: {self.n_violations} value(s) outside {self.rule} (e.g. {shown})"


@dataclass
class PlausibilityReport:
    violations: list[RangeViolation] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return any(v.n_violations for v in self.violations)

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": v.column,
                "rule": v.rule,
                "n_violations": v.n_violations,
                "pct": round(v.pct, 2),
                "examples": ", ".join(str(x) for x in v.examples[:3]),
                "message": v.message,
            }
            for v in self.violations
            if v.n_violations
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return repr(self.summary)


def duplicate_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask (aligned to ``df.index``) marking duplicated rows."""
    return df.duplicated(keep="first")


def _split_rule(rule: Any) -> tuple[float | None, float | None, list[Any] | None]:
    if isinstance(rule, dict):
        return rule.get("min"), rule.get("max"), rule.get("allowed")
    if isinstance(rule, (tuple, list)) and len(rule) == 2:
        low, high = rule
        return low, high, None
    if isinstance(rule, (tuple, list)) and len(rule) > 2:
        return None, None, list(rule)
    raise TypeError(f"Unsupported range rule {rule!r}")


def check_ranges(
    df: pd.DataFrame,
    rules: dict[str, Any] | None = None,
    non_negative: list[str] | None = None,
    percentages: list[str] | None = None,
) -> PlausibilityReport:
    """Validate values against optional range/plausibility rules."""
    effective: dict[str, Any] = dict(rules or {})
    for col in non_negative or []:
        effective.setdefault(col, {"min": 0})
    for col in percentages or []:
        effective.setdefault(col, {"min": 0, "max": 100})

    violations: list[RangeViolation] = []
    for column, rule in effective.items():
        if column not in df.columns:
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce") if not pd.api.types.is_numeric_dtype(series) else series
        low, high, allowed = _split_rule(rule)

        if allowed is not None:
            mask = series.notna() & ~series.isin(allowed)
            rule_text = f"allowed values {allowed}"
        else:
            mask = pd.Series(False, index=series.index)
            if low is not None:
                mask = mask | (numeric < low)
            if high is not None:
                mask = mask | (numeric > high)
            mask = mask & series.notna()
            bounds = f"[{low if low is not None else '-inf'}, {high if high is not None else 'inf'}]"
            rule_text = bounds

        n_valid = int(series.notna().sum())
        count = int(mask.sum())
        violations.append(
            RangeViolation(
                column=str(column),
                rule=rule_text,
                n_violations=count,
                pct=(count / n_valid * 100) if n_valid else 0.0,
                lower=low,
                upper=high,
                examples=series[mask].head(5).tolist(),
            )
        )
    return PlausibilityReport(violations=violations)
