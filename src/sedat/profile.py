"""Column type profiling.

Produces a per-column profile with the inferred semantic type, basic stats
(missing %, unique count, cardinality ratio) and a confidence flag when the
inference is ambiguous.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import pandas as pd

from .types import infer_column_type


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str
    n_non_null: int
    n_unique: int
    missing_pct: float
    cardinality_ratio: float
    confidence: str
    notes: list[str] = field(default_factory=list)
    is_constant: bool = False
    role: str | None = None

    @property
    def ambiguous(self) -> bool:
        return self.confidence in ("low", "medium")


@dataclass
class DataFrameProfile:
    columns: list[ColumnProfile]
    outlier_notes: dict[Hashable, str] = field(default_factory=dict)
    duplicate_row_count: int = 0
    target_name: str | None = None
    target_role: str | None = None

    @property
    def constant_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_constant]

    def set_target(self, name: str, role: str) -> None:
        """Mark ``name`` as the target and resolve its ambiguity in the profile."""
        self.target_name = name
        self.target_role = role
        for column in self.columns:
            if column.name == name:
                column.role = "target"
                column.confidence = "high"
                column.notes = [
                    f"designated target ({role}); ambiguity resolved by target role",
                    *[n for n in column.notes if "ambiguous" not in n],
                ]
                return

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": c.name,
                "dtype": c.dtype,
                "inferred_type": c.inferred_type,
                "role": c.role or "",
                "missing_pct": c.missing_pct,
                "n_unique": c.n_unique,
                "cardinality_ratio": c.cardinality_ratio,
                "constant": c.is_constant,
                "confidence": c.confidence,
                "outliers": self.outlier_notes.get(c.name, ""),
                "notes": "; ".join(c.notes),
            }
            for c in self.columns
        ]
        return pd.DataFrame(rows).set_index("column")

    def __repr__(self) -> str:
        parts = [repr(self.summary)]
        if self.duplicate_row_count:
            parts.append(f"\nDuplicate rows: {self.duplicate_row_count}")
        return "\n".join(parts)


def profile_column(
    series: pd.Series,
    type_override: str | None = None,
    derived: bool = False,
) -> ColumnProfile:
    info = infer_column_type(series)
    notes = list(info["notes"])
    inferred_type = info["inferred_type"]
    confidence = info["confidence"]
    if type_override:
        inferred_type = type_override
        confidence = "high"
        notes = [f"type '{type_override}' from data dictionary", *notes]
    if derived:
        notes.append("derived column (from a data dictionary)")
    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_type=inferred_type,
        n_non_null=info["n_non_null"],
        n_unique=info["n_unique"],
        missing_pct=info["missing_pct"],
        cardinality_ratio=info["cardinality_ratio"],
        confidence=confidence,
        notes=notes,
        is_constant=info.get("is_constant", False),
    )


def profile_dataframe(
    df: pd.DataFrame,
    type_overrides: dict[str, str] | None = None,
    derived_columns: list[str] | None = None,
) -> DataFrameProfile:
    overrides = type_overrides or {}
    derived = set(derived_columns or [])
    return DataFrameProfile(
        [profile_column(df[col], type_override=overrides.get(col), derived=col in derived) for col in df.columns],
        duplicate_row_count=int(df.duplicated().sum()),
    )
