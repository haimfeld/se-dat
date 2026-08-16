"""Column type profiling.

Produces a per-column profile with the inferred semantic type, basic stats
(missing %, unique count, cardinality ratio) and a confidence flag when the
inference is ambiguous.
"""

from __future__ import annotations

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

    @property
    def ambiguous(self) -> bool:
        return self.confidence in ("low", "medium")


@dataclass
class DataFrameProfile:
    columns: list[ColumnProfile]

    @property
    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "column": c.name,
                "dtype": c.dtype,
                "inferred_type": c.inferred_type,
                "missing_pct": c.missing_pct,
                "n_unique": c.n_unique,
                "cardinality_ratio": c.cardinality_ratio,
                "confidence": c.confidence,
                "notes": "; ".join(c.notes),
            }
            for c in self.columns
        ]
        return pd.DataFrame(rows).set_index("column")

    def __repr__(self) -> str:
        return repr(self.summary)


def profile_column(series: pd.Series) -> ColumnProfile:
    info = infer_column_type(series)
    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_type=info["inferred_type"],
        n_non_null=info["n_non_null"],
        n_unique=info["n_unique"],
        missing_pct=info["missing_pct"],
        cardinality_ratio=info["cardinality_ratio"],
        confidence=info["confidence"],
        notes=info["notes"],
    )


def profile_dataframe(df: pd.DataFrame) -> DataFrameProfile:
    return DataFrameProfile([profile_column(df[col]) for col in df.columns])