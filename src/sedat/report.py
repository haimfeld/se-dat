"""High-level facade combining profiling, correlations and encoding."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .correlations import CorrelationReport, correlation_report
from .encoding import (
    DEFAULT_CARDINALITY_THRESHOLD,
    EncodingPlan,
    suggest_encodings,
)
from .profile import DataFrameProfile, profile_dataframe


@dataclass
class EDAReport:
    """One object summarizing profiling, correlations and encodings for a frame."""

    df: pd.DataFrame
    profile: DataFrameProfile
    correlations: CorrelationReport
    encodings: EncodingPlan
    corr_threshold: float = 0.7
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        corr_threshold: float = 0.7,
        cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
        target: pd.Series | None = None,
    ) -> "EDAReport":
        return cls(
            df=df,
            profile=profile_dataframe(df),
            correlations=correlation_report(df, threshold=corr_threshold),
            encodings=suggest_encodings(
                df, target=target, cardinality_threshold=cardinality_threshold
            ),
            corr_threshold=corr_threshold,
            cardinality_threshold=cardinality_threshold,
        )

    @property
    def summary(self) -> pd.DataFrame:
        return self.profile.summary

    @property
    def correlation_summary(self) -> pd.DataFrame:
        return self.correlations.summary

    @property
    def encoding_summary(self) -> pd.DataFrame:
        return self.encodings.summary

    def apply_all_encodings(self, target: pd.Series | None = None) -> pd.DataFrame:
        return self.encodings.apply_all(self.df, target=target)

    def __repr__(self) -> str:
        lines = [repr(self.profile)]
        corr = self.correlations.summary
        if not corr.empty:
            flagged = corr[corr["flagged"]]
            if not flagged.empty:
                lines.append("\nFlagged highly-correlated pairs:")
                lines.append(repr(flagged))
        lines.append("\n" + repr(self.encodings))
        return "\n".join(lines)