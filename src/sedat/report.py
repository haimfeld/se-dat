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
from .imputation import (
    DEFAULT_MISSING_THRESHOLD,
    ImputationPlan,
    suggest_imputations,
)
from .outliers import OutlierReport, detect_outliers, outlier_note
from .profile import DataFrameProfile, profile_dataframe


@dataclass
class EDAReport:
    """One object summarizing profiling, correlations and encodings for a frame."""

    df: pd.DataFrame
    profile: DataFrameProfile
    correlations: CorrelationReport
    encodings: EncodingPlan
    imputations: ImputationPlan
    outliers: OutlierReport
    corr_threshold: float = 0.7
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        corr_threshold: float = 0.7,
        cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
        target: pd.Series | None = None,
        missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    ) -> EDAReport:
        profile = profile_dataframe(df)
        outliers = detect_outliers(df)
        profile.outlier_notes.update({c.column: outlier_note(c) for c in outliers.columns if c.count})
        return cls(
            df=df,
            profile=profile,
            correlations=correlation_report(df, threshold=corr_threshold),
            encodings=suggest_encodings(df, target=target, cardinality_threshold=cardinality_threshold),
            imputations=suggest_imputations(df, missing_threshold=missing_threshold),
            outliers=outliers,
            corr_threshold=corr_threshold,
            cardinality_threshold=cardinality_threshold,
            missing_threshold=missing_threshold,
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

    @property
    def imputation_summary(self) -> pd.DataFrame:
        return self.imputations.summary

    def apply_all_encodings(self, target: pd.Series | None = None) -> pd.DataFrame:
        return self.encodings.apply_all(self.df, target=target)

    def apply_all_imputations(self) -> pd.DataFrame:
        return self.imputations.apply_all(self.df)

    def to_html(self, path: str) -> str:
        """Write a static, self-contained HTML report; returns ``path``."""
        from .report_html import write_html

        return write_html(self, path)

    def __repr__(self) -> str:
        lines = [repr(self.profile)]
        corr = self.correlations.summary
        if not corr.empty:
            flagged = corr[corr["flagged"]]
            if not flagged.empty:
                lines.append("\nFlagged highly-correlated pairs:")
                lines.append(repr(flagged))
        if self.imputations.suggestions:
            lines.append("\nImputation suggestions:")
            lines.append(repr(self.imputations))
        lines.append("\n" + repr(self.encodings))
        return "\n".join(lines)
