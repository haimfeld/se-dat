"""High-level facade combining profiling, correlations, targets and encodings."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .correlations import CorrelationReport, correlation_report
from .datadict import DataDictionary
from .encoding import DEFAULT_CARDINALITY_THRESHOLD, EncodingPlan, suggest_encodings
from .imputation import (
    DEFAULT_MISSING_THRESHOLD,
    ImputationPlan,
    suggest_imputations,
)
from .outliers import IQR_METHOD, OutlierReport, detect_outliers, outlier_note
from .profile import DataFrameProfile, profile_dataframe
from .quality import PlausibilityReport, check_ranges
from .target import (
    DEFAULT_LEAKAGE_THRESHOLD,
    TargetAnalysis,
    analyze_target,
    infer_target_role,
)
from .transforms import DEFAULT_SKEW_THRESHOLD, TransformPlan, suggest_transforms

DEFAULT_FAST_SAMPLE = 50_000
DEFAULT_MAX_CORR_COLUMNS = 100


def _coerce_dictionary(dictionary: DataDictionary | dict | str | None) -> DataDictionary | None:
    if dictionary is None or isinstance(dictionary, DataDictionary):
        return dictionary
    return DataDictionary.load(dictionary)


def _sample(df: pd.DataFrame, sample: int | None, random_state: int | None) -> tuple[pd.DataFrame, bool]:
    """Return ``(analysis_frame, sampled)``.

    When ``sample`` is set and the frame is larger, a random subset of that size
    is used for the (statistically stable) analysis steps so very large frames
    stay fast. The full frame is kept for applying the resulting plans.
    """
    if sample is not None and sample > 0 and len(df) > sample:
        return df.sample(n=sample, random_state=random_state), True
    return df, False


def _target_name(target: pd.Series) -> str | None:
    return str(target.name) if getattr(target, "name", None) is not None else None


def _align_target(target, index, df: pd.DataFrame, target_name: str | None):
    if target_name and target_name in df.columns:
        return df[target_name].loc[index]
    if isinstance(target, pd.Series):
        if target.index.equals(index):
            return target
        if index.isin(target.index).all():
            return target.loc[index]
        return target.reindex(index)
    return pd.Series(target, index=df.index).loc[index]


def _cap_columns(df: pd.DataFrame, profile: DataFrameProfile, max_columns: int | None, exclude: str | None):
    """Limit the frame fed to the O(p^2) correlation steps.

    Ids and constant columns are deprioritised (and truncated) so the columns
    most likely to carry signal survive the cap.
    """
    if exclude:
        df = df.drop(columns=[exclude], errors="ignore")
    if max_columns is None or df.shape[1] <= max_columns:
        return df
    by_name = {c.name: c for c in profile.columns}
    order = list(df.columns)

    def rank(col: str) -> int:
        column = by_name.get(col)
        if column is None:
            return 1
        if column.is_constant or column.inferred_type == "id":
            return 2
        return 0

    kept = sorted(order, key=lambda c: (rank(c), order.index(c)))[:max_columns]
    return df[kept]


@dataclass
class EDAReport:
    """One object summarizing profiling, correlations, targets and encodings."""

    df: pd.DataFrame
    profile: DataFrameProfile
    correlations: CorrelationReport
    encodings: EncodingPlan
    imputations: ImputationPlan
    outliers: OutlierReport
    target_analysis: TargetAnalysis | None = None
    transforms: TransformPlan = field(default_factory=TransformPlan)
    plausibility: PlausibilityReport = field(default_factory=PlausibilityReport)
    dictionary: DataDictionary | None = None
    corr_threshold: float = 0.7
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD
    outlier_method: str = IQR_METHOD
    sampled: bool = False
    n_rows_analyzed: int = 0
    max_columns: int | None = None
    random_state: int | None = 0

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        corr_threshold: float = 0.7,
        cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
        target: pd.Series | None = None,
        target_name: str | None = None,
        missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
        dictionary: DataDictionary | dict | str | None = None,
        range_rules: dict[str, Any] | None = None,
        non_negative: list[str] | None = None,
        percentages: list[str] | None = None,
        outlier_method: str = IQR_METHOD,
        skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
        leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
        sample: int | None = None,
        max_columns: int | None = None,
        fast: bool = False,
        random_state: int | None = 0,
        compute_target: bool = True,
        warn_leakage: bool = True,
        n_jobs: int | None = None,
        progress: bool = False,
    ) -> EDAReport:
        """Build an :class:`EDAReport` for ``df``.

        Parameters
        ----------
        target:
            Optional target Series. Its column (matched by name) is given a
            dedicated ``target`` role and excluded from feature analyses.
        dictionary:
            Optional :class:`~sedat.datadict.DataDictionary`, mapping, JSON path
            or CSV path describing column types, ranges and derived columns.
        sample / fast / max_columns:
            Scalability controls. ``fast=True`` samples to 50k rows (seeded),
            caps the correlation columns and skips VIF; ``sample`` gives an
            explicit row cap and ``max_columns`` caps the correlation columns.
        n_jobs / progress:
            Forwarded to the O(p^2) categorical/eta correlation loops.
        """
        dictionary = _coerce_dictionary(dictionary)
        type_overrides = dictionary.type_overrides() if dictionary else None
        derived_columns = dictionary.derived_columns() if dictionary else None

        if sample is None and fast:
            sample = DEFAULT_FAST_SAMPLE
        if max_columns is None and fast:
            max_columns = DEFAULT_MAX_CORR_COLUMNS
        analysis_df, sampled = _sample(df, sample, random_state)
        # The profile drives all downstream plans; profiling the (possibly
        # sampled) analysis frame keeps wide/large inputs from being re-scanned
        # once per plan. Column facts used by plan *application* (categories,
        # medians, target means) are still read from the full frame.
        profile = profile_dataframe(analysis_df, type_overrides=type_overrides, derived_columns=derived_columns)

        # Resolve the target from the dictionary when none was passed explicitly.
        if target is None and dictionary is not None:
            dict_targets = [c for c in dictionary.target_columns() if c in df.columns]
            if dict_targets:
                target_name = target_name or dict_targets[0]
                target = df[target_name]
        inferred_name = target_name if target_name is not None else _target_name(target)
        target_name = inferred_name

        corr_df = _cap_columns(analysis_df, profile, max_columns, exclude=target_name)
        correlations = correlation_report(
            corr_df,
            threshold=corr_threshold,
            compute_collinearity=not fast,
            n_jobs=n_jobs,
            progress=progress,
        )

        target_analysis = None
        if target is not None and compute_target:
            if target_name and target_name in df.columns:
                profile.set_target(target_name, infer_target_role(df[target_name]))
            aligned = _align_target(target, analysis_df.index, df, target_name)
            features = analysis_df.drop(columns=[target_name], errors="ignore") if target_name else analysis_df
            target_analysis = analyze_target(
                features,
                aligned,
                target_name=target_name,
                leakage_threshold=leakage_threshold,
                warn=warn_leakage,
            )

        if warn_leakage and dictionary is not None and target_name is not None:
            derived = [c for c in dictionary.derived_from(target_name) if c in df.columns]
            if derived:
                warnings.warn(
                    f"Derived column(s) {', '.join(repr(c) for c in derived)} are built from the target "
                    f"'{target_name}'; they leak the target and must be dropped before modelling.",
                    UserWarning,
                    stacklevel=2,
                )

        effective_rules: dict[str, Any] = dict(range_rules or {})
        if dictionary is not None:
            for name, rule in dictionary.range_rules().items():
                effective_rules.setdefault(name, rule)

        outliers = detect_outliers(df, method=outlier_method)
        profile.outlier_notes.update({c.column: outlier_note(c) for c in outliers.columns if c.count})

        return cls(
            df=df,
            profile=profile,
            correlations=correlations,
            encodings=suggest_encodings(
                df, target=target, cardinality_threshold=cardinality_threshold, profile=profile
            ),
            imputations=suggest_imputations(df, missing_threshold=missing_threshold, profile=profile),
            outliers=outliers,
            target_analysis=target_analysis,
            transforms=suggest_transforms(df, skew_threshold=skew_threshold, profile=profile),
            plausibility=check_ranges(
                df,
                rules=effective_rules,
                non_negative=non_negative,
                percentages=percentages,
            ),
            dictionary=dictionary,
            corr_threshold=corr_threshold,
            cardinality_threshold=cardinality_threshold,
            missing_threshold=missing_threshold,
            outlier_method=outlier_method,
            sampled=sampled,
            n_rows_analyzed=len(analysis_df),
            max_columns=max_columns,
            random_state=random_state,
        )

    # -- summary views -------------------------------------------------------
    @property
    def summary(self) -> pd.DataFrame:
        return self.profile.summary

    @property
    def correlation_summary(self) -> pd.DataFrame:
        return self.correlations.summary

    @property
    def collapsed_correlation_summary(self) -> pd.DataFrame:
        return self.correlations.collapsed_summary

    @property
    def vif_summary(self) -> pd.DataFrame:
        return self.correlations.vif

    @property
    def condition_number(self) -> float:
        return self.correlations.condition_number_value

    @property
    def encoding_summary(self) -> pd.DataFrame:
        return self.encodings.summary

    @property
    def imputation_summary(self) -> pd.DataFrame:
        return self.imputations.summary

    @property
    def transform_summary(self) -> pd.DataFrame:
        return self.transforms.summary

    @property
    def plausibility_summary(self) -> pd.DataFrame:
        return self.plausibility.summary

    @property
    def target_summary(self) -> pd.DataFrame:
        """Ranked per-feature association with the target (empty if none)."""
        return self.target_analysis.summary if self.target_analysis is not None else pd.DataFrame()

    @property
    def leakage_features(self) -> list[str]:
        """Features that almost perfectly predict the target (potential leakage)."""
        return [] if self.target_analysis is None else self.target_analysis.leakage_features

    @property
    def constant_columns(self) -> list[str]:
        return self.profile.constant_columns

    @property
    def duplicate_row_count(self) -> int:
        return self.profile.duplicate_row_count

    # -- transforms ----------------------------------------------------------
    def apply_all_encodings(
        self,
        target: pd.Series | None = None,
        warn_target_leakage: bool = True,
    ) -> pd.DataFrame:
        return self.encodings.apply_all(self.df, target=target, warn_target_leakage=warn_target_leakage)

    def apply_all_imputations(self) -> pd.DataFrame:
        return self.imputations.apply_all(self.df)

    def apply_all_transforms(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Apply every skew-correcting transform (defaults to the full frame)."""
        return self.transforms.apply_all(self.df if df is None else df)

    def to_html(self, path: str) -> str:
        """Write a static, self-contained HTML report; returns ``path``."""
        from .report_html import write_html

        return write_html(self, path)

    def __repr__(self) -> str:
        lines = [repr(self.profile)]
        if self.sampled:
            lines.append(
                f"\nAnalysis computed on a {self.n_rows_analyzed}-row sample; check translatable center stats."
            )
        if target_analysis := self.target_analysis:
            lines.append(f"\nTarget association ranking ({target_analysis.target_name}):")
            lines.append(repr(target_analysis.summary))
            if self.leakage_features:
                lines.append(f"\nPotential leakage: {', '.join(self.leakage_features)}")
        if self.constant_columns:
            lines.append(f"\nConstant columns: {', '.join(self.constant_columns)}")
        corr = self.correlations.summary
        if not corr.empty:
            flagged = corr[corr["flagged"]]
            if not flagged.empty:
                lines.append("\nFlagged highly-correlated pairs:")
                lines.append(repr(flagged))
        if self.plausibility.has_violations:
            lines.append("\nPlausibility violations:")
            lines.append(repr(self.plausibility))
        if self.transforms.suggestions:
            lines.append("\nTransform suggestions:")
            lines.append(repr(self.transforms))
        if self.imputations.suggestions:
            lines.append("\nImputation suggestions:")
            lines.append(repr(self.imputations))
        lines.append("\n" + repr(self.encodings))
        return "\n".join(lines)
