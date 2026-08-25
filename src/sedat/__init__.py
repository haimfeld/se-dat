"""se-dat: Simple Exploratory Data Analysis tool.

Provides column type profiling, correlation analysis (numeric, categorical and
mixed) and encoding suggestions with automatic transforms.
"""

from __future__ import annotations

from .correlations import (
    CorrelationReport,
    categorical_correlation,
    correlation_heatmap,
    correlation_ratio,
    correlation_report,
    cramers_v,
    numeric_categorical_correlation,
    numeric_correlation,
)
from .datetime_features import (
    DatetimeFeaturePlan,
    DatetimeFeatureSuggestion,
    apply_datetime_features,
    suggest_datetime_features,
)
from .encoding import (
    EncodingPlan,
    EncodingSuggestion,
    apply_encodings,
    suggest_encodings,
)
from .imputation import (
    ImputationPlan,
    ImputationSuggestion,
    apply_imputations,
    suggest_imputations,
)
from .outliers import (
    ColumnOutliers,
    OutlierReport,
    detect_outliers,
    outlier_note,
)
from .persistence import load_plan, save_plan
from .profile import ColumnProfile, DataFrameProfile, profile_column, profile_dataframe
from .report import EDAReport
from .types import infer_column_type, is_binary_like

__version__ = "0.1.0"

__all__ = [
    "ColumnOutliers",
    "ColumnProfile",
    "CorrelationReport",
    "DataFrameProfile",
    "DatetimeFeaturePlan",
    "DatetimeFeatureSuggestion",
    "EDAReport",
    "EncodingPlan",
    "EncodingSuggestion",
    "ImputationPlan",
    "ImputationSuggestion",
    "OutlierReport",
    "apply_datetime_features",
    "apply_encodings",
    "apply_imputations",
    "categorical_correlation",
    "correlation_heatmap",
    "correlation_ratio",
    "correlation_report",
    "cramers_v",
    "detect_outliers",
    "infer_column_type",
    "is_binary_like",
    "load_plan",
    "numeric_categorical_correlation",
    "numeric_correlation",
    "outlier_note",
    "profile_column",
    "profile_dataframe",
    "save_plan",
    "suggest_datetime_features",
    "suggest_encodings",
    "suggest_imputations",
]
