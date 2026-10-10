"""se-dat: Simple Exploratory Data Analysis tool.

Provides column type profiling, correlation analysis (numeric, categorical and
mixed) and encoding suggestions with automatic transforms.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:  # read the installed distribution version so it never goes stale
    __version__ = _pkg_version("se-dat")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

from .correlations import (
    CorrelationReport,
    categorical_correlation,
    condition_number,
    correlation_heatmap,
    correlation_ratio,
    correlation_report,
    cramers_v,
    numeric_categorical_correlation,
    numeric_correlation,
    variance_inflation_factors,
)
from .datadict import ColumnSpec, DataDictionary
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
from .quality import (
    PlausibilityReport,
    RangeViolation,
    check_ranges,
    duplicate_rows,
)
from .report import EDAReport
from .target import (
    TargetAnalysis,
    analyze_target,
    infer_target_role,
    mutual_information,
)
from .transforms import (
    TransformPlan,
    TransformSuggestion,
    apply_transforms,
    suggest_transforms,
)
from .types import infer_column_type, is_binary_like

__all__ = [
    "ColumnOutliers",
    "ColumnProfile",
    "ColumnSpec",
    "CorrelationReport",
    "DataDictionary",
    "DataFrameProfile",
    "DatetimeFeaturePlan",
    "DatetimeFeatureSuggestion",
    "EDAReport",
    "EncodingPlan",
    "EncodingSuggestion",
    "ImputationPlan",
    "ImputationSuggestion",
    "OutlierReport",
    "PlausibilityReport",
    "RangeViolation",
    "TargetAnalysis",
    "TransformPlan",
    "TransformSuggestion",
    "__version__",
    "analyze_target",
    "apply_datetime_features",
    "apply_encodings",
    "apply_imputations",
    "apply_transforms",
    "categorical_correlation",
    "check_ranges",
    "condition_number",
    "correlation_heatmap",
    "correlation_ratio",
    "correlation_report",
    "cramers_v",
    "detect_outliers",
    "duplicate_rows",
    "infer_column_type",
    "infer_target_role",
    "is_binary_like",
    "load_plan",
    "mutual_information",
    "numeric_categorical_correlation",
    "numeric_correlation",
    "outlier_note",
    "profile_column",
    "profile_dataframe",
    "save_plan",
    "suggest_datetime_features",
    "suggest_encodings",
    "suggest_imputations",
    "suggest_transforms",
    "variance_inflation_factors",
]
