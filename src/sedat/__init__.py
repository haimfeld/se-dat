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
from .encoding import (
    EncodingPlan,
    EncodingSuggestion,
    apply_encodings,
    suggest_encodings,
)
from .profile import ColumnProfile, DataFrameProfile, profile_column, profile_dataframe
from .report import EDAReport
from .types import infer_column_type, is_binary_like

__version__ = "0.1.0"

__all__ = [
    "ColumnProfile",
    "CorrelationReport",
    "DataFrameProfile",
    "EDAReport",
    "EncodingPlan",
    "EncodingSuggestion",
    "apply_encodings",
    "categorical_correlation",
    "correlation_heatmap",
    "correlation_ratio",
    "correlation_report",
    "cramers_v",
    "infer_column_type",
    "is_binary_like",
    "numeric_categorical_correlation",
    "numeric_correlation",
    "profile_column",
    "profile_dataframe",
    "suggest_encodings",
]