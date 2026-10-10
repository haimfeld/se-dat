"""Target-aware association analysis and leakage detection.

A designated ``target`` is given its own role (``binary``, ``numeric``,
``categorical`` or ``datetime``) instead of being treated as just another
column. For every other feature we compute an appropriate effect size
(point-biserial / Pearson for numeric targets, correlation ratio η for mixed
pairs, Cramér's V for categorical pairs) plus a discretisation-based mutual
information, and return them as a single ranked table.

Features whose association with the target is almost perfect (an effect size or
normalised mutual information at/above ``leakage_threshold``) are flagged and
warn about — a strong signal that the feature *encodes* the target (leakage),
which matters a lot for habit/behaviour datasets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .correlations import correlation_ratio, cramers_v
from .types import binary_mapping, infer_column_type

TARGET_ROLE_BINARY = "binary"
TARGET_ROLE_NUMERIC = "numeric"
TARGET_ROLE_CATEGORICAL = "categorical"
TARGET_ROLE_DATETIME = "datetime"

DEFAULT_LEAKAGE_THRESHOLD = 0.98
_DEFAULT_MI_BINS = 10


def _as_target_series(target: pd.Series | Any) -> pd.Series:
    return target if isinstance(target, pd.Series) else pd.Series(target)


def infer_target_role(target: pd.Series) -> str:
    """Classify the target's semantic role for association analysis."""
    target = _as_target_series(target)
    if pd.api.types.is_bool_dtype(target.dtype):
        return TARGET_ROLE_BINARY
    if pd.api.types.is_numeric_dtype(target.dtype):
        n_unique = int(target.nunique(dropna=True))
        if binary_mapping(target) is not None and n_unique <= 2:
            return TARGET_ROLE_BINARY
        return TARGET_ROLE_NUMERIC
    if pd.api.types.is_datetime64_any_dtype(target.dtype):
        return TARGET_ROLE_DATETIME
    role = infer_column_type(target)["inferred_type"]
    if role == "boolean":
        return TARGET_ROLE_BINARY
    if role == "datetime":
        return TARGET_ROLE_DATETIME
    return TARGET_ROLE_CATEGORICAL


def _is_numeric_feature(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return False
    if pd.api.types.is_numeric_dtype(series.dtype):
        return True
    return infer_column_type(series)["inferred_type"] == "numeric"


def _discretize(series: pd.Series, max_bins: int = _DEFAULT_MI_BINS) -> pd.Series:
    """Bin a series so MI is meaningful.

    Numeric columns are quantile-binned; high-cardinality categorical/text
    columns keep their ``max_bins`` most frequent values and roll the rest into
    an ``"__other__"`` bucket. Without this cap a near-unique column would look
    like it "perfectly" predicts anything.
    """
    values = series
    if (
        pd.api.types.is_numeric_dtype(values)
        and not pd.api.types.is_bool_dtype(values)
        and int(values.nunique(dropna=True)) > max_bins
    ):
        try:
            return pd.qcut(values, q=max_bins, duplicates="drop").astype("object")
        except (ValueError, TypeError):  # pragma: no cover - degenerate data
            pass
    values = values.astype("object")
    if int(values.nunique(dropna=True)) > max_bins:
        top = set(values.value_counts().index[:max_bins])
        values = values.where(values.isin(top) | values.isna(), "__other__")
    return values


def mutual_information(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Return ``(mi, normalized_mi)`` between two series.

    Continuous inputs are quantile-binned first; MI is in nats and normalised
    by ``min(H(x), H(y))`` so the result lies in ``[0, 1]``.
    """
    frame = pd.DataFrame({"x": _discretize(pd.Series(x)), "y": _discretize(pd.Series(y))}).dropna()
    if frame.empty:
        return 0.0, 0.0
    table = pd.crosstab(frame["x"], frame["y"]).to_numpy(dtype=float)
    total = table.sum()
    if total <= 0:
        return 0.0, 0.0
    px = table.sum(axis=1, keepdims=True) / total
    py = table.sum(axis=0, keepdims=True) / total
    expected = px @ py
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(table > 0, (table / total) / expected, 1.0)
        mi = float(np.sum((table / total) * np.log(ratio)))
    hx = float(-np.sum(px * np.log(px)))
    hy = float(-np.sum(py * np.log(py)))
    denom = min(hx, hy)
    return mi, (mi / denom if denom > 0 else 0.0)


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(frame["x"], frame["y"])
        return float(r), float(p)
    except (ValueError, ZeroDivisionError):  # pragma: no cover - degenerate data
        return float("nan"), float("nan")


def _effect_size(feature: pd.Series, target: pd.Series, target_role: str) -> dict[str, Any]:
    """Pick and compute an effect size appropriate to the two column types."""
    feature_numeric = _is_numeric_feature(feature)
    if feature_numeric and target_role in (TARGET_ROLE_NUMERIC, TARGET_ROLE_BINARY):
        value, p_value = _pearson(feature, target)
        method = "point_biserial" if target_role == TARGET_ROLE_BINARY else "pearson"
        return {"method": method, "value": value, "effect_size": abs(value), "p_value": p_value}
    if feature_numeric:
        value = correlation_ratio(feature, target)
        return {"method": "eta", "value": value, "effect_size": abs(value), "p_value": float("nan")}
    if target_role in (TARGET_ROLE_NUMERIC, TARGET_ROLE_BINARY):
        value = correlation_ratio(target, feature)
        return {"method": "eta", "value": value, "effect_size": abs(value), "p_value": float("nan")}
    value = cramers_v(feature, target)
    return {"method": "cramers_v", "value": value, "effect_size": abs(value), "p_value": float("nan")}


@dataclass
class TargetAnalysis:
    """Ranked per-feature association with a designated target."""

    target_name: str
    target_role: str
    associations: pd.DataFrame
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD

    @property
    def summary(self) -> pd.DataFrame:
        """Ranked table: strongest feature→target associations first."""
        if self.associations.empty:
            return self.associations
        return self.associations.sort_values(
            ["effect_size", "normalized_mi"], ascending=False, na_position="last"
        ).reset_index(drop=True)

    @property
    def leakage_features(self) -> list[str]:
        if self.associations.empty:
            return []
        flagged = self.associations[self.associations["leakage"]]
        return flagged["feature"].tolist()

    def __repr__(self) -> str:
        return repr(self.summary)


def analyze_target(
    df: pd.DataFrame,
    target: pd.Series,
    target_name: str | None = None,
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    warn: bool = True,
) -> TargetAnalysis:
    """Build a :class:`TargetAnalysis` of ``df`` features against ``target``.

    When ``target_name`` is passed explicitly, a column of that name is treated
    as the target and excluded from the feature list (callers such as
    :class:`~sedat.report.EDAReport` already drop it). When it is derived from
    ``target.name`` it is only used for labelling, so a feature that merely
    shares the name is still analysed — this is what lets a direct copy of the
    target be reported as leakage. Features whose association meets
    ``leakage_threshold`` are marked as potential leakage and, when ``warn`` is
    true, trigger a ``UserWarning``.
    """
    exclude_named = target_name is not None
    target = _as_target_series(target)
    if target_name is None:
        target_name = str(target.name) if target.name is not None else "target"
    target_role = infer_target_role(target)

    rows: list[dict[str, Any]] = []
    for col in df.columns:
        if exclude_named and col == target_name:
            continue
        feature = df[col]
        stats = _effect_size(feature, target, target_role)
        mi, nmi = mutual_information(feature, target)
        feature_type = infer_column_type(feature)["inferred_type"]
        effect = stats["effect_size"]
        leakage = bool(
            (not np.isnan(effect) and effect >= leakage_threshold) or (not np.isnan(nmi) and nmi >= leakage_threshold)
        )
        rows.append(
            {
                "feature": str(col),
                "feature_type": feature_type,
                "method": stats["method"],
                "effect_size": None if np.isnan(effect) else round(float(effect), 4),
                "value": None if np.isnan(stats["value"]) else round(float(stats["value"]), 4),
                "mutual_information": round(float(mi), 4),
                "normalized_mi": round(float(nmi), 4),
                "p_value": None if np.isnan(stats["p_value"]) else float(stats["p_value"]),
                "leakage": leakage,
            }
        )

    analysis = TargetAnalysis(
        target_name=target_name,
        target_role=target_role,
        associations=pd.DataFrame(rows),
        leakage_threshold=leakage_threshold,
    )
    if warn and analysis.leakage_features:
        warnings.warn(
            "Potential target leakage: feature(s) "
            + ", ".join(repr(f) for f in analysis.leakage_features)
            + f" almost perfectly predict the target '{target_name}' "
            f"(effect size / normalised MI >= {leakage_threshold}).",
            UserWarning,
            stacklevel=2,
        )
    return analysis
