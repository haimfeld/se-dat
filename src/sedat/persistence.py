"""JSON persistence for plan objects.

``EncodingPlan``, ``ImputationPlan`` and ``DatetimeFeaturePlan`` can be
saved to a JSON file and loaded back later. Plans serialize their *decisions*
(categories, fill values, feature lists) rather than function references, so
a plan fitted on a training frame can be re-applied to a completely new
DataFrame — e.g. a test set or production batch — without recomputation and
with an identical output schema.

Usage::

    plan = sedat.suggest_encodings(train_df, target=train_y)
    plan.save("encoding_plan.json")

    loaded = sedat.EncodingPlan.load("encoding_plan.json")
    test_encoded = loaded.apply_all(test_df)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .datetime_features import DatetimeFeaturePlan, DatetimeFeatureSuggestion
from .encoding import EncodingPlan, EncodingSuggestion
from .imputation import ImputationPlan, ImputationSuggestion

PLAN_SCHEMA_VERSION = 1

ENCODING_PLAN_TYPE = "encoding"
IMPUTATION_PLAN_TYPE = "imputation"
DATETIME_PLAN_TYPE = "datetime_feature"


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas scalars to JSON-native types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _encoding_plan_to_dict(plan: EncodingPlan) -> dict[str, Any]:
    return {
        "cardinality_threshold": plan.cardinality_threshold,
        # EncodingPlan.target (a Series) is intentionally NOT serialized;
        # target-encoding falls back to the stored per-category means.
        "suggestions": [
            {
                "column": s.column,
                "current_type": s.current_type,
                "suggested_type": s.suggested_type,
                "strategy": s.strategy,
                "cardinality": s.cardinality,
                "rationale": s.rationale,
                "categories": [_jsonable(c) for c in s.categories],
                "target_means": {k: float(v) for k, v in s.target_means.items()},
                "mapping": (
                    None
                    if s.mapping is None
                    else {_jsonable(k): int(v) for k, v in s.mapping.items()}
                ),
            }
            for s in plan.suggestions
        ],
    }


def _encoding_plan_from_dict(payload: dict[str, Any]) -> EncodingPlan:
    suggestions = [
        EncodingSuggestion(
            column=s["column"],
            current_type=s["current_type"],
            suggested_type=s["suggested_type"],
            strategy=s["strategy"],
            cardinality=int(s["cardinality"]),
            rationale=s["rationale"],
            categories=list(s.get("categories", [])),
            target_means=dict(s.get("target_means", {})),
            mapping=(
                None
                if s.get("mapping") is None
                else {str(k): int(v) for k, v in s["mapping"].items()}
            ),
        )
        for s in payload.get("suggestions", [])
    ]
    return EncodingPlan(
        suggestions=suggestions,
        cardinality_threshold=int(payload.get("cardinality_threshold", 10)),
        target=None,
    )


def _imputation_plan_to_dict(plan: ImputationPlan) -> dict[str, Any]:
    return {
        "missing_threshold": plan.missing_threshold,
        "suggestions": [
            {
                "column": s.column,
                "current_type": s.current_type,
                "strategy": s.strategy,
                "n_missing": s.n_missing,
                "missing_pct": s.missing_pct,
                "fill_value": _jsonable(s.fill_value),
                "rationale": s.rationale,
            }
            for s in plan.suggestions
        ],
    }


def _imputation_plan_from_dict(payload: dict[str, Any]) -> ImputationPlan:
    suggestions = []
    for s in payload.get("suggestions", []):
        fill_value = s.get("fill_value")
        # datetime modes round-trip as ISO strings; restore Timestamps so
        # fillna works on datetime64 columns again
        if (
            isinstance(fill_value, str)
            and s.get("current_type") == "datetime"
        ):
            fill_value = pd.Timestamp(fill_value)
        suggestions.append(
            ImputationSuggestion(
                column=s["column"],
                current_type=s["current_type"],
                strategy=s["strategy"],
                n_missing=int(s["n_missing"]),
                missing_pct=float(s["missing_pct"]),
                fill_value=fill_value,
                rationale=s.get("rationale", ""),
            )
        )
    return ImputationPlan(
        suggestions=suggestions,
        missing_threshold=float(payload.get("missing_threshold", 50.0)),
    )


def _datetime_plan_to_dict(plan: DatetimeFeaturePlan) -> dict[str, Any]:
    return {
        "suggestions": [
            {
                "column": s.column,
                "features": list(s.features),
                "has_time_component": bool(s.has_time_component),
            }
            for s in plan.suggestions
        ]
    }


def _datetime_plan_from_dict(payload: dict[str, Any]) -> DatetimeFeaturePlan:
    suggestions = [
        DatetimeFeatureSuggestion(
            column=s["column"],
            features=list(s.get("features", [])),
            has_time_component=bool(s.get("has_time_component", False)),
        )
        for s in payload.get("suggestions", [])
    ]
    return DatetimeFeaturePlan(suggestions=suggestions)


_SERIALIZERS = {
    ENCODING_PLAN_TYPE: (_encoding_plan_to_dict, _encoding_plan_from_dict),
    IMPUTATION_PLAN_TYPE: (_imputation_plan_to_dict, _imputation_plan_from_dict),
    DATETIME_PLAN_TYPE: (_datetime_plan_to_dict, _datetime_plan_from_dict),
}

_PLAN_TYPES = {
    EncodingPlan: ENCODING_PLAN_TYPE,
    ImputationPlan: IMPUTATION_PLAN_TYPE,
    DatetimeFeaturePlan: DATETIME_PLAN_TYPE,
}


def save_plan(plan, path: str) -> str:
    """Serialize ``plan`` to ``path`` as JSON; returns the path."""
    plan_type = _PLAN_TYPES.get(type(plan))
    if plan_type is None:
        raise TypeError(f"Cannot save object of type {type(plan).__name__}")
    to_dict, _ = _SERIALIZERS[plan_type]
    envelope = {
        "plan_type": plan_type,
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "payload": to_dict(plan),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, ensure_ascii=False)
    return path


def load_plan(path: str):
    """Deserialize a plan saved by :func:`save_plan`."""
    with open(path, "r", encoding="utf-8") as fh:
        envelope = json.load(fh)
    plan_type = envelope.get("plan_type")
    if plan_type not in _SERIALIZERS:
        raise ValueError(
            f"Unknown plan type '{plan_type}'. Expected one of: "
            f"{sorted(_SERIALIZERS)}"
        )
    _, from_dict = _SERIALIZERS[plan_type]
    return from_dict(envelope.get("payload", {}))


def _make_save_load(cls):
    def save(self, path: os.PathLike | str) -> str:
        return save_plan(self, path)

    def load(cls_, path: os.PathLike | str):  # noqa: N805 - classmethod
        plan = load_plan(path)
        if not isinstance(plan, cls_):
            raise TypeError(
                f"File contains a {type(plan).__name__}, expected {cls_.__name__}"
            )
        return plan

    save.__name__ = "save"
    load.__name__ = "load"
    cls.save = save
    cls.load = classmethod(load)


_make_save_load(EncodingPlan)
_make_save_load(ImputationPlan)
_make_save_load(DatetimeFeaturePlan)
