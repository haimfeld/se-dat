"""Data dictionary support.

A data dictionary describes columns declaratively — semantic ``type``,
allowed ``min``/``max``/``allowed`` ranges and whether a column is ``derived``
(optionally ``derived_from`` the target). It can be supplied as a Python
mapping, a JSON file, or a CSV with a header. se-dat uses it to override the
inferred column types, to drive plausibility checks, and to raise leakage
warnings for derived columns.

Example JSON::

    {
      "age":       {"type": "numeric", "min": 0, "max": 120},
      "churned":   {"type": "boolean", "target": true},
      "tenure":    {"type": "numeric", "min": 0},
      "churn_score": {"type": "numeric", "derived": true, "derived_from": ["churned"]}
    }
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BOOL_TRUE = {"1", "true", "yes", "y", "t"}


@dataclass
class ColumnSpec:
    name: str
    type: str | None = None
    min: float | None = None
    max: float | None = None
    allowed: list[Any] | None = None
    target: bool = False
    derived: bool = False
    derived_from: list[str] = field(default_factory=list)

    def range_rule(self) -> dict[str, Any] | None:
        rule: dict[str, Any] = {}
        if self.min is not None:
            rule["min"] = self.min
        if self.max is not None:
            rule["max"] = self.max
        if self.allowed is not None:
            rule["allowed"] = self.allowed
        return rule or None


@dataclass
class DataDictionary:
    columns: dict[str, ColumnSpec] = field(default_factory=dict)

    @classmethod
    def load(cls, source: str | Path | dict[str, Any]) -> DataDictionary:
        """Load a dictionary from a mapping, JSON file or CSV file."""
        if isinstance(source, dict):
            return cls.from_mapping(source)
        path = Path(source)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path)
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> DataDictionary:
        columns: dict[str, ColumnSpec] = {}
        for name, spec in mapping.items():
            if isinstance(spec, str):
                spec = {"type": spec}
            spec = dict(spec or {})
            columns[name] = ColumnSpec(
                name=name,
                type=spec.get("type"),
                min=spec.get("min"),
                max=spec.get("max"),
                allowed=spec.get("allowed"),
                target=bool(spec.get("target", False)),
                derived=bool(spec.get("derived", False)),
                derived_from=list(spec.get("derived_from", [])),
            )
        return cls(columns=columns)

    @classmethod
    def from_csv(cls, path: str | Path) -> DataDictionary:
        columns: dict[str, ColumnSpec] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row.get("name") or row.get("column")
                if not name:
                    continue

                def _num(value: str | None) -> float | None:
                    value = (value or "").strip()
                    return float(value) if value else None

                allowed = (row.get("allowed") or "").strip()
                derived_from = (row.get("derived_from") or "").strip()
                columns[name] = ColumnSpec(
                    name=name,
                    type=(row.get("type") or "").strip() or None,
                    min=_num(row.get("min")),
                    max=_num(row.get("max")),
                    allowed=[v.strip() for v in allowed.split("|") if v.strip()] or None,
                    target=(row.get("target") or "").strip().lower() in _BOOL_TRUE,
                    derived=(row.get("derived") or "").strip().lower() in _BOOL_TRUE,
                    derived_from=[v.strip() for v in derived_from.split("|") if v.strip()],
                )
        return cls(columns=columns)

    # -- accessors used by EDAReport -----------------------------------------
    def type_overrides(self) -> dict[str, str]:
        return {name: spec.type for name, spec in self.columns.items() if spec.type}

    def range_rules(self) -> dict[str, dict[str, Any]]:
        return {name: rule for name, spec in self.columns.items() if (rule := spec.range_rule()) is not None}

    def derived_columns(self) -> list[str]:
        return [name for name, spec in self.columns.items() if spec.derived]

    def target_columns(self) -> list[str]:
        return [name for name, spec in self.columns.items() if spec.target]

    def derived_from(self, source: str) -> list[str]:
        return [name for name, spec in self.columns.items() if source in spec.derived_from]
