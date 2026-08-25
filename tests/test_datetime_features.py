"""Tests for datetime feature extraction (sedat.datetime_features)."""

import pandas as pd
import pytest

import sedat
from sedat.datetime_features import (
    ALL_FEATURES,
    BASE_FEATURES,
    DatetimeFeaturePlan,
    DatetimeFeatureSuggestion,
    suggest_datetime_features,
)


def test_date_only_column_gets_five_features():
    df = pd.DataFrame({"when": ["2024-01-06", "2024-01-08", "2024-03-15", None]})
    plan = suggest_datetime_features(df)
    assert plan.get("when").features == list(BASE_FEATURES)
    assert not plan.get("when").has_time_component

    out = plan.apply_all(df)
    for suffix in ("_year", "_month", "_day", "_dow", "_is_weekend"):
        assert f"when{suffix}" in out.columns
    assert "when_hour" not in out.columns
    # original column preserved
    assert "when" in out.columns


def test_weekday_and_weekend_values():
    # 2024-01-06 is a Saturday, 2024-01-08 the Monday after
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-06", "2024-01-08"])})
    out = suggest_datetime_features(df).apply_all(df)
    assert out["d_dow"].tolist() == [5, 0]
    assert out["d_is_weekend"].tolist() == [1, 0]


def test_time_component_adds_hour():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 08:30:00", "2024-06-15 23:05:00"])})
    plan = suggest_datetime_features(df)
    assert plan.get("ts").has_time_component
    assert plan.get("ts").features == list(ALL_FEATURES)
    out = plan.apply_all(df)
    assert out["ts_hour"].tolist() == [8, 23]


def test_all_midnight_times_do_not_suggest_hour():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 00:00:00"] * 3)})
    plan = suggest_datetime_features(df)
    assert not plan.get("ts").has_time_component


def test_missing_timestamps_stay_missing():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-06", None])})
    out = suggest_datetime_features(df).apply_all(df)
    assert out["d_year"].iloc[0] == 2024
    assert pd.isna(out["d_year"].iloc[1])
    assert pd.isna(out["d_is_weekend"].iloc[1])
    # nullable integer dtype, not float sentinel
    assert str(out["d_month"].dtype) == "Int64"


def test_string_dates_are_detected_and_extracted():
    df = pd.DataFrame({"when": ["March 5, 2020", "April 6, 2020"]})
    plan = suggest_datetime_features(df)
    assert plan.get("when") is not None
    out = plan.apply_all(df)
    assert out["when_month"].tolist() == [3, 4]


def test_non_datetime_columns_are_ignored():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "text": ["a", "b"],
            "flag": ["yes", "no"],
        }
    )
    plan = suggest_datetime_features(df)
    assert plan.suggestions == []
    pd.testing.assert_frame_equal(plan.apply_all(df), df)


def test_apply_single_column_and_unknown_raises():
    df = pd.DataFrame(
        {
            "a": pd.to_datetime(["2024-01-01"]),
            "b": pd.to_datetime(["2024-02-01"]),
        }
    )
    plan = suggest_datetime_features(df)
    out = plan.apply(df, column="a")
    assert "a_year" in out.columns and "b_year" not in out.columns
    with pytest.raises(KeyError):
        plan.apply(df, column="c")


def test_summary_shape():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01 10:00:00"])})
    summary = suggest_datetime_features(df).summary
    assert list(summary.columns) == [
        "column",
        "n_features",
        "features",
        "has_time_component",
        "new_columns",
    ]
    row = summary.iloc[0]
    assert row["column"] == "d"
    assert row["n_features"] == 6
    assert row["has_time_component"]


def test_suggestion_for_missing_column_is_noop():
    s = DatetimeFeatureSuggestion(column="ghost")
    df = pd.DataFrame({"x": [1]})
    pd.testing.assert_frame_equal(s.apply(df), df)


def test_exported_from_package():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])})
    plan = sedat.suggest_datetime_features(df)
    assert isinstance(plan, DatetimeFeaturePlan)
    out = sedat.apply_datetime_features(df)
    assert "d_year" in out.columns
