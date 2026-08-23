"""Tests for missing-value handling (sedat.imputation)."""

import numpy as np
import pandas as pd
import pytest

import sedat
from sedat.imputation import (
    DROP_COLUMN_STRATEGY,
    MEDIAN_STRATEGY,
    MODE_STRATEGY,
    ImputationPlan,
    ImputationSuggestion,
    suggest_imputations,
)


def make_df():
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0],
            "cat": ["a", None, "a", "a", "b", "a", "a", "a"],
            "flag": ["yes", "no", None, "yes", "no", "yes", "yes", "no"],
            "gone": [None, None, None, None, "x", "y", None, None],  # 75% missing
            "full": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )


def test_one_suggestion_per_incomplete_column():
    plan = suggest_imputations(make_df())
    assert [s.column for s in plan.suggestions] == ["num", "cat", "flag", "gone"]
    assert plan.get("full") is None


def test_strategy_selection():
    plan = suggest_imputations(make_df())
    by_col = {s.column: s for s in plan.suggestions}
    assert by_col["num"].strategy == MEDIAN_STRATEGY
    assert by_col["cat"].strategy == MODE_STRATEGY
    assert by_col["flag"].strategy == MODE_STRATEGY
    assert by_col["gone"].strategy == DROP_COLUMN_STRATEGY


def test_median_fill_value_and_rationale_mentions_mean():
    plan = suggest_imputations(make_df())
    num = plan.get("num")
    assert num.fill_value == 4.5  # median of [1, 2, 4, 5, 7, 8]
    assert "mean" in num.rationale.lower()


def test_mode_fill_values_preserve_dtype():
    plan = suggest_imputations(make_df())
    assert plan.get("cat").fill_value == "a"
    assert plan.get("flag").fill_value == "yes"


def test_drop_column_above_threshold_only():
    plan = suggest_imputations(make_df(), missing_threshold=80.0)
    strategies = {s.column: s.strategy for s in plan.suggestions}
    # gone (75%) is below an 80% threshold now -> mode instead of drop
    assert strategies["gone"] == MODE_STRATEGY
    assert DROP_COLUMN_STRATEGY not in strategies.values()

    plan2 = suggest_imputations(make_df(), missing_threshold=30.0)
    strategies2 = {s.column: s.strategy for s in plan2.suggestions}
    # cat/flag/num (25%) stay under a 30% threshold; gone still dropped
    assert strategies2["gone"] == DROP_COLUMN_STRATEGY
    assert strategies2["num"] == MEDIAN_STRATEGY


def test_apply_single_column_leaves_others_untouched():
    df = make_df()
    plan = suggest_imputations(df)
    out = plan.apply(df, column="num")
    assert out["num"].isna().sum() == 0
    assert out["cat"].isna().sum() == 1  # untouched


def test_apply_unknown_column_raises():
    plan = suggest_imputations(make_df())
    with pytest.raises(KeyError):
        plan.apply(make_df(), column="nope")


def test_apply_all_removes_all_missing_and_drops_column():
    df = make_df()
    plan = suggest_imputations(df)
    out = plan.apply_all(df)
    assert "gone" not in out.columns
    assert not out.isna().any().any()
    # original frame never mutated
    assert df.isna().any().any()


def test_apply_is_alias_of_apply_all():
    df = make_df()
    plan = suggest_imputations(df)
    pd.testing.assert_frame_equal(plan.apply(df), plan.apply_all(df))


def test_summary_shape():
    plan = suggest_imputations(make_df())
    summary = plan.summary
    assert list(summary.columns) == [
        "column",
        "current_type",
        "strategy",
        "n_missing",
        "missing_pct",
        "fill_value",
        "rationale",
    ]
    assert set(summary["column"]) == {"num", "cat", "flag", "gone"}


def test_empty_plan_when_no_missing():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    plan = suggest_imputations(df)
    assert plan.suggestions == []
    assert plan.summary.empty
    pd.testing.assert_frame_equal(plan.apply_all(df), df)


def test_shortcut_function_matches_plan():
    df = make_df()
    plan = suggest_imputations(df)
    pd.testing.assert_frame_equal(
        sedat.apply_imputations(df), plan.apply_all(df)
    )


def test_report_wiring():
    report = sedat.EDAReport.create(make_df(), target=None)
    assert isinstance(report.imputations, ImputationPlan)
    assert set(report.imputation_summary["column"]) == {"num", "cat", "flag", "gone"}
    out = report.apply_all_imputations()
    assert "gone" not in out.columns
    assert not out.isna().any().any()


def test_suggestion_apply_directly():
    df = make_df()
    suggestion = ImputationSuggestion(
        column="num",
        current_type="numeric",
        strategy=MEDIAN_STRATEGY,
        n_missing=1,
        missing_pct=25.0,
        fill_value=2.0,
    )
    out = suggestion.apply(df)
    assert out["num"].tolist()[:3] == [1.0, 2.0, 2.0]


def test_suggestion_for_missing_column_is_noop():
    ghost = ImputationSuggestion(
        column="ghost",
        current_type="numeric",
        strategy=MEDIAN_STRATEGY,
        n_missing=1,
        missing_pct=25.0,
        fill_value=0.0,
    )
    df = make_df()
    pd.testing.assert_frame_equal(ghost.apply(df), df)


def test_datetime_column_gets_mode():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2021-01-01", None, "2021-03-05", "2021-03-05"])
        }
    )
    plan = suggest_imputations(df)
    when = plan.get("when")
    assert when.strategy == MODE_STRATEGY
    assert when.fill_value == pd.Timestamp("2021-03-05")
