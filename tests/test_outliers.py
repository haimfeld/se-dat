"""Tests for outlier detection (sedat.outliers)."""

import numpy as np
import pandas as pd
import pytest

import sedat
from sedat.outliers import (
    IQR_METHOD,
    ZSCORE_METHOD,
    ColumnOutliers,
    OutlierReport,
    detect_outliers,
    outlier_note,
)


def make_df():
    return pd.DataFrame(
        {
            "clean": list(range(1, 21)),
            "spiky": [float(x) for x in range(1, 20)] + [1000.0],
            "with_nan": [1.0, 2.0, 3.0, np.nan, 5.0, np.nan, 6.0, 7.0]
            + [1.5, 2.5] * 5 + [9.0, 10.0],
            "constant": [7.0] * 20,
            "text": ["a"] * 20,
        }
    )


def test_iqr_flags_planted_extreme_value():
    report = detect_outliers(make_df(), method="iqr")
    spiky = report.get("spiky")
    assert spiky.count == 1
    assert spiky.indices == [19]
    assert spiky.pct == pytest.approx(100 / 20, abs=0.01)
    assert spiky.upper is not None and 1000.0 > spiky.upper


def test_zscore_flags_planted_extreme_value():
    report = detect_outliers(make_df(), method="zscore")
    spiky = report.get("spiky")
    assert spiky.count == 1


def test_clean_and_constant_columns_have_no_outliers():
    for method in (IQR_METHOD, ZSCORE_METHOD):
        report = detect_outliers(make_df(), method=method)
        assert report.get("clean").count == 0
        assert report.get("constant").count == 0
        # non-numeric columns are skipped entirely
        assert report.get("text") is None


def test_nan_values_never_flagged_or_counted():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    report = detect_outliers(df)
    result = report.get("x")
    assert result.count == 0
    assert result.indices == []
    planted = pd.DataFrame({"x": [1.0] * 9 + [np.nan, 500.0]})
    result2 = detect_outliers(planted).get("x")
    assert result2.count == 1
    assert result2.pct == pytest.approx(10.0)  # denominator excludes NaN


def test_indices_are_dataframe_labels_not_positions():
    df = pd.DataFrame(
        {"x": [1.0] * 9 + [999.0]}, index=[f"row{i}" for i in range(10)]
    )
    report = detect_outliers(df)
    assert report.indices("x") == ["row9"]


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        detect_outliers(make_df(), method="mad")


def test_custom_thresholds_change_results():
    base = pd.DataFrame({"x": list(range(100)) + [10_000]})
    tight = detect_outliers(base, iqr_scale=0.5)
    wide = detect_outliers(base, iqr_scale=3.0)
    assert tight.get("x").count >= wide.get("x").count

    zloose = detect_outliers(base, method=ZSCORE_METHOD, z_threshold=50.0)
    assert zloose.get("x").count == 0
    ztight = detect_outliers(base, method=ZSCORE_METHOD, z_threshold=1.0)
    assert ztight.get("x").count >= 1


def test_summary_shape_and_note_text():
    report = detect_outliers(make_df())
    summary = report.summary
    assert set(summary.columns) == {
        "column",
        "method",
        "outlier_count",
        "outlier_pct",
        "lower_bound",
        "upper_bound",
        "note",
    }
    row = summary[summary["column"] == "spiky"].iloc[0]
    assert row["outlier_count"] == 1
    assert row["note"] == "spiky: 1 potential outliers (5.0%)"


def test_outlier_note_zero_is_empty():
    empty = ColumnOutliers(column="x", method=IQR_METHOD, count=0)
    assert outlier_note(empty) == ""
    some = ColumnOutliers(column="age", method=IQR_METHOD, count=12, pct=2.4)
    assert outlier_note(some) == "age: 12 potential outliers (2.4%)"


def test_empty_report_repr_and_summary():
    report = OutlierReport(method=IQR_METHOD)
    assert report.summary.empty
    assert repr(report) == "Empty DataFrame\nColumns: []\nIndex: []"


def test_report_wiring_into_profile_summary():
    report = sedat.EDAReport.create(make_df())
    summary = report.profile.summary
    assert "outliers" in summary.columns
    note = summary.loc["spiky", "outliers"]
    assert note.startswith("spiky:") and "potential outliers" in note
    assert summary.loc["clean", "outliers"] == ""


def test_standalone_profile_has_no_outlier_notes():
    profile = sedat.profile_dataframe(make_df())
    assert profile.outlier_notes == {}
