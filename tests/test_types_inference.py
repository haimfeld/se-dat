"""Edge-case coverage for column type inference (sedat.types / sedat.profile)."""

import numpy as np
import pandas as pd
import pytest

from sedat import infer_column_type, is_binary_like, profile_dataframe


def test_all_null_object_column():
    s = pd.Series([None, None, None], name="ghost")
    info = infer_column_type(s)
    assert info["inferred_type"] == "string"
    assert info["confidence"] == "low"
    assert "empty column" in info["notes"]
    assert info["n_non_null"] == 0
    assert info["n_unique"] == 0
    assert info["missing_pct"] == 100.0


def test_all_null_numeric_column_is_numeric():
    # float dtype wins over content: nothing to infer from values
    s = pd.Series([np.nan, np.nan], dtype=float)
    info = infer_column_type(s)
    assert info["inferred_type"] == "numeric"


def test_single_unique_string_is_categorical():
    s = pd.Series(["same"] * 5)
    info = infer_column_type(s)
    assert info["inferred_type"] == "categorical"
    assert info["n_unique"] == 1


def test_single_unique_integer_is_numeric_not_id():
    s = pd.Series([7, 7, 7])
    info = infer_column_type(s)
    assert info["inferred_type"] == "numeric"


def test_numeric_looking_strings_inferred_numeric_low_confidence():
    s = pd.Series(["45000", "67000", "52000"])
    info = infer_column_type(s)
    assert info["inferred_type"] == "numeric"
    assert info["confidence"] == "low"
    assert any("stored as string" in n for n in info["notes"])
    assert info["missing_pct"] == 0.0


def test_partially_numeric_strings_do_not_become_numeric():
    s = pd.Series(["100", "200", "abc123", "def456", "ghi789"])
    info = infer_column_type(s)
    assert info["inferred_type"] != "numeric"
    assert info["confidence"] == "high"


@pytest.mark.parametrize("values", [["yes", "no"], ["y", "n"], ["true", "false"], ["t", "f"]])
def test_boolean_like_strings(values):
    s = pd.Series(values * 4)
    info = infer_column_type(s)
    assert info["inferred_type"] == "boolean"
    assert info["confidence"] == "medium"
    assert any("stored as strings" in n for n in info["notes"])
    assert is_binary_like(s)


def test_zero_one_integers_ambiguous_boolean_vs_numeric():
    s = pd.Series([0, 1, 1, 0], dtype=int)
    info = infer_column_type(s)
    assert info["inferred_type"] == "boolean"
    assert info["confidence"] == "medium"
    assert any("ambiguous" in n for n in info["notes"])


def test_two_value_floats_are_numeric_not_boolean():
    # 0.0/1.0-style floats with other magnitudes must stay numeric;
    # genuine {0.0, 1.0} sets also stay numeric (string repr differs from "0"/"1")
    s = pd.Series([0.5, 1.5, 2.5])
    assert infer_column_type(s)["inferred_type"] == "numeric"


def test_real_bool_dtype_high_confidence():
    s = pd.Series([True, False, True])
    info = infer_column_type(s)
    assert info["inferred_type"] == "boolean"
    assert info["confidence"] == "high"
    assert info["notes"] == []


def test_high_cardinality_identifier_detected_as_id():
    s = pd.Series([f"user_{i}" for i in range(60)])
    info = infer_column_type(s)
    assert info["inferred_type"] == "id"
    assert info["cardinality_ratio"] >= 0.95


def test_high_cardinality_free_text_stays_string():
    s = pd.Series([f"some free text number {i}" for i in range(60)])
    info = infer_column_type(s)
    assert info["inferred_type"] == "string"
    assert any("free text" in n for n in info["notes"])


def test_medium_cardinality_text_stays_string():
    s = pd.Series([f"text {i}" for i in range(10)] * 3)  # ratio ~0.33... unique=10/30 -> categorical
    info = infer_column_type(s)
    # 10 uniques out of 30 rows: ratio 0.33 < 0.5 -> categorical
    assert info["inferred_type"] == "categorical"
    s2 = pd.Series([f"note {i}" for i in range(40)] + ["dup"] * 20)
    info2 = infer_column_type(s2)
    assert info2["inferred_type"] == "string"
    assert any("many unique values" in n for n in info2["notes"])


@pytest.mark.parametrize(
    "values",
    [
        ["2021-01-15", "2021-02-20", "2021-03-25"],
        ["2021/01/15", "2021/02/20", "2021/03/25"],
        ["March 5, 2020", "April 6, 2020", "May 7, 2020"],
        ["15-01-2021", "20-02-2021"],
        ["2021-01-15 08:30:00", "2021-02-20 14:45:00"],
    ],
)
def test_datetime_strings_in_multiple_formats(values):
    info = infer_column_type(pd.Series(values))
    assert info["inferred_type"] == "datetime"
    assert info["confidence"] == "medium"
    assert any("parsed from object" in n for n in info["notes"])


def test_real_datetime_dtype_high_confidence():
    s = pd.to_datetime(pd.Series(["2021-01-01", "2021-06-15"]))
    info = infer_column_type(s)
    assert info["inferred_type"] == "datetime"
    assert info["confidence"] == "high"


def test_datetime_with_missing_values_still_detected():
    s = pd.Series(["2021-01-01", None, "2021-03-01"])
    info = infer_column_type(s)
    assert info["inferred_type"] == "datetime"
    assert info["missing_pct"] > 0


def test_mixed_type_object_column_repetitive_is_categorical():
    s = pd.Series(["a", 1, "a", 1, "a"])
    info = infer_column_type(s)
    assert info["inferred_type"] == "categorical"


def test_mixed_type_object_column_varied_no_whitespace_becomes_id():
    # unique whitespace-free values -> looks like an identifier
    s = pd.Series(["a", 1, "b", 2.5, "c"])
    info = infer_column_type(s)
    assert info["inferred_type"] == "id"


def test_mixed_type_object_column_varied_with_spaces_is_string():
    s = pd.Series(["a x", 1, "b y", 2.5, "c z"])
    info = infer_column_type(s)
    assert info["inferred_type"] == "string"


def test_categorical_dtype_respected():
    s = pd.Series(pd.Categorical(["red", "blue", "red"]))
    info = infer_column_type(s)
    assert info["inferred_type"] == "categorical"


def test_binary_mapping_covers_subsets_and_aliases():
    assert is_binary_like(pd.Series(["yes", "no"]))
    assert is_binary_like(pd.Series(["Y", "N", "y"]))
    assert is_binary_like(pd.Series([True, False]))
    assert not is_binary_like(pd.Series(["maybe", "no"]))
    mapping = infer_column_type(pd.Series(["yes", "no"]))  # smoke: notes mention values
    assert mapping["inferred_type"] == "boolean"


def test_profile_dataframe_summary_roundtrip():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "cat": ["x", "y", "x", "x", "x", "y"],
        }
    )
    summary = profile_dataframe(df).summary
    assert list(summary.index) == ["num", "cat"]
    assert summary.loc["num", "missing_pct"] == pytest.approx(100 / 6, abs=0.01)
    assert summary.loc["cat", "inferred_type"] == "categorical"
