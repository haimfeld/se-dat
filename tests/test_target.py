"""Tests for target-aware analysis and leakage detection (sedat.target)."""

import numpy as np
import pandas as pd
import pytest

import sedat
from sedat.target import (
    TARGET_ROLE_BINARY,
    TARGET_ROLE_CATEGORICAL,
    TARGET_ROLE_NUMERIC,
    analyze_target,
    infer_target_role,
    mutual_information,
)


def test_infer_target_role_classifies_boolean_numeric_categorical():
    assert infer_target_role(pd.Series([0, 1, 1, 0], dtype=int)) == TARGET_ROLE_BINARY
    assert infer_target_role(pd.Series([True, False, True])) == TARGET_ROLE_BINARY
    assert infer_target_role(pd.Series(["yes", "no", "yes"])) == TARGET_ROLE_BINARY
    assert infer_target_role(pd.Series([1.5, 2.5, 3.5])) == TARGET_ROLE_NUMERIC
    assert infer_target_role(pd.Series(["a", "b", "c", "a"])) == TARGET_ROLE_CATEGORICAL


def test_target_column_gets_own_role_in_profile():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 200),
            "churned": rng.integers(0, 2, 200).astype(int),  # 0/1 -> ambiguous
        }
    )
    report = sedat.EDAReport.create(df, target=df["churned"])
    col = next(c for c in report.profile.columns if c.name == "churned")
    assert col.role == "target"
    assert col.confidence == "high"
    assert any("target" in n for n in col.notes)


def test_mutual_information_independent_is_low_dependent_is_high():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    independent = mutual_information(x, rng.normal(size=2000))
    assert independent[0] < 0.1

    y = (x > 0).astype(int)  # perfect binary relation
    dependent = mutual_information(x, y)
    assert dependent[0] > 0.3
    assert dependent[1] > 0.8


def test_analyze_target_excludes_target_and_ranks_by_effect_size():
    rng = np.random.default_rng(2)
    # target is the parent series; `leak` is a perfect copy (r=1), `other` is noise
    noise = rng.normal(size=500)
    df = pd.DataFrame(
        {"other": rng.normal(size=500), "leak": noise.copy(), "region": rng.choice(["n", "s", "e", "w"], 500)}
    )
    target = noise.copy()
    analysis = analyze_target(df, target, target_name="age", warn=False)
    ranked = analysis.summary
    assert ranked.iloc[0]["feature"] == "leak"
    assert ranked.iloc[0]["effect_size"] == pytest.approx(1.0, abs=1e-6)
    assert ranked.iloc[-1]["feature"] == "other"


def test_perfect_predictor_flagged_as_leakage():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"leak": rng.normal(size=300)})
    target = pd.Series(df["leak"].to_numpy(), name="y")
    with pytest.warns(UserWarning, match="leak"):
        analysis = analyze_target(df, target, warn=True)
    assert analysis.leakage_features == ["leak"]


def test_high_cardinality_text_is_not_spurious_leakage():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "free_text": [f"user {i}" for i in range(300)],
            "other": rng.normal(size=300),
        }
    )
    target = rng.normal(size=300)
    analysis = analyze_target(df, target, warn=False)
    assert analysis.leakage_features == []


def test_binary_target_effect_size_is_point_biserial():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"score": rng.normal(size=500)})
    target = (df["score"].rank() > 250).astype(int)
    target.name = "y"  # must not collide with the feature name
    analysis = analyze_target(df, target, warn=False)
    row = analysis.associations.iloc[0]
    assert row["method"] == "point_biserial"
    assert row["effect_size"] > 0.7


def test_leakage_threshold_is_configurable():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"x": rng.normal(size=400)})
    target = pd.Series(df["x"].to_numpy(), name="y")
    analysis = analyze_target(df, target, leakage_threshold=0.999, warn=False)
    # correlation is exactly 1.0, so it still exceeds the strict threshold too
    assert analysis.leakage_features == ["x"]


def test_report_surfaces_target_summary_and_leakage():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    target = pd.Series(df["a"] * 3, name="y")
    report = sedat.EDAReport.create(df, target=target)
    assert not report.target_summary.empty
    assert "a" in report.leakage_features
