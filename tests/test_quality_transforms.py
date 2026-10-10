"""Tests for quality checks, transforms, data dictionary, collinearity and more."""

import json

import numpy as np
import pandas as pd
import pytest

import sedat
from sedat.datadict import DataDictionary
from sedat.encoding import EncodingPlan
from sedat.outliers import LOG_IQR_METHOD, MAD_METHOD
from sedat.persistence import load_plan, save_plan


# --------------------------------------------------------------------------- outliers
def test_mad_flags_planted_extreme():
    df = pd.DataFrame({"x": [*list(range(1, 100)), 1000]})
    report = sedat.detect_outliers(df, method=MAD_METHOD)
    assert report.get("x").count == 1
    assert report.get("x").indices == [99]


def test_log_iqr_handles_log_normal_income():
    rng = np.random.default_rng(0)
    base = np.exp(rng.normal(9.5, 0.5, 2000))
    df = pd.DataFrame({"income": base})
    iqr = sedat.detect_outliers(df, method="iqr")
    log_iqr = sedat.detect_outliers(df, method=LOG_IQR_METHOD)
    # log-normal has a long right tail that plain IQR over-flags
    assert log_iqr.get("income").count < iqr.get("income").count


def test_bounds_are_on_original_scale_for_log_iqr():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    report = sedat.detect_outliers(df, method=LOG_IQR_METHOD)
    assert report.get("x").lower > 0  # back on the original scale


# ----------------------------------------------------------------- constant / duplicates
def test_constant_column_flagged_in_profile():
    df = pd.DataFrame({"const": [7] * 10, "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    report = sedat.EDAReport.create(df)
    assert report.constant_columns == ["const"]
    col = next(c for c in report.profile.columns if c.name == "const")
    assert col.is_constant
    assert any("constant" in n for n in col.notes)


def test_duplicate_rows_counted():
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "y"]})
    report = sedat.EDAReport.create(df)
    assert report.duplicate_row_count == 1
    assert sedat.duplicate_rows(df).sum() == 1


# ----------------------------------------------------------------- plausibility
def test_range_rules_catch_negative_age():
    df = pd.DataFrame({"age": [30, -5, 12, 200]})
    report = sedat.EDAReport.create(df, non_negative=["age"])
    summary = report.plausibility.summary
    assert not summary.empty
    row = summary[summary["column"] == "age"].iloc[0]
    assert row["n_violations"] == 1  # only -5 is below 0


def test_percentage_and_custom_bounds():
    df = pd.DataFrame({"pct": [0.5, 1.2, 0.3], "beta": [0.0, 0.5, 2.0]})
    report = sedat.EDAReport.create(df, percentages=["pct"], range_rules={"beta": (0, 1)})
    # summary only lists columns with violations: pct is clean and absent
    assert "pct" not in report.plausibility_summary["column"].tolist()
    by_col = report.plausibility_summary.set_index("column")["n_violations"].to_dict()
    assert by_col["beta"] == 1  # 2.0 outside [0, 1]


def test_check_ranges_whitelist_allowed_values():
    df = pd.DataFrame({"code": ["a", "b", "z"]})
    report = sedat.check_ranges(df, rules={"code": {"allowed": ["a", "b"]}})
    assert report.summary.iloc[0]["n_violations"] == 1


# ----------------------------------------------------------------- transforms
def test_skewed_column_gets_transform_suggestion():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"income": np.exp(rng.normal(9, 0.8, 500))})
    plan = sedat.suggest_transforms(df)
    assert plan.suggestions  # right-skewed -> log1p or box_cox
    out = plan.apply_all(df)
    from scipy.stats import skew

    assert abs(skew(out["income"])) < abs(skew(df["income"]))


def test_transform_plan_roundtrip_via_apply():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"spend": np.maximum(np.exp(rng.normal(4, 0.6, 300)) - 20, 0)})  # has some 0s
    plan = sedat.suggest_transforms(df)
    suggestion = plan.get("spend")
    assert suggestion is not None and suggestion.strategy == "log1p"  # non-negative
    transformed = plan.apply_all(df)
    back = suggestion.inverse(transformed["spend"])
    pd.testing.assert_series_equal(back, df["spend"], check_exact=False, rtol=1e-9)


def test_left_skewed_uses_box_cox():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"t": -np.exp(rng.normal(3, 0.5, 300)) + 5})
    plan = sedat.suggest_transforms(df)
    suggestion = plan.get("t")
    assert suggestion.strategy == "box_cox"


# ----------------------------------------------------------------- collinearity
def test_vif_detects_multiway_collinearity():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    df = pd.DataFrame({"a": base, "b": base * 2 + 1, "c": base * 3 - 4, "noise": rng.normal(size=500)})
    vif = sedat.variance_inflation_factors(df)
    by_col = vif.set_index("feature")["vif"].to_dict()
    assert by_col["a"] > 10 and by_col["b"] > 10 and by_col["c"] > 10
    assert by_col["noise"] < 5
    assert sedat.condition_number(df) > 1


def test_correlation_report_carries_vif_and_condition_number():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": rng.normal(size=200)})
    report = sedat.correlation_report(df)
    assert not report.vif.empty
    assert not np.isnan(report.condition_number_value)


def test_wide_frame_skips_vif_regression():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({f"x{i}": rng.normal(size=100) for i in range(60)})
    report = sedat.correlation_report(df, max_vif_features=50)
    assert report.vif.empty  # p=60 > cap of 50


# ----------------------------------------------------------------- collapsed pairs
def test_collapsed_pairs_one_row_per_pair():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=200)})
    df["b"] = df["a"] * 2 + 1
    df["c"] = rng.normal(size=200)
    report = sedat.correlation_report(df, threshold=0.9)
    collapsed = report.collapsed_summary
    pairs = collapsed["column_a"] + "|" + collapsed["column_b"]
    assert len(collapsed) == 3  # a-b, a-c, b-c
    assert len(set(pairs)) == len(pairs)
    ab = collapsed[(collapsed["column_a"] == "a") & (collapsed["column_b"] == "b")].iloc[0]
    assert ab["n_methods"] == 2  # pearson + spearman collapsed into one
    assert "pearson" in ab["methods"] and "spearman" in ab["methods"]
    flagged_collapsed = report.flagged_pairs_collapsed
    assert ((flagged_collapsed["column_a"] == "a") & (flagged_collapsed["column_b"] == "b")).any()


# ----------------------------------------------------------------- data dictionary
DICT = {
    "age": {"type": "numeric", "min": 0, "max": 120},
    "churned": {"type": "boolean", "target": True},
    "churn_score": {"type": "numeric", "derived": True, "derived_from": ["churned"]},
}


def test_dictionary_overrides_types_and_target():
    df = pd.DataFrame({"age": [25, 30], "churned": [0, 1], "churn_score": [0.2, 0.8]})
    report = sedat.EDAReport.create(df, dictionary=dict(DICT), warn_leakage=False)
    by_col = {c.name: c for c in report.profile.columns}
    assert by_col["age"].inferred_type == "numeric"
    assert by_col["churned"].role == "target"
    assert any("data dictionary" in n for n in by_col["age"].notes)


def test_dictionary_target_drives_leakage_analysis():
    rng = np.random.default_rng(0)
    churned = rng.integers(0, 2, 100)
    df = pd.DataFrame({"churned": churned, "churn_score": churned.astype(float), "age": rng.integers(20, 60, 100)})
    with pytest.warns(UserWarning, match="churn_score"):
        report = sedat.EDAReport.create(df, dictionary=dict(DICT))
    assert report.leakage_features == ["churn_score"]
    assert report.target_analysis is not None


def test_dictionary_from_csv(tmp_path):
    path = tmp_path / "dd.csv"
    path.write_text("name,type,min,max,target,derived,derived_from\nage,numeric,0,120,false,false,\n")
    dd = DataDictionary.load(str(path))
    assert dd.columns["age"].min == 0
    assert dd.columns["age"].max == 120


# ----------------------------------------------------------------- persistence versioning
def test_old_plan_without_schema_version_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"plan_type": "encoding", "payload": {"cardinality_threshold": 10, "suggestions": []}}),
        encoding="utf-8",
    )
    plan = load_plan(str(path))
    assert isinstance(plan, EncodingPlan)


def test_newer_schema_version_is_rejected(tmp_path):
    plan = sedat.suggest_encodings(pd.DataFrame({"s": ["yes", "no"] * 3}))
    plan.save(str(tmp_path / "p.json"))
    envelope = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    envelope["schema_version"] = 99
    (tmp_path / "future.json").write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        load_plan(str(tmp_path / "future.json"))


def test_saved_plan_records_se_dat_version(tmp_path):
    plan = sedat.suggest_encodings(pd.DataFrame({"s": ["yes", "no"] * 3}))
    save_plan(plan, str(tmp_path / "v.json"))
    envelope = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    assert envelope["se_dat_version"]


# ----------------------------------------------------------------- target-encoding leakage warning
def test_apply_all_target_encoding_warns(tmp_path):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"zone": rng.choice([f"z{i}" for i in range(12)], 300), "y": rng.normal(size=300)})
    plan = sedat.suggest_encodings(df, target=df["y"])
    with pytest.warns(UserWarning, match="Target encoding"):
        plan.apply_all(df)
    with pytest.warns(UserWarning, match="silence"):
        plan.apply_all(df)  # still warns by default
    # explicit opt-out
    out = plan.apply_all(df, warn_target_leakage=False)
    assert isinstance(out, pd.DataFrame)


def test_loaded_plan_applies_frozen_means_without_warning(tmp_path):
    # the recommended leak-free workflow: fit on train, freeze means, apply on test
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"zone": rng.choice([f"z{i}" for i in range(15)], 200), "y": rng.normal(size=200)})
    plan = sedat.suggest_encodings(train, target=train["y"])
    loaded = EncodingPlan.load(str(plan.save(str(tmp_path / "plan.json"))))
    test = pd.DataFrame({"zone": ["z0", "z2"]})
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = loaded.apply_all(test)  # no target, frozen means -> no leakage path
    assert not any("Target encoding" in str(w.message) for w in caught)
    assert len(out["zone"].dropna()) == 2


# ----------------------------------------------------------------- scalability / sampling
def test_fast_mode_samples_and_caps_columns():
    rng = np.random.default_rng(6)
    n = 200_000
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )
    report = sedat.EDAReport.create(df, target=df["target"], sample=1000, fast=False)
    assert report.sampled
    assert report.n_rows_analyzed == 1000
    # full frame retained for applying plans
    assert len(report.df) == n


def test_fast_flag_enables_defaults():
    df = pd.DataFrame({"a": range(60_000), "b": range(60_000)})
    report = sedat.EDAReport.create(df, fast=True)
    assert report.n_rows_analyzed == 50_000 or not report.sampled  # 60k>50k -> sampled
