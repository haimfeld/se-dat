import numpy as np
import pandas as pd
import pytest

import sedat


def make_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "age": rng.integers(18, 80, n),
            "height": rng.normal(170, 10, n).round(2),
            "income": [str(x) for x in rng.integers(30000, 120000, n)],
            "gender": rng.choice(["M", "F"], n),
            "smoker": rng.choice(["yes", "no"], n),
            "active": rng.choice([True, False], n),
            "flag01": rng.integers(0, 2, n),
            "score_cat": rng.choice(["low", "med", "high"], n),
            "free_text": [f"user comment number {i}" for i in range(n)],
            "joined": pd.to_datetime("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 365 * 5, n), unit="D"),
            "large_card": rng.choice([f"sku_{i}" for i in range(50)], n),
            "target": rng.normal(0, 1, n),
        }
    )
    df.loc[rng.choice(n, 25, replace=False), "age"] = np.nan
    return df


def type_map(profile):
    return {c.name: c.inferred_type for c in profile.columns}


def test_profile_infers_types():
    df = make_df()
    profile = sedat.profile_dataframe(df)
    types = type_map(profile)
    assert types["id"] == "id"
    assert types["age"] == "numeric"
    assert types["height"] == "numeric"
    assert types["income"] == "numeric"
    assert types["gender"] == "categorical"
    assert types["smoker"] == "boolean"
    assert types["active"] == "boolean"
    assert types["flag01"] == "boolean"
    assert types["free_text"] == "string"
    assert types["joined"] == "datetime"
    assert types["target"] == "numeric"


def test_profile_confidence_flags():
    df = make_df()
    profile = sedat.profile_dataframe(df)
    by_name = {c.name: c for c in profile.columns}
    assert by_name["income"].confidence == "low"
    assert by_name["income"].ambiguous
    assert by_name["smoker"].confidence == "medium"
    assert by_name["active"].confidence == "high"
    assert "stored as string" in by_name["income"].notes[0]


def test_profile_basic_stats():
    df = make_df()
    profile = sedat.profile_dataframe(df)
    by_name = {c.name: c for c in profile.columns}
    assert by_name["age"].missing_pct == pytest.approx(2.5, abs=0.05)
    assert by_name["id"].cardinality_ratio == pytest.approx(1.0)
    assert by_name["gender"].n_unique == 2


def test_numeric_correlation():
    df = make_df()
    pearson = sedat.numeric_correlation(df)
    assert {"age", "height"} <= set(pearson.columns)
    assert abs(pearson.loc["height", "age"]) < 0.3
    spearman = sedat.numeric_correlation(df, method="spearman")
    assert "age" in spearman.index


def test_cramers_v_range():
    df = make_df()
    mat = sedat.categorical_correlation(df)
    assert {"gender", "smoker", "score_cat"} <= set(mat.columns)
    assert mat.loc["gender", "gender"] == 1.0
    for col in mat.columns:
        v = mat.loc["gender", col]
        assert np.isnan(v) or 0.0 <= v <= 1.0


def test_eta_range():
    df = make_df()
    mat = sedat.numeric_categorical_correlation(df)
    assert "gender" in mat.columns
    assert "age" in mat.index
    v = mat.loc["age", "gender"]
    assert np.isnan(v) or 0.0 <= v <= 1.0


def test_correlation_report_flags():
    df = make_df()
    df2 = df.copy()
    df2["height2"] = df["height"] * 1.5 + 3
    report = sedat.correlation_report(df2, threshold=0.95)
    flagged = report.flagged_pairs
    assert not flagged.empty
    pair = flagged[(flagged["column_a"] == "height") & (flagged["column_b"] == "height2")]
    assert not pair.empty
    assert pair["value"].iloc[0] == pytest.approx(1.0, abs=0.01)


def test_encode_binary_suggestion():
    df = make_df()
    plan = sedat.suggest_encodings(df)
    by_col = {s.column: s for s in plan.suggestions}
    assert by_col["smoker"].strategy == "binary_encode"
    assert "active" not in by_col
    assert by_col["gender"].strategy == "binary_encode"
    assert by_col["score_cat"].strategy == "one_hot"
    assert by_col["large_card"].strategy == "ordinal_encode"


def test_apply_encodings_all():
    df = make_df()
    plan = sedat.suggest_encodings(df)
    out = plan.apply_all(df)
    object_cols = [c for c in out.columns if out[c].dtype == object]
    assert object_cols == []
    assert out["smoker"].isin([0, 1]).all()
    assert out["score_cat_low"].dtype == int
    assert "large_card" in out.columns and out["large_card"].dtype in (int, float)


def test_target_encoding_high_cardinality():
    df = make_df()
    plan = sedat.suggest_encodings(df, target=df["target"])
    by_col = {s.column: s for s in plan.suggestions}
    assert by_col["large_card"].strategy == "target_encode"
    out = plan.apply_all(df)
    assert out["large_card"].dtype == float


def test_apply_single_suggestion():
    df = make_df()
    plan = sedat.suggest_encodings(df)
    smoker = next(s for s in plan.suggestions if s.column == "smoker")
    out = smoker.apply(df)
    assert out["smoker"].isin([0, 1]).all()


def test_report_end_to_end():
    df = make_df()
    report = sedat.EDAReport.create(df, target=df["target"])
    assert isinstance(report.profile, sedat.DataFrameProfile)
    assert report.encoding_summary is not None
    out = report.apply_all_encodings()
    assert object not in out.dtypes.values
    assert "score_cat" not in out.columns
    assert "score_cat_low" in out.columns


def test_binary_mapping_helpers():
    s = pd.Series(["yes", "no", "yes"])
    assert sedat.is_binary_like(s)
    s2 = pd.Series([0, 1, 1])
    assert sedat.is_binary_like(s2)
    s3 = pd.Series(["alpha", "beta", "gamma"])
    assert not sedat.is_binary_like(s3)


def test_infer_column_type_direct():
    s = pd.Series(["1.5", "2.5", "3.0"])
    info = sedat.infer_column_type(s)
    assert info["inferred_type"] == "numeric"
    assert info["confidence"] == "low"