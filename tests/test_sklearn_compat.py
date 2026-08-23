"""Tests for scikit-learn interoperability (sedat.sklearn_compat)."""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

import sedat  # noqa: E402


def make_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "fare": rng.gamma(2.0, 30.0, n).round(2),
            "smoker": rng.choice(["yes", "no"], n),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "sku": rng.choice([f"sku_{i}" for i in range(12)], n),
            "free_text": [f"text {i}" for i in range(n)],
            "target": rng.normal(0, 1, n),
        }
    )


@pytest.fixture()
def plan_with_target():
    df = make_df()
    return sedat.suggest_encodings(df, target=df["target"]), df


def test_transformer_mirrors_plan_strategies(plan_with_target):
    plan, _ = plan_with_target
    ct = plan.to_column_transformer()
    by_name = {name: est for name, est, _ in ct.transformers if name != "remainder"}
    assert isinstance(by_name["smoker_binary_encode"], sklearn.preprocessing.OrdinalEncoder)
    assert isinstance(by_name["region_one_hot"], sklearn.preprocessing.OneHotEncoder)
    assert isinstance(by_name["sku_target_encode"], sklearn.preprocessing.TargetEncoder)
    # passthrough carries everything without a suggestion
    assert ct.remainder == "passthrough"


def test_fit_transform_output_matches_pandas_apply(plan_with_target):
    plan, df = plan_with_target
    y = df["target"]
    features = df.drop(columns=["target"])
    ct = plan.to_column_transformer()
    Xt = ct.fit_transform(features, y)

    expected = plan.apply_all(features, target=y)
    # same total width: one-hot expands region to its categories
    assert Xt.shape[1] == expected.shape[1]
    out = pd.DataFrame(Xt, columns=ct.get_feature_names_out())
    # passthrough text columns make the assembled array object dtype,
    # so compare values rather than dtypes
    pd.testing.assert_series_equal(
        out["smoker"].astype(int), expected["smoker"], check_names=False
    )
    pd.testing.assert_frame_equal(
        out[["age", "fare"]].astype(float),
        expected[["age", "fare"]],
        check_dtype=False,
        check_names=False,
    )
    for cat in ["east", "north", "south", "west"]:
        col = f"region_{cat}"
        assert col in out.columns
        assert out[col].astype(int).sum() == expected[col].sum()


def test_unseen_categories_do_not_explode_at_transform_time():
    cols = ["age", "fare", "smoker", "region", "sku"]
    train = make_df(seed=2)[cols]
    test = make_df(seed=3)[cols]
    test["region"] = "atlantis"  # category never seen during fit
    plan = sedat.suggest_encodings(train, target=None)
    ct = plan.to_column_transformer()
    ct.fit(train)
    Xt = np.asarray(ct.transform(test), dtype=float)
    assert np.isfinite(Xt).all()


def test_passthrough_only_when_no_suggestions():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    plan = sedat.suggest_encodings(df)
    ct = plan.to_column_transformer()
    Xt = ct.fit_transform(df)
    assert list(ct.get_feature_names_out()) == ["a", "b"]
    np.testing.assert_allclose(np.asarray(Xt), df.to_numpy())


def test_works_without_any_target(plan_with_target):
    plan, df = plan_with_target
    plain = sedat.suggest_encodings(df)  # no target -> ordinal fallback for sku
    ct = plain.to_column_transformer()
    Xt = ct.fit_transform(df.drop(columns=["target"]))
    out = pd.DataFrame(Xt, columns=ct.get_feature_names_out())
    assert set(out["sku"]) <= set(range(-1, 12))  # ordinal codes, -1 = unseen
