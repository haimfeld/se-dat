"""Round-trip tests for plan persistence (sedat.persistence)."""

import json

import numpy as np
import pandas as pd
import pytest

import sedat
from sedat.datetime_features import DatetimeFeaturePlan
from sedat.encoding import EncodingPlan
from sedat.imputation import ImputationPlan
from sedat.persistence import load_plan, save_plan


def make_train(n=400, seed=10):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east", "west"], n),
            "smoker": rng.choice(["yes", "no"], n),
            "zone": rng.choice([f"z{i}" for i in range(15)], n),
            "income": rng.normal(50_000, 10_000, n).round(2),
            "target": rng.normal(0, 1, n),
        }
    )


def test_encoding_round_trip_same_frame_identical(tmp_path):
    train = make_train()
    plan = sedat.suggest_encodings(train, target=train["target"])
    expected = plan.apply_all(train)

    path = plan.save(str(tmp_path / "plan.json"))
    loaded = EncodingPlan.load(path)
    assert isinstance(loaded, EncodingPlan)

    pd.testing.assert_frame_equal(loaded.apply_all(train), expected)


def test_encoding_round_trip_new_frame_stable_schema(tmp_path):
    train = make_train(seed=11)
    # test frame: 'west' never occurs and an unseen category appears
    test = make_train(seed=12)
    test.loc[test["region"] == "west", "region"] = "east"
    test.loc[test.index[:5], "region"] = "atlantis"

    plan = sedat.suggest_encodings(train, target=train["target"])
    path = save_plan(plan, tmp_path / "enc.json")
    loaded = load_plan(path)

    out_train = plan.apply_all(train.drop(columns=["target"]))
    out_test = loaded.apply_all(test.drop(columns=["target"]))
    assert list(out_test.columns) == list(out_train.columns)
    assert (out_test["region_west"] == 0).all()  # known category absent here
    assert out_test["smoker"].isin([0, 1]).all()


def test_loaded_binary_uses_frozen_mapping(tmp_path):
    train = pd.DataFrame({"smoker": ["yes", "no"] * 20})
    plan = sedat.suggest_encodings(train)
    path = plan.save(str(tmp_path / "p.json"))
    loaded = EncodingPlan.load(path)

    # a test set with extra binary aliases must still map via the stored map;
    # unknown values become NaN rather than silently re-deriving codes
    test = pd.DataFrame({"smoker": ["yes", "no", "maybe"]})
    out = loaded.apply_all(test)
    values = out["smoker"].tolist()
    assert values[0] == 1 and values[1] == 0 and pd.isna(values[2])


def test_one_hot_unseen_category_all_zeros_after_load(tmp_path):
    train = pd.DataFrame({"color": ["red", "blue", "green"] * 10})
    plan = sedat.suggest_encodings(train)
    loaded = EncodingPlan.load(str(plan.save(str(tmp_path / "oh.json"))))

    test = pd.DataFrame({"color": ["red", "chartreuse"]})
    out = loaded.apply_all(test)
    # the original column is dropped; schema is frozen to train categories
    assert list(out.columns) == ["color_blue", "color_green", "color_red"]
    assert out.iloc[1].sum() == 0  # unseen category -> all-zero row


def test_target_encode_falls_back_to_stored_means_without_target(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "zone": rng.choice([f"z{i}" for i in range(12)], 200),
            "y": rng.normal(0, 1, 200),
        }
    )
    plan = sedat.suggest_encodings(train, target=train["y"])
    means = next(s for s in plan.suggestions if s.column == "zone").target_means
    loaded = EncodingPlan.load(str(plan.save(str(tmp_path / "te.json"))))

    test = pd.DataFrame({"zone": ["z0", "unseen-zone"]})
    out = loaded.apply_all(test)  # no target passed on purpose
    assert out["zone"].iloc[0] == pytest.approx(means["z0"])
    assert pd.isna(out["zone"].iloc[1])


def test_imputation_round_trip_uses_train_statistics(tmp_path):
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", None, "a", "a"],
            "gone": [None, None, None, "x"],  # 75% missing -> drop_column
        }
    )
    plan = sedat.suggest_imputations(train)
    loaded = ImputationPlan.load(str(plan.save(str(tmp_path / "imp.json"))))
    assert isinstance(loaded, ImputationPlan)

    test = pd.DataFrame(
        {
            "num": [np.nan, 100.0],  # train median is 2.0, NOT test median
            "cat": [None, None],
            "gone": [None, None],
        }
    )
    out = loaded.apply_all(test)
    assert "gone" not in out.columns
    assert out["num"].tolist()[0] == 2.0
    assert out["cat"].tolist() == ["a", "a"]


def test_datetime_feature_plan_round_trip(tmp_path):
    train = pd.DataFrame({"ts": pd.to_datetime(["2024-01-06 08:30:00", "2024-01-08 22:15:00"])})
    plan = sedat.suggest_datetime_features(train)
    loaded = DatetimeFeaturePlan.load(str(plan.save(str(tmp_path / "dt.json"))))

    test = pd.DataFrame({"ts": pd.to_datetime(["2023-07-04 12:00:00"])})
    expected_cols = plan.apply_all(train).columns.tolist()
    out = loaded.apply_all(test)
    assert out.columns.tolist() == expected_cols
    assert out["ts_dow"].iloc[0] == 1  # Tuesday
    assert out["ts_hour"].iloc[0] == 12


def test_load_rejects_wrong_plan_type(tmp_path):
    df = pd.DataFrame({"a": ["x", "y", "x", "x", "y", "y"]})
    enc_path = sedat.suggest_encodings(df).save(str(tmp_path / "e.json"))
    imp_path = sedat.suggest_imputations(pd.DataFrame({"a": [1.0, np.nan]})).save(str(tmp_path / "i.json"))

    with pytest.raises(TypeError, match="ImputationPlan"):
        ImputationPlan.load(enc_path)
    with pytest.raises(TypeError, match="EncodingPlan"):
        EncodingPlan.load(imp_path)


def test_load_rejects_unknown_type(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"plan_type": "mystery", "payload": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown plan type"):
        load_plan(str(path))


def test_save_rejects_non_plan(tmp_path):
    with pytest.raises(TypeError):
        save_plan("not a plan", str(tmp_path / "x.json"))


def test_envelope_metadata(tmp_path):
    plan = sedat.suggest_encodings(pd.DataFrame({"s": ["yes", "no"] * 5}))
    path = plan.save(str(tmp_path / "meta.json"))
    envelope = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert envelope["plan_type"] == "encoding"
    assert envelope["schema_version"] == 1
    assert "created_at" in envelope
    assert path.endswith(".json")


def test_json_file_is_human_readable(tmp_path):
    plan = sedat.suggest_encodings(pd.DataFrame({"s": ["yes", "no"] * 5}))
    plan.save(str(tmp_path / "readable.json"))
    text = (tmp_path / "readable.json").read_text(encoding="utf-8")
    assert '"strategy"' in text and "\n" in text  # indented JSON
