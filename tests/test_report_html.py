"""Tests for HTML report export (sedat.report_html)."""

import numpy as np
import pandas as pd
import pytest

import sedat


def make_df():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "fare": rng.gamma(2.0, 30.0, n).round(2),
            "sex": rng.choice(["male", "female"], n),
            "embarked": rng.choice(["S", "C", "Q"], n),
            "smoker": rng.choice(["yes", "no"], n),
            "target": rng.normal(0, 1, n),
        }
    )
    df.loc[0, "age"] = np.nan
    df.loc[1, "fare"] = 99999.0  # planted outlier
    return df


def make_report():
    return sedat.EDAReport.create(make_df(), target=None)


def test_to_html_writes_selfcontained_file(tmp_path):
    report = make_report()
    out = tmp_path / "report.html"
    result = report.to_html(str(out))
    assert result == str(out)
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert content.rstrip().endswith("</html>")
    assert '<meta charset="utf-8"' in content


def test_html_contains_all_sections(tmp_path):
    report = make_report()
    out = tmp_path / "report.html"
    report.to_html(str(out))
    content = out.read_text(encoding="utf-8")

    assert "<h1>se-dat EDA report</h1>" in content
    for heading in (
        "Column profile",
        "Outliers",
        "Correlations",
        "Encoding suggestions",
        "Imputation suggestions",
    ):
        assert f"<h2>{heading}</h2>" in content

    # profile table content
    assert "inferred_type" in content
    # outlier note surfaced
    assert "potential outliers" in content
    # imputation suggestion present (median strategy text)
    assert "median" in content
    # encoding suggestion present
    assert "one_hot" in content or "binary_encode" in content


def test_heatmaps_embedded_as_base64_png(tmp_path):
    report = make_report()
    out = tmp_path / "report.html"
    report.to_html(str(out))
    content = out.read_text(encoding="utf-8")
    # pearson + spearman + cramers + eta all non-empty here
    assert content.count("data:image/png;base64,") == 4


def test_empty_matrices_are_skipped_not_crashing(tmp_path):
    df = pd.DataFrame(
        {
            "only_num": [1.0, 2.0, 3.0],
            "only_cat": ["a", "b", "a"],
        }
    )
    report = sedat.EDAReport.create(df)
    out = tmp_path / "sparse.html"
    report.to_html(str(out))  # pearson/cramers empty -> no images for them
    content = out.read_text(encoding="utf-8")
    assert content.count("data:image/png;base64,") <= 1


def test_user_content_is_escaped(tmp_path):
    df = pd.DataFrame({"<b>col&x</b>": [1.0, 2.0], "<script>alert(1)</script>": [3.0, 4.0]})
    report = sedat.EDAReport.create(df)
    out = tmp_path / "xss.html"
    report.to_html(str(out))
    content = out.read_text(encoding="utf-8")
    assert "<script>alert" not in content
    assert "&lt;script&gt;" in content


def test_render_html_returns_string():
    report = make_report()
    from sedat.report_html import render_html

    content = render_html(report)
    assert isinstance(content, str) and "</html>" in content
