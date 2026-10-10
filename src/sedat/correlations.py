"""Correlation analysis.

Numeric-numeric: Pearson/Spearman matrices plus a heatmap.
Categorical-categorical: Cramer's V.
Numeric-categorical: correlation ratio (eta), an ANOVA-style association measure.

Highly correlated pairs above a threshold are flagged as potential
multicollinearity.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .types import binary_mapping

CORRELATION_TYPES = ("numeric", "boolean", "categorical")

T = TypeVar("T")

PROGRESS_BAR = os.environ.get("SEDAT_PROGRESS") == "1"


def _tqdm(items: Sequence[T]) -> Sequence[T]:
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]

        return tqdm(items)
    except ImportError:  # pragma: no cover - tqdm strictly optional
        return items


def _map(func: Callable[[T], object], items: list[T], n_jobs: int | None = None, progress: bool = False) -> list:
    """Apply ``func`` to ``items``, optionally in parallel.

    ``n_jobs > 1`` uses a thread pool (GIL-bound work still benefits from the
    C-level pandas/numpy releases); a real progress bar is shown when
    ``progress`` is set and ``tqdm`` is installed.
    """
    if progress:
        items = list(_tqdm(items))
    if (n_jobs is None or n_jobs in (0, 1)) or len(items) < 2:
        return [func(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor

    workers = None if n_jobs < 0 else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])


def _categorical_df(df: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for col in df.columns:
        dtype = df[col].dtype
        if (
            pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or dtype is object
            or isinstance(dtype, pd.StringDtype)
        ):
            picked.append(col)
    return df[picked]


def numeric_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pearson or Spearman correlation matrix over numeric columns."""
    num = _numeric_df(df)
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method)


def _cramers_v_from_codes(codes_a: np.ndarray, codes_b: np.ndarray, correction: bool) -> float:
    """Cramer's V from integer codes, dropping rows with a missing value.

    ``-1`` marks a missing entry (as returned by ``pd.factorize``); those rows
    are dropped so this matches ``pd.crosstab``'s default ``dropna=True``.
    """
    mask = (codes_a >= 0) & (codes_b >= 0)
    codes_a = codes_a[mask]
    codes_b = codes_b[mask]
    if codes_a.size == 0:
        return float("nan")
    codes_a, uniques_a = pd.factorize(codes_a)
    codes_b, uniques_b = pd.factorize(codes_b)
    n_a = int(uniques_a.size)
    n_b = int(uniques_b.size)
    if n_a < 2 or n_b < 2:
        return float("nan")
    table = np.bincount(codes_a * n_b + codes_b, minlength=n_a * n_b).reshape(n_a, n_b)
    if min(table.shape) < 2:
        return float("nan")
    with warnings.catch_warnings():
        # scipy warns about low expected frequencies on sparse tables; that is
        # exactly the degenerate case where we already return nan-adjacent
        # values, so we deliberately keep it out of the user's console.
        warnings.simplefilter("ignore")
        chi2, _, _, _ = chi2_contingency(table, correction=correction)
    n = int(table.sum())
    if n <= 1:
        return float("nan")
    phi2 = chi2 / n
    r, k = table.shape
    if correction:
        phi2 = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r = max(0, r - ((r - 1) ** 2) / (n - 1))
        k = max(0, k - ((k - 1) ** 2) / (n - 1))
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2 / denom))


def cramers_v(a: pd.Series, b: pd.Series, correction: bool = True) -> float:
    """Cramer's V association between two categorical series, in [0, 1]."""
    if a.name is not None and b.name is not None and a.name == b.name:
        return 1.0
    codes_a, _ = pd.factorize(a.astype(str))
    codes_b, _ = pd.factorize(b.astype(str))
    return _cramers_v_from_codes(np.asarray(codes_a), np.asarray(codes_b), correction)


def categorical_correlation(df: pd.DataFrame, n_jobs: int | None = None, progress: bool = False) -> pd.DataFrame:
    """Cramer's V matrix over categorical/boolean columns."""
    cats = _categorical_df(df)
    cols = list(cats.columns)
    if len(cols) < 2:
        return pd.DataFrame()
    # Factorise each column once; the O(p^2) pairs then reuse the integer codes
    # instead of rebuilding a crosstab per pair (the original bottleneck).
    codes: dict[str, np.ndarray] = {}
    for col in cols:
        c, _ = pd.factorize(cats[col].astype(str))
        codes[col] = np.asarray(c)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    def _pair(pair: tuple[str, str]) -> float:
        a, b = pair
        return _cramers_v_from_codes(codes[a], codes[b], True)

    values = _map(_pair, pairs, n_jobs, progress)
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for (a, b), v in zip(pairs, values):
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    for col in cols:
        mat.loc[col, col] = 1.0
    return mat


def correlation_ratio(numeric: pd.Series, categorical: pd.Series) -> float:
    """Eta correlation ratio between one numeric and one categorical series."""
    y = pd.to_numeric(numeric, errors="coerce")
    g = categorical.astype(str)
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    if len(d) < 2 or d["g"].nunique() < 2:
        return float("nan")
    # eta is meaningless when groups are near-singleton (e.g. id/free-text
    # columns): every row would explain itself, yielding a spurious eta of 1.
    if len(d) / d["g"].nunique() < 2:
        return float("nan")
    grand_mean = float(d["y"].mean())
    group = d.groupby("g")["y"].agg(["mean", "count"])
    ss_between = float(((group["mean"] - grand_mean) ** 2 * group["count"]).sum())
    ss_total = float(((d["y"] - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    return float(np.sqrt(ss_between / ss_total))


def _eta_matrix(num: pd.DataFrame, cats: pd.DataFrame) -> pd.DataFrame:
    """Eta matrix computed one grouped reduction per categorical column.

    Pair-by-pair eta is O(p^2) pandas group-bys; reducing *every* numeric column
    against a single categorical column in one grouped aggregation keeps the
    full matrix cheap even on wide frames. Mirrors
    :func:`correlation_ratio` (including the near-singleton guard).
    """
    num_cols = list(num.columns)
    mat = pd.DataFrame(index=num_cols, columns=list(cats.columns), dtype=float)
    for cat in cats.columns:
        g = cats[cat]
        valid = g.notna()
        if not bool(valid.any()):
            continue
        gv = g[valid].astype(str)
        if int(gv.nunique()) < 2:
            continue
        y = num[valid]
        codes = pd.factorize(gv)[0]
        values = y.to_numpy(dtype=float)
        grouped = y.groupby(codes)
        means = grouped.mean().to_numpy(dtype=float)
        counts = grouped.count().to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            grand = np.nanmean(values, axis=0)
            ss_total = np.nansum((values - grand) ** 2, axis=0)
            ss_between = np.nansum(counts * (means - grand) ** 2, axis=0)
            eta = np.sqrt(ss_between / ss_total)
            # near-singleton groups would explain themselves -> drop to NaN
            present = (counts > 0).sum(axis=0)
            per_group = np.divide(
                counts.sum(axis=0), present, out=np.zeros_like(present, dtype=float), where=present > 0
            )
            eta = np.where(per_group < 2, np.nan, eta)
        mat[cat] = eta
    return mat


def numeric_categorical_correlation(
    df: pd.DataFrame, n_jobs: int | None = None, progress: bool = False
) -> pd.DataFrame:
    """Eta matrix with numeric columns as rows and categorical columns as cols."""
    num = _numeric_df(df)
    cats = _categorical_df(df)
    if num.shape[1] < 1 or cats.shape[1] < 1:
        return pd.DataFrame()
    if (n_jobs is None or n_jobs in (0, 1)) or cats.shape[1] < 2:
        return _eta_matrix(num, cats)
    from concurrent.futures import ThreadPoolExecutor

    workers = None if n_jobs < 0 else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = list(executor.map(lambda c: _eta_matrix(num, cats[[c]]), cats.columns))
    return pd.concat(frames, axis=1)


def _upper_pairs(mat: pd.DataFrame) -> Iterator[tuple[str, str, float]]:
    cols = list(mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = mat.iat[i, j]
            if np.isnan(v):
                continue
            yield cols[i], cols[j], float(v)


def variance_inflation_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per numeric column.

    ``VIF_i = 1 / (1 - R2_i)`` where ``R2_i`` comes from regressing column *i*
    on every other numeric column. Values above ~5-10 indicate multi-way
    collinearity that pairwise correlation alone can miss. Zero-variance
    columns are excluded (their VIF is undefined).
    """
    num = _numeric_df(df).dropna()
    num = num.loc[:, num.nunique() > 1]
    cols = list(num.columns)
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature", "vif"])
    x = num.to_numpy(dtype=float)
    std = x.std(axis=0, ddof=0)
    z = (x - x.mean(axis=0)) / np.where(std == 0, 1.0, std)
    rows = []
    for i, col in enumerate(cols):
        y = z[:, i]
        others = np.delete(z, i, axis=1)
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        residual = y - others @ coef
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
        rows.append({"feature": col, "vif": vif})
    return pd.DataFrame(rows)


def condition_number(df: pd.DataFrame) -> float:
    """Condition number of the standardised numeric feature matrix.

    A large value (say > 30) signals near-linear dependencies among features.
    Returns ``nan`` when fewer than two usable numeric columns exist.
    """
    num = _numeric_df(df).dropna()
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return float("nan")
    x = num.to_numpy(dtype=float)
    std = x.std(axis=0, ddof=0)
    z = (x - x.mean(axis=0)) / np.where(std == 0, 1.0, std)
    return float(np.linalg.cond(z))


@dataclass
class CorrelationReport:
    numeric: pd.DataFrame
    numeric_spearman: pd.DataFrame
    categorical: pd.DataFrame
    numeric_categorical: pd.DataFrame
    method_numeric: str = "pearson"
    threshold: float = 0.7
    vif: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["feature", "vif"]))
    condition_number_value: float = float("nan")

    @property
    def summary(self) -> pd.DataFrame:
        rows = []
        for mat, kind, method in (
            (self.numeric, "numeric", self.method_numeric),
            (self.numeric_spearman, "numeric", "spearman"),
            (self.categorical, "categorical", "cramers_v"),
            (self.numeric_categorical, "numeric-categorical", "eta"),
        ):
            if mat.empty:
                continue
            if kind == "numeric-categorical":
                for num_col in mat.index:
                    for cat_col in mat.columns:
                        v = mat.loc[num_col, cat_col]
                        if np.isnan(v):
                            continue
                        rows.append(
                            {
                                "column_a": num_col,
                                "column_b": cat_col,
                                "kind": kind,
                                "method": method,
                                "value": round(float(v), 4),
                                "flagged": abs(float(v)) >= self.threshold,
                            }
                        )
            else:
                for a, b, v in _upper_pairs(mat):
                    rows.append(
                        {
                            "column_a": a,
                            "column_b": b,
                            "kind": kind,
                            "method": method,
                            "value": round(v, 4),
                            "flagged": abs(v) >= self.threshold,
                        }
                    )
        return pd.DataFrame(rows)

    @property
    def flagged_pairs(self) -> pd.DataFrame:
        s = self.summary
        return s[s["flagged"]] if not s.empty else s

    @property
    def collapsed_summary(self) -> pd.DataFrame:
        """One row per unordered pair, collapsing the methods that agree.

        Numeric pairs otherwise appear once for Pearson and once for Spearman;
        here they become a single row with the strongest absolute value, the
        contributing methods and their individual values.
        """
        s = self.summary
        if s.empty:
            return pd.DataFrame(
                columns=["column_a", "column_b", "kind", "methods", "n_methods", "max_abs_value", "flagged", "values"]
            )
        rows = []
        for (a, b, kind), group in s.groupby(["column_a", "column_b", "kind"], sort=False):
            values = "; ".join(f"{m}={v}" for m, v in zip(group["method"], group["value"]))
            rows.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "kind": kind,
                    "methods": ", ".join(group["method"]),
                    "n_methods": len(group),
                    "max_abs_value": float(group["value"].abs().max()),
                    "flagged": bool(group["flagged"].any()),
                    "values": values,
                }
            )
        collapsed = pd.DataFrame(rows)
        return collapsed.sort_values("max_abs_value", ascending=False, ignore_index=True)

    @property
    def flagged_pairs_collapsed(self) -> pd.DataFrame:
        collapsed = self.collapsed_summary
        return collapsed[collapsed["flagged"]] if not collapsed.empty else collapsed


def correlation_report(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.7,
    compute_collinearity: bool = True,
    max_vif_features: int = 50,
    n_jobs: int | None = None,
    progress: bool = PROGRESS_BAR,
) -> CorrelationReport:
    num_cols = _numeric_df(df)
    vif = pd.DataFrame(columns=["feature", "vif"])
    cond = float("nan")
    if compute_collinearity and num_cols.shape[1] >= 2:
        cond = condition_number(df)
        # VIF is O(p^3); skip the expensive regression sweep on very wide frames
        if num_cols.shape[1] <= max_vif_features:
            vif = variance_inflation_factors(df)
    return CorrelationReport(
        numeric=numeric_correlation(df, method=method),
        numeric_spearman=numeric_correlation(df, method="spearman"),
        categorical=categorical_correlation(df, n_jobs=n_jobs, progress=progress),
        numeric_categorical=numeric_categorical_correlation(df, n_jobs=n_jobs, progress=progress),
        method_numeric=method,
        threshold=threshold,
        vif=vif,
        condition_number_value=cond,
    )


def _binary_numeric_map(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {col: mapping for col in df.columns if (mapping := binary_mapping(df[col])) is not None}


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary-like object columns to 0/1 numeric for correlation use."""
    out = df.copy()
    for col, mapping in _binary_numeric_map(df).items():
        out[col] = out[col].astype(str).str.strip().str.lower().map(mapping)
    return out


def correlation_heatmap(matrix: pd.DataFrame, title: str = "Correlation matrix") -> object:
    """Plot a heatmap for a correlation matrix; returns a matplotlib Figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.7), max(5, matrix.shape[0] * 0.6)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
