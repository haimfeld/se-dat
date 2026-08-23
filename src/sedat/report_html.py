"""Static, self-contained HTML export for :class:`sedat.EDAReport`.

The page is assembled with plain string formatting (no templating framework,
no external JS/CSS) and correlation heatmaps are embedded as base64-encoded
PNG images so the output file works offline as a single artifact.
"""

from __future__ import annotations

import base64
import html as _html
import io
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .report import EDAReport

_CSS = """
body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1200px; color: #1f2430; background: #fafbfc; }
h1 { font-size: 1.6rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 2px solid #e3e7ee;
     padding-bottom: .3rem; }
.meta { color: #667085; font-size: .9rem; }
section { margin-bottom: 1.5rem; }
table.data-table { border-collapse: collapse; font-size: .85rem; width: 100%; }
table.data-table th, table.data-table td { border: 1px solid #d8dee9;
     padding: 4px 8px; text-align: left; }
table.data-table th { background: #eef1f6; }
tr:nth-child(even) { background: #f5f7fa; }
figure { margin: 1rem 0; }
figure img { max-width: 100%; border: 1px solid #d8dee9; background: #fff; }
figcaption { color: #667085; font-size: .85rem; margin-top: .3rem; }
.empty { color: #98a2b3; font-style: italic; }
"""


def _escape(text: object) -> str:
    return _html.escape(str(text))


def _fig_png_base64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap_img(matrix: pd.DataFrame, title: str) -> str:
    if matrix is None or getattr(matrix, "empty", True):
        return ""
    from .correlations import correlation_heatmap

    fig = correlation_heatmap(matrix, title=title)
    b64 = _fig_png_base64(fig)
    return (
        "<figure>"
        f'<img src="data:image/png;base64,{b64}" alt="{_escape(title)}"/>'
        f"<figcaption>{_escape(title)}</figcaption>"
        "</figure>"
    )


def _df_table(df: pd.DataFrame | None, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return '<p class="empty">Nothing to report.</p>'
    return df.to_html(index=index, border=0, classes="data-table")


def _section(title: str, body: str) -> str:
    return f"<section><h2>{_escape(title)}</h2>{body}</section>"


def render_html(report: "EDAReport") -> str:
    """Render ``report`` to a complete standalone HTML document string."""
    from . import __version__

    corr = report.correlations
    flagged = corr.flagged_pairs
    n_rows, n_cols = report.df.shape
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta = (
        f'<p class="meta">se-dat {_escape(__version__)} &middot; '
        f"generated {_escape(generated)} &middot; "
        f"{n_rows} rows &times; {n_cols} columns</p>"
    )
    title = f"se-dat EDA report &mdash; {_escape(n_cols)} columns"

    sections = [
        _section("Column profile", _df_table(report.profile.summary, index=True)),
        _section("Outliers", _df_table(report.outliers.summary)),
        _section("Correlations", _df_table(flagged)),
        _heatmap_img(corr.numeric, "Pearson correlation (numeric)"),
        _heatmap_img(corr.numeric_spearman, "Spearman correlation (numeric)"),
        _heatmap_img(corr.categorical, "Cramér's V (categorical)"),
        _heatmap_img(
            corr.numeric_categorical,
            "Correlation ratio η (numeric vs categorical)",
        ),
        _section("Encoding suggestions", _df_table(report.encoding_summary)),
        _section("Imputation suggestions", _df_table(report.imputation_summary)),
    ]

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8"/>\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>se-dat EDA report</h1>\n{meta}\n"
        + "\n".join(sections)
        + "\n</body>\n</html>\n"
    )


def write_html(report: "EDAReport", path: str) -> str:
    """Render ``report`` and write it to ``path``; returns the path."""
    content = render_html(report)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
