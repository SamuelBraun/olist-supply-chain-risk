"""Visualisation helpers for the Olist notebook.

Every helper takes a *small* pandas DataFrame or dict (sourced from a
pre-aggregated / pre-capped Spark aggregate) and either returns a
`pandas.io.formats.style.Styler` (for tables) or displays a
`plotly.graph_objects.Figure` (for charts). Helpers never touch Spark,
never do file I/O, and never materialise raw data — that contract is
the module's reason for existing.

Plotly was chosen over matplotlib because the rendering payload it
emits is a small JSON description of the figure, independent of the
upstream Spark dataset size — i.e. the visualisation layer remains
bounded even if the project scales to TB-class data, provided the
upstream `.agg() / .limit()` contract is preserved. The interactive
HTML is paired with a Kaleido-rendered static PNG (via the
``notebook_connected+png`` renderer) so the executed notebook outputs
survive fresh-kernel reopens required by
``scripts/assert_notebook_outputs.py``.

Driver-side materialisation happens exactly once upstream, annotated
with ``# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ`` (or a more specific
ID). See ``docs/big_data_safety_log.md``.

Palette:
* Categorical: ``plotly.express.colors.qualitative.D3`` (10-class, print-safe).
* Sequential risk / heat: ``"Reds"``.
* Diverging (residuals, correlations): ``"RdBu_r"``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Static PNG via Kaleido. Single image MIME per cell -> survives nbconvert
# and fresh-kernel reopens, passes scripts/assert_notebook_outputs.py.
# We used to use "notebook_connected+png", which emitted BOTH renderers
# and produced two outputs per figure (the double-render bug).
pio.renderers.default = "png"

_CAT_PALETTE: list[str] = list(px.colors.qualitative.D3)
_RISK_SCALE = "Reds"
_DIVERGING_SCALE = "RdBu_r"

# Sequential "Reds" colour stops, used by the styled-table gradients.
_RED_STOPS = (
    "#fff5f0",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#a50f15",
)
_BLUE_STOPS = (
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
)


def _gradient_css(values: pd.Series, stops: Sequence[str]) -> list[str]:
    """Map a numeric series to per-cell CSS background-colour strings.

    Pandas Styler's ``.background_gradient`` depends on matplotlib for the
    colormap; rolling our own keeps the Styler matplotlib-free.
    """
    if len(values) == 0:
        return []
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return [f"background-color: {stops[len(stops) // 2]}"] * len(arr)
    out: list[str] = []
    n = len(stops) - 1
    for v in arr:
        if not np.isfinite(v):
            out.append("background-color: #f0f0f0")
            continue
        t = (v - lo) / (hi - lo)
        idx = min(int(round(t * n)), n)
        out.append(f"background-color: {stops[idx]}")
    return out


def _show(fig: go.Figure) -> go.Figure:
    # Jupyter renders the returned fig once via its mimebundle repr.
    # Used to also call fig.show() here, which produced double output.
    return fig


# ---------------------------------------------------------------------------
# Styled tables (pandas Styler — HTML only, no matplotlib)
# ---------------------------------------------------------------------------


def styled_topn_table(
    df: pd.DataFrame,
    *,
    bar_cols: Sequence[str] | None = None,
    gradient_cols: Sequence[str] | None = None,
    fmt: dict[str, str] | None = None,
    title: str | None = None,
    hide_index: bool = True,
):
    """Render ``df`` (≤50 rows) as a consulting-quality styled HTML table.

    ``bar_cols``      → embedded horizontal bars (for magnitudes).
    ``gradient_cols`` → red sequential gradient (for risk scores).
    ``fmt``           → per-column format string.
    """
    styler = df.style
    if hide_index:
        styler = styler.hide(axis="index")
    if title:
        styler = styler.set_caption(title)
    if fmt:
        styler = styler.format(fmt)
    if bar_cols:
        styler = styler.bar(subset=list(bar_cols), color="#4C72B0")
    if gradient_cols:
        for col in gradient_cols:
            if col in df.columns:
                css = _gradient_css(df[col], _RED_STOPS)
                styler = styler.apply(lambda _s, css=css: css, subset=[col])
    styler = styler.set_table_styles(
        [
            {"selector": "caption", "props": "font-size: 1.1em; font-weight: 600; text-align: left; padding-bottom: 0.4em; color: #1a1a1a;"},
            {"selector": "th", "props": "background-color: #e6e6e6; color: #1a1a1a; text-align: left; padding: 6px 10px; border-bottom: 1px solid #bdbdbd;"},
            {"selector": "td", "props": "padding: 4px 8px;"},
        ]
    )
    return styler


def eda_quantile_table(stats: dict):
    """Render the ``eda_stats(...)`` dict as a styled summary table.

    Expected keys: ``price_quantiles``, ``delay_quantiles`` (each length-4
    list, quartiles p25/p50/p75/p95), and ``approx_counts`` (a Spark DF
    collected to a single pandas row).
    """
    price_q = stats["price_quantiles"]
    delay_q = stats["delay_quantiles"]
    counts = stats["approx_counts"].toPandas().iloc[0].to_dict()
    df = pd.DataFrame(
        [
            ("price (R$)", *price_q),
            ("delivery delay (days)", *delay_q),
        ],
        columns=["metric", "p25", "p50", "p75", "p95"],
    )
    styler = df.style.hide(axis="index").set_caption(
        f"EDA quantiles — approx distinct sellers: {int(counts.get('approx_sellers', 0)):,} | "
        f"products: {int(counts.get('approx_products', 0)):,}"
    )
    styler = styler.format({c: "{:,.2f}" for c in ["p25", "p50", "p75", "p95"]})
    # Per-row blue gradient across the four quantile columns (axis=1 equivalent).
    quant_cols = ["p25", "p50", "p75", "p95"]

    def _row_gradient(row: pd.Series) -> list[str]:
        return _gradient_css(row, _BLUE_STOPS)

    styler = styler.apply(_row_gradient, axis=1, subset=quant_cols)
    styler = styler.set_table_styles(
        [
            {"selector": "caption", "props": "font-size: 1em; text-align: left; padding-bottom: 0.4em; color: #1a1a1a;"},
            {"selector": "th", "props": "background-color: #e6e6e6; color: #1a1a1a; padding: 6px 10px; border-bottom: 1px solid #bdbdbd;"},
        ]
    )
    return styler


# ---------------------------------------------------------------------------
# Bars / distributions
# ---------------------------------------------------------------------------


def class_balance_bar(df: pd.DataFrame, label_col: str = "label", n_col: str = "n") -> go.Figure:
    """Stacked horizontal bar of class counts with percentage labels.

    Intended for the binary-sentiment class balance (positive / negative).
    """
    total = float(df[n_col].sum())
    fig = go.Figure()
    palette = ["#2ca02c", "#d62728"]  # green / red, colour-blind safe enough
    for i, (_, row) in enumerate(df.iterrows()):
        pct = row[n_col] / total * 100 if total else 0.0
        fig.add_bar(
            x=[row[n_col]],
            y=["balance"],
            orientation="h",
            marker_color=palette[i % len(palette)],
            text=[f"{row[label_col]} — {int(row[n_col]):,} ({pct:.1f}%)"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="%{text}<extra></extra>",
            name=str(row[label_col]),
        )
    fig.update_layout(
        barmode="stack",
        title="Review sentiment class balance",
        xaxis_title="reviews",
        yaxis_title="",
        showlegend=False,
        height=160,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    fig.update_yaxes(showticklabels=False)
    return _show(fig)


def state_bar(
    df: pd.DataFrame,
    value_col: str,
    *,
    label_col: str = "seller_state",
    title: str | None = None,
    color_by_value: bool = True,
    sort: str = "desc",
) -> go.Figure:
    """Horizontal bar chart of a per-state metric (≤27 rows)."""
    data = df.copy()
    ascending = sort == "asc"
    data = data.sort_values(value_col, ascending=ascending).reset_index(drop=True)
    color_kw = {}
    if color_by_value:
        color_kw = {"color": value_col, "color_continuous_scale": _RISK_SCALE}
    fig = px.bar(
        data,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        **color_kw,
    )
    fig.update_traces(
        texttemplate="%{x:,.3f}",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(
        title=title or f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ')}",
        xaxis_title=value_col.replace("_", " "),
        yaxis_title=label_col.replace("_", " "),
        yaxis=dict(categoryorder="array", categoryarray=list(data[label_col])),
        height=max(280, 28 * len(data) + 120),
        margin=dict(l=80, r=60, t=60, b=50),
    )
    return _show(fig)


def feature_importance_bar(
    pairs: Iterable[tuple[str, float]],
    title: str = "Feature importance",
) -> go.Figure:
    """Horizontal bar of (feature, importance) pairs, largest on top."""
    data = sorted(list(pairs), key=lambda p: p[1], reverse=True)
    pdf = pd.DataFrame(data, columns=["feature", "importance"])
    fig = px.bar(
        pdf,
        x="importance",
        y="feature",
        orientation="h",
        text="importance",
        color_discrete_sequence=[_CAT_PALETTE[0]],
    )
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=title,
        xaxis_title="importance",
        yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=list(pdf["feature"])[::-1]),
        height=max(280, 32 * len(pdf) + 120),
        margin=dict(l=140, r=60, t=60, b=50),
    )
    return _show(fig)


def lag_corr_bar(df: pd.DataFrame, *, lag_col: str = "lag", corr_col: str = "corr") -> go.Figure:
    """9-row lag-vs-correlation chart with peak annotation.

    ``df`` comes from ``pipeline.sentiment.build_lead_indicator_lags``.
    """
    data = df.sort_values(lag_col).copy()
    colors = ["#d62728" if v < 0 else _CAT_PALETTE[0] for v in data[corr_col]]
    fig = go.Figure(
        go.Bar(
            x=data[lag_col],
            y=data[corr_col],
            marker_color=colors,
            hovertemplate="lag=%{x}w<br>ρ=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="grey")

    peak_idx = data[corr_col].abs().idxmax()
    peak_lag = int(data.loc[peak_idx, lag_col])
    peak_corr = float(data.loc[peak_idx, corr_col])
    fig.add_annotation(
        x=peak_lag,
        y=peak_corr,
        text=f"peak |ρ|={abs(peak_corr):.4f}<br>at lag={peak_lag}w",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        ay=-30 if peak_corr >= 0 else 30,
        bgcolor="white",
        bordercolor="black",
        borderpad=3,
    )
    fig.update_layout(
        title="Does sentiment decline precede volume decline?",
        xaxis_title="Lag k (weeks)",
        yaxis_title="Pearson ρ",
        xaxis=dict(tickmode="array", tickvals=list(data[lag_col])),
        height=380,
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return _show(fig)


# ---------------------------------------------------------------------------
# Heatmaps / scatter
# ---------------------------------------------------------------------------


def heatmap_from_long(
    df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    cmap: str = _RISK_SCALE,
    fmt: str = ".2f",
    title: str | None = None,
) -> go.Figure:
    """Pivot a long DataFrame and render it as a Plotly heatmap."""
    wide = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig = px.imshow(
        wide,
        text_auto=fmt,
        color_continuous_scale=cmap,
        aspect="auto",
        labels=dict(x=columns, y=index, color=values),
    )
    fig.update_layout(
        title=title or f"{values} by {index} × {columns}",
        height=max(280, 30 * len(wide) + 120),
        margin=dict(l=110, r=40, t=60, b=80),
    )
    return _show(fig)


def risk_band_donut(df: pd.DataFrame, *, label_col: str = "risk_class", n_col: str = "n") -> go.Figure:
    """Donut chart of risk-band counts with percentage labels."""
    color_map = {"SAFE": "#2ca02c", "WARNING": "#ff7f0e", "CRITICAL": "#d62728"}
    pie_colors = [color_map.get(c, "#888") for c in df[label_col]]
    total = int(df[n_col].sum())
    fig = go.Figure(
        go.Pie(
            labels=df[label_col],
            values=df[n_col],
            hole=0.5,
            marker=dict(colors=pie_colors, line=dict(color="white", width=2)),
            textinfo="label+value+percent",
            textposition="outside",
            sort=False,
        )
    )
    fig.update_layout(
        title="Seller risk-band distribution",
        annotations=[
            dict(
                text=f"Total<br><b>{total:,}</b><br>sellers",
                x=0.5, y=0.5,
                font=dict(size=13),
                showarrow=False,
            )
        ],
        height=440,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return _show(fig)


def quadrant_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    size: str,
    color: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> go.Figure:
    """Bubble scatter for top-N seller risk views.

    Bubble area scales with ``size`` column, colour with ``color`` column.
    Adds median-crosshairs to split the plot into four quadrants.
    """
    size_max = float(df[size].max()) or 1.0
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        size_max=42,
        color_continuous_scale=_RISK_SCALE,
        hover_data=list(df.columns),
    )
    fig.add_hline(y=float(df[y].median()), line_dash="dash", line_color="grey", line_width=1)
    fig.add_vline(x=float(df[x].median()), line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title=xlabel or x.replace("_", " "),
        yaxis_title=ylabel or y.replace("_", " "),
        height=520,
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return _show(fig)


# ---------------------------------------------------------------------------
# Time series / diagnostics
# ---------------------------------------------------------------------------


def weekly_trend_multiline(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    title: str = "Weekly trend",
) -> go.Figure:
    """Multi-line chart of a weekly metric across a small set of entities."""
    data = df.sort_values(x).copy()
    fig = px.line(
        data,
        x=x,
        y=y,
        color=hue,
        color_discrete_sequence=_CAT_PALETTE,
    )
    fig.update_traces(line=dict(width=1.8), opacity=0.9)
    fig.update_layout(
        title=title,
        xaxis_title=x.replace("_", " "),
        yaxis_title=y.replace("_", " "),
        legend_title=hue.replace("_", " "),
        height=400,
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return _show(fig)


def confusion_matrix_heatmap(
    counts: pd.DataFrame,
    *,
    y_true: str = "label",
    y_pred: str = "prediction",
    n_col: str = "n",
    class_labels: Sequence[str] = ("negative (0)", "positive (1)"),
    title: str = "Confusion matrix — test set",
) -> go.Figure:
    """Render a 2×2 confusion matrix from a 4-row groupBy count."""
    pivot = counts.pivot_table(index=y_true, columns=y_pred, values=n_col, aggfunc="sum").fillna(0.0)
    pivot = pivot.reindex(index=[0, 1], columns=[0, 1], fill_value=0.0)
    total = pivot.values.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        row_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
    text = [
        [
            f"{int(pivot.values[i, j]):,}<br>({row_norm.values[i, j] * 100:.1f}%)"
            for j in range(2)
        ]
        for i in range(2)
    ]
    fig = px.imshow(
        pivot.values,
        text_auto=False,
        color_continuous_scale="Blues",
        aspect="equal",
        labels=dict(x="predicted", y="actual", color="n reviews"),
        x=list(class_labels),
        y=list(class_labels),
    )
    fig.update_traces(text=text, texttemplate="%{text}", hovertemplate="actual=%{y}<br>predicted=%{x}<br>%{text}<extra></extra>")
    fig.update_layout(
        title=f"{title}  (n={int(total):,})",
        height=440,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return _show(fig)


def correlation_heatmap(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    title: str = "Pearson correlation",
    cmap: str = _DIVERGING_SCALE,
) -> go.Figure:
    """Compute and render an NxN Pearson correlation matrix as a heatmap.

    Diverging colour map centred on 0 so positive (+1) and negative (-1)
    correlations are visually distinct.
    """
    corr = df[list(cols)].corr(method="pearson")
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=cmap,
        zmin=-1,
        zmax=1,
        aspect="equal",
        labels=dict(color="Pearson ρ"),
    )
    fig.update_layout(
        title=title,
        height=max(360, 70 * len(cols) + 120),
        margin=dict(l=110, r=40, t=60, b=80),
    )
    return _show(fig)


def kmeans_elbow_plot(
    df: pd.DataFrame,
    *,
    k_col: str = "k",
    wssse_col: str = "wssse",
    chosen_k: int = 4,
    title: str = "K-Means elbow — WSSSE vs k",
) -> go.Figure:
    """Line + marker chart of WSSSE per k with a vertical highlight at the
    chosen elbow.

    Used in notebook §6.3.1 to justify ``k=4`` for risk archetypes.
    """
    data = df.sort_values(k_col).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[k_col],
            y=data[wssse_col],
            mode="lines+markers+text",
            text=[f"{v:.1f}" for v in data[wssse_col]],
            textposition="top center",
            marker=dict(size=10, color=_CAT_PALETTE[0]),
            line=dict(width=2, color=_CAT_PALETTE[0]),
            hovertemplate="k=%{x}<br>WSSSE=%{y:.3f}<extra></extra>",
            name="WSSSE",
        )
    )
    if chosen_k in set(data[k_col].astype(int)):
        chosen_y = float(data.loc[data[k_col] == chosen_k, wssse_col].iloc[0])
        fig.add_vline(
            x=chosen_k, line_dash="dash", line_color="#d62728", line_width=1.5,
            annotation_text=f"chosen k={chosen_k}",
            annotation_position="top right",
        )
        fig.add_trace(
            go.Scatter(
                x=[chosen_k], y=[chosen_y],
                mode="markers",
                marker=dict(size=16, color="#d62728", symbol="star"),
                name=f"k={chosen_k}",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="k (number of clusters)",
        yaxis_title="WSSSE (within-set sum of squared errors)",
        xaxis=dict(tickmode="array", tickvals=list(data[k_col])),
        height=380,
        margin=dict(l=70, r=40, t=60, b=50),
        showlegend=False,
    )
    return _show(fig)


def archetype_scatter(
    df: pd.DataFrame,
    *,
    components: Sequence[str],
    cluster_col: str = "cluster",
    label_col: str = "archetype",
    title: str = "Risk archetypes — pairwise component view",
) -> go.Figure:
    """Three-panel pairwise scatter (one per component pair) coloured by
    cluster + archetype label. Renders well even at ~3000 sellers because
    each cluster is plotted with low alpha.
    """
    pairs = [
        (components[0], components[1]),
        (components[0], components[2]),
        (components[1], components[2]),
    ]
    cluster_to_label = (
        df.drop_duplicates(cluster_col)[[cluster_col, label_col]]
        .set_index(cluster_col)[label_col]
        .to_dict()
    )
    clusters_sorted = sorted(cluster_to_label.keys())
    palette = {c: _CAT_PALETTE[i % len(_CAT_PALETTE)] for i, c in enumerate(clusters_sorted)}

    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{x} vs {y}" for x, y in pairs])
    for col_idx, (x, y) in enumerate(pairs, start=1):
        for c in clusters_sorted:
            sub = df[df[cluster_col] == c]
            fig.add_trace(
                go.Scatter(
                    x=sub[x],
                    y=sub[y],
                    mode="markers",
                    marker=dict(
                        color=palette[c],
                        size=6,
                        opacity=0.55,
                        line=dict(width=0),
                    ),
                    name=f"{cluster_to_label[c]} (n={len(sub):,})",
                    legendgroup=str(c),
                    showlegend=(col_idx == 1),
                    hovertemplate=f"{x}=%{{x:.3f}}<br>{y}=%{{y:.3f}}<extra>{cluster_to_label[c]}</extra>",
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text=x.replace("_", " "), range=[-0.02, 1.02], row=1, col=col_idx)
        fig.update_yaxes(title_text=y.replace("_", " "), range=[-0.02, 1.02], row=1, col=col_idx)
    fig.update_layout(
        title=title,
        height=440,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return _show(fig)


# ---------------------------------------------------------------------------
# Schema + temporal + cardinality (notebook §2)
# ---------------------------------------------------------------------------


def schema_diagram(title: str = "Olist schema — 9 tables, shared keys") -> go.Figure:
    """Render the 9-table Olist schema as boxes + FK arrows.

    Hardcoded layout (the schema is fixed). Box colour encodes role.
    """
    role_color = {
        "transactional": "#4C72B0",
        "dimensional":   "#55A868",
        "free-text":     "#DD8452",
        "geospatial":    "#8172B2",
        "taxonomy":      "#937860",
    }
    boxes = {
        "orders":               (5.0, 5.5, "transactional"),
        "order_items":          (5.0, 3.5, "transactional"),
        "order_reviews":        (8.5, 5.5, "free-text"),
        "order_payments":       (1.5, 5.5, "transactional"),
        "customers":            (1.5, 7.5, "dimensional"),
        "sellers":              (5.0, 1.5, "dimensional"),
        "products":             (8.5, 3.5, "dimensional"),
        "geolocation":          (1.5, 1.5, "geospatial"),
        "category_translation": (8.5, 1.5, "taxonomy"),
    }
    edges = [
        ("orders",         "customers",    "customer_id"),
        ("order_items",    "orders",       "order_id"),
        ("order_items",    "products",     "product_id"),
        ("order_items",    "sellers",      "seller_id"),
        ("order_reviews",  "orders",       "order_id"),
        ("order_payments", "orders",       "order_id"),
        ("products",       "category_translation", "product_category_name"),
        ("customers",      "geolocation",  "zip_prefix"),
        ("sellers",        "geolocation",  "zip_prefix"),
    ]
    box_w, box_h = 1.95, 0.85

    def _edge_point(cx: float, cy: float, tx: float, ty: float) -> tuple[float, float]:
        dx, dy = tx - cx, ty - cy
        if dx == 0 and dy == 0:
            return cx, cy
        half_w, half_h = box_w / 2, box_h / 2
        if dx == 0:
            return cx, cy + (half_h if dy > 0 else -half_h)
        if dy == 0:
            return cx + (half_w if dx > 0 else -half_w), cy
        s = min(half_w / abs(dx), half_h / abs(dy))
        return cx + dx * s, cy + dy * s

    fig = go.Figure()

    # FK edges drawn as annotations with arrowheads.
    for src_name, dst_name, key in edges:
        x1, y1, _ = boxes[src_name]
        x2, y2, _ = boxes[dst_name]
        sx, sy = _edge_point(x1, y1, x2, y2)
        ex, ey = _edge_point(x2, y2, x1, y1)
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor="#7f7f7f", standoff=0,
        )
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        fig.add_annotation(
            x=mx, y=my, text=key, showarrow=False,
            font=dict(size=9, color="black"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0)",
        )

    # Tables drawn as filled rectangle shapes + a label annotation per box.
    for name, (x, y, role) in boxes.items():
        fig.add_shape(
            type="rect",
            x0=x - box_w / 2, y0=y - box_h / 2,
            x1=x + box_w / 2, y1=y + box_h / 2,
            fillcolor=role_color[role], opacity=0.85,
            line=dict(color="black", width=0.8),
            layer="above",
        )
        fig.add_annotation(
            x=x, y=y, text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=11, color="white"),
        )

    # Legend rendered as invisible scatter traces.
    for role, color in role_color.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, opacity=0.85),
                name=role,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        xaxis=dict(range=[0, 10], visible=False),
        yaxis=dict(range=[0.5, 8.5], visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        height=520,
        margin=dict(l=20, r=20, t=70, b=70),
        legend=dict(
            orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
            title=None,
        ),
    )
    return _show(fig)


def temporal_overlap_chart(df: pd.DataFrame, *, title: str = "Temporal coverage of timestamp columns") -> go.Figure:
    """Gantt-style horizontal bars per (table, column) timestamp range.

    `df` is the pandas frame from `data_foundation.temporal_coverage(...).toPandas()`
    — columns: table, column, min_ts, max_ts, n_non_null.
    """
    data = df.copy()
    data["label"] = data["table"] + " · " + data["column"]
    data["min_ts"] = pd.to_datetime(data["min_ts"])
    data["max_ts"] = pd.to_datetime(data["max_ts"])
    data = data.sort_values("min_ts").reset_index(drop=True)

    fig = px.timeline(
        data,
        x_start="min_ts",
        x_end="max_ts",
        y="label",
        color="table",
        color_discrete_sequence=_CAT_PALETTE,
        hover_data=["n_non_null"],
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(
        title=title,
        xaxis_title="date",
        height=max(280, 30 * len(data) + 120),
        margin=dict(l=170, r=40, t=60, b=50),
    )
    return _show(fig)


def shared_key_grouped_bar(df: pd.DataFrame, *, title: str = "Shared-key cardinality across tables") -> go.Figure:
    """Grouped horizontal bar chart: for each shared key, distinct count per table.

    `df` from `data_foundation.shared_key_cardinality(...).toPandas()` —
    columns: shared_key, table, approx_distinct.
    """
    fig = px.bar(
        df,
        x="approx_distinct",
        y="shared_key",
        color="table",
        orientation="h",
        barmode="group",
        color_discrete_sequence=_CAT_PALETTE,
        log_x=True,
    )
    fig.update_layout(
        title=title,
        xaxis_title="approx_count_distinct (log scale)",
        yaxis_title="",
        height=max(320, 60 * df["shared_key"].nunique() + 120),
        margin=dict(l=140, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return _show(fig)


def residual_plot(
    df: pd.DataFrame,
    *,
    y_true: str,
    y_pred: str,
    title: str = "Residuals — actual vs predicted",
) -> go.Figure:
    """Predicted vs actual scatter + residual histogram side-by-side."""
    data = df.copy()
    data["residual"] = data[y_true] - data[y_pred]
    lim = float(max(data[y_true].max(), data[y_pred].max()))
    mean = float(data["residual"].mean())
    std = float(data["residual"].std())

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(title, f"Residual histogram  (μ={mean:.2f}, σ={std:.2f})"),
    )
    fig.add_trace(
        go.Scatter(
            x=data[y_pred], y=data[y_true],
            mode="markers",
            marker=dict(size=5, opacity=0.4, color=_CAT_PALETTE[0]),
            name="observations",
            hovertemplate=f"{y_pred}=%{{x:.2f}}<br>{y_true}=%{{y:.2f}}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, lim], y=[0, lim],
            mode="lines",
            line=dict(color="grey", width=1, dash="dash"),
            name="y = x",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=data["residual"], nbinsx=40,
            marker=dict(color="#d62728", line=dict(color="white", width=0.5)),
            name="residual",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1, row=1, col=2)

    fig.update_xaxes(title_text=y_pred.replace("_", " "), row=1, col=1)
    fig.update_yaxes(title_text=y_true.replace("_", " "), row=1, col=1)
    fig.update_xaxes(title_text="residual  (actual − predicted)", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)
    fig.update_layout(
        height=420,
        margin=dict(l=70, r=40, t=70, b=60),
        showlegend=False,
    )
    return _show(fig)
