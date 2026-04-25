"""Static visualisation helpers for the Olist notebooks.

Every helper here takes a *small* pandas DataFrame or dict (sourced from
a pre-aggregated / pre-capped Spark aggregate) and returns a
`matplotlib.figure.Figure` or a `pandas.io.formats.style.Styler`.
Helpers never touch Spark, never do file I/O, and never materialise raw
data — that contract is the module's reason for existing. Driver-side
materialisation happens exactly once upstream, annotated with
`# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ` (or a more specific
ID). See `docs/big_data_safety_log.md`.

Palette:
* Categorical: seaborn ``"deep"`` (colour-blind-safe, print-friendly).
* Sequential risk / heat: matplotlib ``"Reds"``.
* Diverging (residuals): matplotlib ``"RdBu_r"``.

All helpers add a title, axis labels, grid, and call `tight_layout()`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

_CAT_PALETTE = sns.color_palette("deep")
_RISK_CMAP = "Reds"


# ---------------------------------------------------------------------------
# Styled tables
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
    """Render ``df`` (≤50 rows) as a consulting-quality styled table.

    ``bar_cols``        → embedded horizontal bars (for magnitudes).
    ``gradient_cols``   → background colour gradient (for risk scores).
    ``fmt``             → per-column format string (e.g. ``{"revenue": "R$ {:,.0f}"}``).
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
        styler = styler.background_gradient(subset=list(gradient_cols), cmap=_RISK_CMAP)
    styler = styler.set_table_styles(
        [
            {"selector": "caption", "props": "font-size: 1.1em; font-weight: 600; text-align: left; padding-bottom: 0.4em;"},
            {"selector": "th", "props": "background-color: #f0f0f0; text-align: left;"},
            {"selector": "td", "props": "padding: 4px 8px;"},
        ]
    )
    return styler


def eda_quantile_table(stats: dict) -> "pd.io.formats.style.Styler":
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
    styler = styler.background_gradient(subset=["p25", "p50", "p75", "p95"], cmap="Blues", axis=1)
    styler = styler.set_table_styles(
        [
            {"selector": "caption", "props": "font-size: 1em; text-align: left; padding-bottom: 0.4em;"},
            {"selector": "th", "props": "background-color: #f0f0f0;"},
        ]
    )
    return styler


# ---------------------------------------------------------------------------
# Bars / distributions
# ---------------------------------------------------------------------------


def class_balance_bar(df: pd.DataFrame, label_col: str = "label", n_col: str = "n") -> Figure:
    """Stacked horizontal bar of class counts with percentage labels.

    Intended for the binary-sentiment class balance (positive / negative).
    """
    total = float(df[n_col].sum())
    fig, ax = plt.subplots(figsize=(6, 1.6))
    colors = [_CAT_PALETTE[2], _CAT_PALETTE[3]]  # green / red-ish
    left = 0.0
    for (_, row), color in zip(df.iterrows(), colors):
        pct = row[n_col] / total * 100
        ax.barh(0, row[n_col], left=left, color=color, edgecolor="white", linewidth=1.2)
        ax.text(
            left + row[n_col] / 2,
            0,
            f"{row[label_col]}\n{int(row[n_col]):,} ({pct:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        left += row[n_col]
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("reviews")
    ax.set_title("Review sentiment class balance")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def state_bar(
    df: pd.DataFrame,
    value_col: str,
    *,
    label_col: str = "seller_state",
    title: str | None = None,
    color_by_value: bool = True,
    sort: str = "desc",
) -> Figure:
    """Horizontal bar chart of a per-state metric (≤27 rows)."""
    data = df.copy()
    ascending = sort == "asc"
    data = data.sort_values(value_col, ascending=ascending).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(data) + 1)))
    if color_by_value:
        norm = plt.Normalize(vmin=data[value_col].min(), vmax=data[value_col].max())
        cmap = plt.get_cmap(_RISK_CMAP)
        colors = [cmap(norm(v)) for v in data[value_col]]
    else:
        colors = [_CAT_PALETTE[0]] * len(data)
    ax.barh(data[label_col], data[value_col], color=colors, edgecolor="white")
    for i, v in enumerate(data[value_col]):
        ax.text(v, i, f"  {v:,.3f}" if v < 1 else f"  {v:,.1f}", va="center", fontsize=9)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_ylabel(label_col.replace("_", " "))
    ax.set_title(title or f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ')}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def feature_importance_bar(pairs: Iterable[tuple[str, float]], title: str = "Feature importance") -> Figure:
    """Horizontal bar of (feature, importance) pairs, largest on top."""
    data = sorted(list(pairs), key=lambda p: p[1], reverse=True)
    names = [p[0] for p in data]
    values = [p[1] for p in data]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(data) + 1)))
    ax.barh(names[::-1], values[::-1], color=_CAT_PALETTE[0], edgecolor="white")
    for i, v in enumerate(values[::-1]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def lag_corr_bar(df: pd.DataFrame, *, lag_col: str = "lag", corr_col: str = "corr") -> Figure:
    """9-row lag-vs-correlation chart with peak annotation.

    ``df`` comes from ``pipeline.sentiment.build_lead_indicator_lags``.
    """
    data = df.sort_values(lag_col).copy()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [_CAT_PALETTE[3] if v < 0 else _CAT_PALETTE[0] for v in data[corr_col]]
    ax.bar(data[lag_col], data[corr_col], color=colors, edgecolor="white")
    ax.axhline(0, color="grey", linewidth=0.8)

    peak_idx = data[corr_col].abs().idxmax()
    peak_lag = int(data.loc[peak_idx, lag_col])
    peak_corr = float(data.loc[peak_idx, corr_col])
    ax.annotate(
        f"peak |ρ|={abs(peak_corr):.4f}\nat lag={peak_lag}w",
        xy=(peak_lag, peak_corr),
        xytext=(peak_lag, peak_corr + (0.005 if peak_corr >= 0 else -0.005)),
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax.set_xlabel("Lag k (weeks)")
    ax.set_ylabel("Pearson ρ")
    ax.set_title("Does sentiment decline precede volume decline?")
    ax.set_xticks(data[lag_col])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmaps / scatter
# ---------------------------------------------------------------------------


def heatmap_from_long(
    df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    cmap: str = _RISK_CMAP,
    fmt: str = ".2f",
    title: str | None = None,
) -> Figure:
    """Pivot a long DataFrame and render it as a seaborn heatmap."""
    wide = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(wide.columns) + 2), max(3, 0.35 * len(wide) + 1)))
    sns.heatmap(
        wide,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": values},
    )
    ax.set_title(title or f"{values} by {index} × {columns}")
    fig.tight_layout()
    return fig


def risk_band_donut(df: pd.DataFrame, *, label_col: str = "risk_class", n_col: str = "n") -> Figure:
    """Donut chart of risk-band counts with percentage labels."""
    colors = {"SAFE": "#55A868", "WARNING": "#DD8452", "CRITICAL": "#C44E52"}
    pie_colors = [colors.get(c, "#888") for c in df[label_col]]
    total = int(df[n_col].sum())

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    wedges, _ = ax.pie(
        df[n_col],
        colors=pie_colors,
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
    )
    for w, cls, n in zip(wedges, df[label_col], df[n_col]):
        ang = (w.theta2 + w.theta1) / 2
        x = 0.82 * np.cos(np.deg2rad(ang))
        y = 0.82 * np.sin(np.deg2rad(ang))
        pct = n / total * 100 if total else 0
        ax.text(x, y, f"{cls}\n{int(n):,}\n({pct:.1f}%)", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0, 0, f"Total\n{total:,}\nsellers", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Seller risk-band distribution")
    fig.tight_layout()
    return fig


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
) -> Figure:
    """Bubble scatter for top-N seller risk views.

    Bubble area scales with ``size`` column, colour with ``color`` column.
    Adds median-crosshairs to split the plot into four quadrants.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    size_max = float(df[size].max()) or 1.0
    sizes = 30 + 3500 * (df[size] / size_max)
    sc = ax.scatter(
        df[x],
        df[y],
        s=sizes,
        c=df[color],
        cmap=_RISK_CMAP,
        alpha=0.78,
        edgecolors="black",
        linewidths=0.6,
    )
    ax.axhline(df[y].median(), color="grey", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.axvline(df[x].median(), color="grey", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_xlabel(xlabel or x.replace("_", " "))
    ax.set_ylabel(ylabel or y.replace("_", " "))
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color.replace("_", " "))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


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
) -> Figure:
    """Multi-line chart of a weekly metric across a small set of entities.

    ``df`` is a capped long DataFrame (one row per (hue, x)). ``hue`` is
    typically ``seller_id``; the caller caps the unique hue count to ~5.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("deep", n_colors=df[hue].nunique())
    for color, (name, sub) in zip(palette, df.sort_values(x).groupby(hue)):
        ax.plot(sub[x], sub[y], label=str(name)[:12], color=color, linewidth=1.8, alpha=0.9)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title)
    ax.legend(title=hue.replace("_", " "), loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def confusion_matrix_heatmap(
    counts: pd.DataFrame,
    *,
    y_true: str = "label",
    y_pred: str = "prediction",
    n_col: str = "n",
    class_labels: Sequence[str] = ("negative (0)", "positive (1)"),
    title: str = "Confusion matrix — test set",
) -> Figure:
    """Render a 2×2 confusion matrix from a 4-row groupBy count."""
    pivot = counts.pivot_table(index=y_true, columns=y_pred, values=n_col, aggfunc="sum").fillna(0.0)
    pivot = pivot.reindex(index=[0, 1], columns=[0, 1], fill_value=0.0)
    total = pivot.values.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        row_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
    annot = np.empty_like(pivot.values, dtype=object)
    for i in range(2):
        for j in range(2):
            n = int(pivot.values[i, j])
            pct = row_norm.values[i, j] * 100
            annot[i, j] = f"{n:,}\n({pct:.1f}%)"
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=0.4,
        linecolor="white",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "n reviews"},
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{title}  (n={int(total):,})")
    fig.tight_layout()
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    title: str = "Pearson correlation",
    cmap: str = "RdBu_r",
) -> Figure:
    """Compute and render an NxN Pearson correlation matrix as a heatmap.

    Diverging colour map centred on 0 so positive (+1) and negative (-1)
    correlations are visually distinct.
    """
    corr = df[list(cols)].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(cols) + 2), max(3.2, 0.7 * len(cols) + 1.5)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pearson ρ"},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def archetype_scatter(
    df: pd.DataFrame,
    *,
    components: Sequence[str],
    cluster_col: str = "cluster",
    label_col: str = "archetype",
    title: str = "Risk archetypes — pairwise component view",
) -> Figure:
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
    palette = sns.color_palette("deep", n_colors=len(clusters_sorted))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (x, y) in zip(axes, pairs):
        for color, c in zip(palette, clusters_sorted):
            sub = df[df[cluster_col] == c]
            ax.scatter(
                sub[x], sub[y],
                color=color, alpha=0.55, s=18, edgecolors="none",
                label=f"{cluster_to_label[c]} (n={len(sub):,})",
            )
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def schema_diagram(title: str = "Olist schema — 9 tables, shared keys") -> Figure:
    """Render the 9-table Olist schema as boxes + FK arrows.

    Hardcoded layout (the schema is fixed). Box colour encodes role
    (transactional / dimensional / free-text / geospatial / taxonomy).
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

    fig, ax = plt.subplots(figsize=(11, 7.5))
    box_w, box_h = 1.7, 0.7
    for name, (x, y, role) in boxes.items():
        ax.add_patch(
            plt.Rectangle(
                (x - box_w / 2, y - box_h / 2),
                box_w, box_h,
                facecolor=role_color[role],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
            )
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

    for src_name, dst_name, key in edges:
        x1, y1, _ = boxes[src_name]
        x2, y2, _ = boxes[dst_name]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.9, alpha=0.7),
        )
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        ax.text(midx, midy, key, fontsize=7.5, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
                ha="center", va="center")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", alpha=0.85, label=role)
        for role, color in role_color.items()
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(role_color), frameon=False, fontsize=9)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def temporal_overlap_chart(df: pd.DataFrame, *, title: str = "Temporal coverage of timestamp columns") -> Figure:
    """Gantt-style horizontal bars per (table, column) timestamp range.

    `df` is the pandas frame returned by `data_foundation.temporal_coverage(...).toPandas()`
    — columns: table, column, min_ts, max_ts, n_non_null.
    """
    data = df.copy()
    data["label"] = data["table"] + " · " + data["column"]
    data["min_ts"] = pd.to_datetime(data["min_ts"])
    data["max_ts"] = pd.to_datetime(data["max_ts"])
    data = data.sort_values("min_ts").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(data) + 1)))
    palette = sns.color_palette("deep", n_colors=data["table"].nunique())
    table_to_color = {t: palette[i] for i, t in enumerate(sorted(data["table"].unique()))}
    for i, row in data.iterrows():
        width = (row["max_ts"] - row["min_ts"]).days
        ax.barh(i, width, left=row["min_ts"], color=table_to_color[row["table"]],
                edgecolor="white", height=0.7)
        ax.text(row["max_ts"], i, f"  n={int(row['n_non_null']):,}", va="center", fontsize=8)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("date")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def shared_key_grouped_bar(df: pd.DataFrame, *, title: str = "Shared-key cardinality across tables") -> Figure:
    """Grouped horizontal bar chart: for each shared key, distinct count per table.

    `df` from `data_foundation.shared_key_cardinality(...).toPandas()` —
    columns: shared_key, table, approx_distinct.
    """
    keys = sorted(df["shared_key"].unique())
    tables = sorted(df["table"].unique())
    table_to_color = {t: c for t, c in zip(tables, sns.color_palette("deep", n_colors=len(tables)))}

    fig, ax = plt.subplots(figsize=(8.5, max(3, 0.45 * len(keys) * len(tables) + 1)))
    bar_h = 0.8 / max(len(tables), 1)
    y_positions = list(range(len(keys)))
    for ti, t in enumerate(tables):
        sub = df[df["table"] == t].set_index("shared_key").reindex(keys)
        offsets = [y + (ti - (len(tables) - 1) / 2) * bar_h for y in y_positions]
        ax.barh(offsets, sub["approx_distinct"].fillna(0).values,
                height=bar_h, color=table_to_color[t], edgecolor="white", label=t)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("approx_count_distinct (log scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def residual_plot(
    df: pd.DataFrame,
    *,
    y_true: str,
    y_pred: str,
    title: str = "Residuals — actual vs predicted",
) -> Figure:
    """Predicted vs actual scatter + residual histogram side-by-side."""
    data = df.copy()
    data["residual"] = data[y_true] - data[y_pred]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})

    ax0 = axes[0]
    lim = float(max(data[y_true].max(), data[y_pred].max()))
    ax0.scatter(data[y_pred], data[y_true], alpha=0.35, s=14, color=_CAT_PALETTE[0], edgecolors="none")
    ax0.plot([0, lim], [0, lim], color="grey", linewidth=0.8, linestyle="--")
    ax0.set_xlabel(y_pred.replace("_", " "))
    ax0.set_ylabel(y_true.replace("_", " "))
    ax0.set_title(title)
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    ax1.hist(data["residual"], bins=40, color=_CAT_PALETTE[3], edgecolor="white")
    ax1.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    mean = float(data["residual"].mean())
    std = float(data["residual"].std())
    ax1.set_xlabel("residual  (actual − predicted)")
    ax1.set_ylabel("count")
    ax1.set_title(f"Residual histogram  (μ={mean:.2f}, σ={std:.2f})")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    return fig
