"""Emit notebooks/01_demand_forecasting.ipynb — CRISP-DM edition.

Thin-wrapper: every transformation lives in src/olist/pipeline/demand.py.
Every rendered chart/table goes through src/olist/viz.py.

Structure (CRISP-DM):
  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Evaluation
  6. Deployment
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "01_demand_forecasting.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md("""# NB1 — Demand Forecasting & Delivery-Risk Scores

**Scope.** Produces two consumer parquets for downstream notebooks:
- `outputs/nb1_weekly_order_volume.parquet` — weekly order count per seller (consumed by NB2's lead-indicator section).
- `outputs/nb1_seller_demand_scores.parquet` — per-seller forecast uplift, average delivery delay, delay-risk flag (feeds the convergence layer).

**Rubric surface in this notebook.** RDDs (`textFile → filter → map → reduceByKey → typed DF`) · DataFrames · SparkSQL (≥3 queries on temp views) · ML `Pipeline(Imputer → VectorAssembler)` · MLlib `GBTRegressor` + `RandomForestRegressor` under `CrossValidator(folds=3)` with `RegressionEvaluator(rmse)` · Window functions (lag / rolling) · `approxQuantile` / `approx_count_distinct`.

**Narrative structure.** This notebook follows the **CRISP-DM** methodology — six sections from *Business Understanding* to *Deployment*. Every code cell calls into `src/olist/pipeline/demand.py`; every chart/table goes through `src/olist/viz.py`.""")


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
md("""## 0. Boot — `SparkSession` + `JAVA_HOME`

Local Homebrew OpenJDK 11, `spark.driver.memory=6g`, `shuffle_partitions=64`, time zone `America/Sao_Paulo` — all configured in `src/olist/spark_session.py`.""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark
from pyspark.sql import functions as F

spark = get_spark("nb1-demand-forecasting")
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')


# ---------------------------------------------------------------------------
# 1. Business Understanding
# ---------------------------------------------------------------------------
md("""## 1. Business Understanding

**Why demand forecasting matters to Olist.** A seller's health is the combination of *where the demand is going* and *whether they can deliver it on time*. This notebook builds the **demand component** of the Seller Risk Index:

- **Forecast-uplift signal.** Predicted next-4-week order volume vs. trailing-4-week actuals — is this seller growing, stable, or shrinking?
- **Delay signal.** Average delivery delay (days beyond the estimated date) and a `delay_risk_flag` (1 iff `avg_delay_days > 3`).
- **Structural signal.** Weekly-volume trajectories per seller drive both the forecast and the lead-indicator work in NB2.

**Stakeholder question.** *"Which sellers are trending down in volume, shipping late, or both — so account management can intervene before the pipeline dries up?"* — the per-seller demand parquet written in §6 answers this row-by-row.""")


# ---------------------------------------------------------------------------
# 2. Data Understanding
# ---------------------------------------------------------------------------
md("""## 2. Data Understanding

Seven typed CSVs combine into the hot DataFrame: `orders ⋈ order_items ⋈ broadcast(sellers) ⋈ broadcast(products)`, filtered to *delivered* orders. Every load uses a pre-declared `StructType` from `src/olist/schemas.py` — no `inferSchema=True`, so no extra full-file pass.""")

md("""### 2.1 RDD warm-up — `textFile → filter → map → reduceByKey → typed DF`

The rubric explicitly asks to demonstrate the RDD API even on a dataset this size. We read `olist_orders_dataset.csv` as raw text, strip the header, map each row to `(purchase_date, 1)`, filter bad rows, `reduceByKey`, and rebuild a typed DataFrame. The chain itself is printed below — no duplicated logic.""")

code('''from olist.pipeline.demand import rdd_daily_order_count

print(inspect.getsource(rdd_daily_order_count))
''')

code('''daily_orders_rdd_df = rdd_daily_order_count(spark)
print(f"RDD-derived daily rows: {daily_orders_rdd_df.count():,}")
daily_orders_rdd_df.orderBy("purchase_date").limit(5).show()
''')

md("""### 2.2 Typed loads + row audit""")

code('''from olist.pipeline.demand import load_core_tables

tables = load_core_tables(spark)
for name, df in tables.items():
    print(f"{name:>24}: {df.count():>8,}")
''')

md("""### 2.3 Filter to delivered orders""")

code('''from olist.pipeline.demand import filter_delivered

orders = tables["orders"]
orders_delivered = filter_delivered(orders)
dropped = orders.count() - orders_delivered.count()
print(f"orders dropped (not-yet-delivered / cancelled): {dropped:,}")
print(f"orders retained: {orders_delivered.count():,}")
''')

md("""### 2.4 EDA primitives — `approxQuantile` + `approx_count_distinct`

Both are big-data-safe: quantiles are computed on a sketch (1% relative error here), `approx_count_distinct` is a HyperLogLog sketch — neither requires a full shuffle.""")

code('''from olist.pipeline.demand import build_order_lines, eda_stats
from olist import viz

order_lines = build_order_lines(spark)
print(f"order_lines rows: {order_lines.count():,}")

stats = eda_stats(order_lines)
print("price quantiles (p25/p50/p75/p95):", stats["price_quantiles"])
print("delay quantiles (p25/p50/p75/p95):", stats["delay_quantiles"])
stats["approx_counts"].show()

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — 2-row styled table
viz.eda_quantile_table(stats)
''')

md("""### 2.5 SparkSQL — three queries on a temp view

The rubric asks for ≥3 SparkSQL queries on temp views. We print the SQL text inline so both the query and its result are visible. Two of the three results are rendered as consulting-quality visuals in the next subsections.""")

code('''from olist.pipeline.demand import sparksql_queries

queries = sparksql_queries(order_lines)
for name, (sql, result) in queries.items():
    print(f"\\n--- {name} ---")
    print(sql.strip())
    result.show(truncate=False)
''')

md("""### 2.6 Top-10 sellers by revenue — styled table

The revenue distribution is heavily skewed: a small fraction of sellers produce the majority of GMV. The styled table below embeds a magnitude bar on `total_revenue` so the long-tail shape is visible at a glance.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 10 rows
top10_revenue = queries["top_sellers_by_revenue"][1].toPandas()
top10_revenue["seller_id"] = top10_revenue["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_revenue,
    bar_cols=["total_revenue"],
    fmt={"total_revenue": "R$ {:,.0f}", "line_count": "{:,d}"},
    title="Top-10 sellers by total revenue",
)
''')

md("""### 2.7 Late-delivery rate by state — bar chart

State-level context for the delay signal. The bar is coloured by value (more red = later); 15-state cap comes from the SparkSQL `LIMIT 15` to keep the chart readable.""")

code('''# BIG-DATA-SAFETY-ESCAPE: STATE_AGG_VIZ — ≤15-row per-state aggregate
late_state = queries["late_rate_by_state"][1].toPandas()
# Spark ROUND returns DECIMAL → pandas Decimal; matplotlib needs float
late_state["late_rate"] = late_state["late_rate"].astype(float)
late_state["avg_delay_days"] = late_state["avg_delay_days"].astype(float)
viz.state_bar(
    late_state,
    value_col="late_rate",
    label_col="seller_state",
    title="Late-delivery rate by state (top 15)",
    sort="desc",
)
''')


# ---------------------------------------------------------------------------
# 3. Data Preparation
# ---------------------------------------------------------------------------
md("""## 3. Data Preparation

Two transformations feed the modelling step:

1. **Geolocation centroids.** The raw `olist_geolocation_dataset.csv` has many rows per zip; we reduce to centroids once and broadcast (≤10 MB) to every join that needs it. `@step` caches the parquet.
2. **Weekly order volume per seller** via Window functions, plus lag / rolling / calendar features — the "Pipelines & Data Engineering" rubric surface.""")

md("""### 3.1 Geolocation centroids""")

code('''from olist.pipeline.demand import build_geo_centroids

geo_centroids = build_geo_centroids(spark)
print(f"geo_centroids rows: {geo_centroids.count():,}")
geo_centroids.limit(3).show()
''')

md("""### 3.2 Weekly order volume — parquet consumed by NB2""")

code('''from olist.pipeline.demand import build_weekly_order_volume

weekly_order_volume = build_weekly_order_volume(spark)
print(f"weekly rows: {weekly_order_volume.count():,}")
weekly_order_volume.orderBy("seller_id", "year_week").limit(5).show()
''')

md("""### 3.3 Feature engineering — `Imputer → VectorAssembler` + Window features

Two Pipeline stages (`Imputer` median-imputes `lag_1`, `lag_4`, `rolling_4w_mean`; `VectorAssembler` bundles the six model features). Window primitives: `Window.partitionBy(seller_id).orderBy(year_week)` plus a `rowsBetween(-4, -1)` frame for the rolling mean.""")

code('''from olist.pipeline.demand import build_feature_pipeline, add_weekly_features, FEATURE_COLS

feature_pipeline = build_feature_pipeline()
print("Pipeline stages:")
for stage in feature_pipeline.getStages():
    print(" ", stage)
print("\\nFeature columns:", FEATURE_COLS)
''')

code('''weekly_features = add_weekly_features(weekly_order_volume)
print(f"weekly_features rows: {weekly_features.count():,}")
weekly_features.select("seller_id", "year_week", "weekly_order_count", *FEATURE_COLS).limit(5).show()
''')


# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------
md("""## 4. Modeling

Two regressors share the same labelled input, each wrapped in a `CrossValidator(numFolds=3, parallelism=2, seed=42)` with a small param grid (`maxDepth ∈ {3, 5}` for GBT; `maxDepth ∈ {5, 10}` for RF). The lower-test-RMSE model wins and scores every seller.""")

md("""### 4.1 CrossValidator setup""")

code('''from pyspark.ml.evaluation import RegressionEvaluator
from olist.pipeline.demand import build_cv_estimators

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
cv = build_cv_estimators(evaluator)
print("GBT CrossValidator:")
print("  numFolds:", cv["gbt_cv"].getNumFolds(), "| param grid size:", len(cv["gbt_cv"].getEstimatorParamMaps()))
print("RF  CrossValidator:")
print("  numFolds:", cv["rf_cv"].getNumFolds(), "| param grid size:", len(cv["rf_cv"].getEstimatorParamMaps()))
''')

md("""### 4.2 Fit + score (`@step`-cached)

`fit_and_score` fits both CVs, picks the winner by test RMSE, writes three cache parquets (`demand_predictions`, `demand_metrics`, `demand_feature_importances`) and the committed consumer `nb1_seller_demand_scores.parquet`. Warm reruns skip this entirely.""")

code('''from olist.pipeline.demand import fit_and_score

scoring = fit_and_score(spark)
print("--- demand_metrics ---")
scoring["demand_metrics"].show()
''')


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
md("""## 5. Evaluation

Three diagnostic views: GBT-vs-RF comparison (styled), feature importance of the winning model (bar), and a residual plot to show how prediction error is distributed.""")

md("""### 5.1 GBT vs RF — test-RMSE comparison

Both regressors land at nearly the same test RMSE (~5.06–5.09 orders/week). RF edges out GBT by a hair and is faster to re-score, so it is the deployed model for the per-seller scores.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — 1-row metrics table
import pandas as pd

metrics_pd = scoring["demand_metrics"].toPandas()
comparison = pd.DataFrame({
    "model": ["GBTRegressor", "RandomForestRegressor"],
    "test_rmse": [float(metrics_pd.loc[0, "gbt_rmse"]), float(metrics_pd.loc[0, "rf_rmse"])],
    "selected": [
        "✓" if metrics_pd.loc[0, "best_name"] == "GBT" else "",
        "✓" if metrics_pd.loc[0, "best_name"] == "RandomForest" else "",
    ],
})
viz.styled_topn_table(
    comparison,
    bar_cols=["test_rmse"],
    fmt={"test_rmse": "{:.3f}"},
    title="GBT vs RandomForest — test RMSE",
)
''')

md("""### 5.2 Feature importance of the winning model

Feature importances from the fitted best model (read from the cached `demand_feature_importances.parquet`). The lagged-volume features dominate — the model's signal is mostly "what happened last week / last month", with calendar features (month, `is_q4`) contributing the remaining explanatory power.""")

code('''from olist.pipeline.demand import best_model_feature_importances

fi = best_model_feature_importances(spark)
fi.orderBy(F.col("importance").desc()).show()

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — 6-row aggregate
fi_pd = fi.toPandas()
viz.feature_importance_bar(
    list(fi_pd[["feature", "importance"]].itertuples(index=False, name=None)),
    title=f"Feature importance — {metrics_pd.loc[0, 'best_name']} regressor",
)
''')

md("""### 5.3 Residual plot — actual vs predicted + histogram

A 1000-row sample of the full-model predictions is pulled to pandas for the two-panel diagnostic. A well-calibrated model clusters tightly around the y=x line and produces residuals centred on zero with no obvious heteroskedasticity. Large residual outliers are sellers whose weekly volume spikes unpredictably — the model's natural ceiling at this feature set.""")

code('''from olist.pipeline.demand import predictions_sample

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 1000 rows
preds_pd = predictions_sample(spark, n=1000).toPandas()
print(f"sampled predictions: {len(preds_pd):,} rows")
viz.residual_plot(preds_pd, y_true="label", y_pred="prediction")
''')


# ---------------------------------------------------------------------------
# 6. Deployment
# ---------------------------------------------------------------------------
md("""## 6. Deployment

The deployable artefacts are two committed parquets — one feeds NB2, one feeds the convergence layer.""")

md("""### 6.1 Per-seller demand scores""")

code('''demand_scores = scoring["nb1_seller_demand_scores"]
print(f"seller_demand_scores rows: {demand_scores.count():,}")
demand_scores.groupBy("delay_risk_flag").agg(F.count("*").alias("n")).orderBy("delay_risk_flag").show()
''')

md("""### 6.2 Top-10 sellers by forecast uplift — deployment-ready table

These are the sellers with the strongest predicted growth — the short-list for inventory-ramp-up conversations with the operations team. Bar-embedded on the uplift column, gradient on `avg_delay_days` (redder = later — growth + delay is the dangerous combination).""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 10 rows
top10_uplift = (
    demand_scores.orderBy(F.col("forecast_uplift_pct").desc()).limit(10).toPandas()
)
top10_uplift["seller_id"] = top10_uplift["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_uplift,
    bar_cols=["forecast_uplift_pct"],
    gradient_cols=["avg_delay_days"],
    fmt={
        "forecast_uplift_pct": "{:+.1f}%",
        "avg_delay_days": "{:+.1f}",
        "delay_risk_flag": "{:d}",
    },
    title="Top-10 sellers by forecast uplift %",
)
''')

md("""### 6.3 Clean up""")

code('''ROOT_OUT = Path.cwd().parent / "outputs" if Path.cwd().name == "notebooks" else Path.cwd() / "outputs"
print("Notebook 1 outputs (committed):")
for path in sorted(ROOT_OUT.glob("*.parquet")):
    print(" ", path.name)
spark.stop()
print("\\nSpark stopped.")
''')


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9.6"}
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    with OUT_NB.open("w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT_NB.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
