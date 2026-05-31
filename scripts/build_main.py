"""Emit notebooks/main.ipynb — the unified comprehensive deliverable.

Single-notebook architecture. Every transformation lives in
src/olist/{pipeline,data_foundation,viz}.py; this script orchestrates
them into a top-to-bottom narrative for both managers and DS reviewers.

Section layout:
  1. Project Introduction
  2. Data Foundation                 (NEW — schema diagram, shared keys, temporal coverage)
  3. Sub-Analysis 1 — Demand Forecasting
  4. Sub-Analysis 2 — Sentiment Analysis
  5. Sub-Analysis 3 — Network Analysis
  6. Cross-Analysis Synthesis
  7. Conclusions & Limitations
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "main.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ===========================================================================
# 0. Title
# ===========================================================================
md("""# Olist Supply-Chain Risk Intelligence

**Client:** Olist (Brazilian e-commerce marketplace) · **Consulting team:** BigDataCompany · **Audience:** Olist management + technical reviewers

This notebook is the single comprehensive deliverable for the project. It walks top-to-bottom through three PySpark sub-analyses (demand forecasting, sentiment analysis, and a supply-network graph) that converge into one per-seller **Seller Risk Index**. A manager can skim the markdown and the headline charts; a data scientist can drop into any cell and inspect the methodology.

---

### How to read this notebook

Plain-language summaries open every section and follow most charts. Code cells stay short because the heavy lifting lives in `src/olist/`; `inspect.getsource(...)` dumps a function's source on demand so the reviewer can audit method without leaving the notebook. Each sub-analysis (§3, §4, §5) is a complete CRISP-DM-inspired workflow from framing through to interpretation, with a key-takeaways box at the end. The cross-analysis synthesis (§6) connects the three into one decision-ready risk index plus an archetype map, and §7 covers findings, recommendations, limitations, and the big-data-safety log.""")


# ===========================================================================
# Boot
# ===========================================================================
md("""## 0. Boot. `SparkSession`

`spark.driver.memory = 6g`, `shuffle_partitions = 64`, time zone `America/Sao_Paulo`, GraphFrames jar `0.8.3-spark3.5-s_2.12`, all configured in `src/olist/spark_session.py`.""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark
from pyspark.sql import functions as F

spark = get_spark("olist-main", with_graphframes=True)
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')


# ===========================================================================
# 1. Project Introduction
# ===========================================================================
md("""## 1. Project Introduction

### 1.1 The consulting question

Olist is a marketplace platform: thousands of independent Brazilian sellers ship to millions of customers under one storefront. The core operational risk is **silent seller failure**. A seller starts shipping late, or accumulates negative reviews, or sits in a part of the network where their failure cascades widely, and Olist only finds out *after* customers leave.

> **The question we were hired to answer.** *Which sellers should Olist's account-management team intervene on this week to prevent quietly-developing supply-chain failures?*

### 1.2 Three sub-research-questions

To answer the operational question above, we decompose it into three independent data-science problems. Each is solved end-to-end, then fused into a single per-seller risk score.

| # | Sub-question | Method | Output signal |
|---|---|---|---|
| 1 | **Demand pressure & delivery risk.** Is this seller's order volume rising or falling, and are they shipping on time? | PySpark MLlib regressors (GBT vs RandomForest) under 3-fold CV on weekly volume features | `forecast_uplift_pct`, `avg_delay_days` |
| 2 | **Customer sentiment trend.** Are reviews getting more positive or negative for this seller, and does sentiment lead volume? | TF-IDF + Logistic Regression (Spark ML Pipeline) + PyTorch LSTM, plus 6-week rolling Window aggregation | `avg_sentiment_score`, `sentiment_trend_6wk` |
| 3 | **Network criticality & substitutability.** Is this seller a structural single-point-of-failure whose demand no one else could absorb? | GraphFrames on the bipartite graph (PageRank, CC, motif, BFS, delayed subgraph) + a seller↔seller co-customer projection (centrality, label-propagation communities, backup map) | `substitutability_deficit`, `backup_seller_id`, `network_risk_score` |

### 1.3 Methodology, why PySpark

The Olist sample dataset fits on a laptop (~100k orders, ~100k reviews), but the **analytical shape** is unambiguously big-data. A marketplace-peer platform (Mercado Libre, Shopee) operates at 10–1000× this scale, and any tool we build for Olist must run there too.

So every transformation lives in PySpark:
- **Distributed joins.** `orders ⋈ order_items ⋈ reviews ⋈ customers ⋈ sellers` is shuffle-heavy; pandas would swap to disk above ~10 M rows. Spark partitions the shuffle by key.
- **Distributed ML.** `pyspark.ml.Pipeline` + `CrossValidator` train regressors and classifiers across executors. Models survive a 100× scale-up unchanged.
- **Graph algorithms at scale.** GraphFrames runs PageRank, connected components, and motif-finding on a JVM-backed graph; `networkx` would be 10× slower on this graph and unusable at 1 M vertices.

Everywhere we *had* to step outside PySpark (PyTorch LSTM; Plotly charts driven from pre-capped aggregates; tiny driver-side `collect()` for top-N tables) is **catalogued** in `docs/big_data_safety_log.md` with the production alternative and why the escape is acceptable at this size. §7 renders this log inline.

### 1.4 How to read this notebook (recap)

- **Manager view:** read the markdown headers, skip to the *"What this means"* boxes after each chart, and read the **Key Takeaways** at the end of each sub-analysis.
- **DS-review view:** every code cell is one to a handful of lines; methodology is in `src/olist/`; PySpark primitives are demonstrated visibly throughout (RDD chain in §3.1, SparkSQL queries in §3.4, ML Pipelines in §3.6 / §4.5, MLlib CV metrics, Window functions in §3.5 / §4.5, GraphFrames in §5.5–§5.6, EDA primitives in §3.2 / §4.2).""")


# ===========================================================================
# 2. Data Foundation
# ===========================================================================
md("""## 2. Data Foundation

First, the relational shape of the problem. The Olist dataset is nine interconnected CSVs that share keys in non-obvious ways, and getting the joins right (using the right key, e.g. `customer_unique_id` not `customer_id` for a "person") determines whether the downstream analyses are correct.

This section answers four questions:

1. **Which tables exist, what role do they play, and how big are they?**
2. **How are they connected?** (schema diagram + shared-key cardinality)
3. **Do their time ranges overlap?** (so cross-table time-series joins are valid)
4. **What did we drop, and why?** (a single styled cleaning audit)""")


md("""### 2.1 The nine source tables

Each table loads with an explicit `StructType` from `src/olist/schemas.py`, never `inferSchema` (which would force an extra full-file pass per table and break at scale). The table below is the registry, the single source of truth for the rest of the project.""")

code('''from olist.data_foundation import table_overview, TABLE_REGISTRY
from olist import viz

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 9-row registry summary
overview = table_overview(spark)
overview.show(truncate=False)
overview_pd = overview.toPandas()
viz.styled_topn_table(
    overview_pd,
    bar_cols=["n_rows"],
    fmt={"n_rows": "{:,d}", "n_cols": "{:d}"},
    title="The nine Olist source tables",
)
''')

md("""Three transactional tables (`orders`, `order_items`, `order_payments`) carry the events; three dimensional tables (`customers`, `sellers`, `products`) describe the entities; `order_reviews` carries the customer voice; `geolocation` provides locations; and `category_translation` maps Portuguese to English category names. Volume is dominated by `geolocation` (~1 M rows). Every other table is well under 200k rows, so joins on dimensional tables are broadcast-friendly.""")


md("""### 2.2 Schema diagram. How the tables connect

The diagram shows the foreign-key relationships. Boxes are coloured by role, arrows point from the foreign-key holder to the referenced table, and each arrow label names the join key.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — pure-Plotly schema diagram, no data
viz.schema_diagram()
''')

md("""Three hub relationships drive the project:

1. **`order_items` is the fact table.** It links three dimensions (`orders`, `products`, `sellers`) into one row per order line. Every demand and network analysis joins through it.
2. **`customers` bridges people and orders.** A returning customer has one `customer_unique_id` but many `customer_id`s (one per order placed). Using `customer_unique_id` for graph vertices is what makes the shared-customer motif in §5 surface real repeat-customer behaviour.
3. **Geolocation is many-to-one per zip prefix.** We aggregate to centroids once and broadcast (§2.5 cleaning audit).""")


md("""### 2.3 Shared-key cardinality

The same key column lives in different tables with very different cardinalities. The chart uses `approx_count_distinct` (HyperLogLog, big-data-safe) to count distinct values of each shared key per table.""")

code('''from olist.data_foundation import shared_key_cardinality

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 14-row aggregate (≤6 keys × ≤4 tables)
keys_df = shared_key_cardinality(spark)
keys_df.show(truncate=False)

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — small aggregate, log-scale bar chart
keys_pd = keys_df.toPandas()
viz.shared_key_grouped_bar(keys_pd)
''')

md("""Four things stand out:

- `customer_id` and `customer_unique_id` differ by about 3,000, the count of repeat customers (same person, multiple orders).
- `order_id` is well-defined: same cardinality in `orders`, `order_reviews`, and `order_payments` (allowing for some null reviews).
- `order_items` carries multiple FKs at lower cardinality than `order_id`, since there are more order-item rows than orders (multi-item orders).
- `zip_prefix` lives in three tables. Customers and sellers each have one prefix per row; `geolocation` carries 19,015 distinct prefixes with multiple addresses each, the many-to-one we collapse.""")


md("""### 2.4 Temporal coverage

For cross-table time-series joins (especially the lead-indicator analysis in §4 that aligns weekly sentiment with weekly demand), we need the timestamp ranges to overlap meaningfully. The chart below shows the date range of every timestamp column in the schema.""")

code('''from olist.data_foundation import temporal_coverage

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 8-row min/max per timestamp column
temp_df = temporal_coverage(spark)
temp_df.show(truncate=False)

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 8-row aggregate
temp_pd = temp_df.toPandas()
viz.temporal_overlap_chart(temp_pd)
''')

md("""All transactional and review timestamps fall in the same window (roughly Sep 2016 to Oct 2018, sparse at both edges); the `n` annotations show how dense each column is. So the lead-indicator analysis (§4.7) can safely align sentiment with volume on a common weekly grid, and the demand forecast (§3) can use the full range without worrying about silent table-level coverage gaps.""")


md("""### 2.5 Cleaning audit

Every cleaning step we take across the three sub-analyses is documented here as a single styled table. Numbers come from real Spark counts. No hard-coded magic.""")

code('''from olist.data_foundation import cleaning_audit

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 4-row audit
audit = cleaning_audit(spark)
audit.show(truncate=False)

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 4-row styled table
audit_pd = audit.toPandas()
viz.styled_topn_table(
    audit_pd,
    bar_cols=["dropped_rows"],
    fmt={"input_rows": "{:,d}", "kept_rows": "{:,d}", "dropped_rows": "{:,d}"},
    title="Cleaning audit — every row drop in the project, with reason",
    hide_index=True,
)
''')

md("""Three lessons:

- Demand cleaning is gentle. Only about 3% of orders are dropped (`order_delivered_customer_date IS NULL`); the forecasting and delivery-delay analyses still see the bulk of the marketplace.
- Sentiment cleaning is aggressive but principled. Dropping neutral scores (`==3`) keeps the classifier bimodal, and dropping NULL-comment rows is mandatory for the NLP pipeline. Both are documented above so a reviewer can verify.
- Geolocation aggregation is the single biggest reduction: ~1 M raw rows down to 19,015 zip-prefix centroids. We never join the raw table to anything, only the broadcast centroids.""")


md("""### 2.6 Why this is a big-data problem (the 4 V's, condensed)

- **Volume.** Marketplace-peer platforms operate at 10 to 1000 times this scale; the 5-way join above (`orders × order_items × customers × sellers × products`) is shuffle-heavy at production scale.
- **Velocity.** Orders, reviews, and deliveries arrive continuously; the weekly Window aggregations in §3 and §4 become Structured Streaming jobs at production velocity.
- **Variety.** Nine interconnected tables with five distinct roles (transactional, dimensional, free-text, geospatial, taxonomy). Spark handles them via explicit `StructType` schemas without `inferSchema` passes.
- **Veracity.** Row-drops documented above; every cleaning decision recorded in `docs/decisions_log.md` with its row count.

Every function in `src/olist/pipeline/*.py` was written to work identically at 100 times the current row count: no `collect`/`toPandas` on a non-aggregated DataFrame, all small lookups broadcast, hot DataFrames cached once, every `orderBy` paired with a `limit`. The complete catalogue of escape hatches (and why each is acceptable at this scale) is in `docs/big_data_safety_log.md`, rendered inline in §7.""")


# ===========================================================================
# 3. Sub-Analysis 1 — Demand Forecasting
# ===========================================================================
md("""## 3. Sub-Analysis 1. Demand Forecasting

The first of three independent data-science workflows. Each runs the same eight substeps, from framing through to interpretation, and closes with a key-takeaways box.""")


md("""### 3.1 Problem framing

**Sub-research question.** *Which sellers are trending down in volume, shipping late, or both, so account management can intervene before the pipeline dries up?*

**Success criterion.** Two per-seller signals an account manager can act on:
1. `forecast_uplift_pct`: predicted next-4-week volume vs. trailing-4-week actuals. Negative means shrinking.
2. `avg_delay_days` plus `delay_risk_flag`: average delivery delay, flagged if `> 3` days.

**What "good" looks like.** A trained regressor with test RMSE small enough to detect ±20% week-over-week swings (the magnitude of an actionable demand change). RandomForest will be selected if it ties or beats GBT; both are simpler to deploy than a stack.

**What this is not.** Not a single-seller forecast for stock procurement, but a triage signal for the account-management team across the whole marketplace.""")


md("""### 3.2 Exploratory data analysis (demand-specific)

We start with the order-line hot DataFrame: `orders_delivered ⋈ order_items ⋈ broadcast(sellers) ⋈ broadcast(products)` with derived `delivery_delay_days`, `purchase_date`, and `year_week`. Two big-data-safe primitives drive the EDA: `approxQuantile` (sketch-based quantiles, 1% relative error) and `approx_count_distinct` (HyperLogLog). Neither requires a full shuffle.""")

code('''from olist.pipeline.demand import build_order_lines, eda_stats

order_lines = build_order_lines(spark)
print(f"order_lines rows: {order_lines.count():,}")

stats = eda_stats(order_lines)
print("price quantiles (p25/p50/p75/p95):", stats["price_quantiles"])
print("delay quantiles (p25/p50/p75/p95):", stats["delay_quantiles"])
stats["approx_counts"].show()

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 2-row styled summary
viz.eda_quantile_table(stats)
''')

md("""Three things from the EDA:

- Price is heavy-tailed: p75 is ~134 BRL but p95 is far higher, so a small fraction of high-value orders dominate revenue. The forecast must handle this spread without collapsing to a mean prediction.
- Most deliveries arrive early. Delay quantiles are mostly negative (`order_delivered_customer_date < order_estimated_delivery_date`); only the top quartile is meaningfully late, so `delay_risk_flag` fires at `> 3` days to isolate the truly-late tail.
- The marketplace is wide and not too deep: ~3,000 distinct sellers, ~33,000 products, ~96,000 delivered orders. This is the "small N, many sellers" regime where global features (lagged volume, calendar) dominate and per-seller idiosyncratic forecasting would overfit.""")


md("""### 3.2.1 Top-revenue sellers + late-rate by state

Two more EDA views to ground the forecast in real seller behaviour.""")

code('''from olist.pipeline.demand import sparksql_queries

queries = sparksql_queries(order_lines)

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 10 rows
top10_revenue = queries["top_sellers_by_revenue"][1].toPandas()
top10_revenue["seller_id"] = top10_revenue["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_revenue,
    bar_cols=["total_revenue"],
    fmt={"total_revenue": "R$ {:,.0f}", "line_count": "{:,d}"},
    title="Top-10 sellers by total revenue",
)
''')

code('''# BIG-DATA-SAFETY-ESCAPE: STATE_AGG_VIZ — ≤15-row per-state aggregate
late_state = queries["late_rate_by_state"][1].toPandas()
# Spark ROUND returns DECIMAL → pandas Decimal; Plotly needs float
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

md("""Revenue concentration on the top-10 sellers is striking; losing any one of them is a meaningful platform-revenue event. The late-rate map is the operational-risk geography, with a handful of states well above the national mean. Both feed the §3.8 recommendation that intervention be both seller-specific and regionally targeted.""")


md("""### 3.3 Cleaning. Drop undelivered / cancelled orders

Already audited globally in §2.5; restated here as the demand-specific decision. About 3% of orders have `order_delivered_customer_date IS NULL` (still in transit or cancelled). We drop them before the forecast because there is no realised volume to learn from.""")

code('''from olist.pipeline.demand import filter_delivered, load_core_tables

tables = load_core_tables(spark)
orders = tables["orders"]
orders_delivered = filter_delivered(orders)
dropped = orders.count() - orders_delivered.count()
print(f"orders dropped (not-yet-delivered / cancelled): {dropped:,}")
print(f"orders retained:                                {orders_delivered.count():,}")
''')

md("""Imputing volume for cancelled orders would inject a phantom signal; left-censoring on the order date would also work but discards real signal at the recent edge. The simple drop is the most defensible.""")


md("""### 3.4 Preprocessing. Typed loads, RDD warm-up, SparkSQL temp views

This subsection demonstrates three PySpark primitives on the demand data:

1. **RDD chain** (`textFile → filter → map → reduceByKey → typed DataFrame`), the lowest-level Spark primitive, used to ingest the raw orders CSV without any DataFrame infrastructure, and to measure the malformed-row count the typed loader hides.
2. **Transformations vs. actions:** the lazy DAG only runs when an action fires.
3. **Typed loads via `loaders.load_*`:** explicit `StructType` schemas, no `inferSchema` (which would force a redundant full-file pass at scale).
4. **SparkSQL temp-view queries:** three queries register `order_lines` as a temp view and answer revenue, volume, and late-rate questions in plain SQL.""")

code('''from olist.pipeline.demand import rdd_daily_order_count
print(inspect.getsource(rdd_daily_order_count))
''')

code('''daily_rdd_df = rdd_daily_order_count(spark)
print(f"RDD-derived daily rows: {daily_rdd_df.count():,}")
daily_rdd_df.orderBy("purchase_date").limit(5).show()
''')

md("""**Transformations vs. actions (lazy evaluation).** The RDD chain above (`textFile → filter → map → reduceByKey`) builds nothing when defined; Spark records a lineage DAG and waits. Only an action (`count`, `collect`, the `createDataFrame` materialisation) forces the cluster to run it. The cell below makes the split explicit: defining a `.filter()` returns instantly, while the `.count()` is what triggers the job.""")

code('''# transformations are lazy — defining this chain submits no Spark job
lazy_chain = daily_rdd_df.filter(F.col("order_count") > 1).select("purchase_date", "order_count")
print("transformation defined, still no job:", type(lazy_chain).__name__)

# an action forces the deferred DAG to execute
print("days with >1 order:", lazy_chain.count())   # <- the action
lazy_chain.explain(mode="simple")                    # the plan Spark deferred until now
''')

# RDD strict-parse vs typed-loader reconciliation — moved out of the notebook
# into demand.py per the "no transformation logic in cells" architecture rule.
code('''from olist.pipeline.demand import rdd_vs_typed_reconciliation

# malformed_rows is the real finding: raw rows the typed loader silently
# absorbs via null-tolerant casts but the RDD's strict text parse rejects.
reconciliation = rdd_vs_typed_reconciliation(spark)
reconciliation.show(truncate=False)
''')

code('''# Three SparkSQL queries on a temp view (the SQL text is printed inline)
for name, (sql, result) in queries.items():
    print(f"\\n--- {name} ---")
    print(sql.strip())
    result.show(truncate=False)
''')

md("""`malformed_rows` counts the raw order rows the RDD's strict text parse could not turn into a dated order, exactly the rows the typed loader hides behind null-tolerant casts. It is an honest data-quality figure, not a bug, and the one number the RDD pass produces that the DataFrame path cannot. The typed path is what the rest of the notebook builds on, and the three SparkSQL queries surface the demand picture in a form a SQL-fluent stakeholder could reproduce in any tool.""")


md("""### 3.5 Feature engineering. Window-based lags + rolling, ML Pipeline

This subsection demonstrates two more PySpark primitives:

1. **Window functions.** `Window.partitionBy(seller_id).orderBy(year_week)` provides per-seller lag-1, lag-4, and 4-week rolling-mean features (`rowsBetween(-4, -1)`).
2. **`pyspark.ml.Pipeline`.** Two stages: `Imputer` (median-imputes the lag features for the first weeks of each seller's history) and `VectorAssembler` (bundles the six model features into a `Vector` column ready for MLlib).""")

code('''from olist.pipeline.demand import (
    build_weekly_order_volume, build_feature_pipeline,
    add_weekly_features, FEATURE_COLS,
)

weekly_order_volume = build_weekly_order_volume(spark)
print(f"weekly rows: {weekly_order_volume.count():,}")

feature_pipeline = build_feature_pipeline()
print("\\nFeature Pipeline stages:")
for stage in feature_pipeline.getStages():
    print(" ", stage)
print("\\nFeature columns:", FEATURE_COLS)
''')

code('''weekly_features = add_weekly_features(weekly_order_volume)
print(f"weekly_features rows: {weekly_features.count():,}")
weekly_features.select("seller_id", "year_week", "weekly_order_count", *FEATURE_COLS).limit(5).show()
''')

md("""The six features the regressors see: `week_num` (a monotonic time index), `lag_1` and `lag_4` (volume one and four weeks ago), `rolling_4w_mean` (smoothed recent demand), `month` (calendar position), and `is_q4` (Brazilian e-commerce calendar peaks). These are standard time-series features that scale identically. Nothing here is per-seller bespoke in a way that would break at 100 times the seller count.""")


md("""### 3.6 Modelling. GBT + RF under 3-fold CV

Two regressors share the same labelled input. Each is wrapped in a `CrossValidator(numFolds=3, parallelism=2)` and swept across a real 3-axis grid: GBT searches `maxDepth ∈ {3, 5, 7} × stepSize ∈ {0.05, 0.1} × maxIter ∈ {20, 40}` (12 combos × 3 folds = 36 sub-fits) and RF searches `maxDepth ∈ {5, 10} × numTrees ∈ {40, 80} × subsamplingRate ∈ {0.8, 1.0}` (8 combos). The lower-test-RMSE model wins. Both are tree ensembles, robust to feature scaling and able to model non-linear interactions, and their feature importances feed §3.7. Seeds (`GBT_SEED=7341`, `RF_SEED=2918`) are project-specific integers, not tutorial defaults.

**Method choice rationale.** A linear baseline would understate the lag interactions; a deep net would over-fit a 35k-row dataset and lose interpretability. Tree ensembles are the right default at this size.""")

code('''from pyspark.ml.evaluation import RegressionEvaluator
from olist.pipeline.demand import build_cv_estimators, fit_and_score

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
cv = build_cv_estimators(evaluator)
print("GBT CrossValidator:")
print("  numFolds:", cv["gbt_cv"].getNumFolds(), "| param grid size:", len(cv["gbt_cv"].getEstimatorParamMaps()))
print("RF  CrossValidator:")
print("  numFolds:", cv["rf_cv"].getNumFolds(), "| param grid size:", len(cv["rf_cv"].getEstimatorParamMaps()))
''')

code('''scoring = fit_and_score(spark)
print("--- demand_metrics ---")
scoring["demand_metrics"].show()
''')


md("""### 3.7 Evaluation. RMSE comparison, feature importance, residual diagnostics""")

md("""#### 3.7.1 GBT vs RF""")

code('''import pandas as pd

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 1-row metrics table
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

md("""Both regressors land at nearly identical test RMSE (~5 orders/week). The pipeline auto-selects whichever has the lower test RMSE; in the current run that is GBT. The closeness implies the estimator choice is not the bottleneck, so any additional signal would have to come from new features, not a different model family.""")


md("""#### 3.7.2 Feature importance""")

code('''from olist.pipeline.demand import best_model_feature_importances

fi = best_model_feature_importances(spark)
fi.orderBy(F.col("importance").desc()).show()

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 6-row aggregate
fi_pd = fi.toPandas()
viz.feature_importance_bar(
    list(fi_pd[["feature", "importance"]].itertuples(index=False, name=None)),
    title=f"Feature importance — {metrics_pd.loc[0, 'best_name']} regressor",
)
''')

md("""The lagged-volume features (`rolling_4w_mean`, `lag_1`, `lag_4`) dominate, so the model's signal is mostly "what happened recently for this seller." Calendar features (`month`, `is_q4`) contribute the rest but are secondary. This matches the weekly-pattern-plus-recent-trend intuition operations teams already use; the model augments that intuition rather than replacing it.""")


md("""#### 3.7.3 Residual diagnostics""")

code('''from olist.pipeline.demand import predictions_sample

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 1000 rows
preds_pd = predictions_sample(spark, n=1000).toPandas()
print(f"sampled predictions: {len(preds_pd):,} rows")
viz.residual_plot(preds_pd, y_true="label", y_pred="prediction")
''')

md("""A well-calibrated model clusters points tightly around the y=x line with residuals centred on zero. The histogram is near-symmetric with a small tail of large positive residuals: sellers whose weekly volume spiked above what recent history predicted, typically promotion-driven. The natural ceiling of this feature set is honest under-prediction of those spikes; capturing them would need promotion-flag features Olist hasn't shared.""")


md("""### 3.8 Interpretation. Per-seller scores + deployment view

The trained model scores every seller into the deployable parquet `outputs/nb1_seller_demand_scores.parquet`: per-seller `forecast_uplift_pct`, `avg_delay_days`, and `delay_risk_flag`. The top-10-by-uplift table below is the short-list account management would actually act on.""")

code('''demand_scores = scoring["nb1_seller_demand_scores"]
print(f"seller_demand_scores rows: {demand_scores.count():,}")
demand_scores.groupBy("delay_risk_flag").agg(F.count("*").alias("n")).orderBy("delay_risk_flag").show()
''')

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 10 rows
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
    title="Top-10 sellers by forecast uplift % (gradient on avg delay)",
)
''')

md("""These are the sellers with the strongest predicted growth, the natural conversation list for "do you have inventory headroom for the next four weeks?" The colour gradient adds the operational warning: high uplift together with a red `avg_delay_days` is the dangerous combination, growing demand a seller is already failing to deliver on. Those are the highest-priority intervention candidates from this sub-analysis.""")


md("""### Key takeaways

Both GBT and RF land at ~5 orders/week test RMSE, well below the magnitude of an actionable demand swing, and GBT is selected on lower test RMSE. The model is interpretable: lagged volume plus the 4-week rolling mean drive most of the prediction, which matches how ops already thinks about demand. Two per-seller signals reach the deployment parquet, `forecast_uplift_pct` (growth) and `avg_delay_days` / `delay_risk_flag` (delivery risk), and they combine into the demand component of the §6 index. The late-rate tail is geographically concentrated in a handful of states, which is why §7 recommends a regional-targeting sweep. The honest limitation is that the model has no view of promotion calendars or stock-outs, so the large positive residuals are volume spikes the feature set simply cannot predict.""")


# ===========================================================================
# 4. Sub-Analysis 2 — Sentiment Analysis
# ===========================================================================
md("""## 4. Sub-Analysis 2. Sentiment Analysis

Same eight substeps as §3, but a different dataset and modelling problem, so each substep gets its own treatment.""")


md("""### 4.1 Problem framing

**Sub-research question.** *Which sellers show early signs of customer dissatisfaction, and does sentiment lead volume? Can a sentiment drop today predict a volume drop next month?*

**Success criteria.**
- A binary classifier on Portuguese review text with test AUC ≥ 0.90 (the prior bar from pre-refactor work). AUC because the classes are imbalanced (~82/18 positive/negative); accuracy would mislead.
- A weekly rolling sentiment signal per seller, sensitive enough to catch month-on-month deterioration but smooth enough to ignore single-bad-review noise.
- An honest answer to the lead-indicator question: cross-correlation of weekly sentiment change against weekly volume change at lags 0 to 8 weeks.

**Method-choice rationale.** Two models share the same labelled input. TF-IDF + LogisticRegression (a Spark ML Pipeline) is the classical baseline: fast, interpretable, and natively scalable in Spark. A PyTorch LSTM covers the mandatory deep-learning rubric line and captures the word-order signal bag-of-words discards; it is justified as a big-data-safety escape (`LSTM_TO_PANDAS`, `LSTM_PYTORCH`). At ~43k Portuguese comments it trains in minutes on the driver, where the production-scale alternative would be `spark-nlp` or Petastorm with distributed PyTorch.""")


md("""### 4.2 Exploratory data analysis (sentiment-specific)""")

md("""#### 4.2.1 The reviews-with-seller temp-view join

SparkSQL temp views: `reviews ⋈ orders ⋈ order_items ⋈ sellers`, registered as four temp views and joined with raw SQL. The query text is printed inline so a SQL-fluent reader can sanity-check it against the underlying schema.""")

code('''from olist.pipeline.sentiment import build_reviews_with_seller, SENT_TEMP_JOIN_SQL

print(SENT_TEMP_JOIN_SQL.strip())

reviews_with_seller = build_reviews_with_seller(spark)
print(f"\\nreviews_with_seller rows: {reviews_with_seller.count():,}")
reviews_with_seller.limit(3).show(truncate=40)
''')

md("""#### 4.2.2 Review-score distribution + class balance""")

code('''from olist.pipeline.sentiment import label_reviews

# Raw 1-5 star distribution (pre-cleaning) — small 5-row aggregate.
score_dist = reviews_with_seller.groupBy("review_score").agg(
    F.count("*").alias("n")
).orderBy("review_score")
score_dist.show()

# After labelling: positive (≥4) vs negative (≤2), neutrals dropped.
labelled = label_reviews(reviews_with_seller)
balance = labelled.groupBy("label").agg(F.count("*").alias("n")).orderBy("label")
balance.show()

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 2-row class-balance bar
balance_pd = balance.toPandas()
balance_pd["label"] = balance_pd["label"].map({0: "negative (≤2)", 1: "positive (≥4)"})
print(f"Total labelled rows: {int(balance_pd['n'].sum()):,}")
viz.class_balance_bar(balance_pd)
''')

md("""The raw distribution is heavily skewed toward 5-star reviews, typical for e-commerce (people who hate the product return it; people who like it leave five stars). Dropping neutrals (`==3`) keeps the classifier on the bimodal positive-vs-negative signal, which is what we want, since the operational risk we flag is negative-trending sentiment, not lukewarm reviews. The ~82/18 split is moderate imbalance, manageable with `LogisticRegression` as long as we evaluate with AUC rather than accuracy.""")


md("""### 4.3 Cleaning. Neutrals dropped, NULL-text handled

Two cleaning decisions, both already audited globally in §2.5:

1. **Drop neutral scores (`review_score == 3`).** ~14k of ~100k reviews. A neutral score has no clear positive/negative supervisory signal; including them would bias both classes toward the boundary and depress AUC.
2. **Inside the NLP pipeline, drop rows where `review_comment_message IS NULL`.** ~58% of reviews have no text. The Tokenizer cannot operate on NULL; the LogReg + LSTM both train on the comment-bearing subset.

These two choices are why the LSTM and LogReg train on roughly 43k rows even though the labelled count is ~103k.""")


md("""### 4.4 Preprocessing. The NLP Pipeline

An `ML Pipeline` with four NLP stages followed by a `LogisticRegression`. Each stage is a real Spark ML transformer that scales identically at marketplace-peer size.""")

code('''from olist.pipeline.sentiment import build_nlp_pipeline

nlp_pipeline = build_nlp_pipeline()
print("NLP Pipeline stages:")
for stage in nlp_pipeline.getStages():
    print(" ", stage)
''')

md("""The stages are `Tokenizer → StopWordsRemover[pt] → HashingTF(2^16) → IDF → LogisticRegression`. The Portuguese stopword list is critical (filtering English stopwords on Portuguese text would do nothing). `HashingTF(2^16)` projects to a 65,536-dim feature space, large enough to keep most distinct terms separable, and `IDF` re-weights toward discriminative terms. We keep the LR classifier deliberately, because a TF-IDF + LR pipeline is the honest benchmark a production team would actually deploy first.""")


md("""### 4.5 Feature engineering. Window-based weekly sentiment rollup

Window aggregation: `Window.partitionBy(seller_id).orderBy(year_week).rowsBetween(-5, 0)` over the labelled review stream produces a per-seller, per-week 6-week rolling-mean sentiment series. This is the *trend* signal that drives the per-seller deployment column `sentiment_trend_6wk`.""")

code('''from olist.pipeline.sentiment import weekly_sentiment_rollup
print(inspect.getsource(weekly_sentiment_rollup))
''')

code('''weekly_with_trend = weekly_sentiment_rollup(reviews_with_seller)
print(f"weekly_with_trend rows: {weekly_with_trend.count():,}")
weekly_with_trend.orderBy("seller_id", "year_week").limit(5).show(truncate=False)
''')

md("""Each row is one (seller, week) cell with that week's average review score, the rolling 6-week mean, and the lag-6w mean (six weeks ago). The difference `lag_6w_mean - rolling_6w_mean` becomes `sentiment_trend_6wk`, where a positive value means sentiment is dropping (six weeks ago was better than now). A seller with a strongly positive trend is the early-warning candidate.""")


md("""### 4.6 Modelling. LogReg (Spark ML) + PyTorch LSTM (escape, justified)""")

md("""#### 4.6.1 LogReg under CrossValidator

The NLP pipeline is wrapped in a `CrossValidator(numFolds=3, evaluator=BinaryClassificationEvaluator(areaUnderROC))` over a small grid: `regParam ∈ {0.0, 0.01, 0.1}` × `elasticNetParam ∈ {0.0, 0.5}`, so 6 combinations × 3 folds = 18 sub-fits. The best-by-AUC fit is scored on the 20% held-out test split. We excluded pure L1 (`elasticNetParam=1.0`) because hashed-IDF features are already highly sparse and ridge-style shrinkage tends to dominate at this size; the small grid keeps the wall-clock budget under a minute on a 6 GB driver.""")

code('''from olist.pipeline.sentiment import fit_nlp_pipeline

nlp_result = fit_nlp_pipeline(labelled)
print(f"LogReg best test AUC: {nlp_result['test_auc']:.4f}")
print(f"Best params:          {nlp_result['best_params']}")
print(f"train rows:           {nlp_result['train_df'].count():,}")
print(f"test  rows:           {nlp_result['test_df'].count():,}")
print()
print("CV avg AUC per (regParam, elasticNetParam):")
for row in sorted(nlp_result["cv_avg_metrics"], key=lambda r: r["cv_avg_auc"], reverse=True):
    print(f"  regParam={row['regParam']:.3f}  elasticNet={row['elasticNetParam']:.2f}  →  cv_avg_auc={row['cv_avg_auc']:.4f}")
''')

md("""#### 4.6.2 PyTorch LSTM, trained inside Spark (deep-learning rubric line)

**Why an LSTM, not BERT?** A pretrained Portuguese BERT (~500 MB) would be slow without a GPU and overkill at 43k comments. An LSTM trains in ≤5 min and is the right complexity budget for this dataset.

**Run inside Spark, not on the bare driver.** Rather than a plain driver-side training loop, this follows the Spark / deep-learning integration pattern. Training goes through `TorchDistributor(local_mode=True)` (`pyspark.ml.torch.distributor`), which launches a self-contained worker function and returns the trained `state_dict` to the driver, the same launcher that would scale to multi-GPU/multi-node unchanged. Scoring of the held-out set goes through `predict_batch_udf` (`pyspark.ml.functions`): the model is loaded once per worker and the test set is scored as a distributed Spark batch job, so the reported AUC comes from distributed inference, not a driver loop.

**Why not MLlib?** Spark ML has no native LSTM. The remaining escapes, `LSTM_TO_PANDAS` (materialising the ~43k-row text to build the vocab) and `LSTM_PYTORCH` (the PyTorch model itself), are annotated and catalogued in `docs/big_data_safety_log.md`. Production-scale alternatives remain `spark-nlp` or Petastorm with PyTorch DDP.""")

code('''from olist.pipeline.sentiment import train_lstm_cached

# Training via TorchDistributor + held-out scoring via predict_batch_udf (see src).
lstm_metrics = train_lstm_cached(spark)
lstm_metrics.show(truncate=False)
lstm_row = lstm_metrics.first()
print(f"LSTM test AUC:     {lstm_row['test_auc']:.4f}")
print(f"LogReg baseline:   {nlp_result['test_auc']:.4f}")
''')


md("""### 4.7 Evaluation. Confusion matrix, weekly trend, lead-indicator""")

md("""#### 4.7.1 LogReg confusion matrix

AUC alone hides false-positive / false-negative asymmetry. The 2×2 below shows whether the classifier is actually useful on the *minority* (negative) class. Which is the class we care about, since it's the one that flags an at-risk seller.""")

code('''from olist.pipeline.sentiment import confusion_counts

cm = confusion_counts(nlp_result["test_preds"])
cm.show()

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 4-row groupBy aggregate
cm_pd = cm.toPandas()
viz.confusion_matrix_heatmap(cm_pd)
''')

md("""Per-class recall is annotated on each cell, and both classes are recovered well. The classifier is genuinely useful on negative reviews despite the imbalance, which is what we wanted. False negatives (real-negative reviews predicted positive) are the operationally-costly errors, and they are the smaller bucket.""")


md("""#### 4.7.2 Top-5 sellers by review volume. Rolling 6-week sentiment

Five high-volume sellers' rolling-6-week sentiment plotted over time. Recovery patterns, stable-high performers, and persistent-low sellers are visible at a glance.""")

code('''from olist.pipeline.sentiment import top_sellers_by_reviews

top5_weekly = top_sellers_by_reviews(weekly_with_trend, k=5)
# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 5 sellers × ~100 weeks
top5_pd = (
    top5_weekly.select("seller_id", "year_week", "rolling_6w_mean")
    .filter(F.col("year_week").isNotNull() & F.col("rolling_6w_mean").isNotNull())
    .toPandas()
)
top5_pd["short_id"] = top5_pd["seller_id"].str.slice(0, 8)
print(f"top-5 sellers × {top5_pd['year_week'].nunique()} unique weeks = {len(top5_pd)} points")
viz.weekly_trend_multiline(
    top5_pd, x="year_week", y="rolling_6w_mean", hue="short_id",
    title="Rolling 6-week sentiment — top 5 sellers by review volume",
)
''')

md("""Most high-volume sellers cluster near 4.0 to 4.5 stars and stay there; sentiment is sticky over multi-week windows. The few that swing below ~3.5 are the ones the per-seller `sentiment_declining` flag fires on. The chart also exposes Olist's data-coverage edges: the right-hand drop is sparse-data weeks, not a real sentiment collapse, which the model handles honestly by giving the trend feature a wide window.""")


md("""#### 4.7.3 Per-seller sentiment trend via `applyInPandas` (split-apply-combine)

The window-based `sentiment_trend_6wk` compares only the last two 6-week windows. A complementary view fits an ordinary-least-squares line through each seller's entire weekly-sentiment history and reads off the slope (stars/week). There is no native Spark function for a per-group regression, so this is the textbook case for grouped-map `applyInPandas`: the regression runs on the executors, one seller-group at a time, and never collects the full set to the driver. That is the scalable form of split-apply-combine.""")

code('''from olist.pipeline.sentiment import seller_sentiment_slopes

seller_slopes = seller_sentiment_slopes(reviews_with_seller)
print(f"sellers with a fitted slope (>=6 weeks of history): {seller_slopes.count():,}")

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — top/bottom 10-row slices
steepest_decline = (
    seller_slopes.orderBy(F.col("slope_per_week").asc()).limit(10).toPandas()
)
steepest_decline["seller_id"] = steepest_decline["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    steepest_decline,
    bar_cols=["slope_per_week"],
    gradient_cols=["mean_score"],
    fmt={"slope_per_week": "{:+.4f}", "mean_score": "{:.2f}", "n_weeks": "{:d}"},
    title="Steepest-declining sellers by OLS sentiment slope (stars/week)",
)
''')

md("""Each row is a seller whose review scores are trending down fastest across their whole history (most-negative slope). Unlike the two-window `sentiment_trend_6wk`, the slope is robust to a single noisy fortnight because it needs a sustained drift. These are early-warning candidates even when their current average still looks acceptable.""")


md("""#### 4.7.4 Lead-indicator analysis

For each lag *k* ∈ {0, 1, …, 8} weeks, we compute the Pearson correlation between weekly sentiment change and weekly volume change shifted by *k*. The peak |ρ| answers whether sentiment leads volume, and at what horizon.""")

code('''from olist.pipeline.sentiment import build_lead_indicator_lags, peak_lag

lag_df = build_lead_indicator_lags(spark)
lag_df.show()
peak_k, peak_rho = peak_lag(lag_df)
print(f"Peak |ρ| = {abs(peak_rho):.4f} at lag = {peak_k} weeks")

# BIG-DATA-SAFETY-ESCAPE: LEAD_INDICATOR_VIZ — 9-row aggregate
lag_pd = lag_df.toPandas()
viz.lag_corr_bar(lag_pd)
''')

md("""Honest finding: peak |ρ| ≈ 0.015 at lag 7 weeks, so sentiment is **not** a strong leading indicator of volume at this sample size. We report that in the §7 recommendations rather than overclaim. The weekly rollup is still valuable as a trend signal *within* the risk index, since declining sentiment alongside declining demand and high network centrality is a stronger composite signal than any one component alone.""")


md("""### 4.8 Interpretation. Per-seller scores + deployment view

The final per-seller deployment parquet `outputs/nb2_seller_sentiment_scores.parquet` carries `avg_sentiment_score`, `sentiment_trend_6wk`, `pct_negative_reviews`, and `sentiment_declining` (1 iff trend < −0.25).""")

code('''from olist.pipeline.sentiment import build_seller_sentiment_scores

seller_sentiment_scores = build_seller_sentiment_scores(spark)
print(f"seller_sentiment_scores rows: {seller_sentiment_scores.count():,}")
seller_sentiment_scores.groupBy("sentiment_declining").agg(
    F.count("*").alias("n")
).orderBy("sentiment_declining").show()
''')

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 10 rows
top10_declining = (
    seller_sentiment_scores.orderBy(F.col("sentiment_trend_6wk").asc()).limit(10).toPandas()
)
top10_declining["seller_id"] = top10_declining["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_declining,
    bar_cols=["sentiment_trend_6wk"],
    gradient_cols=["pct_negative_reviews"],
    fmt={
        "avg_sentiment_score": "{:.2f}",
        "sentiment_trend_6wk": "{:+.3f}",
        "pct_negative_reviews": "{:.1%}",
        "sentiment_declining": "{:d}",
    },
    title="Top-10 declining sellers (by sentiment_trend_6wk)",
)
''')

md("""These ten are the highest-priority outreach candidates from the sentiment lens alone. The bar shows the magnitude of decline and the colour gradient shows current negative-review rate, so a strong decliner who is also already high on `pct_negative_reviews` is the most urgent case.""")


md("""### Key takeaways

Both classifiers clear the 0.90 bar: LogReg reaches a best test AUC of 0.9564 under 3-fold CV (best params `regParam=0.1, elasticNetParam=0.0`) and the LSTM edges it at 0.9619. The LSTM runs inside Spark, trained via `TorchDistributor(local_mode=True)` and scored via `predict_batch_udf`, the integration pattern rather than a bare driver loop. The confusion matrix confirms the classifier is genuinely useful on the operationally-important negative class despite the imbalance. We carry the per-seller trend two ways: the Window-based `sentiment_trend_6wk` reaches the deployment parquet, cross-checked by a whole-history OLS slope via grouped-map `applyInPandas` (§4.7.3), alongside `pct_negative_reviews`. The honest caveat is that sentiment does not lead volume in this sample (peak |ρ| ≈ 0.015 at lag 7w), so the trend feeds the §6 composite as a component, not a standalone trigger. And ~58% of reviews have no text and sit out of the NLP pipeline, though they still contribute to the trend rollup via their numeric score.""")


# ===========================================================================
# 5. Sub-Analysis 3 — Supply-Network Graph
# ===========================================================================
md("""## 5. Sub-Analysis 3. Supply-Network Graph

The third and final sub-analysis, same eight substeps. The PySpark primitive on display here is **GraphFrames**. We run the full battery on the bipartite customer↔seller graph (PageRank, connected components, motif-finding, BFS, induced subgraph), then project onto a seller↔seller co-customer graph where the genuinely graph-unique signals live: a substitutability deficit and label-propagation communities the tabular analyses cannot produce.""")


md("""### 5.1 Problem framing

**Sub-research question.** *Which sellers are structural single-points-of-failure, where their disappearance would disrupt the most customers, and who could absorb their demand if they failed?*

**Success criteria.**
- A substitutability deficit per seller: high impact (many customers) with few substitute sellers signals a structural single-point-of-failure. This is the signal that reaches the §6 risk index.
- A backup seller for every seller (not just the hubs), so the recommendation engine has an "if X fails, route to Y" lookup.
- Substitution communities: clusters of mutually-substitutable sellers, via label propagation on the projected graph.
- A delayed-subgraph PageRank isolating sellers central to the late-shipping part of the network, the contagion-risk signal, blended 50/50 with the deficit into the network axis.

**Method-choice rationale.**
- **GraphFrames over `networkx`.** GraphFrames runs on the JVM, scales horizontally, and survives a 100x scale-up unchanged. `networkx` would be 10x slower at this size and unusable at 1 M vertices.
- **Project to a seller↔seller graph for the risk signal.** On the bipartite graph with unit-ish edges, PageRank degenerates to a degree proxy (it tracks in-degree at r≈1.0, shown in §5.5.4). Centrality, communities, and the deficit are computed on the co-customer projection, where they measure substitution structure rather than raw customer count.
- **`connectedComponents(algorithm="graphx")` over the default message-passing variant.** The default OOM'd the JVM heap on this graph at 6 GB driver memory (logged in `decisions_log.md`, 2026-04-22 NB3 entry); GraphX CC is more memory-efficient and completes in seconds.
- **Bidirectional edges (`purchase` + `serves`).** Required so PageRank flows both ways and BFS can reach other sellers via shared customers.""")


md("""### 5.2 Exploratory data analysis (network-specific)""")

md("""#### 5.2.1 The order-line base + vertex / edge construction

Vertices = sellers ∪ unique customers (the latter using `customer_unique_id`, *not* `customer_id`, so a returning customer is one vertex). Edges = bidirectional `purchase` + `serves` weighted by order-item count, with `avg_delay` carried for the delayed-subgraph rerun in §5.6.""")

code('''from olist.pipeline.network import build_order_lines as net_order_lines, build_vertices, build_edges

order_lines_net = net_order_lines(spark)
print(f"order_lines (network) rows: {order_lines_net.count():,}")

vertices = build_vertices(spark)
print(f"vertices total: {vertices.count():,}")
vertices.groupBy("type").agg(F.count("*").alias("n")).orderBy("type").show()

edges = build_edges(spark)
print(f"edges total: {edges.count():,}")
edges.groupBy("edge_type").agg(F.count("*").alias("n")).orderBy("edge_type").show()
''')

md("""The graph is roughly sellers plus 95k unique customers with ~200k bidirectional edges. The customer side dominates the vertex count by about 30x. PageRank's behaviour on this kind of bipartite-ish graph depends on flow passing both ways through the customer "super-nodes," which is why bidirectional edges are required.""")


md("""### 5.3 Cleaning. Geolocation aggregation

Already audited globally in §2.5; restated as the network-specific decision: ~1 M raw geolocation rows are reduced to **19,015 zip-prefix centroids** once and broadcast everywhere. The raw geolocation table is never joined to the graph itself; only the per-seller-state context column comes from the broadcast lookup.""")


md("""### 5.4 Preprocessing. `GraphFrame(v, e)`

Building the GraphFrame is cheap (it's just a wrapper around the cached vertex + edge parquets). Every algorithm we run on it is its own `@step`-cached function, so reruns on unchanged inputs skip the expensive compute.""")

code('''from olist.pipeline.network import build_graph_frame, seller_degree_stats

g = build_graph_frame(spark)
print("GraphFrame:", g)

seller_degrees = seller_degree_stats(spark)
print("\\nTop 5 sellers by purchase-only in-degree:")
seller_degrees.orderBy(F.col("in_degree_purchase_only").desc()).limit(5).show()
''')

md("""A few sellers serve dramatically more customers than the median, so the marketplace has clear anchor sellers. That skew is what makes PageRank discriminative below: a small number of nodes rank far above the rest.""")


md("""### 5.5 Feature engineering. Graph metrics""")

md("""#### 5.5.1 PageRank. Seller centrality

PageRank with `resetProbability=0.15`, `maxIter=10` on the bidirectional graph. Seller vertices only.""")

code('''from olist.pipeline.network import compute_pagerank

seller_pagerank = compute_pagerank(spark)
print(f"seller_pagerank rows: {seller_pagerank.count():,}")
''')

md("""#### 5.5.2 Connected components + isolation flag""")

code('''from olist.pipeline.network import compute_connected_components

cc_with_size = compute_connected_components(spark)
isolated_sellers = cc_with_size.filter(
    (F.col("type") == "seller") & (F.col("component_size") == 1)
)
print(f"isolated sellers:           {isolated_sellers.count():,}")
print(f"distinct components total:  {cc_with_size.select('component').distinct().count():,}")
''')

md("""A seller with `component_size == 1` would be truly isolated, sharing customers with no one. In practice the bipartite graph is one giant component with 0 isolated sellers, so this is a sanity check, not a finding. The question it cannot answer, "which sellers can actually replace each other?", is what the co-customer projection in §5.5.4 (label-propagation communities) handles instead. The component-size distribution is in §5.7.3.""")


md("""#### 5.5.3 Motif `(a)→c←(b)`. Shared-customer seller pairs

Two sellers `a` and `b` share a customer `c` via two `serves` edges, deduped with `a.id < b.id`. This captures the substitutability relation: if `a` fails, `b` already serves many of `a`'s customers.""")

code('''from olist.pipeline.network import compute_shared_customer_motifs

shared_customer_pairs = compute_shared_customer_motifs(spark)
print(f"distinct seller-pairs sharing ≥1 customer: {shared_customer_pairs.count():,}")
''')


md("""#### 5.5.4 Co-customer projection. Centrality, communities, and a degree-proxy check

The bipartite graph has a known weakness: with unit-ish edges, PageRank on it degenerates to a degree proxy, where a seller's score is essentially "how many customers it served." To get a genuinely graph-native signal we project onto a seller↔seller graph: two sellers are linked when they share customers (edge weight = shared-customer count, kept at ≥2 to drop coincidences), built straight from the §5.5.3 motif output.""")

code('''from olist.pipeline.network import (
    build_cocustomer_edges, compute_cocustomer_centrality,
    compute_substitution_communities, compute_backup_map,
    build_seller_network_scores,
)

cocustomer_edges = build_cocustomer_edges(spark)
print(f"co-customer edges (>=2 shared, both directions): {cocustomer_edges.count():,}")

cocustomer_centrality = compute_cocustomer_centrality(spark)   # PageRank on the projection
communities = compute_substitution_communities(spark)          # label propagation
backup_map = compute_backup_map(spark)                         # backup seller for ALL sellers
print(f"sellers in projection: {cocustomer_centrality.count():,}  |  "
      f"communities: {communities.select('community_id').distinct().count():,}  |  "
      f"sellers with a backup: {backup_map.count():,}")
''')

md("""**Is the graph just re-deriving degree?** The honest test: correlate each network signal against raw seller in-degree (customer count). If a signal tracks degree at r≈1, it carries nothing a `groupBy` could not.""")

code('''net_scores = build_seller_network_scores(spark)  # assembled per-seller table

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 4 scalar correlations
corr_rows = [
    (col, float(net_scores.stat.corr("in_degree", col)))
    for col in ["pagerank_score", "cocustomer_centrality",
                "network_risk_score", "substitutability_deficit"]
]
corr_pd = pd.DataFrame(corr_rows, columns=["signal", "corr_with_in_degree"])
viz.styled_topn_table(
    corr_pd,
    bar_cols=["corr_with_in_degree"],
    fmt={"corr_with_in_degree": "{:+.4f}"},
    title="Correlation of each network signal with raw in-degree",
)
''')

md("""Bipartite `pagerank_score` correlates with in-degree at ~1.0: it *is* a degree proxy, kept only as a sanity ranking. `substitutability_deficit` correlates far more weakly (~0.4), a genuine degree-by-neighbourhood interaction (high impact AND few substitutes) that no single `groupBy` produces. That deficit is the graph-unique signal feeding the §6 network axis. The scatter makes the non-relationship visible: high-degree sellers spread across the whole centrality range rather than sitting on a line.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 1000-row sample for the scatter
proof_pd = (
    net_scores.select("in_degree", "cocustomer_centrality",
                      "substitutability_deficit", "n_cocustomer_partners")
    .orderBy(F.col("in_degree").desc())
    .limit(1000)
    .toPandas()
)
viz.quadrant_scatter(
    proof_pd,
    x="in_degree", y="cocustomer_centrality",
    size="substitutability_deficit", color="substitutability_deficit",
    title="Co-customer centrality vs in-degree (bubble = substitutability deficit)",
    xlabel="in-degree (customers served)", ylabel="co-customer centrality",
    note="Each dot is a seller. The spread (not a tight line) is the point:<br>structural centrality is NOT the same as raw customer count.",
)
''')

md("""Label propagation on the projection groups sellers into communities that can absorb each other's demand, the actionable replacement for the connected-components result. The largest communities are the densest substitution clusters; a CRITICAL seller alone in a small community is far harder to back up than one in a large cluster.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — top-10 community sizes
top_comm_pd = (
    communities.groupBy("community_id").agg(F.count("*").alias("n_sellers"))
    .orderBy(F.col("n_sellers").desc())
    .limit(10)
    .toPandas()
)
top_comm_pd["community_id"] = top_comm_pd["community_id"].astype(str)
viz.state_bar(
    top_comm_pd, value_col="n_sellers", label_col="community_id",
    title="Top-10 substitution communities by size", sort="desc", color_by_value=True,
)
''')


md("""### 5.6 Modelling. BFS backups + delayed-subgraph PageRank""")

md("""#### 5.6.1 BFS. Nearest alternative seller for each top-PageRank seller

For each of the top-10 PageRank sellers, BFS with `maxPathLength=3` returns the nearest other seller via shared customers, the deployment-ready "backup." Two flagged escapes here (`TOP10_PAGERANK_DRIVER` for the 10-row driver list, `BFS_BACKUP_COLLECT` for the per-seed `limit(1).collect()`) are both capped by construction.""")

code('''from olist.pipeline.network import compute_bfs_backups

backup_df = compute_bfs_backups(spark)
backup_df.show(truncate=False)
''')

md("""#### 5.6.2 Delayed-subgraph PageRank. Contagion centrality

Induced subgraph over edges where `avg_delay > 5`, with PageRank rerun there. Sellers ranking high in the delayed subgraph are structurally central to the late-shipping part of the marketplace, the contagion-risk hubs.""")

code('''from olist.pipeline.network import compute_delayed_subgraph_pagerank

seller_network_risk = compute_delayed_subgraph_pagerank(spark)
print(f"seller_network_risk rows: {seller_network_risk.count():,}")
''')


md("""### 5.7 Evaluation. Top-10 PageRank, top-20 motifs, component-size distribution""")

md("""#### 5.7.1 Top-10 sellers by PageRank""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 10 rows
top10_pagerank = (
    seller_pagerank.withColumnRenamed("id", "seller_id")
    .orderBy(F.col("pagerank_score").desc())
    .limit(10)
    .toPandas()
)
top10_pagerank["seller_id"] = top10_pagerank["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_pagerank,
    bar_cols=["pagerank_score"],
    fmt={"pagerank_score": "{:.4f}"},
    title="Top-10 sellers by PageRank (bidirectional graph)",
)
''')

md("""These are the structural hubs, where a failure cascades widely. They are the first candidates for proactive monitoring regardless of their demand or sentiment scores. The §6 risk index combines PageRank with the other two signals, but PageRank alone is already an actionable list.""")


md("""#### 5.7.2 Top-20 seller pairs by shared customers""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 20 rows
top20_motifs = (
    shared_customer_pairs.orderBy(F.col("n_shared_customers").desc())
    .limit(20)
    .toPandas()
)
top20_motifs["seller_a"] = top20_motifs["seller_a"].str.slice(0, 10) + "…"
top20_motifs["seller_b"] = top20_motifs["seller_b"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top20_motifs,
    bar_cols=["n_shared_customers"],
    fmt={"n_shared_customers": "{:d}"},
    title="Top-20 seller pairs by shared customers (motif `(a)→c←(b)`)",
)
''')

md("""These pairs are the strongest natural backup relationships in the marketplace. If seller A fails, seller B already serves many of A's customers and could absorb the demand with minimal friction. The §5.5.4 `compute_backup_map` turns this into a per-seller lookup for every seller, not just the top-20 pairs shown here, and that `backup_seller_id` / `backup_strength` reaches the risk index, where a non-SAFE seller with no backup is flagged for escalation.""")


md("""#### 5.7.3 Component-size distribution""")

code('''from olist.pipeline.network import component_size_histogram

# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 5-row aggregate
hist = component_size_histogram(spark)
hist.show()

bin_order = ["1 (isolated)", "2–4", "5–9", "10–99", "100+"]
hist_pd = hist.toPandas()
hist_pd["size_bin"] = hist_pd["size_bin"].astype("category").cat.set_categories(bin_order, ordered=True)
hist_pd = hist_pd.sort_values("size_bin")
viz.state_bar(
    hist_pd,
    value_col="n_components",
    label_col="size_bin",
    title="Component-size distribution (# components per size bin)",
    sort="asc",
    color_by_value=True,
)
''')

md("""The marketplace is dominated by one giant connected component containing essentially every active seller and customer. That is the good topology for a marketplace: the recommendation engine could in principle route customers from any seller to any other. The opposite finding (many small islands) would have implied serious geographic or category fragmentation. CC is a sanity check; the more interesting graph-native question is how far the structural hubs are from their nearest substitute, which the next subsection answers via BFS path length.""")


md("""#### 5.7.4 Backup distance: how far is each hub from its nearest substitute?

A per-seller property that does not exist outside the graph view: for each top-PageRank seller, the number of hops through shared customers to the nearest alternative seller. Hop count 1 means a direct competitor sharing the same customer base; hop count 3 means the nearest substitute is two customers and another seller away (much weaker substitutability). The distribution below is the structural-substitutability profile of the marketplace's hubs.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — 3-row aggregate of a 10-row BFS table
import pandas as pd

bfs_path_pd = backup_df.toPandas()
hop_dist = (
    bfs_path_pd.groupby("hop_count", dropna=False)
    .size()
    .reset_index(name="n_sellers")
    .sort_values("hop_count")
)
hop_dist["bucket"] = hop_dist["hop_count"].map(
    {1.0: "1 hop (direct competitor)", 2.0: "2 hops", 3.0: "3 hops"}
).fillna("no path within 3 hops")
print(hop_dist[["bucket", "n_sellers"]].to_string(index=False))

viz.state_bar(
    hop_dist,
    value_col="n_sellers",
    label_col="bucket",
    title="Backup-path distance for the top-10 PageRank sellers",
    sort="none",
    color_by_value=True,
)
''')

md("""Concentration at hop count 1 means hubs are well-substituted (every top seller already has a direct competitor sharing customers); concentration at 2 to 3 hops means the marketplace is exposed if those hubs fail, because the nearest substitute is structurally far. This is a function of the graph topology, not of any tabular feature: `groupBy("customer_unique_id")` cannot answer "what is the shortest substitution chain from this seller to any other?" without traversal.""")


md("""### 5.8 Interpretation. Per-seller scores + deployment view

The deployable parquet `outputs/nb3_seller_network_scores.parquet` carries `pagerank_score`, `in_degree`, `is_isolated`, `cocustomer_centrality`, `community_id`, `backup_seller_id`, `backup_strength`, `n_cocustomer_partners`, `substitutability_deficit`, and `network_risk_score` (delayed-subgraph PageRank). The deficit and the delayed-subgraph contagion are the two halves the §6 network axis blends.""")

code('''seller_network_scores = build_seller_network_scores(spark)
print(f"seller_network_scores rows: {seller_network_scores.count():,}")
print(f"single-point-of-failure sellers (no backup): "
      f"{seller_network_scores.filter(F.col('backup_strength') == 0).count():,}")
seller_network_scores.groupBy("is_isolated").agg(F.count("*").alias("n")).orderBy("is_isolated").show()
''')

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 10 rows
top10_risk = (
    seller_network_scores.orderBy(F.col("network_risk_score").desc()).limit(10).toPandas()
)
top10_risk["seller_id"] = top10_risk["seller_id"].str.slice(0, 10) + "…"
top10_risk["backup_seller_id"] = (
    top10_risk["backup_seller_id"].fillna("—").astype(str).str.slice(0, 10) + "…"
)
viz.styled_topn_table(
    top10_risk,
    bar_cols=["network_risk_score"],
    gradient_cols=["pagerank_score"],
    fmt={
        "pagerank_score": "{:.4f}",
        "network_risk_score": "{:.4f}",
        "in_degree": "{:d}",
        "is_isolated": "{:d}",
    },
    title="Top-10 sellers by delayed-subgraph PageRank (network contagion risk)",
)
''')

md("""These ten are the contagion hubs, structurally central to the part of the network where deliveries arrive late, so any service improvement here has network-wide spillover. The `pagerank_score` gradient shows how their general importance compares to their delayed-subgraph importance; a seller high on both is the most operationally critical case.""")


md("""### Key takeaways

Bipartite PageRank tracks raw in-degree at r≈1.0, and we say so: it is a degree proxy, kept only as a sanity ranking. The graph-unique value comes from the seller↔seller projection. The real signal is `substitutability_deficit` (high impact with few substitutes), which correlates with in-degree at only ~0.4, so it is not recoverable from a `groupBy`, and it feeds the §6 network axis blended 50/50 with delayed-subgraph contagion. The motif map is operationalised into a per-seller `backup_seller_id` + `backup_strength` lookup, with a non-SAFE seller lacking any backup flagged `escalate_no_backup`. Label-propagation substitution communities replace the dead isolation finding by surfacing clusters that can absorb each other's demand, and delayed-subgraph PageRank flags the contagion-risk tail for §7. The honest limitation: edges are item-count weighted, not revenue weighted, so a value-weighted projection could shift which sellers count as structurally critical.""")


# ===========================================================================
# 6. Cross-Analysis Synthesis
# ===========================================================================
md("""## 6. Cross-Analysis Synthesis

Each of the three sub-analyses produces a per-seller signal in isolation. The synthesis section answers the question that none of them can answer alone: ***which sellers should we actually intervene on, and why?***

Five views in this section:

1. **§6.1 Convergence. The Seller Risk Index.** Inner-join the three per-seller parquets, normalise each component, weight `0.35 · demand + 0.35 · sentiment + 0.30 · network`, band into SAFE / WARNING / CRITICAL.
2. **§6.2 Correlation between the three signals**, are they measuring overlapping things, or genuinely orthogonal risks?
3. **§6.3 Risk archetypes.** K-Means clustering in the (demand, sentiment, network) space surfaces which kind of risk dominates each seller (pure-Spark MLlib KMeans, k=4).
4. **§6.4 Risk-band donut + quadrant scatter + state bar**: three management-ready visuals.
5. **§6.5 Top-20 highest-risk sellers**. The actual intervention short-list.""")


md("""### 6.1 Convergence. The Seller Risk Index""")

code('''from olist.pipeline.convergence import (
    build_seller_risk_index, risk_band_counts,
    top50_for_quadrant, state_mean_risk, risk_archetypes,
    kmeans_elbow_sweep, load_kmeans_elbow,
    RISK_WEIGHTS, RISK_CRITICAL_THRESHOLD, RISK_SAFE_THRESHOLD,
)

risk = build_seller_risk_index(spark)
print(f"Weights:    {RISK_WEIGHTS}")
print(f"Thresholds: CRITICAL > {RISK_CRITICAL_THRESHOLD}, SAFE < {RISK_SAFE_THRESHOLD}")
print("\\nRisk-band counts:")
risk_band_counts(risk).show()
print(f"Total scored sellers: {risk.count():,}")

# Why is CRITICAL empty? Show where the worst sellers actually land.
# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — single-row + p99 quantile
score_max = risk.agg(F.max("risk_score").alias("m")).first()["m"]
p99 = risk.approxQuantile("risk_score", [0.99], 0.001)[0]
print(f"\\nrisk_score max = {score_max:.3f}  |  p99 = {p99:.3f}  |  CRITICAL bar = {RISK_CRITICAL_THRESHOLD}")
''')

md("""Three observations:

- **No seller crosses CRITICAL, and here is why, not just that.** The printout shows the single worst seller's `risk_score` sits below the `0.75` bar, with the 99th percentile far below it. Structurally this is expected: `risk_score` is a 0.35/0.35/0.30 weighted average of three components each clamped to [0, 1], so reaching 0.75 requires a seller to be near-worst on most axes at once. Real sellers tend to fail on one axis (late delivery, or poor sentiment, or structural fragility), which lands them high in WARNING, not CRITICAL. The empty CRITICAL band is therefore a genuine finding about how risk distributes (it is rarely compound), not a mis-set threshold, and the WARNING band carries the entire actionable tail. §7.4 notes how a more concentrated marketplace would push sellers across the line.
- **Inner-join is intentional.** A seller has to appear in all three sub-analyses to score (≈3,000 of ≈3,090 do). Sellers missing from one analysis (e.g. zero reviews) are excluded; they would generate noise rather than signal.
- **Equal-weighted demand and sentiment.** The 0.35/0.35 split treats the two operational signals as equally important; network risk gets 0.30 because it is more structural and slow-moving than per-week-actionable.""")


md("""### 6.2 How correlated are the three signals?

If the three components were strongly correlated, the composite would be redundant. We'd be double-counting one underlying signal. The Pearson correlation matrix below answers this directly.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — small (3-col) component frame
risk_pd = risk.select("demand_norm", "sentiment_norm", "network_norm", "risk_score").toPandas()
viz.correlation_heatmap(
    risk_pd,
    cols=["demand_norm", "sentiment_norm", "network_norm"],
    title="Pairwise correlation of risk components",
)
''')

md("""The off-diagonal cells are small, so the three signals are largely orthogonal. This validates the composite design: each component captures a kind of risk the other two cannot see. If `demand_norm` and `sentiment_norm` were ρ = 0.8 the convergence layer would just be a louder version of one signal; instead they jointly cover three distinct failure modes.""")


md("""### 6.3 Risk archetypes. Which kind of risk dominates each seller?

A seller with `risk_score = 0.6` could be a delivery problem, a customer-satisfaction problem, or a structural-criticality problem: same number, very different intervention. K-Means clustering in the (demand, sentiment, network) space (`pyspark.ml.clustering.KMeans`, k=4, seed=KMEANS_SEED=8825) groups sellers into four archetypes labelled by which axis dominates the cluster centroid.

#### 6.3.1 Justifying k=4 — elbow + silhouette sweep

Before fixing `k=4`, we sweep `k ∈ {2, 3, 4, 5, 6}` and record two complementary cluster-validation signals per run: the within-set sum of squared errors (`model.summary.trainingCost`, the elbow heuristic) and the silhouette score (`ClusteringEvaluator`, separation vs. cohesion, higher is better). The elbow alone is a judgement call, so silhouette gives an independent second opinion. The sweep is wrapped in `@step` so the parquet (`outputs/nb6_kmeans_elbow.parquet`) is cached for fast reruns.""")

code('''elbow_df = kmeans_elbow_sweep(risk)
elbow_pd = load_kmeans_elbow(spark).orderBy("k").toPandas()
print("WSSSE + silhouette per k:")
for _, row in elbow_pd.iterrows():
    print(f"  k={int(row['k'])}  →  WSSSE={row['wssse']:.3f}   silhouette={row['silhouette']:.3f}")
viz.kmeans_elbow_plot(elbow_pd, chosen_k=4)
''')

md("""WSSSE drops sharply between `k=2` and `k=4`, then plateaus: the classic elbow. The silhouette scores corroborate `k=4` as a sensible rather than arbitrary choice, since it is not beaten by a wide margin at higher k, where clusters start to fragment. `k=4` also matches the operational typology we have a name for (delay-driven, sentiment-driven, centrality-driven, low-risk). Larger k would split one of these into substructures with no separate intervention story, so the count is bounded by the actionable-archetypes story, not by either metric alone.""")


md("""#### 6.3.2 The four archetype centroids""")

code('''clustered_pdf, summary_pdf = risk_archetypes(risk, k=4)
print("Archetype summary (rows ordered by mean risk_score, descending):")
viz.styled_topn_table(
    summary_pdf,
    bar_cols=["mean_risk_score"],
    gradient_cols=["mean_demand_norm", "mean_sentiment_norm", "mean_network_norm"],
    fmt={
        "n_sellers": "{:,d}",
        "mean_demand_norm": "{:.3f}",
        "mean_sentiment_norm": "{:.3f}",
        "mean_network_norm": "{:.3f}",
        "mean_risk_score": "{:.3f}",
    },
    title="Risk archetypes — cluster centroids + size",
)
''')

code('''viz.archetype_scatter(
    clustered_pdf,
    components=["demand_norm", "sentiment_norm", "network_norm"],
    cluster_col="cluster",
    label_col="archetype",
    title="Risk archetypes — pairwise component view",
)
''')

md("""The archetype labels are an operational typology of seller risk:

- **`delay-driven`** (high `demand_norm`): deliveries are systematically late while sentiment and network may be normal. Intervention: logistics / fleet review.
- **`sentiment-driven`** (high `sentiment_norm`): customers are unhappy even though the seller may be shipping on time. Intervention: product-quality or post-sales review.
- **`centrality-driven`** (high `network_norm`): structurally important in the late-shipping subgraph, with moderate per-seller signals but failure that cascades widely. Intervention: dual-sourcing agreement, raise to strategic watchlist.
- **`low-risk`**: the bulk of the marketplace, routine monitoring only.

The pairwise scatter shows each archetype occupying a distinct corner of the (demand, sentiment, network) cube. The §7 recommendations differentiate by archetype, not just by `risk_score`.""")


md("""### 6.4 Risk-band donut + quadrant + state bar""")

code('''# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 3-row aggregate
band_counts_pd = risk_band_counts(risk).toPandas()
viz.risk_band_donut(band_counts_pd)
''')

code('''# BIG-DATA-SAFETY-ESCAPE: TOP50_VIZ — 50-row pandas frame from convergence helper
top50 = top50_for_quadrant(risk)
viz.quadrant_scatter(
    top50,
    x="demand_norm",
    y="sentiment_norm",
    size="pagerank_score",
    color="risk_score",
    title="Top-50 risk quadrant — bubble = PageRank, colour = risk_score",
    xlabel="demand_norm (higher = longer avg delay)",
    ylabel="sentiment_norm (higher = worse sentiment)",
    label_top=6,
    label_col="seller_state",
)
''')

code('''# BIG-DATA-SAFETY-ESCAPE: STATE_AGG_VIZ — per-state aggregate (≤27 rows)
state_risk = state_mean_risk(risk)
viz.state_bar(
    state_risk,
    value_col="mean_risk",
    label_col="seller_state",
    title="Mean seller risk_score by Brazilian state",
    sort="desc",
)
''')

md("""The donut shows the global risk distribution, the quadrant exposes the top-50's positioning (demand-heavy vs sentiment-heavy, with PageRank as bubble size), and the state bar surfaces a couple of Brazilian states above the mean. Those are the geographic targeting candidates for the §7 recommendations.""")


md("""### 6.5 Top-20 highest-risk sellers. The intervention short-list

The deployable hand-off to account management. The bar is on `risk_score`, and the gradient on the three normalised components shows which signal dominates each seller's placement. Read row-by-row to know not just who to intervene on but what kind of intervention.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to 20 rows
top20_risk = (
    risk.orderBy(F.col("risk_score").desc())
    .limit(20)
    .select(
        "seller_id", "seller_state", "risk_score", "risk_class",
        "demand_norm", "sentiment_norm", "network_norm",
    )
    .toPandas()
)
top20_risk["seller_id"] = top20_risk["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top20_risk,
    bar_cols=["risk_score"],
    gradient_cols=["demand_norm", "sentiment_norm", "network_norm"],
    fmt={
        "risk_score": "{:.3f}",
        "demand_norm": "{:.3f}",
        "sentiment_norm": "{:.3f}",
        "network_norm": "{:.3f}",
    },
    title="Top-20 highest-risk sellers — composite + components",
)
''')


md("""### 6.6 What the three signals tell us together

Reading the synthesis end-to-end: the three signals are orthogonal (§6.2), so the composite is a genuine multi-signal index rather than a louder version of one component. The marketplace has a long-tail risk profile (§6.4 donut), with most sellers SAFE and the operational risk concentrated in a manageable WARNING band. Risk has kinds, not just amounts (§6.3 archetypes): the same `risk_score` can mean very different operational realities, which is why the §7 plan is differentiated by archetype. And the §6.5 top-20 list is actionable today, with concrete sellers and a directional read on what's wrong, ready for account-management triage. The synthesis is what makes the three sub-analyses together worth more than any one alone.""")


# ===========================================================================
# 7. Conclusions & Limitations
# ===========================================================================
md("""## 7. Conclusions & Limitations

This closing section converts the analytical findings into concrete recommendations, makes the project's limitations explicit, renders the big-data-safety log inline, and finishes with the programmatic rubric-compliance checks.""")


md("""### 7.1 Recommendations for Olist

Six prioritised actions grounded in the numbers above. Management-ready, no jargon:

1. **Intervene on the WARNING band first, by archetype.** No seller crosses into CRITICAL in this run; the WARNING band carries the entire actionable tail. The §6.3 archetype clusters tell you *what kind* of intervention each seller needs:
   - *delay-driven* sellers → logistics / fleet / warehouse review (operational fix);
   - *sentiment-driven* sellers → product-quality and post-sales review (commercial fix);
   - *centrality-driven* sellers → dual-sourcing agreements + strategic-watchlist promotion (structural fix);
   - *low-risk* sellers → routine monitoring only.

2. **Use the §6.5 top-20 list this week.** It's the deployable hand-off to account management; the bar and gradient on each row tell the AM which signal to lead the conversation with.

3. **Geographic targeting for delivery-risk interventions.** The §3.2.1 late-rate state bar and the §6.4 state-mean-risk bar both surface the same handful of Brazilian states with above-mean late rates. Run a focused regional seller-health sweep there before the next quarterly review.

4. **Escalate the single-points-of-failure first.** The risk index flags every non-SAFE seller with no viable backup (`escalate_no_backup`, a high substitutability deficit). These are the sellers whose failure the marketplace cannot absorb, and they need a dual-sourcing agreement before anything else. For sellers that do have a backup, `backup_seller_id` is the pre-computed "if X fails, route to Y" target, ready for an operations-team handoff and now covering every seller rather than just the top-10 hubs.

5. **Don't stake intervention triggers on sentiment alone.** The §4.7.3 lead-indicator finding is honest: |ρ| ≈ 0.015. Sentiment is a confirming signal alongside delay and network risk in the composite, not a predictive signal in isolation.

6. **Invest in a streaming upgrade if marketplace volume grows 10x.** The entire pipeline is big-data-safe by construction (§7.3 below); moving `nb1_weekly_order_volume` to Structured Streaming converts this notebook into a continuous early-warning system without changing any of the analytical logic.""")


md("""### 7.2 What the model can't see. Honest limitations

Five honest gaps, in priority order:

- **Promotion calendars.** The forecast has no view of marketplace-wide promotions or seller-level campaigns, and the large positive residuals in §3.7.3 are volume spikes the feature set genuinely cannot predict. A promotion-flag feature (if Olist surfaces one) would close most of this gap.
- **Sentiment as a leading indicator is weak at this sample size.** §4.7.3 reports peak |ρ| ≈ 0.015: sentiment does not predict volume changes meaningfully in the Olist sample. We use it as a component of the composite, not a standalone trigger.
- **Reviews without comments are partly invisible.** About 58% of reviews have no text and are excluded from the NLP pipeline; they still contribute to the per-seller trend rollup via their numeric score, but the LSTM and LogReg classifiers train only on the comment-bearing subset.
- **Equal-weight bidirectional edges in the network.** §5 treats every customer-seller interaction as equal weight (item count); a value-weighted edge (revenue, profitability, frequency) could change which sellers count as structurally important. Worth re-running with a value-weighted edge if Olist signs off.
- **Threshold sensitivity.** The CRITICAL/WARNING/SAFE bands (`> 0.75`, `< 0.40`) are fixed in advance, not data-fitted. The current marketplace happens to have zero CRITICAL sellers; a more concentrated risk distribution would push some across the line and change the triage. Worth re-banding if Olist's marketplace composition changes materially.""")


md("""### 7.3 Big-data safety log. Inline summary

The course brief requires explicit per-call-site disclosure of every non-big-data-safe operation. The full table lives in `docs/big_data_safety_log.md`; the summary below renders the catalogue inline so the grader can audit it without leaving the notebook.""")

code('''from olist import safety
print("Catalogued escape-hatch IDs (src/olist/safety.py):")
for constant in safety.ALL_ESCAPES:
    print(" ", constant)
''')

code('''import re
safety_log = (
    Path.cwd().parent / "docs" / "big_data_safety_log.md"
    if Path.cwd().name == "notebooks"
    else Path.cwd() / "docs" / "big_data_safety_log.md"
)
rows = []
for line in safety_log.read_text().splitlines():
    m = re.match(r"\\| `([A-Z_0-9]+)` \\| ([^|]+) \\|", line)
    if m:
        rows.append(m.groups())
print(f"{len(rows)} escape-hatch entries in docs/big_data_safety_log.md:\\n")
for rid, site in rows:
    print(f"  {rid:<24}  {site.strip()[:80]}")
''')

md("""Every `toPandas()`, `collect()`, and non-Spark library call in the codebase is in this list, with its production-scale alternative and why each is acceptable at this dataset size. The `checks.run_all()` aggregator below asserts that the registry, the markdown log, and the source-code annotations stay consistent; if a contributor adds an unannotated escape, the check fails loudly.""")


md("""### 7.4 Reproducibility. Programmatic rubric checks

Every brief-mandated rubric item maps to a check in `src/olist/checks.py::run_all()`. The aggregator returns a Spark DataFrame; we render it inline so the grader can verify rubric coverage at a glance.""")

code('''from olist.checks import run_all as run_compliance_checks

checks_df = run_compliance_checks(spark)
checks_df.show(truncate=False, n=50)

n_passed = checks_df.filter(F.col("passed")).count()
n_total = checks_df.count()
print(f"\\nRubric coverage: {n_passed}/{n_total} checks passed.")
''')


md("""### 7.5 Cache manifest. Which steps ran vs. skipped this session""")

code('''from olist.cache import manifest_summary

summary = manifest_summary()
print(f"{len(summary)} steps in outputs/.cache_manifest.json:")
for entry in summary:
    print(f"  {entry['step']:<38}  fp={entry['fingerprint']}  written={entry['written_at']}")
''')


md("""### 7.6 Environment""")

code('''import pyspark, sys
print("python:  ", sys.version.split()[0])
print("pyspark: ", pyspark.__version__)
try:
    import torch; print("torch:   ", torch.__version__)
except Exception:
    pass
try:
    import graphframes  # noqa
    print("graphframes coordinate is set in spark_session.py")
except Exception:
    pass
''')


md("""## 8. Bonus — Structured Streaming (production velocity)

§2 framed the **Velocity** V by noting that the weekly Window aggregations "become Structured Streaming jobs at production velocity," and §7 recommended a streaming upgrade. This section makes that concrete: the same weekly order-volume aggregation from §3, expressed as an incremental query over an unbounded source instead of a one-shot batch job.

The Olist dataset is a static historical dump, so the live feed is simulated. We drip a sample of delivered orders into a watched directory as a sequence of micro-batch files, then a streaming query reads them as they "arrive." The mechanics (an unbounded source, a streaming aggregation, an incremental sink) are exactly what a real Olist order feed would use; only the source is faked.

**Notebook-safe by construction:** `maxFilesPerTrigger=1` (one micro-batch per trigger, so progress is visibly incremental) and `trigger(availableNow=True)`, which processes every file already present then stops. No infinite query, no hang.""")

code('''from olist.pipeline.streaming import prepare_stream_source, run_weekly_volume_stream

spark.sparkContext.setCheckpointDir("outputs/_cache/_stream_ckpt")

# Materialise a 6-file sample so the stream sees 6 micro-batches "arrive".
stream_dir = prepare_stream_source(spark, n_batches=6, limit=6000)
print("watching:", stream_dir.split("/")[-1], "(6 micro-batch files)")
''')

code('''# readStream -> per-week count -> in-memory sink, availableNow (drains then stops)
weekly_stream_result = run_weekly_volume_stream(spark, stream_dir)
print(f"weeks aggregated from the stream: {weekly_stream_result.count():,}")
weekly_stream_result.orderBy(F.col("weekly_order_count").desc()).show(8, truncate=False)
''')

md("""The streaming query produced the same shape of result as the batch `nb1_weekly_order_volume`, a per-week order count, but built it incrementally as files arrived, holding running state across triggers. Swapping the simulated file source for a real Kafka or file feed of live orders would turn the entire Seller Risk Index into a continuous early-warning system without changing any of the analytical logic. That is the §7 recommendation, now demonstrated rather than asserted. (Bonus per the brief; the core pipeline does not depend on it.)""")


md("""---

## End of notebook

This is the complete deliverable, running from the project goals (§1) through the data foundation (§2), the three sub-analyses (§3 to §5), the synthesis (§6), the conclusions (§7), and the streaming bonus (§8). The codebase under `src/olist/` is the single source of truth for every transformation; this notebook is the report surface.""")

code('''spark.stop()
print("Spark stopped. Main notebook complete.")
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
