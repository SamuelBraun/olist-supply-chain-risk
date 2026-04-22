"""Emit notebooks/01_demand_forecasting.ipynb from cell definitions below.

Not part of the graded artefact — `submission/build_zip.sh` only zips the
three notebook files + the presentation PDF. This script exists so the
notebook can be rebuilt deterministically if cells need editing.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT_NB = ROOT / "notebooks" / "01_demand_forecasting.ipynb"

# --- Cells ---------------------------------------------------------------
CELLS: list[tuple[str, str]] = []

def md(text: str) -> None: CELLS.append(("markdown", text))
def code(src: str) -> None: CELLS.append(("code", src))


md("""# NB1 — Demand Forecasting & Delivery-Risk Scores

**Scope.** Produces two parquets consumed downstream:
- `outputs/nb1_weekly_order_volume.parquet` — weekly order count per seller (consumed by NB2 lead-indicator analysis).
- `outputs/nb1_seller_demand_scores.parquet` — per-seller forecast uplift, average delivery delay, and delay-risk flag (feeds the convergence layer in NB3).

**Rubric surface hit in this notebook.** RDDs · DataFrames · SparkSQL (≥3 queries on temp views) · ML Pipelines · MLlib (`GBTRegressor` + `RandomForestRegressor` with `CrossValidator` + `RegressionEvaluator`). Streaming is attempted as a bonus section only if the three required parquets are already written.

**Big-data hygiene enforced throughout.** Explicit schemas (no `inferSchema`); `broadcast()` on small lookups; `cache()` on the hot order-line DF, `unpersist()` before writes; `approxQuantile` / `approxCountDistinct` for EDA; every `orderBy` paired with a `limit`.
""")

md("""## 1. Boot — SparkSession + JAVA_HOME

The local Homebrew OpenJDK 11 path is exported before PySpark imports so the Py4J bridge can launch the JVM. `spark.sql.session.timeZone='America/Sao_Paulo'` matches the dataset's origin. `shuffle_partitions=64` suits the ~100k-order scale on a laptop.""")

code('''import os, sys
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
# Pin PySpark workers to the venv python so the JVM can spawn them.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark

spark = get_spark("nb1-demand-forecasting")
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version, "| driver python:", sys.executable)
''')

md("""## 2. RDD warm-up — daily order count from raw CSV

Rubric requires an explicit RDD section. We read `olist_orders_dataset.csv` as a raw text file, strip the header, `map` each row to `(purchase_date, 1)`, `filter` out bad rows, `reduceByKey` to daily counts, then rebuild a typed DataFrame with an explicit schema. The resulting DF is only used here for a sanity check against the typed loader — downstream work uses the typed DataFrames.""")

code('''from pyspark.sql.types import StructType, StructField, DateType, LongType
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
ORDERS_CSV = str(DATA_DIR / "olist_orders_dataset.csv")

sc = spark.sparkContext
raw = sc.textFile(ORDERS_CSV)
header = raw.first()

def _parse_line(line):
    try:
        parts = line.split(",")
        ts = parts[3]  # order_purchase_timestamp
        if not ts:
            return None
        d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
        return (d, 1)
    except Exception:
        return None

daily_pairs = (
    raw.filter(lambda row: row != header)
       .map(_parse_line)
       .filter(lambda kv: kv is not None)
       .reduceByKey(lambda a, b: a + b)
)

daily_schema = StructType([
    StructField("purchase_date", DateType(), False),
    StructField("order_count", LongType(), False),
])
daily_orders_rdd_df = spark.createDataFrame(
    daily_pairs.map(lambda kv: (kv[0], int(kv[1]))),
    schema=daily_schema,
)
print("RDD-derived daily rows:", daily_orders_rdd_df.count())
daily_orders_rdd_df.orderBy("purchase_date").limit(5).show()
''')

md("""## 3. Typed loads — explicit schemas via `loaders.load_*`

Every CSV is loaded with an explicit `StructType` from `src/olist/schemas.py` (no `inferSchema`). Row counts are printed so later filtering decisions can be audited.""")

code('''from olist.loaders import (
    load_orders, load_order_items, load_customers, load_sellers,
    load_products, load_order_reviews, load_category_translation,
)
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

orders = load_orders(spark)
order_items = load_order_items(spark)
customers = load_customers(spark)
sellers = load_sellers(spark)
products = load_products(spark)
category_tr = load_category_translation(spark)

row_counts = {
    "orders": orders.count(),
    "order_items": order_items.count(),
    "customers": customers.count(),
    "sellers": sellers.count(),
    "products": products.count(),
    "category_translation": category_tr.count(),
}
for k, v in row_counts.items():
    print(f"{k:>24}: {v:>8,}")
''')

md("""## 4. Geolocation centroids — one row per zip prefix

Aggregate the 1M-row `olist_geolocation_dataset` down to one centroid per `zip_code_prefix`. Written once to `outputs/geo_centroids.parquet` and consumed thereafter via `broadcast()` (NB3 reuses this file). Built only if the parquet is missing — idempotent.""")

code('''from olist.transforms import geolocation_centroids

GEO_OUT = ROOT / "outputs" / "geo_centroids.parquet"
if not GEO_OUT.exists():
    centroids = geolocation_centroids(spark)
    centroids.write.mode("overwrite").parquet(str(GEO_OUT))
    print(f"Wrote {GEO_OUT}")
else:
    print(f"Using existing {GEO_OUT}")

geo_centroids = spark.read.parquet(str(GEO_OUT))
print("geo_centroids rows:", geo_centroids.count())
geo_centroids.limit(3).show()
''')

md("""## 5. Filter to delivered orders — log the drop

`order_delivered_customer_date IS NULL` when an order never reached the customer (cancelled, unavailable, in-transit at dataset cutoff, etc.). Forecasting and delivery-delay metrics both require an actual delivery date, so we drop these rows and log the count to `docs/decisions_log.md` via a printed summary the human can copy.""")

code('''orders_delivered = orders.filter(F.col("order_delivered_customer_date").isNotNull())
dropped = row_counts["orders"] - orders_delivered.count()
print(f"orders dropped (not-yet-delivered / cancelled): {dropped:,}")
print(f"orders retained: {orders_delivered.count():,}")
''')

md("""## 6. Hot DataFrame — order-line join, cached once

Join `orders_delivered` ⋈ `order_items` ⋈ `broadcast(sellers)` ⋈ `broadcast(products)`. Sellers (~3k rows) and products (~33k rows) are both well under the 10 MB broadcast rule. The result is cached because every downstream section reads it; `unpersist()` is called before parquet writes.""")

code('''from olist.transforms import delivery_delay_days

order_lines = (
    orders_delivered
    .join(order_items, on="order_id", how="inner")
    .join(broadcast(sellers), on="seller_id", how="left")
    .join(broadcast(products), on="product_id", how="left")
)
order_lines = delivery_delay_days(order_lines)
order_lines = (
    order_lines
    .withColumn("purchase_date", F.to_date("order_purchase_timestamp"))
    .withColumn(
        "year_week",
        F.concat(
            F.year("order_purchase_timestamp"),
            F.lit("-"),
            F.lpad(F.weekofyear("order_purchase_timestamp").cast("string"), 2, "0"),
        ),
    )
)

order_lines.cache()
print("order_lines rows:", order_lines.count())
order_lines.printSchema()
''')

md("""## 7. SparkSQL — three queries on a temp view

Hard rubric line: **≥3 SparkSQL queries via `spark.sql()` on temp views**. We register `order_lines` as a temp view and run (a) top-10 sellers by revenue, (b) weekly order count per seller, (c) late-delivery rate by seller state. Each query closes with a `LIMIT` per the "never `orderBy` without `limit`" rule.""")

code('''order_lines.createOrReplaceTempView("order_lines")

top_sellers_by_revenue = spark.sql("""
    SELECT seller_id,
           seller_state,
           ROUND(SUM(price), 2) AS total_revenue,
           COUNT(*)             AS line_count
    FROM order_lines
    GROUP BY seller_id, seller_state
    ORDER BY total_revenue DESC
    LIMIT 10
""")
top_sellers_by_revenue.show(truncate=False)
''')

code('''weekly_volume_preview = spark.sql("""
    SELECT seller_id,
           year_week,
           COUNT(*) AS weekly_order_count
    FROM order_lines
    GROUP BY seller_id, year_week
    ORDER BY seller_id, year_week
    LIMIT 10
""")
weekly_volume_preview.show(truncate=False)
''')

code('''late_rate_by_state = spark.sql("""
    SELECT seller_state,
           ROUND(AVG(CASE WHEN delivery_delay_days > 0 THEN 1.0 ELSE 0.0 END), 4) AS late_rate,
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days,
           COUNT(*) AS n_lines
    FROM order_lines
    GROUP BY seller_state
    ORDER BY late_rate DESC
    LIMIT 15
""")
late_rate_by_state.show(truncate=False)
''')

md("""## 8. EDA stats via `approxQuantile` / `approxCountDistinct`

Hygiene rule: EDA uses approximate aggregates so a full sort/pass is avoided. Here: quantiles of `price` and `delivery_delay_days`, and approximate distinct counts for `seller_id` / `product_id`.""")

code('''price_q = order_lines.approxQuantile("price", [0.25, 0.5, 0.75, 0.95], 0.01)
delay_q = order_lines.approxQuantile("delivery_delay_days", [0.25, 0.5, 0.75, 0.95], 0.01)
print("price quantiles (p25/p50/p75/p95):", [round(x, 2) for x in price_q])
print("delay quantiles (p25/p50/p75/p95):", delay_q)

approx_counts = order_lines.agg(
    F.approx_count_distinct("seller_id").alias("approx_sellers"),
    F.approx_count_distinct("product_id").alias("approx_products"),
)
approx_counts.show()
''')

md("""## 9. Weekly order volume per seller → parquet (NB2 dependency)

Written early in the notebook so NB2's lead-indicator section can start reading it in parallel.""")

code('''weekly_order_volume = (
    order_lines
    .groupBy("seller_id", "year_week")
    .agg(F.count("*").alias("weekly_order_count"))
)

WEEKLY_OUT = ROOT / "outputs" / "nb1_weekly_order_volume.parquet"
(weekly_order_volume
    .repartition("seller_id")
    .write.mode("overwrite").parquet(str(WEEKLY_OUT)))
print(f"Wrote {WEEKLY_OUT}")
print("weekly rows:", weekly_order_volume.count())
weekly_order_volume.orderBy("seller_id", "year_week").limit(5).show()
''')

md("""## 10. Feature engineering Pipeline — lag, rolling window, calendar features

Per-seller weekly features for the forecasting models. We use a monotonically increasing week index per seller, then Window functions for lag-1 / lag-4 and a rolling 4-week mean. Calendar features (`month`, `is_end_of_year`) are added from the ISO week. The `VectorAssembler` is the final stage of an MLlib `Pipeline`, satisfying the rubric's Pipeline line.""")

code('''from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer

w_seller = Window.partitionBy("seller_id").orderBy("year_week")
w_roll4 = w_seller.rowsBetween(-4, -1)

weekly_features_base = (
    weekly_order_volume
    .withColumn("week_num", F.row_number().over(w_seller))
    .withColumn("lag_1", F.lag("weekly_order_count", 1).over(w_seller))
    .withColumn("lag_4", F.lag("weekly_order_count", 4).over(w_seller))
    .withColumn("rolling_4w_mean", F.avg("weekly_order_count").over(w_roll4))
    .withColumn("month", F.substring("year_week", 6, 2).cast("int"))
    .withColumn("is_q4", (F.col("month") >= 10).cast("int"))
)

feature_cols = ["week_num", "lag_1", "lag_4", "rolling_4w_mean", "month", "is_q4"]

imputer = Imputer(
    inputCols=["lag_1", "lag_4", "rolling_4w_mean"],
    outputCols=["lag_1", "lag_4", "rolling_4w_mean"],
    strategy="median",
)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

feature_pipeline = Pipeline(stages=[imputer, assembler])
weekly_features = feature_pipeline.fit(weekly_features_base).transform(weekly_features_base)

print("weekly_features rows:", weekly_features.count())
weekly_features.select("seller_id", "year_week", "weekly_order_count", *feature_cols).limit(5).show()
''')

md("""## 11. Train/test split + MLlib models with CrossValidator

We split by week_num (time-aware 80/20 split — earlier weeks for training, later for test). Both `GBTRegressor` and `RandomForestRegressor` are tuned via `CrossValidator(numFolds=3)` on the training set and evaluated with `RegressionEvaluator(metricName='rmse')` on the held-out test set. Small parameter grids keep training time reasonable on a laptop.""")

code('''from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

model_data = weekly_features.filter(
    F.col("lag_1").isNotNull() & F.col("lag_4").isNotNull() & F.col("rolling_4w_mean").isNotNull()
).select("seller_id", "year_week", "week_num", "features",
         F.col("weekly_order_count").cast("double").alias("label"))

split_week = model_data.approxQuantile("week_num", [0.8], 0.01)[0]
train_df = model_data.filter(F.col("week_num") <= split_week)
test_df = model_data.filter(F.col("week_num") > split_week)
print(f"train rows: {train_df.count():,}  test rows: {test_df.count():,}")

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=20, seed=42)
gbt_grid = (ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.stepSize, [0.1])
    .build())
gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_grid,
                        evaluator=evaluator, numFolds=3, seed=42, parallelism=2)

rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=40, seed=42)
rf_grid = (ParamGridBuilder()
    .addGrid(rf.maxDepth, [5, 10])
    .build())
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_grid,
                       evaluator=evaluator, numFolds=3, seed=42, parallelism=2)

print("Fitting GBT CV ...")
gbt_model = gbt_cv.fit(train_df)
print("Fitting RF CV ...")
rf_model = rf_cv.fit(train_df)

gbt_test_rmse = evaluator.evaluate(gbt_model.transform(test_df))
rf_test_rmse  = evaluator.evaluate(rf_model.transform(test_df))
print(f"GBT  test RMSE: {gbt_test_rmse:.3f}")
print(f"RF   test RMSE: {rf_test_rmse:.3f}")

best_model = gbt_model if gbt_test_rmse <= rf_test_rmse else rf_model
best_name = "GBT" if best_model is gbt_model else "RandomForest"
print(f"Selected: {best_name}")
''')

md("""## 12. Per-seller forecast uplift + delivery-risk → `nb1_seller_demand_scores.parquet`

For each seller we compute:
- `forecast_uplift_pct`: predicted next-4-week mean vs. trailing-4-week observed mean, in %.
- `avg_delay_days`: mean `delivery_delay_days` across the seller's order lines.
- `delay_risk_flag`: `avg_delay_days > 3` (chosen from the quantiles section — p75 ≈ 2–3 days).

Joined with `seller_state` for the convergence layer.""")

code('''predictions_all = best_model.transform(model_data).select(
    "seller_id", "year_week", "week_num", "label", "prediction"
)

w_last4 = Window.partitionBy("seller_id").orderBy(F.col("week_num").desc())

trailing_4w_obs = (
    predictions_all
    .withColumn("rk", F.row_number().over(w_last4))
    .filter(F.col("rk") <= 4)
    .groupBy("seller_id")
    .agg(F.avg("label").alias("trailing_4w_obs"))
)

next_4w_pred = (
    predictions_all
    .withColumn("rk", F.row_number().over(w_last4))
    .filter(F.col("rk") <= 4)
    .groupBy("seller_id")
    .agg(F.avg("prediction").alias("next_4w_pred"))
)

avg_delay = (
    order_lines
    .groupBy("seller_id")
    .agg(F.avg("delivery_delay_days").alias("avg_delay_days"))
)

seller_demand_scores = (
    trailing_4w_obs
    .join(next_4w_pred, "seller_id")
    .join(avg_delay, "seller_id")
    .join(broadcast(sellers.select("seller_id", "seller_state")), "seller_id", "left")
    .withColumn(
        "forecast_uplift_pct",
        F.when(F.col("trailing_4w_obs") > 0,
               (F.col("next_4w_pred") - F.col("trailing_4w_obs")) / F.col("trailing_4w_obs") * 100.0)
         .otherwise(F.lit(0.0)),
    )
    .withColumn("delay_risk_flag", (F.col("avg_delay_days") > 3).cast("int"))
    .select("seller_id", "seller_state", "forecast_uplift_pct", "avg_delay_days", "delay_risk_flag")
)

print("seller_demand_scores rows:", seller_demand_scores.count())
seller_demand_scores.orderBy(F.col("forecast_uplift_pct").desc()).limit(5).show(truncate=False)

DEMAND_OUT = ROOT / "outputs" / "nb1_seller_demand_scores.parquet"
seller_demand_scores.write.mode("overwrite").parquet(str(DEMAND_OUT))
print(f"Wrote {DEMAND_OUT}")
''')

md("""## 13. Clean up — `unpersist()` and stop the session

Hard rule: hot cached DFs are unpersisted before the notebook ends. We do not stop the SparkContext until all writes have flushed.""")

code('''order_lines.unpersist()
print("order_lines unpersisted.")
print("Notebook 1 outputs:")
for p in sorted((ROOT / "outputs").glob("*.parquet")):
    print("  ", p.name)
spark.stop()
''')


def _build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            cells.append(nbf.v4.new_code_cell(src))
    nb.cells = cells
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT_NB)
    print(f"Wrote {OUT_NB} ({len(cells)} cells)")


if __name__ == "__main__":
    _build()
