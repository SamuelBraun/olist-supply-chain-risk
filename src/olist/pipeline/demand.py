"""NB1 — demand forecasting and delivery-risk scores.

Every public function is either pure (returns a Spark object without
side effects) or `@step`-wrapped (caches a parquet output on disk).

Writes (committed):
* `outputs/geo_centroids.parquet` — shared with NB3.
* `outputs/nb1_weekly_order_volume.parquet` — consumed by NB2.
* `outputs/nb1_seller_demand_scores.parquet` — feeds convergence.

Writes (cache-only, gitignored):
* `outputs/_cache/demand_order_lines.parquet` — the 4-way joined hot DF.
* `outputs/_cache/demand_predictions.parquet` — full-model_data predictions.
* `outputs/_cache/demand_metrics.parquet` — 1-row (gbt_rmse, rf_rmse, best_name).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import DateType, LongType, StructField, StructType
from pyspark.sql.window import Window

from ..cache import resolve_path, step
from ..loaders import (
    load_category_translation,
    load_customers,
    load_order_items,
    load_order_reviews,
    load_orders,
    load_products,
    load_sellers,
)
from ..transforms import delivery_delay_days, geolocation_centroids

FEATURE_COLS = ["week_num", "lag_1", "lag_4", "rolling_4w_mean", "month", "is_q4"]
LAG_IMPUTE_COLS = ["lag_1", "lag_4", "rolling_4w_mean"]

_DEMAND_CODE_DEPS = [
    "src/olist/pipeline/demand.py",
    "src/olist/loaders.py",
    "src/olist/schemas.py",
    "src/olist/transforms.py",
]


def rdd_daily_order_count(spark: SparkSession) -> DataFrame:
    """Demonstrate the required RDD primitive: read the raw orders CSV as text,
    split / filter / reduceByKey into a (date → order-count) pair, and rebuild
    a typed DataFrame with an explicit schema.

    The returned DataFrame is only used for a sanity check against the typed
    loader — downstream work reads the typed DataFrame via `loaders.load_orders`.
    """
    data_dir = Path(__file__).resolve().parents[3] / "data"
    orders_csv = str(data_dir / "olist_orders_dataset.csv")
    raw = spark.sparkContext.textFile(orders_csv)
    header = raw.first()

    def _parse(line: str):
        try:
            parts = line.split(",")
            ts = parts[3]  # order_purchase_timestamp
            if not ts:
                return None
            return (datetime.strptime(ts[:10], "%Y-%m-%d").date(), 1)
        except Exception:
            return None

    daily_pairs = (
        raw.filter(lambda row: row != header)
        .map(_parse)
        .filter(lambda kv: kv is not None)
        .reduceByKey(lambda a, b: a + b)
    )
    schema = StructType(
        [
            StructField("purchase_date", DateType(), False),
            StructField("order_count", LongType(), False),
        ]
    )
    return spark.createDataFrame(
        daily_pairs.map(lambda kv: (kv[0], int(kv[1]))), schema=schema
    )


def load_core_tables(spark: SparkSession) -> dict[str, DataFrame]:
    """Load the six CSVs NB1 touches, via explicit-schema typed loaders.

    Returns a dict with keys: orders, order_items, customers, sellers, products,
    order_reviews, category_translation. Row counts are available on each
    DataFrame via `.count()` — caller should print them as an audit.
    """
    return {
        "orders": load_orders(spark),
        "order_items": load_order_items(spark),
        "customers": load_customers(spark),
        "sellers": load_sellers(spark),
        "products": load_products(spark),
        "order_reviews": load_order_reviews(spark),
        "category_translation": load_category_translation(spark),
    }


@step(
    name="demand.geo_centroids",
    inputs=["data/olist_geolocation_dataset.csv"],
    outputs=["outputs/geo_centroids.parquet"],
    code_deps=_DEMAND_CODE_DEPS,
    version=1,
)
def build_geo_centroids(spark: SparkSession) -> DataFrame:
    """Aggregate the raw geolocation table down to one row per zip prefix.

    Shared with NB3 (state-level joins, distance features).
    """
    return geolocation_centroids(spark)


def filter_delivered(orders: DataFrame) -> DataFrame:
    """Drop orders without `order_delivered_customer_date` (not-yet-delivered /
    cancelled). Decision and row-count impact are logged in
    `docs/decisions_log.md` (2026-04-22 NB1 entry: ≈2,965 dropped of 99,441).
    """
    return orders.filter(F.col("order_delivered_customer_date").isNotNull())


@step(
    name="demand.order_lines",
    inputs=[
        "data/olist_orders_dataset.csv",
        "data/olist_order_items_dataset.csv",
        "data/olist_sellers_dataset.csv",
        "data/olist_products_dataset.csv",
    ],
    outputs=["outputs/_cache/demand_order_lines.parquet"],
    code_deps=_DEMAND_CODE_DEPS,
    version=1,
)
def build_order_lines(spark: SparkSession) -> DataFrame:
    """Build the NB1 hot DataFrame: delivered orders ⋈ order_items ⋈
    broadcast(sellers) ⋈ broadcast(products), with `delivery_delay_days`,
    `purchase_date`, and `year_week` columns. Small lookups (sellers, products)
    are broadcast per CLAUDE.md §3. One row per order line.
    """
    tables = load_core_tables(spark)
    orders_delivered = filter_delivered(tables["orders"])
    order_lines = (
        orders_delivered.join(tables["order_items"], on="order_id", how="inner")
        .join(broadcast(tables["sellers"]), on="seller_id", how="left")
        .join(broadcast(tables["products"]), on="product_id", how="left")
    )
    order_lines = delivery_delay_days(order_lines)
    return (
        order_lines.withColumn("purchase_date", F.to_date("order_purchase_timestamp"))
        .withColumn(
            "year_week",
            F.concat(
                F.year("order_purchase_timestamp"),
                F.lit("-"),
                F.lpad(F.weekofyear("order_purchase_timestamp").cast("string"), 2, "0"),
            ),
        )
    )


def sparksql_queries(order_lines: DataFrame) -> dict[str, tuple[str, DataFrame]]:
    """Register `order_lines` as a temp view and run the three SparkSQL queries
    required by the rubric (≥3 queries on temp views). Returns a dict mapping
    query name → (sql_text, result_df). The sql text is surfaced so the
    thin-wrapper notebook can display the query itself alongside the result.
    """
    order_lines.createOrReplaceTempView("order_lines")
    spark = order_lines.sparkSession

    top_sellers_sql = """
    SELECT seller_id,
           seller_state,
           ROUND(SUM(price), 2) AS total_revenue,
           COUNT(*)             AS line_count
    FROM order_lines
    GROUP BY seller_id, seller_state
    ORDER BY total_revenue DESC
    LIMIT 10
    """
    weekly_preview_sql = """
    SELECT seller_id,
           year_week,
           COUNT(*) AS weekly_order_count
    FROM order_lines
    GROUP BY seller_id, year_week
    ORDER BY seller_id, year_week
    LIMIT 10
    """
    late_rate_sql = """
    SELECT seller_state,
           ROUND(AVG(CASE WHEN delivery_delay_days > 0 THEN 1.0 ELSE 0.0 END), 4) AS late_rate,
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days,
           COUNT(*) AS n_lines
    FROM order_lines
    GROUP BY seller_state
    ORDER BY late_rate DESC
    LIMIT 15
    """
    return {
        "top_sellers_by_revenue": (top_sellers_sql, spark.sql(top_sellers_sql)),
        "weekly_volume_preview": (weekly_preview_sql, spark.sql(weekly_preview_sql)),
        "late_rate_by_state": (late_rate_sql, spark.sql(late_rate_sql)),
    }


def eda_stats(order_lines: DataFrame) -> dict[str, object]:
    """Run the big-data-safe EDA primitives: `approxQuantile` on price and
    delay, plus `approx_count_distinct` on seller_id / product_id. All driver-
    side results are aggregates, not row-level collects.
    """
    price_q = order_lines.approxQuantile("price", [0.25, 0.5, 0.75, 0.95], 0.01)
    delay_q = order_lines.approxQuantile(
        "delivery_delay_days", [0.25, 0.5, 0.75, 0.95], 0.01
    )
    approx_counts = order_lines.agg(
        F.approx_count_distinct("seller_id").alias("approx_sellers"),
        F.approx_count_distinct("product_id").alias("approx_products"),
    )
    return {
        "price_quantiles": [round(x, 2) for x in price_q],
        "delay_quantiles": delay_q,
        "approx_counts": approx_counts,
    }


@step(
    name="demand.weekly_order_volume",
    inputs=["outputs/_cache/demand_order_lines.parquet"],
    outputs=["outputs/nb1_weekly_order_volume.parquet"],
    code_deps=_DEMAND_CODE_DEPS,
    version=1,
)
def build_weekly_order_volume(spark: SparkSession) -> DataFrame:
    """Aggregate `order_lines` to (seller_id, year_week, weekly_order_count).

    Repartitioned by seller_id so NB2's lead-indicator join co-locates
    partitions. Consumed by both the feature pipeline (below) and NB2.
    """
    order_lines = spark.read.parquet(resolve_path("outputs/_cache/demand_order_lines.parquet"))
    return (
        order_lines.groupBy("seller_id", "year_week")
        .agg(F.count("*").alias("weekly_order_count"))
        .repartition("seller_id")
    )


def build_feature_pipeline() -> Pipeline:
    """Return the unfit feature pipeline (Imputer → VectorAssembler).

    Cheap to reconstruct, so not cached; the notebook prints the stages
    for the "Pipelines & Data Engineering" rubric line.
    """
    imputer = Imputer(
        inputCols=LAG_IMPUTE_COLS,
        outputCols=LAG_IMPUTE_COLS,
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip"
    )
    return Pipeline(stages=[imputer, assembler])


def add_weekly_features(weekly_order_volume: DataFrame) -> DataFrame:
    """Add lag_1, lag_4, rolling_4w_mean (Window-function features), plus
    calendar features (month, is_q4) and week_num. Imputer + VectorAssembler
    are applied via `build_feature_pipeline()` after this step.
    """
    w_seller = Window.partitionBy("seller_id").orderBy("year_week")
    w_roll4 = w_seller.rowsBetween(-4, -1)
    base = (
        weekly_order_volume.withColumn("week_num", F.row_number().over(w_seller))
        .withColumn("lag_1", F.lag("weekly_order_count", 1).over(w_seller))
        .withColumn("lag_4", F.lag("weekly_order_count", 4).over(w_seller))
        .withColumn("rolling_4w_mean", F.avg("weekly_order_count").over(w_roll4))
        .withColumn("month", F.substring("year_week", 6, 2).cast("int"))
        .withColumn("is_q4", (F.col("month") >= 10).cast("int"))
    )
    return build_feature_pipeline().fit(base).transform(base)


def build_cv_estimators(evaluator: RegressionEvaluator) -> dict:
    """Construct both CrossValidators with their param grids. Seed 42 and
    `parallelism=2` are preserved from the pre-refactor NB1 so that post-
    refactor reruns match the committed numbers within Spark's tolerance.
    """
    gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=20, seed=42)
    gbt_grid = (
        ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.stepSize, [0.1]).build()
    )
    gbt_cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=gbt_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=42,
        parallelism=2,
    )
    rf = RandomForestRegressor(
        featuresCol="features", labelCol="label", numTrees=40, seed=42
    )
    rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).build()
    rf_cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=rf_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=42,
        parallelism=2,
    )
    return {"gbt_cv": gbt_cv, "rf_cv": rf_cv, "gbt": gbt, "rf": rf}


@step(
    name="demand.fit_and_score",
    inputs=[
        "outputs/nb1_weekly_order_volume.parquet",
        "outputs/_cache/demand_order_lines.parquet",
        "data/olist_sellers_dataset.csv",
    ],
    outputs=[
        "outputs/_cache/demand_predictions.parquet",
        "outputs/_cache/demand_metrics.parquet",
        "outputs/_cache/demand_feature_importances.parquet",
        "outputs/nb1_seller_demand_scores.parquet",
    ],
    code_deps=_DEMAND_CODE_DEPS,
    version=2,
)
def fit_and_score(spark: SparkSession) -> dict[str, DataFrame]:
    """Fit GBT + RF via `CrossValidator(numFolds=3)`, pick the best by test
    RMSE, then score every seller.

    Returns three DataFrames keyed by their parquet stem:
    * `demand_predictions` — (seller_id, year_week, week_num, label, prediction)
      on the full `model_data`.
    * `demand_metrics` — one row: (gbt_rmse, rf_rmse, best_name).
    * `nb1_seller_demand_scores` — (seller_id, seller_state, forecast_uplift_pct,
      avg_delay_days, delay_risk_flag).
    """
    weekly = spark.read.parquet(resolve_path("outputs/nb1_weekly_order_volume.parquet"))
    order_lines = spark.read.parquet(resolve_path("outputs/_cache/demand_order_lines.parquet"))
    sellers = load_sellers(spark)

    weekly_features = add_weekly_features(weekly)
    model_data = weekly_features.filter(
        F.col("lag_1").isNotNull()
        & F.col("lag_4").isNotNull()
        & F.col("rolling_4w_mean").isNotNull()
    ).select(
        "seller_id",
        "year_week",
        "week_num",
        "features",
        F.col("weekly_order_count").cast("double").alias("label"),
    )

    split_week = model_data.approxQuantile("week_num", [0.8], 0.01)[0]
    train_df = model_data.filter(F.col("week_num") <= split_week)
    test_df = model_data.filter(F.col("week_num") > split_week)

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    cv = build_cv_estimators(evaluator)

    print("Fitting GBT CV ...")
    gbt_model = cv["gbt_cv"].fit(train_df)
    print("Fitting RF CV ...")
    rf_model = cv["rf_cv"].fit(train_df)

    gbt_rmse = evaluator.evaluate(gbt_model.transform(test_df))
    rf_rmse = evaluator.evaluate(rf_model.transform(test_df))
    print(f"GBT  test RMSE: {gbt_rmse:.3f}")
    print(f"RF   test RMSE: {rf_rmse:.3f}")
    best_model = gbt_model if gbt_rmse <= rf_rmse else rf_model
    best_name = "GBT" if best_model is gbt_model else "RandomForest"
    print(f"Selected: {best_name}")

    predictions_all = best_model.transform(model_data).select(
        "seller_id", "year_week", "week_num", "label", "prediction"
    )

    w_last4 = Window.partitionBy("seller_id").orderBy(F.col("week_num").desc())
    trailing = (
        predictions_all.withColumn("rk", F.row_number().over(w_last4))
        .filter(F.col("rk") <= 4)
        .groupBy("seller_id")
        .agg(F.avg("label").alias("trailing_4w_obs"))
    )
    next_pred = (
        predictions_all.withColumn("rk", F.row_number().over(w_last4))
        .filter(F.col("rk") <= 4)
        .groupBy("seller_id")
        .agg(F.avg("prediction").alias("next_4w_pred"))
    )
    avg_delay = order_lines.groupBy("seller_id").agg(
        F.avg("delivery_delay_days").alias("avg_delay_days")
    )

    seller_demand_scores = (
        trailing.join(next_pred, "seller_id")
        .join(avg_delay, "seller_id")
        .join(
            broadcast(sellers.select("seller_id", "seller_state")),
            "seller_id",
            "left",
        )
        .withColumn(
            "forecast_uplift_pct",
            F.when(
                F.col("trailing_4w_obs") > 0,
                (F.col("next_4w_pred") - F.col("trailing_4w_obs"))
                / F.col("trailing_4w_obs")
                * 100.0,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("delay_risk_flag", (F.col("avg_delay_days") > 3).cast("int"))
        .select(
            "seller_id",
            "seller_state",
            "forecast_uplift_pct",
            "avg_delay_days",
            "delay_risk_flag",
        )
    )

    metrics_row = spark.createDataFrame(
        [(float(gbt_rmse), float(rf_rmse), best_name)],
        schema="gbt_rmse double, rf_rmse double, best_name string",
    )

    fi_vector = best_model.bestModel.featureImportances.toArray()
    fi_rows = [(FEATURE_COLS[i], float(fi_vector[i])) for i in range(len(FEATURE_COLS))]
    feature_importances = spark.createDataFrame(
        fi_rows, schema="feature string, importance double"
    )

    return {
        "demand_predictions": predictions_all,
        "demand_metrics": metrics_row,
        "demand_feature_importances": feature_importances,
        "nb1_seller_demand_scores": seller_demand_scores,
    }


def best_model_feature_importances(spark: SparkSession) -> DataFrame:
    """Read the cached (feature, importance) table written by `fit_and_score`.
    Drives the feature-importance bar in NB1 §5.
    """
    return spark.read.parquet(
        resolve_path("outputs/_cache/demand_feature_importances.parquet")
    )


def predictions_sample(spark: SparkSession, n: int = 1000) -> DataFrame:
    """Return a `limit(n)` sample of the full-model predictions parquet.
    Feeds the residual plot in NB1 §5 — a 1000-row cap is big-data-safe.
    """
    return spark.read.parquet(
        resolve_path("outputs/_cache/demand_predictions.parquet")
    ).limit(n)
