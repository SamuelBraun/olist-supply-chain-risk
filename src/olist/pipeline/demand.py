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

    Feeds `rdd_vs_typed_reconciliation`, which derives the malformed-row count
    surfaced in the §3 cleaning audit. Downstream forecasting still reads the
    typed DataFrame via `loaders.load_orders` — the manual `split(',')` parse
    here is for the RDD demo, not production ingestion.
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


def rdd_vs_typed_reconciliation(spark: SparkSession) -> DataFrame:
    """Cross-check the RDD daily-count path against the typed loader on the
    same daily aggregation, and surface how many raw rows the RDD's strict
    text parse rejected (empty / unparseable order_purchase_timestamp).

    Returns a one-row summary: (raw_body_rows, rdd_counted_orders,
    typed_counted_orders, rdd_minus_typed, malformed_rows, days_disagreeing).

    `malformed_rows` is a real data-quality figure the typed loader hides
    behind null-tolerant casts — the RDD pass is the only place we measure it,
    so this output feeds the §3 cleaning audit instead of being thrown away.
    """
    daily_rdd = rdd_daily_order_count(spark)  # (purchase_date, order_count)
    data_dir = Path(__file__).resolve().parents[3] / "data"
    orders_csv = str(data_dir / "olist_orders_dataset.csv")
    raw = spark.sparkContext.textFile(orders_csv)
    header = raw.first()
    raw_body_rows = raw.filter(lambda row: row != header).count()

    rdd_counted = daily_rdd.agg(F.sum("order_count")).first()[0] or 0
    malformed = int(raw_body_rows) - int(rdd_counted)

    typed_daily = (
        load_orders(spark)
        .withColumn("purchase_date", F.to_date("order_purchase_timestamp"))
        .groupBy("purchase_date")
        .agg(F.count("*").alias("typed_n"))
    )
    typed_counted = typed_daily.agg(F.sum("typed_n")).first()[0] or 0
    days_disagreeing = (
        daily_rdd.join(typed_daily, "purchase_date", "full_outer")
        .filter(
            F.coalesce(F.col("order_count"), F.lit(0))
            != F.coalesce(F.col("typed_n"), F.lit(0))
        )
        .count()
    )
    return spark.createDataFrame(
        [(
            int(raw_body_rows),
            int(rdd_counted),
            int(typed_counted),
            int(rdd_counted) - int(typed_counted),
            int(malformed),
            int(days_disagreeing),
        )],
        schema=(
            "raw_body_rows long, rdd_counted_orders long, "
            "typed_counted_orders long, rdd_minus_typed long, "
            "malformed_rows long, days_disagreeing long"
        ),
    )


def winsorize(df: DataFrame, col: str, lower_q: float = 0.01, upper_q: float = 0.99):
    """Clip `col` to its [lower_q, upper_q] approxQuantile bounds. Returns
    (winsorised_df, lo, hi, n_clipped). Used to tame the delivery-delay tail
    before it reaches the §6 min-max normalisation range.
    """
    lo, hi = df.approxQuantile(col, [lower_q, upper_q], 0.01)
    n_clipped = df.filter((F.col(col) < lo) | (F.col(col) > hi)).count()
    clipped = df.withColumn(
        col,
        F.when(F.col(col) < lo, lo).when(F.col(col) > hi, hi).otherwise(F.col(col)),
    )
    return clipped, lo, hi, n_clipped


def load_core_tables(spark: SparkSession) -> dict[str, DataFrame]:
    """Seven typed CSV loads NB1 touches."""
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
    # one row per zip prefix; reused by NB3
    return geolocation_centroids(spark)


def filter_delivered(orders: DataFrame) -> DataFrame:
    # drops ~2,965 of 99,441 rows; see decisions_log 2026-04-22
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
    # Winsorise the delivery-delay tail: a handful of extreme late deliveries
    # otherwise stretch the §6 min-max range and compress everyone else's
    # demand_norm. p1/p99 bounds; logged in decisions_log 2026-05-29.
    order_lines, delay_lo, delay_hi, n_clipped = winsorize(
        order_lines, "delivery_delay_days"
    )
    print(
        f"[winsorize] delivery_delay_days -> [{delay_lo:.1f}, {delay_hi:.1f}] days; "
        f"{n_clipped:,} order-lines clipped"
    )
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
    # Imputer (median, for lag NaNs at series-start) -> VectorAssembler.
    # Cheap to rebuild, so unfit each call.
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
    """Window lags + rolling + calendar features, then run through the
    fit feature pipeline. Output has a ready-to-fit ``features`` vector."""
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


GBT_SEED = 7341
RF_SEED = 2918


def build_cv_estimators(evaluator: RegressionEvaluator) -> dict:
    """Construct both CrossValidators with their param grids.

    Seeds: GBT_SEED=7341, RF_SEED=2918 (distinct, not tutorial defaults).
    `parallelism=2` keeps fold-fit workers bounded on a single driver.
    Grids: GBT 3x2x2 = 12 combos, RF 2x2x2 = 8 combos.
    """
    gbt = GBTRegressor(featuresCol="features", labelCol="label", seed=GBT_SEED)
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5, 7])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .addGrid(gbt.maxIter, [20, 40])
        .build()
    )
    gbt_cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=gbt_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=GBT_SEED,
        parallelism=2,
    )
    rf = RandomForestRegressor(featuresCol="features", labelCol="label", seed=RF_SEED)
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [5, 10])
        .addGrid(rf.numTrees, [40, 80])
        .addGrid(rf.subsamplingRate, [0.8, 1.0])
        .build()
    )
    rf_cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=rf_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=RF_SEED,
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
    # Cache the train split: each CrossValidator re-reads it numFolds × gridSize
    # times (3×12 for GBT, 3×8 for RF = 60 fits). Without this, every fit
    # recomputes the window-feature lineage from parquet and re-shuffles, which
    # spills tens of GB of shuffle data and exhausts local disk (CLAUDE.md §3).
    train_df = model_data.filter(F.col("week_num") <= split_week).cache()
    test_df = model_data.filter(F.col("week_num") > split_week)
    train_df.count()  # materialise the cache before the CV loop

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
    train_df.unpersist()  # done with the train split; free it before scoring

    def _winning_params(cv_model, param_names):
        bm = cv_model.bestModel
        return {p: bm.getOrDefault(p) for p in param_names}

    gbt_best = _winning_params(gbt_model, ["maxDepth", "stepSize", "maxIter"])
    rf_best = _winning_params(rf_model, ["maxDepth", "numTrees", "subsamplingRate"])
    print(f"GBT  test RMSE: {gbt_rmse:.3f}  | best params: {gbt_best}")
    print(f"RF   test RMSE: {rf_rmse:.3f}  | best params: {rf_best}")
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
