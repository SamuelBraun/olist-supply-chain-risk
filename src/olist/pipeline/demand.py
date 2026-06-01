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

import os
from datetime import datetime
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
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

FEATURE_COLS = [
    "week_num",
    "lag_1",
    "lag_4",
    "rolling_4w_mean",
    "decay_wtd_8w",
    "month",
    "is_q4",
    # Retail-calendar spikes (WS9)
    "is_black_friday",
    "is_year_end",
    # Per-seller static covariates (WS9)
    "avg_price",
    "avg_freight",
    "n_categories",
]
LAG_IMPUTE_COLS = ["lag_1", "lag_4", "rolling_4w_mean", "decay_wtd_8w"]

#: Geometric-decay weighting for the time-weighted recent-demand feature.
#: Weight on the k-th most recent week is DECAY_ALPHA**(k-1), so the most
#: recent week dominates and the contribution fades over an 8-week lookback.
#: A bounded window (not a true infinite-history EWMA) keeps it big-data-safe:
#: it is plain Window `lag()` arithmetic, no sequential per-partition scan.
DECAY_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]
DECAY_ALPHA = 0.6

#: Validation switch (CLAUDE.md workflow). When OLIST_LIGHT=1, every
#: ParamGridBuilder shrinks to 1–2 combinations and CrossValidator uses
#: numFolds=2 so a smoke run completes in minutes. Unset (the heavy nightly
#: run) keeps the full grids and 3-fold CV.
LIGHT = os.environ.get("OLIST_LIGHT") == "1"

#: Regional-forecast feature layout (see fit_regional_forecast).
REGIONAL_FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_4w_mean",
    "decay_wtd_8w",
    "month",
    "week_of_year",
    "state_index",
]
REGIONAL_LAG_IMPUTE_COLS = ["lag_1", "lag_2", "lag_4", "rolling_4w_mean", "decay_wtd_8w"]

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


def decay_weighted_recent_mean(value_col, window, horizons=None, alpha=None):
    """Geometric-decay weighted average of `value_col` over the last
    `len(horizons)` weeks within `window`. The k-th most recent week carries
    weight ``alpha**(k-1)`` (recent-dominant), normalised to sum to 1.

    This is a *bounded* recent-demand momentum feature, not a true EWMA: a real
    exponentially weighted moving average integrates the entire history and
    needs a sequential per-partition scan, which does not parallelise. A fixed
    8-week decayed lookback is the same intuition (recent weeks matter more)
    expressed as plain Window `lag()` arithmetic, so it stays big-data-safe and
    distributes like any other windowed feature.

    Returns a Column. Null at the series start (when the deepest lag is null),
    which the train-only Imputer fills downstream — consistent with `lag_4`.
    """
    horizons = horizons or DECAY_HORIZONS
    alpha = DECAY_ALPHA if alpha is None else alpha
    weights = [alpha ** i for i in range(len(horizons))]
    weight_sum = sum(weights)
    weighted_terms = [
        F.lag(value_col, h).over(window) * F.lit(w / weight_sum)
        for h, w in zip(horizons, weights)
    ]
    expr = weighted_terms[0]
    for term in weighted_terms[1:]:
        expr = expr + term
    return expr


def add_weekly_features(weekly_order_volume: DataFrame) -> DataFrame:
    """Window lags + rolling + calendar features. Also attaches a *global*
    calendar-week ordinal (`global_week_idx`) so downstream code can do a
    calendar-time train/test split that is comparable across sellers, rather
    than the per-seller `week_num` that biased the old split toward
    long-tenure sellers.

    NOTE: this returns the *un-imputed* feature frame (no `features` vector).
    The Imputer + VectorAssembler must be fit on the TRAIN split only to avoid
    leakage; `fit_and_score` does that fit after the calendar split.
    """
    w_seller = Window.partitionBy("seller_id").orderBy("year_week")
    w_roll4 = w_seller.rowsBetween(-4, -1)
    base = (
        weekly_order_volume.withColumn("week_num", F.row_number().over(w_seller))
        .withColumn("lag_1", F.lag("weekly_order_count", 1).over(w_seller))
        .withColumn("lag_4", F.lag("weekly_order_count", 4).over(w_seller))
        .withColumn("rolling_4w_mean", F.avg("weekly_order_count").over(w_roll4))
        .withColumn(
            "decay_wtd_8w",
            decay_weighted_recent_mean("weekly_order_count", w_seller),
        )
        .withColumn("month", F.substring("year_week", 6, 2).cast("int"))
        .withColumn("is_q4", (F.col("month") >= 10).cast("int"))
        # Brazilian retail-calendar spikes (week-of-year). Black Friday lands in
        # ISO weeks 47–48 (late Nov) — the single biggest spike in this dataset;
        # year-end (weeks >=49) is the Christmas season. These binary flags let
        # the trees isolate the demand surges §3.7.3 flagged as unpredictable.
        .withColumn(
            "is_black_friday", F.col("month").isin(47, 48).cast("int")
        )
        .withColumn("is_year_end", (F.col("month") >= 49).cast("int"))
    )
    return with_global_week_index(base)


def with_global_week_index(df: DataFrame, week_col: str = "year_week") -> DataFrame:
    """Attach a dense global calendar-week ordinal `global_week_idx` derived
    from the lexicographic order of distinct `year_week` strings (`YYYY-WW`,
    zero-padded, so string order == calendar order). The same ordinal is shared
    across all sellers/states, which is what makes a calendar-time split
    unbiased w.r.t. series length.
    """
    # The single-partition sort below is over a tiny bounded frame (the distinct
    # weeks, ~100 rows), then broadcast-joined back — safe at any data scale.
    distinct_weeks = (
        df.select(week_col).distinct().withColumn(
            "global_week_idx",
            # SCALABLE-WINDOW: ranks the DISTINCT week set (~100 rows), not the full data
            F.dense_rank().over(Window.orderBy(week_col)) - 1,
        )
    )
    return df.join(broadcast(distinct_weeks), on=week_col, how="left")


def calendar_split_threshold(
    df: DataFrame, holdout_frac: float = 0.2, idx_col: str = "global_week_idx"
) -> int:
    """Return the global-week ordinal at/under which a row is TRAIN. The latest
    ~`holdout_frac` of distinct calendar weeks become the test set. Computed
    from min/max of the dense ordinal, so it does not depend on per-seller
    tenure.
    """
    # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT
    bounds = df.agg(
        F.min(idx_col).alias("lo"), F.max(idx_col).alias("hi")
    ).first()
    lo, hi = int(bounds["lo"]), int(bounds["hi"])
    span = hi - lo
    return lo + int(round(span * (1.0 - holdout_frac)))


def fit_feature_split(
    base: DataFrame,
    feature_cols: list[str],
    impute_cols: list[str],
    holdout_frac: float = 0.2,
) -> tuple[DataFrame, DataFrame, float]:
    """Calendar-time split + leakage-free feature assembly.

    1. Split on the global calendar-week ordinal (latest `holdout_frac` of
       weeks = test) so the test set is not biased toward long-tenure series.
    2. Fit the Imputer (median, for series-start lag NaNs) + VectorAssembler
       on TRAIN ONLY, then transform both splits with that fit model.

    Returns (train_df, test_df, split_idx). Both frames carry a `features`
    vector plus `label` and the passthrough id columns already on `base`.
    """
    split_idx = calendar_split_threshold(base, holdout_frac)
    train_raw = base.filter(F.col("global_week_idx") <= split_idx)
    test_raw = base.filter(F.col("global_week_idx") > split_idx)

    imputer = Imputer(
        inputCols=impute_cols, outputCols=impute_cols, strategy="median"
    )
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )
    fit_pipeline = Pipeline(stages=[imputer, assembler]).fit(train_raw)  # TRAIN ONLY
    train_df = fit_pipeline.transform(train_raw)
    test_df = fit_pipeline.transform(test_raw)
    return train_df, test_df, float(split_idx)


def naive_baseline_rmses(
    test_df: DataFrame,
    train_df: DataFrame,
    label_col: str = "label",
) -> dict[str, float]:
    """RMSE of three naive baselines on the SAME held-out test rows:
    persistence (`lag_1`), `rolling_4w_mean`, and the global TRAIN mean.

    `lag_1` / `rolling_4w_mean` are post-imputation here (so no NaNs leak in),
    which matches how the ML models see them. The train-mean baseline uses the
    mean of TRAIN labels only — no test leakage.
    """
    # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT
    train_mean = float(train_df.agg(F.avg(label_col)).first()[0] or 0.0)
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse"
    )

    def _rmse(pred_col_expr) -> float:
        scored = test_df.withColumn("prediction", pred_col_expr.cast("double"))
        return float(evaluator.evaluate(scored))

    return {
        "persistence_lag1_rmse": _rmse(F.col("lag_1")),
        "rolling_4w_mean_rmse": _rmse(F.col("rolling_4w_mean")),
        "train_mean_rmse": _rmse(F.lit(train_mean)),
    }


GBT_SEED = 7341
RF_SEED = 2918


def build_cv_estimators(
    evaluator: RegressionEvaluator, features_col: str = "features"
) -> dict:
    """Construct both CrossValidators with their param grids.

    Seeds: GBT_SEED=7341, RF_SEED=2918 (distinct, not tutorial defaults).
    `parallelism=2` keeps fold-fit workers bounded on a single driver.

    Grid size honours the LIGHT switch:
    * full (LIGHT off): GBT 3x2x2 = 12 combos, RF 2x2x2 = 8 combos, numFolds=3.
    * LIGHT on: GBT 1x1x2 = 2 combos, RF 1x1x2 = 2 combos, numFolds=2 — a fast
      smoke run that still exercises the ParamGridBuilder/CrossValidator path.
    """
    num_folds = 2 if LIGHT else 3
    gbt = GBTRegressor(featuresCol=features_col, labelCol="label", seed=GBT_SEED)
    if LIGHT:
        gbt_grid = (
            ParamGridBuilder()
            .addGrid(gbt.maxDepth, [3])
            .addGrid(gbt.stepSize, [0.1])
            .addGrid(gbt.maxIter, [20, 40])
            .build()
        )
    else:
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
        numFolds=num_folds,
        seed=GBT_SEED,
        parallelism=2,
    )
    rf = RandomForestRegressor(
        featuresCol=features_col, labelCol="label", seed=RF_SEED
    )
    if LIGHT:
        rf_grid = (
            ParamGridBuilder()
            .addGrid(rf.maxDepth, [10])
            .addGrid(rf.numTrees, [40])
            .addGrid(rf.subsamplingRate, [0.8, 1.0])
            .build()
        )
    else:
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
        numFolds=num_folds,
        seed=RF_SEED,
        parallelism=2,
    )
    return {
        "gbt_cv": gbt_cv,
        "rf_cv": rf_cv,
        "gbt": gbt,
        "rf": rf,
        "gbt_grid": gbt_grid,
        "rf_grid": rf_grid,
    }


RO_SPLITS = 2 if LIGHT else 3


def rolling_origin_select(
    train_df: DataFrame,
    estimator,
    param_maps: list,
    evaluator: RegressionEvaluator,
    week_idx_col: str = "global_week_idx",
    n_splits: int = RO_SPLITS,
):
    """Forward-chaining (rolling-origin) hyperparameter selection for time series.

    Random k-fold CV assumes rows are exchangeable; weekly demand is not — a fold
    can train on future weeks and validate on past ones, leaking look-ahead into
    *model selection*. This expands the training window through calendar time
    instead: split the TRAIN weeks into `n_splits`+1 contiguous blocks by the
    shared `global_week_idx`; for split s, fit on blocks [0..s] and validate on
    block s+1. Each ParamMap's score is the mean validation RMSE across splits;
    the lowest wins. No test rows are ever touched here.

    Returns `(best_param_map, best_mean_rmse, per_combo)` where `per_combo` is a
    list of `(combo_index, mean_val_rmse)` for notebook display.
    """
    # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 2-scalar week-range agg
    bounds = train_df.agg(
        F.min(week_idx_col).alias("lo"), F.max(week_idx_col).alias("hi")
    ).first()
    lo, hi = int(bounds["lo"]), int(bounds["hi"])
    span = hi - lo
    # n_splits+1 equal segments; the first segment is always training.
    boundaries = [
        lo + int(round(span * s / (n_splits + 1))) for s in range(n_splits + 2)
    ]
    combo_rmses: list[list[float]] = [[] for _ in param_maps]
    for s in range(1, n_splits + 1):
        train_hi = boundaries[s]
        val_lo, val_hi = boundaries[s], boundaries[s + 1]
        fold_train = train_df.filter(F.col(week_idx_col) <= train_hi)
        fold_val = train_df.filter(
            (F.col(week_idx_col) > val_lo) & (F.col(week_idx_col) <= val_hi)
        )
        if not fold_val.head(1):  # empty validation block — skip this split
            continue
        fold_train = fold_train.cache()
        fold_train.count()  # materialise before fitting the grid against it
        for combo_idx, param_map in enumerate(param_maps):
            fold_model = estimator.fit(fold_train, param_map)
            rmse = float(evaluator.evaluate(fold_model.transform(fold_val)))
            combo_rmses[combo_idx].append(rmse)
        fold_train.unpersist()
    per_combo = [
        (combo_idx, sum(rmses) / len(rmses))
        for combo_idx, rmses in enumerate(combo_rmses)
        if rmses
    ]
    best_idx, best_mean = min(per_combo, key=lambda pair: pair[1])
    return param_maps[best_idx], best_mean, per_combo


@step(
    name="demand.seller_covariates",
    inputs=["outputs/_cache/demand_order_lines.parquet"],
    outputs=["outputs/_cache/demand_seller_covariates.parquet"],
    code_deps=_DEMAND_CODE_DEPS,
    version=1,
)
def build_seller_covariates(spark: SparkSession) -> DataFrame:
    """Per-seller static covariates that contextualise the weekly forecast:
    average item price and freight (price tier / shipping profile) and the number
    of distinct product categories the seller sells (breadth). These are stable
    seller attributes joined into the demand feature base so the model can
    distinguish, e.g., a single-category low-price seller from a broad premium
    one. Plain group-by aggregations — fully distributable.

    Columns: seller_id, avg_price, avg_freight, n_categories.
    """
    order_lines = spark.read.parquet(
        resolve_path("outputs/_cache/demand_order_lines.parquet")
    )
    return order_lines.groupBy("seller_id").agg(
        F.avg("price").alias("avg_price"),
        F.avg("freight_value").alias("avg_freight"),
        F.countDistinct("product_category_name").alias("n_categories"),
    )


@step(
    name="demand.fit_and_score",
    inputs=[
        "outputs/nb1_weekly_order_volume.parquet",
        "outputs/_cache/demand_order_lines.parquet",
        "outputs/_cache/demand_seller_covariates.parquet",
        "data/olist_sellers_dataset.csv",
    ],
    outputs=[
        "outputs/_cache/demand_predictions.parquet",
        "outputs/_cache/demand_metrics.parquet",
        "outputs/_cache/demand_feature_importances.parquet",
        "outputs/nb1_seller_demand_scores.parquet",
    ],
    code_deps=_DEMAND_CODE_DEPS,
    version=5,
)
def fit_and_score(spark: SparkSession) -> dict[str, DataFrame]:
    """Fit GBT + RF, select hyperparameters with forward-chaining (rolling-origin)
    validation, pick the best by held-out test RMSE, compare against three naive
    baselines on the SAME test rows, then score every seller.

    `CrossValidator` (3-fold; 2-fold under LIGHT) is still fit for both models to
    demonstrate the MLlib tuning API, but its random-fold result is NOT what
    selects the reported model — random k-fold leaks future-into-past on a time
    series. `rolling_origin_select` makes the reported choice; the metrics row
    carries both so the notebook can show "k-fold picked X, time-aware picked Y."

    The split is a *global calendar-time* split (latest ~20% of calendar weeks
    held out across all sellers), and the Imputer/VectorAssembler are fit on the
    TRAIN split only — both fixes vs. the earlier per-seller-index split that
    biased the test set toward long-tenure sellers and leaked the imputer median.

    Returns four DataFrames keyed by their parquet stem:
    * `demand_predictions` — (seller_id, year_week, week_num, label, prediction)
      on the full `model_data`.
    * `demand_metrics` — one row: (gbt_rmse, rf_rmse, best_name,
      persistence_lag1_rmse, rolling_4w_mean_rmse, train_mean_rmse).
    * `demand_feature_importances` — (feature, importance) for the winner.
    * `nb1_seller_demand_scores` — (seller_id, seller_state, forecast_uplift_pct,
      avg_delay_days, delay_risk_flag).
    """
    weekly = spark.read.parquet(resolve_path("outputs/nb1_weekly_order_volume.parquet"))
    order_lines = spark.read.parquet(resolve_path("outputs/_cache/demand_order_lines.parquet"))
    sellers = load_sellers(spark)

    seller_covariates = spark.read.parquet(
        resolve_path("outputs/_cache/demand_seller_covariates.parquet")
    )
    weekly_features = add_weekly_features(weekly)  # un-imputed base + global_week_idx
    model_base = (
        weekly_features.filter(
            F.col("lag_1").isNotNull()
            & F.col("lag_4").isNotNull()
            & F.col("rolling_4w_mean").isNotNull()
        )
        .join(broadcast(seller_covariates), "seller_id", "left")
        .fillna({"avg_price": 0.0, "avg_freight": 0.0, "n_categories": 0})
        .select(
            "seller_id",
            "year_week",
            "week_num",
            "global_week_idx",
            *LAG_IMPUTE_COLS,
            "month",
            "is_q4",
            "is_black_friday",
            "is_year_end",
            "avg_price",
            "avg_freight",
            "n_categories",
            F.col("weekly_order_count").cast("double").alias("label"),
        )
    )

    # Calendar-time split + train-only imputer/assembler fit (no leakage).
    train_df, test_df, split_idx = fit_feature_split(
        model_base, FEATURE_COLS, LAG_IMPUTE_COLS, holdout_frac=0.2
    )
    # Cache the train split: each CrossValidator re-reads it numFolds × gridSize
    # times. Without this, every fit recomputes the window-feature lineage from
    # parquet and re-shuffles, which spills shuffle data and exhausts local disk
    # (CLAUDE.md §3).
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_df.count()  # materialise the cache before the CV loop
    test_df.count()

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    cv = build_cv_estimators(evaluator)

    print(f"Calendar-time split at global_week_idx={int(split_idx)} "
          f"(LIGHT={LIGHT})")

    # --- k-fold CrossValidator: demonstrates the MLlib tuning API and serves as
    # the "naive" comparison. Its random folds shuffle calendar order, so we do
    # NOT let it select the reported model — see rolling_origin_select below.
    print("Fitting GBT CV (k-fold demo) ...")
    gbt_cv_model = cv["gbt_cv"].fit(train_df)
    print("Fitting RF CV (k-fold demo) ...")
    rf_cv_model = cv["rf_cv"].fit(train_df)
    gbt_cv_rmse = float(evaluator.evaluate(gbt_cv_model.transform(test_df)))
    rf_cv_rmse = float(evaluator.evaluate(rf_cv_model.transform(test_df)))

    # --- Forward-chaining (rolling-origin) selection: the REPORTED model. Picks
    # hyperparameters with an expanding time window so no future week informs the
    # choice, then refits the winner on the full train split.
    print("Rolling-origin selection (GBT) ...")
    gbt_best_pm, gbt_ro_mean, _ = rolling_origin_select(
        train_df, cv["gbt"], cv["gbt_grid"], evaluator
    )
    print("Rolling-origin selection (RF) ...")
    rf_best_pm, rf_ro_mean, _ = rolling_origin_select(
        train_df, cv["rf"], cv["rf_grid"], evaluator
    )
    gbt_model = cv["gbt"].fit(train_df, gbt_best_pm)
    rf_model = cv["rf"].fit(train_df, rf_best_pm)

    # Guard the model-vs-baseline comparison: VectorAssembler(handleInvalid=
    # "skip") could silently drop test rows, which would put model RMSE and the
    # naive baselines on different denominators. Assert identical row counts.
    test_n = test_df.count()
    gbt_scored = gbt_model.transform(test_df)
    rf_scored = rf_model.transform(test_df)
    assert gbt_scored.count() == test_n and rf_scored.count() == test_n, (
        "VectorAssembler(handleInvalid='skip') dropped held-out rows — model "
        "RMSE and naive baselines would use different denominators."
    )
    gbt_rmse = float(evaluator.evaluate(gbt_scored))
    rf_rmse = float(evaluator.evaluate(rf_scored))

    # Naive baselines on the SAME held-out rows: persistence, rolling-4w, mean.
    baselines = naive_baseline_rmses(test_df, train_df)
    print(
        "Baselines (test RMSE): "
        f"persistence(lag_1)={baselines['persistence_lag1_rmse']:.3f}  "
        f"rolling_4w_mean={baselines['rolling_4w_mean_rmse']:.3f}  "
        f"train_mean={baselines['train_mean_rmse']:.3f}"
    )

    train_df.unpersist()  # done with the train split; free it before scoring
    test_df.unpersist()

    gbt_best = {p.name: v for p, v in gbt_best_pm.items()}
    rf_best = {p.name: v for p, v in rf_best_pm.items()}
    print(
        f"GBT  test RMSE: {gbt_rmse:.3f} (k-fold demo {gbt_cv_rmse:.3f})  "
        f"| time-aware params: {gbt_best}"
    )
    print(
        f"RF   test RMSE: {rf_rmse:.3f} (k-fold demo {rf_cv_rmse:.3f})  "
        f"| time-aware params: {rf_best}"
    )
    best_model = gbt_model if gbt_rmse <= rf_rmse else rf_model
    best_name = "GBT" if best_model is gbt_model else "RandomForest"
    print(f"Selected: {best_name} (selection=rolling_origin)")

    # Score the full series. The full frame must carry the same `features`
    # vector, built with the SAME train-fit imputer/assembler (refit on train
    # only inside fit_feature_split); rebuild it here on the un-split base.
    full_imputer = Imputer(
        inputCols=LAG_IMPUTE_COLS, outputCols=LAG_IMPUTE_COLS, strategy="median"
    )
    full_assembler = VectorAssembler(
        inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip"
    )
    train_only = model_base.filter(F.col("global_week_idx") <= split_idx)
    full_fit = Pipeline(stages=[full_imputer, full_assembler]).fit(train_only)
    model_data_vec = full_fit.transform(model_base)
    predictions_all = best_model.transform(model_data_vec).select(
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
        [(
            float(gbt_rmse),
            float(rf_rmse),
            best_name,
            float(baselines["persistence_lag1_rmse"]),
            float(baselines["rolling_4w_mean_rmse"]),
            float(baselines["train_mean_rmse"]),
            float(gbt_cv_rmse),
            float(rf_cv_rmse),
            str(gbt_best),
            str(rf_best),
            "rolling_origin",
        )],
        schema=(
            "gbt_rmse double, rf_rmse double, best_name string, "
            "persistence_lag1_rmse double, rolling_4w_mean_rmse double, "
            "train_mean_rmse double, gbt_cv_rmse double, rf_cv_rmse double, "
            "gbt_best_params string, rf_best_params string, selection_method string"
        ),
    )

    fi_vector = best_model.featureImportances.toArray()
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


# ---------------------------------------------------------------------------
# Regional demand forecast
#
# The seller-level model (above) is intentionally noisy: most sellers have
# short, sparse weekly series, so per-seller forecasting is closer to a
# warm-up than a planning tool. Aggregating to (seller_state, year_week) gives
# ~27 dense, long state-level series — the level at which a forecast is
# actually decision-grade for regional capacity planning. This is the genuine
# forecasting deliverable; the seller model stays for the per-seller risk
# `avg_delay_days`/`forecast_uplift_pct` features the §6 index consumes.
# ---------------------------------------------------------------------------


@step(
    name="demand.regional_weekly_volume",
    inputs=["outputs/_cache/demand_order_lines.parquet"],
    outputs=["outputs/_cache/demand_regional_weekly_volume.parquet"],
    code_deps=_DEMAND_CODE_DEPS,
    version=1,
)
def build_regional_weekly_volume(spark: SparkSession) -> DataFrame:
    """Aggregate delivered order-lines to (seller_state, year_week,
    weekly_order_count). One dense weekly series per Brazilian state — the
    base for the regional forecast.
    """
    order_lines = spark.read.parquet(
        resolve_path("outputs/_cache/demand_order_lines.parquet")
    )
    return (
        order_lines.filter(F.col("seller_state").isNotNull())
        .groupBy("seller_state", "year_week")
        .agg(F.count("*").alias("weekly_order_count"))
        .repartition("seller_state")
    )


def add_regional_features(regional_weekly_volume: DataFrame) -> DataFrame:
    """Window lags (1/2/4) + rolling-4 mean + calendar features (month,
    week_of_year) + a `state_index` per seller_state, plus the shared global
    calendar-week ordinal. Returns the un-imputed feature base (no `features`
    vector — the Imputer/assembler are fit train-only in `fit_regional_forecast`).
    """
    w_state = Window.partitionBy("seller_state").orderBy("year_week")
    w_roll4 = w_state.rowsBetween(-4, -1)
    state_indexer = StringIndexer(
        inputCol="seller_state", outputCol="state_index", handleInvalid="keep"
    )
    base = (
        regional_weekly_volume.withColumn("lag_1", F.lag("weekly_order_count", 1).over(w_state))
        .withColumn("lag_2", F.lag("weekly_order_count", 2).over(w_state))
        .withColumn("lag_4", F.lag("weekly_order_count", 4).over(w_state))
        .withColumn("rolling_4w_mean", F.avg("weekly_order_count").over(w_roll4))
        .withColumn(
            "decay_wtd_8w",
            decay_weighted_recent_mean("weekly_order_count", w_state),
        )
        .withColumn("month", F.substring("year_week", 6, 2).cast("int"))
        .withColumn("week_of_year", F.substring("year_week", 6, 2).cast("int"))
    )
    base = state_indexer.fit(base).transform(base)
    return with_global_week_index(base)


@step(
    name="demand.regional_forecast",
    inputs=["outputs/_cache/demand_regional_weekly_volume.parquet"],
    outputs=[
        "outputs/nb1_regional_demand_forecast.parquet",
        "outputs/_cache/demand_regional_metrics.parquet",
    ],
    code_deps=_DEMAND_CODE_DEPS,
    version=1,
)
def fit_regional_forecast(spark: SparkSession) -> dict[str, DataFrame]:
    """Train GBT + RF under CrossValidator on the regional weekly series,
    compare against persistence (`lag_1`) and rolling-4w baselines on a
    calendar-time test split, and write per-(state, week) actual-vs-predicted.

    The split holds out the latest ~20% of CALENDAR weeks (shared global
    ordinal) and the Imputer/VectorAssembler are fit on TRAIN only.

    Writes:
    * `nb1_regional_demand_forecast` — (seller_state, year_week, actual,
      predicted, model_rmse, baseline_rmse). `model_rmse`/`baseline_rmse` are
      broadcast constants (the winning model's test RMSE and the best naive
      baseline's test RMSE) repeated on every row for easy charting.
    * `demand_regional_metrics` — one row with the full metric breakdown.

    Returns a small metrics summary dict (the `demand_regional_metrics` frame).
    """
    regional = spark.read.parquet(
        resolve_path("outputs/_cache/demand_regional_weekly_volume.parquet")
    )
    feature_base = add_regional_features(regional)
    model_base = feature_base.filter(
        F.col("lag_1").isNotNull()
        & F.col("lag_2").isNotNull()
        & F.col("lag_4").isNotNull()
        & F.col("rolling_4w_mean").isNotNull()
    ).select(
        "seller_state",
        "year_week",
        "global_week_idx",
        *REGIONAL_FEATURE_COLS,
        F.col("weekly_order_count").cast("double").alias("label"),
    )

    train_df, test_df, split_idx = fit_feature_split(
        model_base, REGIONAL_FEATURE_COLS, REGIONAL_LAG_IMPUTE_COLS, holdout_frac=0.2
    )
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_df.count()
    test_df.count()

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    cv = build_cv_estimators(evaluator)

    print(f"[regional] calendar-time split at global_week_idx={int(split_idx)} "
          f"(LIGHT={LIGHT})")
    print("[regional] Fitting GBT CV ...")
    gbt_model = cv["gbt_cv"].fit(train_df)
    print("[regional] Fitting RF CV ...")
    rf_model = cv["rf_cv"].fit(train_df)

    gbt_rmse = evaluator.evaluate(gbt_model.transform(test_df))
    rf_rmse = evaluator.evaluate(rf_model.transform(test_df))
    baselines = naive_baseline_rmses(test_df, train_df)
    print(
        f"[regional] GBT={gbt_rmse:.3f}  RF={rf_rmse:.3f}  "
        f"persistence={baselines['persistence_lag1_rmse']:.3f}  "
        f"rolling_4w={baselines['rolling_4w_mean_rmse']:.3f}  "
        f"train_mean={baselines['train_mean_rmse']:.3f}"
    )

    best_model = gbt_model if gbt_rmse <= rf_rmse else rf_model
    best_name = "GBT" if best_model is gbt_model else "RandomForest"
    model_rmse = min(gbt_rmse, rf_rmse)
    baseline_rmse = min(
        baselines["persistence_lag1_rmse"], baselines["rolling_4w_mean_rmse"]
    )
    print(f"[regional] selected {best_name} (model_rmse={model_rmse:.3f} "
          f"vs best baseline_rmse={baseline_rmse:.3f})")

    # Score the full series with a train-only-fit imputer/assembler.
    full_imputer = Imputer(
        inputCols=REGIONAL_LAG_IMPUTE_COLS,
        outputCols=REGIONAL_LAG_IMPUTE_COLS,
        strategy="median",
    )
    full_assembler = VectorAssembler(
        inputCols=REGIONAL_FEATURE_COLS, outputCol="features", handleInvalid="skip"
    )
    train_only = model_base.filter(F.col("global_week_idx") <= split_idx)
    full_fit = Pipeline(stages=[full_imputer, full_assembler]).fit(train_only)
    scored_full = best_model.transform(full_fit.transform(model_base))

    forecast = (
        scored_full.select(
            "seller_state",
            "year_week",
            F.col("label").alias("actual"),
            F.col("prediction").alias("predicted"),
        )
        .withColumn("model_rmse", F.lit(float(model_rmse)))
        .withColumn("baseline_rmse", F.lit(float(baseline_rmse)))
    )

    train_df.unpersist()
    test_df.unpersist()

    metrics_row = spark.createDataFrame(
        [(
            float(gbt_rmse),
            float(rf_rmse),
            best_name,
            float(baselines["persistence_lag1_rmse"]),
            float(baselines["rolling_4w_mean_rmse"]),
            float(baselines["train_mean_rmse"]),
            int(split_idx),
        )],
        schema=(
            "gbt_rmse double, rf_rmse double, best_name string, "
            "persistence_lag1_rmse double, rolling_4w_mean_rmse double, "
            "train_mean_rmse double, split_week_idx int"
        ),
    )

    return {
        "nb1_regional_demand_forecast": forecast,
        "demand_regional_metrics": metrics_row,
    }
