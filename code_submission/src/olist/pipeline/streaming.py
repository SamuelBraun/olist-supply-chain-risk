"""NB §8 (bonus) — Structured Streaming twin of the weekly order-volume
aggregation.

The dataset is a static historical dump, so a live feed is *simulated*: we drip
the delivered orders into a watched directory as a sequence of micro-batch files,
then a streaming query reads them as they "arrive" and maintains a windowed
order count per week — the streaming form of `nb1_weekly_order_volume`.

Designed to run cleanly inside `nbconvert`:
* file source with `maxFilesPerTrigger=1` (one micro-batch per trigger so the
  run is visibly incremental),
* `trigger(availableNow=True)` — process every file already present, then STOP
  (no infinite query, no `awaitTermination` hang),
* `memory` sink so the result is queryable as a temp table after the run.

Everything is bounded by construction; nothing collects a non-aggregated frame.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ..cache import resolve_path
from ..loaders import load_orders

# Schema of the streamed micro-batch files. File-source streaming requires an
# explicit schema (it cannot infer from an unbounded source).
STREAM_SCHEMA = StructType([
    StructField("order_id", StringType(), True),
    StructField("order_purchase_timestamp", TimestampType(), True),
])

_STREAM_DIR = "outputs/_cache/_stream_orders"


def prepare_stream_source(spark: SparkSession, *, n_batches: int = 6, limit: int = 6000) -> str:
    """Materialise a small sample of delivered orders as `n_batches` parquet
    files in a watched directory, simulating orders arriving over time. Returns
    the directory path. Idempotent: the directory is recreated each call.
    """
    stream_dir = Path(resolve_path(_STREAM_DIR))
    if stream_dir.exists():
        shutil.rmtree(stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)

    orders = (
        load_orders(spark)
        .filter(F.col("order_purchase_timestamp").isNotNull())
        .select("order_id", "order_purchase_timestamp")
        .limit(limit)
        .repartition(n_batches)  # n_batches files = n_batches micro-batches
    )
    orders.write.mode("overwrite").parquet(str(stream_dir))
    return str(stream_dir)


def run_weekly_volume_stream(spark: SparkSession, source_dir: str) -> DataFrame:
    """Read the watched directory as a stream (one file per trigger), aggregate
    a per-ISO-week order count, write to an in-memory sink with
    `trigger(availableNow=True)`, and return the final result DataFrame once the
    query has drained and stopped.

    This illustrates `demand.build_weekly_order_volume` *as a stream*: the same
    per-ISO-week count, expressed as an incremental query over an unbounded
    source instead of a one-shot batch job. It is a mechanics demo, not a numeric
    replica — it counts orders marketplace-wide (no seller grouping, no
    delivered-only filter, capped sample), so the figures won't match NB1's.
    """
    stream = (
        spark.readStream.schema(STREAM_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .parquet(source_dir)
    )
    weekly = (
        stream.withColumn(
            "year_week",
            F.concat(
                F.year("order_purchase_timestamp"),
                F.lit("-"),
                F.lpad(F.weekofyear("order_purchase_timestamp").cast("string"), 2, "0"),
            ),
        )
        .groupBy("year_week")
        .agg(F.count("*").alias("weekly_order_count"))
    )
    query = (
        weekly.writeStream.format("memory")
        .queryName("weekly_volume_stream")
        .outputMode("complete")
        .trigger(availableNow=True)
        .start()
    )
    try:
        query.awaitTermination()  # availableNow → returns once all files drained
    finally:
        query.stop()  # never leave a started query running, even on error
    return spark.sql(
        "SELECT * FROM weekly_volume_stream ORDER BY year_week"
    )
