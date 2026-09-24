"""Shared transforms reused across notebooks.

Add a function here only when at least two notebooks need it; one-shot
transforms belong inline in their notebook.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .loaders import load_geolocation


def geolocation_centroids(spark: SparkSession) -> DataFrame:
    """Aggregate the 1M-row geolocation table down to one row per zip prefix.

    The raw file has many rows per zip (one per address); for joins we only
    need the centroid. Cache or persist as parquet at the call site.
    """
    geo = load_geolocation(spark)
    return geo.groupBy("geolocation_zip_code_prefix").agg(
        F.mean("geolocation_lat").alias("lat"),
        F.mean("geolocation_lng").alias("lng"),
        F.first("geolocation_state", ignorenulls=True).alias("state"),
    )


def delivery_delay_days(df: DataFrame) -> DataFrame:
    """Add `delivery_delay_days = delivered - estimated` (positive = late)."""
    return df.withColumn(
        "delivery_delay_days",
        F.datediff("order_delivered_customer_date", "order_estimated_delivery_date"),
    )
