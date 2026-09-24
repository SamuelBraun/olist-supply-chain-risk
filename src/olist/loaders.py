"""Typed CSV loaders.

Each loader applies the explicit schema from `schemas.py` and wraps
`spark.read.csv(...)`. Path defaults to repo-relative `data/`.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from .schemas import CSV_FILES

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load(
    spark: SparkSession,
    key: str,
    data_dir: Path | str | None = None,
    multiline: bool = False,
) -> DataFrame:
    filename, schema = CSV_FILES[key]
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = base / filename
    reader = spark.read.option("header", "true").option("encoding", "UTF-8")
    if multiline:
        # The order_reviews file embeds newlines inside quoted free-text
        # fields, so the default line-per-record reader splits ~3.85k reviews
        # mid-text and over-counts rows (104,162 vs the true 99,224). multiLine
        # + an explicit quote-escape makes Spark honour the RFC-4180 quoting
        # and parse one record per review.
        #
        # Big-data-safety note: multiLine disables input-file splitting (each
        # file is read by a single task), so it is NOT the scalable default and
        # is applied ONLY to the one affected file. At production scale the
        # correct fix is to land reviews in a splittable format (Parquet/JSON
        # lines) or pre-clean the embedded newlines in an upstream batch job;
        # here the file is a few MB, so the single-task read is acceptable.
        reader = reader.option("multiLine", "true").option("escape", '"')
    return reader.schema(schema).csv(str(path))


def load_orders(spark, data_dir=None): return _load(spark, "orders", data_dir)
def load_order_items(spark, data_dir=None): return _load(spark, "order_items", data_dir)
def load_order_reviews(spark, data_dir=None): return _load(spark, "order_reviews", data_dir, multiline=True)
def load_customers(spark, data_dir=None): return _load(spark, "customers", data_dir)
def load_sellers(spark, data_dir=None): return _load(spark, "sellers", data_dir)
def load_products(spark, data_dir=None): return _load(spark, "products", data_dir)
def load_order_payments(spark, data_dir=None): return _load(spark, "order_payments", data_dir)
def load_geolocation(spark, data_dir=None): return _load(spark, "geolocation", data_dir)
def load_category_translation(spark, data_dir=None): return _load(spark, "category_translation", data_dir)
