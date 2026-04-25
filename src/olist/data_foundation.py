"""Data Foundation helpers — render the relational shape of the 9-table
Olist schema as small driver-side aggregates that feed `viz.py` charts.

Every function here returns a *small* aggregate (≤27 rows) — table-level
metadata, key cardinalities, date-range summaries. The raw rows are
never materialised; only the reduced statistics travel to the driver.

Used by §2 (Data Foundation) of `notebooks/main.ipynb`.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .loaders import (
    load_category_translation,
    load_customers,
    load_geolocation,
    load_order_items,
    load_order_payments,
    load_order_reviews,
    load_orders,
    load_products,
    load_sellers,
)


#: Canonical metadata for the 9 source tables: name → loader fn, primary key,
#: foreign-key columns, role (transactional / dimensional / geo / taxonomy /
#: free-text). Rendering helpers below derive everything from this single
#: source of truth.
TABLE_REGISTRY = {
    "orders":               {"loader": load_orders,               "pk": "order_id",         "fks": ["customer_id"],                 "role": "transactional"},
    "order_items":          {"loader": load_order_items,          "pk": ["order_id", "order_item_id"], "fks": ["order_id", "product_id", "seller_id"], "role": "transactional"},
    "order_reviews":        {"loader": load_order_reviews,        "pk": "review_id",        "fks": ["order_id"],                    "role": "free-text"},
    "order_payments":       {"loader": load_order_payments,       "pk": ["order_id", "payment_sequential"], "fks": ["order_id"],     "role": "transactional"},
    "customers":            {"loader": load_customers,            "pk": "customer_id",      "fks": ["customer_unique_id", "customer_zip_code_prefix"], "role": "dimensional"},
    "sellers":              {"loader": load_sellers,              "pk": "seller_id",        "fks": ["seller_zip_code_prefix"],      "role": "dimensional"},
    "products":             {"loader": load_products,             "pk": "product_id",       "fks": ["product_category_name"],       "role": "dimensional"},
    "geolocation":          {"loader": load_geolocation,          "pk": None,               "fks": ["geolocation_zip_code_prefix"], "role": "geospatial"},
    "category_translation": {"loader": load_category_translation, "pk": "product_category_name", "fks": [],                          "role": "taxonomy"},
}

#: Every (table, column) pair that carries a date / timestamp — used by
#: `temporal_coverage`. Other tables are dimensional / static.
TEMPORAL_COLUMNS = [
    ("orders", "order_purchase_timestamp"),
    ("orders", "order_approved_at"),
    ("orders", "order_delivered_carrier_date"),
    ("orders", "order_delivered_customer_date"),
    ("orders", "order_estimated_delivery_date"),
    ("order_items", "shipping_limit_date"),
    ("order_reviews", "review_creation_date"),
    ("order_reviews", "review_answer_timestamp"),
]

#: The four shared keys that stitch the schema together. Each helper checks
#: how many distinct values of the key live in each table that contains it.
SHARED_KEYS = [
    ("seller_id",          ["sellers", "order_items"]),
    ("customer_id",        ["customers", "orders"]),
    ("customer_unique_id", ["customers"]),
    ("order_id",           ["orders", "order_items", "order_reviews", "order_payments"]),
    ("product_id",         ["products", "order_items"]),
    ("zip_prefix",         ["customers", "sellers", "geolocation"]),
]

#: Column name per table that represents `zip_prefix` (used by `SHARED_KEYS`).
_ZIP_PREFIX_COL = {
    "customers": "customer_zip_code_prefix",
    "sellers": "seller_zip_code_prefix",
    "geolocation": "geolocation_zip_code_prefix",
}


def table_overview(spark: SparkSession) -> DataFrame:
    """One row per source CSV: (table, role, n_rows, n_cols, primary_key).
    Driver-safe — counts are reduced inside Spark, not collected raw.
    """
    rows = []
    for name, meta in TABLE_REGISTRY.items():
        df = meta["loader"](spark)
        n_rows = df.count()
        n_cols = len(df.columns)
        pk = meta["pk"]
        pk_str = ", ".join(pk) if isinstance(pk, list) else (pk or "—")
        rows.append((name, meta["role"], int(n_rows), int(n_cols), pk_str))
    return spark.createDataFrame(
        rows,
        schema="table string, role string, n_rows long, n_cols long, primary_key string",
    )


def shared_key_cardinality(spark: SparkSession) -> DataFrame:
    """For every (shared_key, table) pair: distinct value count of that key
    in that table. Drives the shared-key bar chart in §2.

    `approx_count_distinct` (HyperLogLog) keeps the call big-data-safe
    even at 10–100× current scale.
    """
    rows = []
    for key, table_names in SHARED_KEYS:
        for tname in table_names:
            df = TABLE_REGISTRY[tname]["loader"](spark)
            col_name = _ZIP_PREFIX_COL.get(tname, key) if key == "zip_prefix" else key
            if col_name not in df.columns:
                continue
            distinct = df.agg(F.approx_count_distinct(col_name).alias("d")).first()["d"]
            rows.append((key, tname, int(distinct)))
    return spark.createDataFrame(
        rows,
        schema="shared_key string, table string, approx_distinct long",
    )


def temporal_coverage(spark: SparkSession) -> DataFrame:
    """Min / max date per timestamp column across the schema. One row per
    (table, column) in `TEMPORAL_COLUMNS`. Driver-safe single-row aggregates.
    """
    rows = []
    for tname, col in TEMPORAL_COLUMNS:
        df = TABLE_REGISTRY[tname]["loader"](spark)
        bounds = df.agg(
            F.min(col).alias("min_ts"),
            F.max(col).alias("max_ts"),
            F.count(col).alias("n_non_null"),
        ).first()
        rows.append((tname, col, bounds["min_ts"], bounds["max_ts"], int(bounds["n_non_null"])))
    return spark.createDataFrame(
        rows,
        schema="table string, column string, min_ts timestamp, max_ts timestamp, n_non_null long",
    )


def cleaning_audit(spark: SparkSession) -> DataFrame:
    """Render the cleaning decisions taken across the three sub-analyses
    as a single styled table: stage → input rows → kept rows → dropped → reason.

    The numbers come from real Spark counts — no hard-coded magic.
    """
    orders = load_orders(spark)
    reviews = load_order_reviews(spark)
    geo = load_geolocation(spark)

    n_orders_total = orders.count()
    n_orders_delivered = orders.filter(F.col("order_delivered_customer_date").isNotNull()).count()

    n_reviews_total = reviews.count()
    n_reviews_with_score = reviews.filter(F.col("review_score").isin(1, 2, 4, 5)).count()
    n_reviews_with_text = reviews.filter(F.col("review_comment_message").isNotNull()).count()

    n_geo_raw = geo.count()
    n_geo_centroids = geo.select("geolocation_zip_code_prefix").distinct().count()

    rows = [
        ("Demand: drop undelivered / cancelled orders",
         int(n_orders_total), int(n_orders_delivered),
         int(n_orders_total - n_orders_delivered),
         "order_delivered_customer_date IS NULL"),
        ("Sentiment: drop neutral reviews (score == 3)",
         int(n_reviews_total), int(n_reviews_with_score),
         int(n_reviews_total - n_reviews_with_score),
         "neutral score has no clear positive/negative label"),
        ("Sentiment: keep reviews with comment text",
         int(n_reviews_total), int(n_reviews_with_text),
         int(n_reviews_total - n_reviews_with_text),
         "NLP pipeline cannot tokenise NULL text"),
        ("Network: aggregate raw geolocation to zip-prefix centroids",
         int(n_geo_raw), int(n_geo_centroids),
         int(n_geo_raw - n_geo_centroids),
         "many-to-one address→zip; broadcast centroids only"),
    ]
    return spark.createDataFrame(
        rows,
        schema="stage string, input_rows long, kept_rows long, dropped_rows long, reason string",
    )
