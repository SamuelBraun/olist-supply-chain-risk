"""Explicit StructType schemas for the nine Olist CSVs.

Avoids `inferSchema=True` (which forces a full data pass per file) and
guarantees timestamp columns load as TimestampType, not StringType.
"""

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

ORDERS_SCHEMA = StructType([
    StructField("order_id", StringType(), nullable=False),
    StructField("customer_id", StringType(), nullable=False),
    StructField("order_status", StringType(), nullable=True),
    StructField("order_purchase_timestamp", TimestampType(), nullable=True),
    StructField("order_approved_at", TimestampType(), nullable=True),
    StructField("order_delivered_carrier_date", TimestampType(), nullable=True),
    StructField("order_delivered_customer_date", TimestampType(), nullable=True),
    StructField("order_estimated_delivery_date", TimestampType(), nullable=True),
])

ORDER_ITEMS_SCHEMA = StructType([
    StructField("order_id", StringType(), nullable=False),
    StructField("order_item_id", IntegerType(), nullable=False),
    StructField("product_id", StringType(), nullable=False),
    StructField("seller_id", StringType(), nullable=False),
    StructField("shipping_limit_date", TimestampType(), nullable=True),
    StructField("price", DoubleType(), nullable=True),
    StructField("freight_value", DoubleType(), nullable=True),
])

ORDER_REVIEWS_SCHEMA = StructType([
    StructField("review_id", StringType(), nullable=False),
    StructField("order_id", StringType(), nullable=False),
    StructField("review_score", IntegerType(), nullable=True),
    StructField("review_comment_title", StringType(), nullable=True),
    StructField("review_comment_message", StringType(), nullable=True),
    StructField("review_creation_date", TimestampType(), nullable=True),
    StructField("review_answer_timestamp", TimestampType(), nullable=True),
])

CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", StringType(), nullable=False),
    StructField("customer_unique_id", StringType(), nullable=False),
    StructField("customer_zip_code_prefix", IntegerType(), nullable=True),
    StructField("customer_city", StringType(), nullable=True),
    StructField("customer_state", StringType(), nullable=True),
])

SELLERS_SCHEMA = StructType([
    StructField("seller_id", StringType(), nullable=False),
    StructField("seller_zip_code_prefix", IntegerType(), nullable=True),
    StructField("seller_city", StringType(), nullable=True),
    StructField("seller_state", StringType(), nullable=True),
])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id", StringType(), nullable=False),
    StructField("product_category_name", StringType(), nullable=True),
    StructField("product_name_lenght", IntegerType(), nullable=True),
    StructField("product_description_lenght", IntegerType(), nullable=True),
    StructField("product_photos_qty", IntegerType(), nullable=True),
    StructField("product_weight_g", IntegerType(), nullable=True),
    StructField("product_length_cm", IntegerType(), nullable=True),
    StructField("product_height_cm", IntegerType(), nullable=True),
    StructField("product_width_cm", IntegerType(), nullable=True),
])

ORDER_PAYMENTS_SCHEMA = StructType([
    StructField("order_id", StringType(), nullable=False),
    StructField("payment_sequential", IntegerType(), nullable=True),
    StructField("payment_type", StringType(), nullable=True),
    StructField("payment_installments", IntegerType(), nullable=True),
    StructField("payment_value", DoubleType(), nullable=True),
])

GEOLOCATION_SCHEMA = StructType([
    StructField("geolocation_zip_code_prefix", IntegerType(), nullable=False),
    StructField("geolocation_lat", DoubleType(), nullable=True),
    StructField("geolocation_lng", DoubleType(), nullable=True),
    StructField("geolocation_city", StringType(), nullable=True),
    StructField("geolocation_state", StringType(), nullable=True),
])

CATEGORY_TRANSLATION_SCHEMA = StructType([
    StructField("product_category_name", StringType(), nullable=False),
    StructField("product_category_name_english", StringType(), nullable=True),
])

CSV_FILES = {
    "orders": ("olist_orders_dataset.csv", ORDERS_SCHEMA),
    "order_items": ("olist_order_items_dataset.csv", ORDER_ITEMS_SCHEMA),
    "order_reviews": ("olist_order_reviews_dataset.csv", ORDER_REVIEWS_SCHEMA),
    "customers": ("olist_customers_dataset.csv", CUSTOMERS_SCHEMA),
    "sellers": ("olist_sellers_dataset.csv", SELLERS_SCHEMA),
    "products": ("olist_products_dataset.csv", PRODUCTS_SCHEMA),
    "order_payments": ("olist_order_payments_dataset.csv", ORDER_PAYMENTS_SCHEMA),
    "geolocation": ("olist_geolocation_dataset.csv", GEOLOCATION_SCHEMA),
    "category_translation": ("product_category_name_translation.csv", CATEGORY_TRANSLATION_SCHEMA),
}
