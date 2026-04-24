"""Convergence layer — unify the three per-seller parquets into a single
Seller Risk Index.

Inputs:
* `outputs/nb1_seller_demand_scores.parquet`
* `outputs/nb2_seller_sentiment_scores.parquet`
* `outputs/nb3_seller_network_scores.parquet`

Writes:
* `outputs/seller_risk_index.parquet` — the consulting deliverable.

Weights and thresholds are fixed in advance (CLAUDE.md §4):
* 0.35 × demand_norm + 0.35 × sentiment_norm + 0.30 × network_norm.
* CRITICAL > 0.75, SAFE < 0.40, WARNING otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..cache import step
from ..safety import (  # noqa: F401 — referenced by annotation comments
    RISK_NORM_AGG,
    STATE_AGG_VIZ,
    TOP50_VIZ,
)

_CONV_CODE_DEPS = ["src/olist/pipeline/convergence.py"]

RISK_WEIGHTS = {"demand": 0.35, "sentiment": 0.35, "network": 0.30}
RISK_CRITICAL_THRESHOLD = 0.75
RISK_SAFE_THRESHOLD = 0.40


@dataclass
class NormalisationRanges:
    """Min/max triples extracted from single-row aggregates, used as
    broadcast literals for min-max normalisation.
    """

    delay_lo: float
    delay_hi: float
    sentiment_lo: float
    sentiment_hi: float
    network_lo: float
    network_hi: float


def compute_normalisation_ranges(joined: DataFrame) -> NormalisationRanges:
    """Extract min/max for the three pre-normalised columns via three
    single-row aggregates. Flagged as RISK_NORM_AGG — the `.first()` calls
    are on one-row DataFrames by construction.
    """
    # BIG-DATA-SAFETY-ESCAPE: RISK_NORM_AGG — single-row aggregates
    d = joined.agg(
        F.min("avg_delay_days").alias("lo"),
        F.max("avg_delay_days").alias("hi"),
    ).first()
    s = joined.agg(
        F.min("avg_sentiment_score").alias("lo"),
        F.max("avg_sentiment_score").alias("hi"),
    ).first()
    n = joined.agg(
        F.min("network_risk_score").alias("lo"),
        F.max("network_risk_score").alias("hi"),
    ).first()
    return NormalisationRanges(
        delay_lo=float(d["lo"]),
        delay_hi=float(d["hi"]),
        sentiment_lo=float(s["lo"]),
        sentiment_hi=float(s["hi"]),
        network_lo=float(n["lo"]),
        network_hi=float(n["hi"]),
    )


def _min_max(col: str, lo: float, hi: float, invert: bool = False):
    span = (hi - lo) if (hi - lo) != 0 else 1.0
    scaled = (F.col(col) - lo) / span
    return (F.lit(1.0) - scaled) if invert else scaled


@step(
    name="convergence.seller_risk_index",
    inputs=[
        "outputs/nb1_seller_demand_scores.parquet",
        "outputs/nb2_seller_sentiment_scores.parquet",
        "outputs/nb3_seller_network_scores.parquet",
    ],
    outputs=["outputs/seller_risk_index.parquet"],
    code_deps=_CONV_CODE_DEPS,
    version=1,
)
def build_seller_risk_index(spark: SparkSession) -> DataFrame:
    """Inner-join the three per-seller parquets, min-max normalise each
    risk component, weight 0.35/0.35/0.30, band CRITICAL/WARNING/SAFE,
    write `outputs/seller_risk_index.parquet`.

    Sentiment is inverted (higher = worse) so all three `*_norm` columns
    point in the same direction: higher = more risky.
    """
    demand = spark.read.parquet(resolve_path("outputs/nb1_seller_demand_scores.parquet"))
    sentiment = spark.read.parquet(resolve_path("outputs/nb2_seller_sentiment_scores.parquet"))
    network = spark.read.parquet(resolve_path("outputs/nb3_seller_network_scores.parquet"))
    joined = (
        demand.join(sentiment, "seller_id", "inner").join(
            network.select("seller_id", "pagerank_score", "network_risk_score"),
            "seller_id",
            "inner",
        )
    )
    ranges = compute_normalisation_ranges(joined)
    return (
        joined.withColumn(
            "demand_norm",
            _min_max("avg_delay_days", ranges.delay_lo, ranges.delay_hi),
        )
        .withColumn(
            "sentiment_norm",
            _min_max(
                "avg_sentiment_score",
                ranges.sentiment_lo,
                ranges.sentiment_hi,
                invert=True,
            ),
        )
        .withColumn(
            "network_norm",
            _min_max(
                "network_risk_score", ranges.network_lo, ranges.network_hi
            ),
        )
        .withColumn(
            "risk_score",
            RISK_WEIGHTS["demand"] * F.col("demand_norm")
            + RISK_WEIGHTS["sentiment"] * F.col("sentiment_norm")
            + RISK_WEIGHTS["network"] * F.col("network_norm"),
        )
        .withColumn(
            "risk_class",
            F.when(F.col("risk_score") > RISK_CRITICAL_THRESHOLD, "CRITICAL")
            .when(F.col("risk_score") < RISK_SAFE_THRESHOLD, "SAFE")
            .otherwise("WARNING"),
        )
        .select(
            "seller_id",
            "seller_state",
            "risk_score",
            "risk_class",
            "demand_norm",
            "sentiment_norm",
            "network_norm",
            "avg_delay_days",
            "avg_sentiment_score",
            "pagerank_score",
            "sentiment_declining",
        )
    )


def risk_band_counts(risk: DataFrame) -> DataFrame:
    """Count of sellers per risk_class, ordered SAFE → WARNING → CRITICAL."""
    return (
        risk.groupBy("risk_class").agg(F.count("*").alias("n")).orderBy(
            F.when(F.col("risk_class") == "SAFE", 0)
            .when(F.col("risk_class") == "WARNING", 1)
            .otherwise(2)
        )
    )


def top50_for_quadrant(risk: DataFrame):
    """Return the top-50 highest-risk sellers as a pandas DataFrame for the
    quadrant scatter chart.
    """
    # BIG-DATA-SAFETY-ESCAPE: TOP50_VIZ — capped at 50 rows
    return risk.orderBy(F.col("risk_score").desc()).limit(50).toPandas()


def state_mean_risk(risk: DataFrame):
    """Per-state mean risk_score across *all* sellers as a pandas DataFrame
    (≤27 rows after groupBy seller_state).
    """
    # BIG-DATA-SAFETY-ESCAPE: STATE_AGG_VIZ — 27-row aggregate
    return (
        risk.groupBy("seller_state")
        .agg(
            F.avg("risk_score").alias("mean_risk"),
            F.count("*").alias("n_sellers"),
        )
        .filter(F.col("seller_state").isNotNull())
        .orderBy(F.col("mean_risk").desc())
        .toPandas()
    )
