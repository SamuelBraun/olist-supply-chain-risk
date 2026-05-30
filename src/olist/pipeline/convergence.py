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

from ..cache import resolve_path, step
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
    """p1/p99 bounds per pre-normalised column, used as broadcast literals for
    min-max normalisation. Percentile bounds (not raw min/max) so one extreme
    seller can't compress everyone else's normalised score.
    """

    delay_lo: float
    delay_hi: float
    sentiment_lo: float
    sentiment_hi: float
    contagion_lo: float
    contagion_hi: float
    deficit_lo: float
    deficit_hi: float


def compute_normalisation_ranges(joined: DataFrame) -> NormalisationRanges:
    """Extract p1/p99 bounds for the four pre-normalised columns via
    `approxQuantile`. Percentile bounds tame the long tails (delay outliers,
    the 68%-zero contagion column) that raw min/max would otherwise let
    dominate the scale.
    """
    # BIG-DATA-SAFETY-ESCAPE: RISK_NORM_AGG — approxQuantile on a ~3k-row frame
    delay_lo, delay_hi = joined.approxQuantile("avg_delay_days", [0.01, 0.99], 0.01)
    sent_lo, sent_hi = joined.approxQuantile("avg_sentiment_score", [0.01, 0.99], 0.01)
    cont_lo, cont_hi = joined.approxQuantile("network_risk_score", [0.01, 0.99], 0.01)
    def_lo, def_hi = joined.approxQuantile("substitutability_deficit", [0.01, 0.99], 0.01)
    return NormalisationRanges(
        delay_lo=float(delay_lo),
        delay_hi=float(delay_hi),
        sentiment_lo=float(sent_lo),
        sentiment_hi=float(sent_hi),
        contagion_lo=float(cont_lo),
        contagion_hi=float(cont_hi),
        deficit_lo=float(def_lo),
        deficit_hi=float(def_hi),
    )


def _min_max(col: str, lo: float, hi: float, invert: bool = False):
    # Scale to [0, 1] then clamp, so sellers past the p1/p99 bounds saturate
    # instead of producing scores outside [0, 1].
    span = (hi - lo) if (hi - lo) != 0 else 1.0
    scaled = (F.col(col) - lo) / span
    clamped = F.greatest(F.lit(0.0), F.least(F.lit(1.0), scaled))
    return (F.lit(1.0) - clamped) if invert else clamped


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

    The network axis is a 50/50 blend of *contagion* (delayed-subgraph
    PageRank — a late-shipping hub) and *substitutability deficit* (high
    impact, no backup). The deficit half is graph-unique and non-degenerate
    across the whole population, so the network axis no longer collapses to a
    degree proxy and is no longer zero for the two-thirds of sellers outside
    the late-shipping subgraph.
    """
    demand = spark.read.parquet(resolve_path("outputs/nb1_seller_demand_scores.parquet"))
    sentiment = spark.read.parquet(resolve_path("outputs/nb2_seller_sentiment_scores.parquet"))
    network = spark.read.parquet(resolve_path("outputs/nb3_seller_network_scores.parquet"))
    joined = (
        demand.join(sentiment, "seller_id", "inner").join(
            network.select(
                "seller_id",
                "pagerank_score",
                "network_risk_score",
                "substitutability_deficit",
                "backup_seller_id",
                "backup_strength",
                "community_id",
            ),
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
            "contagion_norm",
            _min_max("network_risk_score", ranges.contagion_lo, ranges.contagion_hi),
        )
        .withColumn(
            "deficit_norm",
            _min_max(
                "substitutability_deficit", ranges.deficit_lo, ranges.deficit_hi
            ),
        )
        .withColumn(
            "network_norm",
            0.5 * F.col("contagion_norm") + 0.5 * F.col("deficit_norm"),
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
        .withColumn(
            "escalate_no_backup",
            (
                (F.col("risk_class") != "SAFE")
                & (F.coalesce(F.col("backup_strength"), F.lit(0)) == 0)
            ).cast("int"),
        )
        .select(
            "seller_id",
            "seller_state",
            "risk_score",
            "risk_class",
            "demand_norm",
            "sentiment_norm",
            "network_norm",
            "contagion_norm",
            "deficit_norm",
            "avg_delay_days",
            "avg_sentiment_score",
            "pagerank_score",
            "substitutability_deficit",
            "backup_seller_id",
            "backup_strength",
            "community_id",
            "escalate_no_backup",
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
    # 50-row pandas for the quadrant scatter; safety escape tagged below.
    # BIG-DATA-SAFETY-ESCAPE: TOP50_VIZ — capped at 50 rows
    return risk.orderBy(F.col("risk_score").desc()).limit(50).toPandas()


RISK_ARCHETYPE_LABELS = {
    "delay-driven":     "high demand_norm — operationally late, otherwise OK",
    "sentiment-driven": "high sentiment_norm — customers unhappy, delivery may still be on time",
    "centrality-driven":"high network_norm — structurally critical, low individual signal",
    "low-risk":         "low on all three axes — the bulk of the marketplace",
}


KMEANS_SEED = 8825


@step(
    name="convergence.kmeans_elbow",
    inputs=["outputs/seller_risk_index.parquet"],
    outputs=["outputs/nb6_kmeans_elbow.parquet"],
    code_deps=_CONV_CODE_DEPS,
    version=1,
)
def kmeans_elbow_sweep(
    risk: DataFrame,
    *,
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6),
    seed: int = KMEANS_SEED,
) -> DataFrame:
    """Sweep K-Means over ``k_values`` and return a small DataFrame
    ``(k, wssse)`` for the elbow-method justification of k=4.

    ``wssse`` is ``model.summary.trainingCost`` — the within-set sum of
    squared errors. The elbow is the point where WSSSE stops dropping
    sharply as k increases. Persisted to ``outputs/nb6_kmeans_elbow.parquet``
    so the chart in §6.3 of the notebook reads from parquet, deterministically.
    """
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler

    components = ["demand_norm", "sentiment_norm", "network_norm"]
    assembler = VectorAssembler(inputCols=components, outputCol="features_elbow")
    feats = assembler.transform(risk).cache()
    try:
        rows = []
        for k in k_values:
            km = KMeans(
                k=k,
                seed=seed,
                featuresCol="features_elbow",
                predictionCol="cluster",
            )
            model = km.fit(feats)
            wssse = float(model.summary.trainingCost)
            rows.append({"k": int(k), "wssse": wssse})
    finally:
        feats.unpersist()
    spark = risk.sparkSession
    return spark.createDataFrame(rows).orderBy("k")


def load_kmeans_elbow(spark: SparkSession) -> DataFrame:
    """Read the cached elbow-sweep parquet (one row per k)."""
    return spark.read.parquet(resolve_path("outputs/nb6_kmeans_elbow.parquet"))


def risk_archetypes(risk: DataFrame, k: int = 4, seed: int = KMEANS_SEED):
    """Cluster sellers in the (demand_norm, sentiment_norm, network_norm)
    space via Spark ML K-Means, then label each cluster by which axis
    dominates its centroid. Returns (clustered_pdf, summary_pdf):

    * `clustered_pdf` — pandas, one row per seller, columns include
      `cluster` (int) and `archetype` (label).
    * `summary_pdf` — pandas, one row per cluster: n_sellers, mean of
      each component, mean risk_score, archetype label.

    Pure-Spark KMeans avoids a sklearn driver-side fit; the only escape
    is the final `.toPandas()` on the small (~3000-row) clustered frame
    used by the scatter chart and the summary table.
    """
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler

    components = ["demand_norm", "sentiment_norm", "network_norm"]
    assembler = VectorAssembler(inputCols=components, outputCol="features_archetype")
    risk_with_features = assembler.transform(risk)

    km = KMeans(
        k=k,
        seed=seed,
        featuresCol="features_archetype",
        predictionCol="cluster",
    )
    model = km.fit(risk_with_features)
    clustered = model.transform(risk_with_features).drop("features_archetype")

    centroids = model.clusterCenters()  # list of np.ndarray, one per cluster

    def _label_for(centroid):
        idx = int(centroid.argmax())
        component = components[idx]
        if max(centroid) < 0.30:
            return "low-risk"
        if component == "demand_norm":
            return "delay-driven"
        if component == "sentiment_norm":
            return "sentiment-driven"
        return "centrality-driven"

    cluster_to_label = {i: _label_for(c) for i, c in enumerate(centroids)}
    label_udf = F.udf(lambda c: cluster_to_label.get(int(c), "unknown"))
    clustered = clustered.withColumn("archetype", label_udf("cluster"))

    summary = (
        clustered.groupBy("cluster", "archetype")
        .agg(
            F.count("*").alias("n_sellers"),
            F.avg("demand_norm").alias("mean_demand_norm"),
            F.avg("sentiment_norm").alias("mean_sentiment_norm"),
            F.avg("network_norm").alias("mean_network_norm"),
            F.avg("risk_score").alias("mean_risk_score"),
        )
        .orderBy(F.col("mean_risk_score").desc())
    )

    # BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — ~3000-row scatter feed
    clustered_pdf = clustered.select(
        "seller_id", "seller_state", "demand_norm", "sentiment_norm",
        "network_norm", "risk_score", "risk_class", "cluster", "archetype",
    ).toPandas()

    # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — ≤k-row aggregate
    summary_pdf = summary.toPandas()

    return clustered_pdf, summary_pdf


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
