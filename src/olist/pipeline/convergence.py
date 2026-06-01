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
* Percentile bands on the composite: CRITICAL = top 1%, WARNING = next 4%,
  SAFE = bottom 95%.

Normalisation is percentile-rank (not min-max on p1/p99 bounds). Min-max left
``demand_norm`` (std 0.034) and ``network_norm`` (std 0.056) nearly constant —
both source columns are heavily right-skewed with a long thin tail, so once you
clamp to p1/p99 almost every seller lands near the same value. That made the
composite ~96% the star rating (``corr(risk_score, sentiment_norm)=0.96``) and
left the CRITICAL band mathematically dead (max risk_score 0.705 < 0.75).
Percentile rank makes each axis ~uniform on [0, 1] so the three signals
contribute equally under the existing weights, and percentile bands guarantee
the CRITICAL band actually fires.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..cache import resolve_path, step
from ..safety import (  # noqa: F401 — referenced by annotation comments
    RISK_NORM_AGG,
    STATE_AGG_VIZ,
    TOP50_VIZ,
)

_CONV_CODE_DEPS = ["src/olist/pipeline/convergence.py"]

RISK_WEIGHTS = {"demand": 0.35, "sentiment": 0.35, "network": 0.30}
# Percentile cut-points on the composite risk_score (fraction of population).
RISK_CRITICAL_PCTL = 0.99  # top 1% → CRITICAL
RISK_WARNING_PCTL = 0.95   # next 4% (>0.95, ≤0.99) → WARNING; ≤0.95 → SAFE

# Retained for import-compatibility with scripts/build_main.py. Banding is now
# percentile-based (see RISK_*_PCTL), so these absolute thresholds are no longer
# used to assign risk_class — kept only so existing imports/prints don't break.
RISK_CRITICAL_THRESHOLD = 0.75
RISK_SAFE_THRESHOLD = 0.40


def _percentile_rank(col: str, invert: bool = False):
    """Percentile-rank a column into [0, 1] via a window ``percent_rank()``.

    ``percent_rank`` returns ``(rank - 1) / (n - 1)`` over the ordered
    partition, so the lowest value maps to 0.0, the highest to 1.0, and the
    population is spread ~uniformly in between regardless of the raw column's
    skew. This is what lets the three axes contribute equally under fixed
    weights — each becomes a rank, not a clamped distance from a percentile
    bound. ``invert=True`` ranks descending (used for sentiment, where a higher
    star rating means *lower* risk).
    """
    order = F.col(col).desc() if invert else F.col(col).asc()
    return F.percent_rank().over(Window.orderBy(order))


@step(
    name="convergence.seller_risk_index",
    inputs=[
        "outputs/nb1_seller_demand_scores.parquet",
        "outputs/nb2_seller_sentiment_scores.parquet",
        "outputs/nb3_seller_network_scores.parquet",
    ],
    outputs=["outputs/seller_risk_index.parquet"],
    code_deps=_CONV_CODE_DEPS,
    version=2,
)
def build_seller_risk_index(spark: SparkSession) -> DataFrame:
    """Inner-join the three per-seller parquets, percentile-rank normalise each
    risk component, weight 0.35/0.35/0.30, band the composite by percentile
    (CRITICAL top 1% / WARNING next 4% / SAFE bottom 95%), write
    `outputs/seller_risk_index.parquet`.

    Each `*_norm` column is the seller's percentile rank in [0, 1] for that
    component, so all four point the same direction (higher = more risky) and
    each axis is ~uniform on [0, 1]. Sentiment is ranked descending (a higher
    star rating is *lower* risk). Percentile ranking replaced the old
    min-max(p1/p99) scaling, which left demand and network nearly constant and
    let the star rating dominate the composite.

    The network axis is a 50/50 blend of *contagion* (delayed-subgraph
    PageRank — a late-shipping hub) and *substitutability deficit* (high
    impact, no backup), each percentile-ranked first. The deficit half is
    graph-unique and non-degenerate across the whole population, so the network
    axis no longer collapses to a degree proxy and is no longer zero for the
    two-thirds of sellers outside the late-shipping subgraph.
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
    return (
        # Percentile-rank each raw component into [0, 1] (sentiment descending).
        joined.withColumn(
            "demand_norm",
            _percentile_rank("avg_delay_days"),
        )
        .withColumn(
            "sentiment_norm",
            _percentile_rank("avg_sentiment_score", invert=True),
        )
        .withColumn(
            "contagion_norm",
            _percentile_rank("network_risk_score"),
        )
        .withColumn(
            "deficit_norm",
            _percentile_rank("substitutability_deficit"),
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
        # Band by the composite's own percentile so CRITICAL always fires.
        .withColumn(
            "risk_pctl",
            F.percent_rank().over(Window.orderBy(F.col("risk_score").asc())),
        )
        .withColumn(
            "risk_class",
            F.when(F.col("risk_pctl") > RISK_CRITICAL_PCTL, "CRITICAL")
            .when(F.col("risk_pctl") > RISK_WARNING_PCTL, "WARNING")
            .otherwise("SAFE"),
        )
        .drop("risk_pctl")
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


def risk_component_correlations(spark: SparkSession) -> DataFrame:
    """Read ``outputs/seller_risk_index.parquet`` and return the Pearson
    correlation of ``risk_score`` with each axis as a one-row Spark DataFrame
    with columns ``corr_demand``, ``corr_sentiment``, ``corr_network``.

    The notebook prints this to prove the three signals now contribute in a
    balanced way: under the old min-max normalisation ``corr_sentiment`` was
    ~0.96 (the composite was essentially the star rating). With percentile-rank
    normalisation the three correlations should be close to one another.
    """
    risk = spark.read.parquet(resolve_path("outputs/seller_risk_index.parquet"))
    return risk.select(
        F.corr("risk_score", "demand_norm").alias("corr_demand"),
        F.corr("risk_score", "sentiment_norm").alias("corr_sentiment"),
        F.corr("risk_score", "network_norm").alias("corr_network"),
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
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import VectorAssembler

    components = ["demand_norm", "sentiment_norm", "network_norm"]
    assembler = VectorAssembler(inputCols=components, outputCol="features_elbow")
    feats = assembler.transform(risk).cache()
    # Two complementary cluster-validation signals: WSSSE (elbow) and the
    # silhouette score (separation/cohesion, higher is better). The elbow is a
    # heuristic; silhouette gives a defensible second opinion on k.
    silhouette_eval = ClusteringEvaluator(
        featuresCol="features_elbow",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
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
            silhouette = float(silhouette_eval.evaluate(model.transform(feats)))
            rows.append({"k": int(k), "wssse": wssse, "silhouette": silhouette})
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
