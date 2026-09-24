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

Normalisation is (approximate) percentile-rank, not min-max on p1/p99 bounds.
Min-max left ``demand_norm`` (std 0.034) and ``network_norm`` (std 0.056) nearly
constant — both source columns are heavily right-skewed with a long thin tail,
so once you clamp to p1/p99 almost every seller lands near the same value. That
made the composite ~96% the star rating (``corr(risk_score, sentiment_norm)=
0.96``) and left the CRITICAL band mathematically dead (max risk_score 0.705 <
0.75). Percentile rank makes each axis ~uniform on [0, 1] so the three signals
contribute equally under the existing weights, and percentile bands guarantee
the CRITICAL band actually fires.

Both legs are computed with **distributable** primitives — ``QuantileDiscretizer``
for the per-axis ranks and ``approxQuantile`` for the band cut-points (both use
per-partition sketches) — rather than ``percent_rank().over(Window.orderBy(...))``,
which has no ``partitionBy`` and would move the whole population onto one
partition to sort. At this dataset size the approximation is indistinguishable
from exact percentile rank; at scale it is the difference between a job that
distributes and one that does not.
"""

from __future__ import annotations

from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..cache import resolve_path, step
from ..safety import (  # noqa: F401 — referenced by annotation comments
    STATE_AGG_VIZ,
    TOP50_VIZ,
)

_CONV_CODE_DEPS = ["src/olist/pipeline/convergence.py"]

RISK_WEIGHTS = {"demand": 0.35, "sentiment": 0.35, "network": 0.30}
# These weights are a business judgement fixed in advance, NOT fitted to an
# outcome label (none exists in the data — validating the index against
# realised churn/failure is logged as future work). Demand and sentiment are
# weighted equally as the two direct seller-health signals; the network axis
# is slightly lower as a structural modifier. No sensitivity analysis has been
# run, so treat the exact 0.35/0.35/0.30 split as a prior, not a tuned value.
# Percentile cut-points on the composite risk_score (fraction of population).
RISK_CRITICAL_PCTL = 0.99  # top 1% → CRITICAL
RISK_WARNING_PCTL = 0.95   # next 4% (>0.95, ≤0.99) → WARNING; ≤0.95 → SAFE

# Retained for import-compatibility with scripts/build_main.py. Banding is now
# percentile-based (see RISK_*_PCTL), so these absolute thresholds are no longer
# used to assign risk_class — kept only so existing imports/prints don't break.
RISK_CRITICAL_THRESHOLD = 0.75
RISK_SAFE_THRESHOLD = 0.40


#: Granularity of the distributed percentile-rank approximation. 1000 buckets
#: gives ~0.1% resolution — indistinguishable from exact percent_rank at this
#: population, but computed via approxQuantile sketches that scale across
#: partitions (unlike an unpartitioned Window.orderBy, which drags the whole
#: population onto one partition to sort).
NORM_BUCKETS = 1000


def _scalable_percentile_norm(
    df: DataFrame, in_col: str, out_col: str, invert: bool = False
) -> DataFrame:
    """Approximate percentile rank of ``in_col`` into ``out_col`` ∈ [0, 1],
    computed so it scales across partitions.

    Replaces ``percent_rank().over(Window.orderBy(col))`` — which has no
    ``partitionBy`` and so moves the entire population onto a single partition to
    sort, the textbook "won't scale across partitions" anti-pattern. Instead
    ``QuantileDiscretizer`` bins the column into equal-frequency buckets using
    per-partition ``approxQuantile`` sketches merged across the cluster, so the
    work distributes. The bucket index normalised by the max realised bucket
    gives an ~uniform [0, 1] rank, which is what lets the three axes contribute
    equally under fixed weights. ``invert=True`` ranks descending (sentiment: a
    higher star rating means *lower* risk). Ties share a bucket — the natural,
    and arguably more correct, handling for the heavily-tied delay column.
    """
    src = in_col
    if invert:
        src = f"_neg_{in_col}"
        df = df.withColumn(src, -F.col(in_col))
    bkt = f"_bkt_{out_col}"
    discretizer = QuantileDiscretizer(
        numBuckets=NORM_BUCKETS,
        inputCol=src,
        outputCol=bkt,
        relativeError=0.001,
        handleInvalid="keep",
    )
    df = discretizer.fit(df).transform(df)
    # Max bucket as a single scalar (distributed agg, one row to the driver) so
    # each axis spans the full [0, 1] regardless of how many buckets it realised.
    # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — single-scalar max bucket
    max_bkt = df.agg(F.max(bkt)).first()[0]
    denom = float(max_bkt) if max_bkt and max_bkt > 0 else 1.0
    df = df.withColumn(out_col, F.col(bkt) / F.lit(denom))
    return df.drop(*([bkt] + ([src] if invert else [])))


@step(
    name="convergence.seller_risk_index",
    inputs=[
        "outputs/nb1_seller_demand_scores.parquet",
        "outputs/nb2_seller_sentiment_scores.parquet",
        "outputs/nb3_seller_network_scores.parquet",
    ],
    outputs=["outputs/seller_risk_index.parquet"],
    code_deps=_CONV_CODE_DEPS,
    version=5,
)
def build_seller_risk_index(spark: SparkSession) -> DataFrame:
    """Inner-join the three per-seller parquets, normalise each risk component to
    an approximate percentile rank, weight 0.35/0.35/0.30, band the composite
    (CRITICAL top 1% / WARNING next 4% / SAFE bottom 95%), write
    `outputs/seller_risk_index.parquet`.

    Each `*_norm` column is the seller's approximate percentile rank in [0, 1]
    for that component (via `_scalable_percentile_norm` → `QuantileDiscretizer`),
    so all four point the same direction (higher = more risky) and each axis is
    ~uniform on [0, 1]. Sentiment is ranked descending (a higher star rating is
    *lower* risk). Both the normalisation and the banding are **distributable**:
    `QuantileDiscretizer` and `approxQuantile` use per-partition sketches rather
    than the single-partition `percent_rank().over(Window.orderBy(...))` they
    replaced (which dragged the whole population onto one partition to sort).
    The percentile framing itself replaced the earlier min-max(p1/p99) scaling,
    which left demand and network nearly constant and let the star rating
    dominate the composite.

    The network axis is a 50/50 blend of *contagion* (delayed-subgraph
    PageRank — a late-shipping hub) and *supply concentration* (few same-category,
    same-state substitutes = a single point of failure), each percentile-ranked
    first. The supply-concentration half comes from the DENSE co-category+region
    projection (`supply_concentration_risk`), which covers nearly all sellers and
    measures real substitutability — replacing the old degree-proxy
    `substitutability_deficit` (retained only for the §5.5 honesty diagnostic).
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
                "supply_concentration_risk",
                "n_category_substitutes",
                "catregion_backup_seller_id",
                "backup_seller_id",
                "backup_strength",
                "community_id",
                "two_hop_reach_count",
            ),
            "seller_id",
            "inner",
        )
    )
    # Approximate-percentile-rank each raw component into [0, 1] (sentiment
    # descending). Distributable — no single-partition window.
    normed = _scalable_percentile_norm(joined, "avg_delay_days", "demand_norm")
    normed = _scalable_percentile_norm(
        normed, "avg_sentiment_score", "sentiment_norm", invert=True
    )
    normed = _scalable_percentile_norm(
        normed, "network_risk_score", "contagion_norm"
    )
    # Deficit half of the network axis is now the DENSE co-category+region
    # supply-concentration (few same-category, same-state substitutes = high
    # risk), replacing the old degree-proxy substitutability_deficit.
    normed = _scalable_percentile_norm(
        normed, "supply_concentration_risk", "supply_norm"
    )
    normed = normed.withColumn(
        "network_norm",
        0.5 * F.col("contagion_norm") + 0.5 * F.col("supply_norm"),
    ).withColumn(
        "risk_score",
        RISK_WEIGHTS["demand"] * F.col("demand_norm")
        + RISK_WEIGHTS["sentiment"] * F.col("sentiment_norm")
        + RISK_WEIGHTS["network"] * F.col("network_norm"),
    )
    # Band by distributed approxQuantile cut-points on the composite (the 95th
    # and 99th percentiles) instead of an unpartitioned percent_rank window. Two
    # scalars returned to the driver; CRITICAL = top 1%, WARNING = next 4%.
    warn_cut, crit_cut = normed.approxQuantile(
        "risk_score", [RISK_WARNING_PCTL, RISK_CRITICAL_PCTL], 0.001
    )
    return (
        normed.withColumn(
            "risk_class",
            F.when(F.col("risk_score") > crit_cut, "CRITICAL")
            .when(F.col("risk_score") > warn_cut, "WARNING")
            .otherwise("SAFE"),
        )
        # Escalate a non-SAFE seller only when it has NO substitute at all — no
        # same-category/same-state competitor (the dense, decision-grade test), no
        # direct co-customer backup, and no 2-hop one. With the co-category graph
        # this flags genuine single-points-of-failure (no one else sells the
        # category in the region), not artefacts of co-customer sparsity.
        .withColumn(
            "escalate_no_backup",
            (
                (F.col("risk_class") != "SAFE")
                & (F.coalesce(F.col("n_category_substitutes"), F.lit(0)) == 0)
                & (F.coalesce(F.col("backup_strength"), F.lit(0)) == 0)
                & (F.coalesce(F.col("two_hop_reach_count"), F.lit(0)) == 0)
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
            "supply_norm",
            "avg_delay_days",
            "avg_sentiment_score",
            "pagerank_score",
            "substitutability_deficit",
            "supply_concentration_risk",
            "n_category_substitutes",
            "catregion_backup_seller_id",
            "backup_seller_id",
            "backup_strength",
            "two_hop_reach_count",
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


#: Short, management-facing action label per archetype. Surfaced as the
#: `recommended_action` column on the §6.5 watchlist and mirrored on the
#: presentation's watchlist slide — the archetype typology (§6.3) drives the
#: *kind* of intervention each flagged seller needs.
ARCHETYPE_ACTIONS = {
    "delay-driven":     "Logistics audit",
    "sentiment-driven": "Account-manager call",
    "centrality-driven":"Dual-source + watchlist",
    "low-risk":         "Routine monitoring",
}


def recommend_action(archetype: str, escalate_no_backup: bool = False) -> str:
    """Map a seller's risk archetype to a single management-facing action.

    A seller flagged ``escalate_no_backup`` (non-SAFE with no substitute
    anywhere — neither a co-customer/2-hop backup nor a same-category one,
    see §5.5/§6) is escalated regardless of archetype: a single point of
    failure is the first thing to fix. Otherwise the action follows the
    archetype typology of §6.3.
    """
    if escalate_no_backup:
        return "Escalate — no backup"
    return ARCHETYPE_ACTIONS.get(archetype, "Routine monitoring")


def top20_watchlist(risk: DataFrame, clustered_pdf, n: int = 20):
    """Top-``n`` sellers by ``risk_score`` as a presentation-ready pandas
    frame, with the per-seller risk ``archetype`` (from ``risk_archetypes``)
    and a single ``recommended_action`` (see ``recommend_action``) joined on.

    This is the deployable intervention short-list rendered in §6.5 and
    mirrored on the presentation's watchlist slide: who to act on, and what
    action — derived from the archetype, with no-backup sellers escalated.
    """
    # BIG-DATA-SAFETY-ESCAPE: PLOTLY_STATIC_VIZ — capped to n rows
    top = (
        risk.orderBy(F.col("risk_score").desc())
        .limit(n)
        .select(
            "seller_id", "seller_state", "risk_score", "risk_class",
            "demand_norm", "sentiment_norm", "network_norm",
            "escalate_no_backup",
        )
        .toPandas()
    )
    archetype_lookup = clustered_pdf[["seller_id", "archetype"]]
    top = top.merge(archetype_lookup, on="seller_id", how="left")
    top["recommended_action"] = [
        recommend_action(archetype, bool(escalate))
        for archetype, escalate in zip(top["archetype"], top["escalate_no_backup"])
    ]
    return top.drop(columns=["escalate_no_backup"])


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
        # "low-risk" when even the largest centroid axis is below 0.30 on the
        # [0,1] percentile-normalised scale — i.e. the cluster is not elevated
        # on ANY of the three signals. 0.30 is a low-but-non-trivial cut chosen
        # so the bulk of the marketplace (well below the risk bands) is not
        # given a misleading "driven-by-X" label; the three named archetypes
        # are reserved for clusters that genuinely peak on an axis.
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
        # BIG-DATA-SAFETY-ESCAPE: STATE_AGG_VIZ — 27-row sorted aggregate
        .toPandas()
    )
