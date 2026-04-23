"""NB3 — supply-network graph analytics via GraphFrames.

Writes:
* `outputs/nb3_seller_network_scores.parquet` — feeds convergence.

Caches:
* `outputs/_cache/network_order_lines.parquet`
* `outputs/_cache/network_vertices.parquet`
* `outputs/_cache/network_edges.parquet`
* `outputs/_cache/network_pagerank.parquet`
* `outputs/_cache/network_connected_components.parquet`
* `outputs/_cache/network_motifs.parquet`
* `outputs/_cache/network_bfs_backups.parquet`
* `outputs/_cache/network_delayed_pagerank.parquet`
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from ..cache import step
from ..loaders import load_customers, load_order_items, load_orders, load_sellers
from ..safety import (  # noqa: F401 — referenced by annotation comments
    BFS_BACKUP_COLLECT,
    TOP10_PAGERANK_DRIVER,
)
from ..transforms import delivery_delay_days

_NET_CODE_DEPS = [
    "src/olist/pipeline/network.py",
    "src/olist/loaders.py",
    "src/olist/schemas.py",
    "src/olist/transforms.py",
]


# ---------------------------------------------------------------------------
# Order-line / vertex / edge construction
# ---------------------------------------------------------------------------


@step(
    name="network.order_lines",
    inputs=[
        "data/olist_orders_dataset.csv",
        "data/olist_order_items_dataset.csv",
        "data/olist_customers_dataset.csv",
        "data/olist_sellers_dataset.csv",
    ],
    outputs=["outputs/_cache/network_order_lines.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def build_order_lines(spark: SparkSession) -> DataFrame:
    """Delivered orders ⋈ order_items ⋈ broadcast(customers) ⋈ broadcast(sellers)
    projected to (order_id, customer_unique_id, seller_id, seller_state,
    delivery_delay_days, price). One row per delivered order line.
    """
    orders = load_orders(spark)
    order_items = load_order_items(spark)
    customers = load_customers(spark)
    sellers = load_sellers(spark)
    orders_delivered = orders.filter(
        F.col("order_delivered_customer_date").isNotNull()
    )
    order_lines = (
        orders_delivered.join(order_items, "order_id", "inner")
        .join(
            broadcast(customers.select("customer_id", "customer_unique_id")),
            "customer_id",
            "inner",
        )
        .join(
            broadcast(sellers.select("seller_id", "seller_state")),
            "seller_id",
            "left",
        )
    )
    order_lines = delivery_delay_days(order_lines)
    return order_lines.select(
        "order_id",
        "customer_unique_id",
        "seller_id",
        "seller_state",
        "delivery_delay_days",
        "price",
    )


@step(
    name="network.vertices",
    inputs=["outputs/_cache/network_order_lines.parquet"],
    outputs=["outputs/_cache/network_vertices.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def build_vertices(spark: SparkSession) -> DataFrame:
    """Vertex set = union of distinct sellers and distinct customer_unique_ids,
    tagged with a `type` column ('seller' / 'customer'). Uses
    `customer_unique_id` (not `customer_id`) per CLAUDE.md §4.
    """
    order_lines = spark.read.parquet("outputs/_cache/network_order_lines.parquet")
    sellers_v = (
        order_lines.select(F.col("seller_id").alias("id"))
        .distinct()
        .withColumn("type", F.lit("seller"))
    )
    customers_v = (
        order_lines.select(F.col("customer_unique_id").alias("id"))
        .distinct()
        .withColumn("type", F.lit("customer"))
    )
    return sellers_v.unionByName(customers_v)


@step(
    name="network.edges",
    inputs=["outputs/_cache/network_order_lines.parquet"],
    outputs=["outputs/_cache/network_edges.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def build_edges(spark: SparkSession) -> DataFrame:
    """Bidirectional edges: for each (customer_unique_id, seller_id) pair,
    emit one 'purchase' edge (customer→seller) and one 'serves' edge
    (seller→customer), with weight = order-item count and avg_delay.
    Bidirectionality is required so PageRank flows both ways and BFS can
    reach other sellers via shared customers.
    """
    order_lines = spark.read.parquet("outputs/_cache/network_order_lines.parquet")
    cust_seller_agg = order_lines.groupBy(
        "customer_unique_id", "seller_id"
    ).agg(
        F.count("*").alias("n_items"),
        F.avg("delivery_delay_days").alias("avg_delay"),
    )
    purchase_edges = cust_seller_agg.select(
        F.col("customer_unique_id").alias("src"),
        F.col("seller_id").alias("dst"),
        F.col("n_items").alias("weight"),
        F.col("avg_delay"),
        F.lit("purchase").alias("edge_type"),
    )
    serves_edges = cust_seller_agg.select(
        F.col("seller_id").alias("src"),
        F.col("customer_unique_id").alias("dst"),
        F.col("n_items").alias("weight"),
        F.col("avg_delay"),
        F.lit("serves").alias("edge_type"),
    )
    return purchase_edges.unionByName(serves_edges)


def build_graph_frame(spark: SparkSession):
    """Assemble the `GraphFrame(v, e)` from cached parquet vertex / edge sets.
    Cheap to reconstruct — the expensive compute is the GraphFrame algorithms,
    each of which is its own cached step.
    """
    from graphframes import GraphFrame

    vertices = spark.read.parquet("outputs/_cache/network_vertices.parquet")
    edges = spark.read.parquet("outputs/_cache/network_edges.parquet")
    return GraphFrame(vertices, edges)


# ---------------------------------------------------------------------------
# GraphFrame algorithms
# ---------------------------------------------------------------------------


def seller_degree_stats(spark: SparkSession) -> DataFrame:
    """Seller in-degree (total + purchase-only). Not cached — cheap."""
    vertices = spark.read.parquet("outputs/_cache/network_vertices.parquet")
    edges = spark.read.parquet("outputs/_cache/network_edges.parquet")
    gf = build_graph_frame(spark)
    in_degrees = gf.inDegrees
    purchase_in = (
        edges.filter(F.col("edge_type") == "purchase")
        .groupBy("dst")
        .agg(F.count("*").alias("in_degree_purchase_only"))
        .withColumnRenamed("dst", "id")
    )
    return (
        vertices.filter(F.col("type") == "seller")
        .join(in_degrees, "id", "left")
        .join(purchase_in, "id", "left")
        .withColumnRenamed("inDegree", "in_degree_total")
        .fillna({"in_degree_total": 0, "in_degree_purchase_only": 0})
    )


@step(
    name="network.pagerank",
    inputs=[
        "outputs/_cache/network_vertices.parquet",
        "outputs/_cache/network_edges.parquet",
    ],
    outputs=["outputs/_cache/network_pagerank.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_pagerank(spark: SparkSession) -> DataFrame:
    """PageRank over the bidirectional graph. Seller vertices only,
    column renamed to `pagerank_score` for downstream consumers.
    `resetProbability=0.15` + `maxIter=10` match the pre-refactor NB3.
    """
    gf = build_graph_frame(spark)
    pr = gf.pageRank(resetProbability=0.15, maxIter=10)
    return pr.vertices.filter(F.col("type") == "seller").select(
        "id", F.col("pagerank").alias("pagerank_score")
    )


@step(
    name="network.connected_components",
    inputs=[
        "outputs/_cache/network_vertices.parquet",
        "outputs/_cache/network_edges.parquet",
    ],
    outputs=["outputs/_cache/network_connected_components.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_connected_components(spark: SparkSession) -> DataFrame:
    """Connected components via the GraphX-backed algorithm.

    The default message-passing variant OOMs the JVM heap on ~100k vertices
    even at 6g driver memory (see decisions_log.md 2026-04-22 NB3 entry);
    GraphX CC is more memory-efficient. Returns per-vertex (id, type,
    component, component_size). Flagged sellers with component_size == 1
    are "isolated".
    """
    gf = build_graph_frame(spark)
    root = Path(__file__).resolve().parents[3]
    checkpoint_dir = root / "outputs" / "_gf_checkpoints"
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir))
    cc = gf.connectedComponents(algorithm="graphx")
    component_sizes = cc.groupBy("component").agg(
        F.count("*").alias("component_size")
    )
    return cc.join(component_sizes, "component", "left")


@step(
    name="network.motifs_shared_customers",
    inputs=[
        "outputs/_cache/network_vertices.parquet",
        "outputs/_cache/network_edges.parquet",
    ],
    outputs=["outputs/_cache/network_motifs.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_shared_customer_motifs(spark: SparkSession) -> DataFrame:
    """Motif `(a)-[e1]->(c); (b)-[e2]->(c)` — two sellers `a` and `b` sharing
    a common customer `c`. Filter to pairs where both edges are 'serves'
    (seller→customer) so the customer `c` is a true shared customer, and
    a.id < b.id to dedupe unordered pairs. Aggregate to (seller_a,
    seller_b, n_shared_customers).
    """
    gf = build_graph_frame(spark)
    motifs = gf.find("(a)-[e1]->(c); (b)-[e2]->(c)")
    return (
        motifs.filter(
            (F.col("e1.edge_type") == "serves")
            & (F.col("e2.edge_type") == "serves")
            & (F.col("a.id") < F.col("b.id"))
        )
        .groupBy(
            F.col("a.id").alias("seller_a"), F.col("b.id").alias("seller_b")
        )
        .agg(F.count("*").alias("n_shared_customers"))
    )


@step(
    name="network.bfs_backups",
    inputs=[
        "outputs/_cache/network_pagerank.parquet",
        "outputs/_cache/network_vertices.parquet",
        "outputs/_cache/network_edges.parquet",
    ],
    outputs=["outputs/_cache/network_bfs_backups.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_bfs_backups(spark: SparkSession) -> DataFrame:
    """For each of the top-10 PageRank sellers, run a BFS up to path-length 3
    to the nearest other seller vertex. The first hit is reported as the
    backup seller. Returns (seller_id, backup_seller_id).

    Two flagged escapes: TOP10_PAGERANK_DRIVER (driver-side list of 10 ids)
    and BFS_BACKUP_COLLECT (one row per loop iteration). Both are capped
    by construction.
    """
    seller_pagerank = spark.read.parquet("outputs/_cache/network_pagerank.parquet")
    gf = build_graph_frame(spark)
    # BIG-DATA-SAFETY-ESCAPE: TOP10_PAGERANK_DRIVER — 10-row driver list
    top10 = [
        row["id"]
        for row in seller_pagerank.orderBy(F.col("pagerank_score").desc())
        .limit(10)
        .collect()
    ]
    backup_rows: list[tuple[str, str | None]] = []
    for seller_id in top10:
        paths = gf.bfs(
            fromExpr=f"id = '{seller_id}'",
            toExpr=f"type = 'seller' AND id != '{seller_id}'",
            maxPathLength=3,
        )
        # BIG-DATA-SAFETY-ESCAPE: BFS_BACKUP_COLLECT — limit(1) per iteration
        first = paths.limit(1).collect()
        backup = first[0]["to"]["id"] if first else None
        backup_rows.append((seller_id, backup))
    return spark.createDataFrame(
        backup_rows, "seller_id string, backup_seller_id string"
    )


@step(
    name="network.delayed_pagerank",
    inputs=[
        "outputs/_cache/network_vertices.parquet",
        "outputs/_cache/network_edges.parquet",
    ],
    outputs=["outputs/_cache/network_delayed_pagerank.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_delayed_subgraph_pagerank(spark: SparkSession) -> DataFrame:
    """Induced subgraph over edges with avg_delay > 5 days; re-run PageRank.
    Seller vertices with high rank here are structurally central to the
    *late-shipping* part of the network. Column renamed `network_risk_score`.
    """
    from graphframes import GraphFrame

    vertices = spark.read.parquet("outputs/_cache/network_vertices.parquet")
    edges = spark.read.parquet("outputs/_cache/network_edges.parquet")
    delayed_edges = edges.filter(F.col("avg_delay") > 5)
    delayed_vertex_ids = (
        delayed_edges.select(F.col("src").alias("id"))
        .unionByName(delayed_edges.select(F.col("dst").alias("id")))
        .distinct()
    )
    delayed_vertices = vertices.join(delayed_vertex_ids, "id", "inner")
    g_delayed = GraphFrame(delayed_vertices, delayed_edges)
    pr_delayed = g_delayed.pageRank(resetProbability=0.15, maxIter=10)
    return pr_delayed.vertices.filter(F.col("type") == "seller").select(
        "id", F.col("pagerank").alias("network_risk_score")
    )


# ---------------------------------------------------------------------------
# Final per-seller network scores (feeds convergence)
# ---------------------------------------------------------------------------


@step(
    name="network.seller_scores",
    inputs=[
        "outputs/_cache/network_vertices.parquet",
        "outputs/_cache/network_pagerank.parquet",
        "outputs/_cache/network_connected_components.parquet",
        "outputs/_cache/network_bfs_backups.parquet",
        "outputs/_cache/network_delayed_pagerank.parquet",
    ],
    outputs=["outputs/nb3_seller_network_scores.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def build_seller_network_scores(spark: SparkSession) -> DataFrame:
    """Assemble per-seller network scores from the cached GraphFrame outputs.

    Columns: seller_id, pagerank_score, in_degree, is_isolated,
    backup_seller_id, network_risk_score.
    """
    vertices = spark.read.parquet("outputs/_cache/network_vertices.parquet")
    seller_pagerank = spark.read.parquet("outputs/_cache/network_pagerank.parquet")
    cc = spark.read.parquet("outputs/_cache/network_connected_components.parquet")
    bfs_backups = spark.read.parquet("outputs/_cache/network_bfs_backups.parquet")
    delayed = spark.read.parquet("outputs/_cache/network_delayed_pagerank.parquet")

    # Recompute per-seller degree stats (cheap).
    seller_degrees = seller_degree_stats(spark)
    isolated_sellers = (
        cc.filter((F.col("type") == "seller") & (F.col("component_size") == 1))
        .select("id")
        .withColumn("is_isolated", F.lit(1))
    )
    return (
        vertices.filter(F.col("type") == "seller")
        .select(F.col("id").alias("seller_id"))
        .join(
            seller_pagerank.withColumnRenamed("id", "seller_id"),
            "seller_id",
            "left",
        )
        .join(
            seller_degrees.select(
                F.col("id").alias("seller_id"),
                F.col("in_degree_purchase_only").alias("in_degree"),
            ),
            "seller_id",
            "left",
        )
        .join(
            isolated_sellers.withColumnRenamed("id", "seller_id"),
            "seller_id",
            "left",
        )
        .join(bfs_backups, "seller_id", "left")
        .join(
            delayed.withColumnRenamed("id", "seller_id"),
            "seller_id",
            "left",
        )
        .fillna(
            {
                "is_isolated": 0,
                "pagerank_score": 0.0,
                "in_degree": 0,
                "network_risk_score": 0.0,
            }
        )
    )
