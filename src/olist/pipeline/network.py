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
* `outputs/_cache/network_cocustomer_edges.parquet`
* `outputs/_cache/network_cocustomer_centrality.parquet`
* `outputs/_cache/network_communities.parquet`
* `outputs/_cache/network_backup_map.parquet`
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from ..cache import resolve_path, step
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
    order_lines = spark.read.parquet(resolve_path("outputs/_cache/network_order_lines.parquet"))
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
    order_lines = spark.read.parquet(resolve_path("outputs/_cache/network_order_lines.parquet"))
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
    # Cheap; algorithms are cached individually so we just re-load v, e here.
    from graphframes import GraphFrame

    vertices = spark.read.parquet(resolve_path("outputs/_cache/network_vertices.parquet"))
    edges = spark.read.parquet(resolve_path("outputs/_cache/network_edges.parquet"))
    return GraphFrame(vertices, edges)


# ---------------------------------------------------------------------------
# GraphFrame algorithms
# ---------------------------------------------------------------------------


def seller_degree_stats(spark: SparkSession) -> DataFrame:
    """Seller in-degree (total + purchase-only). Not cached — cheap."""
    vertices = spark.read.parquet(resolve_path("outputs/_cache/network_vertices.parquet"))
    edges = spark.read.parquet(resolve_path("outputs/_cache/network_edges.parquet"))
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
    """PageRank on the bidirectional graph; sellers only. resetProb=0.15, maxIter=10."""
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
    # GraphX variant: the default CC OOMs the 6g driver on ~100k vertices.
    # See decisions_log 2026-04-22. component_size == 1 => isolated seller.
    gf = build_graph_frame(spark)
    root = Path(__file__).resolve().parents[3]
    checkpoint_dir = root / "outputs" / "_gf_checkpoints"
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir))
    cc = gf.connectedComponents(algorithm="graphx")
    component_sizes = cc.groupBy("component").agg(
        F.count("*").alias("component_size")
    )
    return cc.join(component_sizes, "component", "left")


def component_size_histogram(spark: SparkSession) -> DataFrame:
    """Bin distinct components by size (isolated / 2–4 / 5–9 / 10–99 / 100+)
    and count components per bin. Five-row small aggregate, drives the
    component-size distribution bar in NB3 §5.
    """
    cc = spark.read.parquet(resolve_path("outputs/_cache/network_connected_components.parquet"))
    distinct_components = cc.select("component", "component_size").distinct()
    return (
        distinct_components.withColumn(
            "size_bin",
            F.when(F.col("component_size") == 1, "1 (isolated)")
            .when(F.col("component_size") <= 4, "2–4")
            .when(F.col("component_size") <= 9, "5–9")
            .when(F.col("component_size") <= 99, "10–99")
            .otherwise("100+"),
        )
        .groupBy("size_bin")
        .agg(F.count("*").alias("n_components"))
    )


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


# ---------------------------------------------------------------------------
# Seller↔seller co-customer projection — the graph-unique layer
# ---------------------------------------------------------------------------


@step(
    name="network.cocustomer_edges",
    inputs=["outputs/_cache/network_motifs.parquet"],
    outputs=["outputs/_cache/network_cocustomer_edges.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def build_cocustomer_edges(spark: SparkSession) -> DataFrame:
    """Project the bipartite graph onto sellers: two sellers are linked when
    they share customers. Edges come straight from the shared-customer motif,
    kept only when a pair shares >= 2 customers (drops one-off coincidences,
    keeps the projection sparse). Emitted both directions so PageRank and label
    propagation treat the projection as undirected.
    """
    motifs = spark.read.parquet(resolve_path("outputs/_cache/network_motifs.parquet"))
    strong = motifs.filter(F.col("n_shared_customers") >= 2)
    forward = strong.select(
        F.col("seller_a").alias("src"),
        F.col("seller_b").alias("dst"),
        F.col("n_shared_customers").alias("weight"),
    )
    backward = strong.select(
        F.col("seller_b").alias("src"),
        F.col("seller_a").alias("dst"),
        F.col("n_shared_customers").alias("weight"),
    )
    return forward.unionByName(backward)


def build_cocustomer_graph(spark: SparkSession):
    """Reconstruct the seller-projection GraphFrame from cached edges.
    Vertices = sellers appearing in any co-customer edge.
    """
    from graphframes import GraphFrame

    edges = spark.read.parquet(resolve_path("outputs/_cache/network_cocustomer_edges.parquet"))
    vertices = (
        edges.select(F.col("src").alias("id"))
        .unionByName(edges.select(F.col("dst").alias("id")))
        .distinct()
    )
    return GraphFrame(vertices, edges)


@step(
    name="network.cocustomer_centrality",
    inputs=["outputs/_cache/network_cocustomer_edges.parquet"],
    outputs=["outputs/_cache/network_cocustomer_centrality.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_cocustomer_centrality(spark: SparkSession) -> DataFrame:
    """PageRank on the seller co-customer projection = how embedded a seller is
    in the substitution network. Structurally distinct from raw in-degree
    (customer count): a seller can serve many customers yet sit at the edge of
    the substitution graph, or serve few yet bridge clusters. We report the
    correlation against in-degree in the notebook to prove it is not a proxy.
    """
    gf = build_cocustomer_graph(spark)
    pr = gf.pageRank(resetProbability=0.15, maxIter=10)
    return pr.vertices.select(
        F.col("id").alias("seller_id"),
        F.col("pagerank").alias("cocustomer_centrality"),
    )


@step(
    name="network.communities",
    inputs=["outputs/_cache/network_cocustomer_edges.parquet"],
    outputs=["outputs/_cache/network_communities.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_substitution_communities(spark: SparkSession) -> DataFrame:
    """Label propagation on the co-customer projection groups sellers into
    substitution communities — clusters that can absorb each other's demand if
    one member fails. Far more actionable than the bipartite connected-
    components result (one giant component, 0 isolated sellers).
    Columns: seller_id, community_id, community_size.
    """
    gf = build_cocustomer_graph(spark)
    communities = gf.labelPropagation(maxIter=5)
    sizes = communities.groupBy("label").agg(F.count("*").alias("community_size"))
    return communities.join(sizes, "label", "left").select(
        F.col("id").alias("seller_id"),
        F.col("label").alias("community_id"),
        "community_size",
    )


@step(
    name="network.backup_map",
    inputs=["outputs/_cache/network_motifs.parquet"],
    outputs=["outputs/_cache/network_backup_map.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=1,
)
def compute_backup_map(spark: SparkSession) -> DataFrame:
    """For every seller with at least one co-customer partner, find the partner
    sharing the most customers (its natural backup) and count how many distinct
    substitutes exist. Unlike the top-10 BFS, this covers ALL sellers and feeds
    the risk index. Columns: seller_id, backup_seller_id, backup_strength,
    n_cocustomer_partners.
    """
    from pyspark.sql.window import Window

    motifs = spark.read.parquet(resolve_path("outputs/_cache/network_motifs.parquet"))
    directed = motifs.select(
        F.col("seller_a").alias("seller_id"),
        F.col("seller_b").alias("partner"),
        F.col("n_shared_customers"),
    ).unionByName(
        motifs.select(
            F.col("seller_b").alias("seller_id"),
            F.col("seller_a").alias("partner"),
            F.col("n_shared_customers"),
        )
    )
    partner_counts = directed.groupBy("seller_id").agg(
        F.count("*").alias("n_cocustomer_partners")
    )
    pick_best = Window.partitionBy("seller_id").orderBy(
        F.col("n_shared_customers").desc(), F.col("partner")
    )
    best = (
        directed.withColumn("rk", F.row_number().over(pick_best))
        .filter(F.col("rk") == 1)
        .select(
            "seller_id",
            F.col("partner").alias("backup_seller_id"),
            F.col("n_shared_customers").alias("backup_strength"),
        )
    )
    return best.join(partner_counts, "seller_id", "left")


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
    """For each top-10 PageRank seller, BFS up to length 3 to the nearest
    other seller. Returns (seller_id, backup_seller_id, hop_count).

    hop_count = path length to the backup (1, 2, or 3); null if no path
    found within the cap. The distribution of hop_count is a graph-native
    finding: it tells us how far the structural hubs are from their
    nearest substitute, which is impossible to derive without traversal.
    """
    seller_pagerank = spark.read.parquet(resolve_path("outputs/_cache/network_pagerank.parquet"))
    gf = build_graph_frame(spark)
    # BIG-DATA-SAFETY-ESCAPE: TOP10_PAGERANK_DRIVER — 10-row driver list
    top10 = [
        row["id"]
        for row in seller_pagerank.orderBy(F.col("pagerank_score").desc())
        .limit(10)
        .collect()
    ]
    backup_rows: list[tuple[str, str | None, int | None]] = []
    for seller_id in top10:
        paths = gf.bfs(
            fromExpr=f"id = '{seller_id}'",
            toExpr=f"type = 'seller' AND id != '{seller_id}'",
            maxPathLength=3,
        )
        # BIG-DATA-SAFETY-ESCAPE: BFS_BACKUP_COLLECT — limit(1) per iteration
        first = paths.limit(1).collect()
        if not first:
            backup_rows.append((seller_id, None, None))
            continue
        # BFS returns paths of identical length per call (level-wise expansion),
        # so the schema's e* column count == hop count for every returned row.
        n_hops = sum(1 for c in paths.columns if c.startswith("e"))
        backup = first[0]["to"]["id"]
        backup_rows.append((seller_id, backup, int(n_hops)))
    return spark.createDataFrame(
        backup_rows,
        "seller_id string, backup_seller_id string, hop_count integer",
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
    # Induced subgraph on slow edges (avg_delay > 5d), then PageRank again.
    # The high-ranked sellers here are contagion hubs for late shipping.
    from graphframes import GraphFrame

    vertices = spark.read.parquet(resolve_path("outputs/_cache/network_vertices.parquet"))
    edges = spark.read.parquet(resolve_path("outputs/_cache/network_edges.parquet"))
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
        "outputs/_cache/network_delayed_pagerank.parquet",
        "outputs/_cache/network_cocustomer_centrality.parquet",
        "outputs/_cache/network_communities.parquet",
        "outputs/_cache/network_backup_map.parquet",
    ],
    outputs=["outputs/nb3_seller_network_scores.parquet"],
    code_deps=_NET_CODE_DEPS,
    version=2,
)
def build_seller_network_scores(spark: SparkSession) -> DataFrame:
    """Assemble per-seller network scores from the cached GraphFrame outputs.

    Columns: seller_id, pagerank_score, in_degree, is_isolated,
    cocustomer_centrality, community_id, backup_seller_id, backup_strength,
    n_cocustomer_partners, substitutability_deficit, network_risk_score.

    `substitutability_deficit = in_degree / (1 + n_cocustomer_partners)` is the
    graph-unique risk signal: high impact (many customers) with few substitute
    sellers = a single point of failure. It is a degree×neighbourhood
    interaction, so it does NOT track raw in-degree — a well-substituted hub
    scores low, an isolated hub scores high.
    """
    vertices = spark.read.parquet(resolve_path("outputs/_cache/network_vertices.parquet"))
    seller_pagerank = spark.read.parquet(resolve_path("outputs/_cache/network_pagerank.parquet"))
    cc = spark.read.parquet(resolve_path("outputs/_cache/network_connected_components.parquet"))
    delayed = spark.read.parquet(resolve_path("outputs/_cache/network_delayed_pagerank.parquet"))
    centrality = spark.read.parquet(resolve_path("outputs/_cache/network_cocustomer_centrality.parquet"))
    communities = spark.read.parquet(resolve_path("outputs/_cache/network_communities.parquet"))
    backup = spark.read.parquet(resolve_path("outputs/_cache/network_backup_map.parquet"))

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
        .join(
            delayed.withColumnRenamed("id", "seller_id"),
            "seller_id",
            "left",
        )
        .join(centrality, "seller_id", "left")
        .join(communities.select("seller_id", "community_id"), "seller_id", "left")
        .join(backup, "seller_id", "left")
        .fillna(
            {
                "is_isolated": 0,
                "pagerank_score": 0.0,
                "in_degree": 0,
                "network_risk_score": 0.0,
                "cocustomer_centrality": 0.0,
                "backup_strength": 0,
                "n_cocustomer_partners": 0,
                "community_id": -1,
            }
        )
        .withColumn(
            "substitutability_deficit",
            F.col("in_degree") / (F.lit(1.0) + F.col("n_cocustomer_partners")),
        )
    )
