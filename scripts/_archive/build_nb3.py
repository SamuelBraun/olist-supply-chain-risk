"""Emit notebooks/03_supply_network_graph.ipynb — CRISP-DM edition.

Thin-wrapper: every transformation lives in src/olist/pipeline/network.py.
Every rendered chart/table goes through src/olist/viz.py.

Structure (CRISP-DM):
  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Evaluation
  6. Deployment
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "03_supply_network_graph.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md("""# NB3 — Supply-Network Graph

**Scope.** Produces one consumer parquet for the convergence layer:
- `outputs/nb3_seller_network_scores.parquet` — per-seller graph metrics (PageRank, in-degree, isolation flag, BFS backup, high-delay PageRank).

**Rubric surface in this notebook.** GraphFrames (vertices = sellers ∪ `customer_unique_id`; edges = purchase + serves) · PageRank · `connectedComponents(algorithm="graphx")` · motif finding `(a)-[]->(c); (b)-[]->(c)` · BFS · induced high-delay subgraph.

**Convergence layer** (merging this parquet with NB1 + NB2 into `seller_risk_index.parquet`) lives in `00_main.ipynb` and `src/olist/pipeline/convergence.py`. This notebook stops at the network-scores write.

**Narrative structure.** This notebook follows the **CRISP-DM** methodology — six sections from *Business Understanding* to *Deployment*. Every code cell calls into `src/olist/pipeline/network.py`; every chart/table goes through `src/olist/viz.py`.""")


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
md("""## 0. Boot — `SparkSession` with the GraphFrames jar""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark
from pyspark.sql import functions as F

spark = get_spark("nb3-supply-network", with_graphframes=True)
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')


# ---------------------------------------------------------------------------
# 1. Business Understanding
# ---------------------------------------------------------------------------
md("""## 1. Business Understanding

**Why network structure matters to Olist.** Two sellers might ship identical products to the same customer base, but if one is a *structural hub* of the purchase network (many customers, many shared customers with peers), its failure cascades far wider than a peripheral seller's. This notebook builds the **network component** of the Seller Risk Index:

- **Centrality signal.** PageRank on the bidirectional customer-seller graph — how structurally important is each seller?
- **Isolation signal.** Connected-components analysis — sellers sitting alone in a tiny component have no backup if they fail.
- **Contagion signal.** PageRank re-run on the *late-shipping* subgraph — which sellers are central in the part of the network where delays propagate?
- **Resilience signal.** BFS from top-PageRank sellers — what is the nearest alternative seller if this one goes down?

**Stakeholder question.** *"If a seller fails, who else is affected and who can absorb the demand?"* — the per-seller network parquet written in §6 answers this row-by-row.""")


# ---------------------------------------------------------------------------
# 2. Data Understanding
# ---------------------------------------------------------------------------
md("""## 2. Data Understanding

Four source tables combine into the graph: `orders ⋈ order_items ⋈ broadcast(customers) ⋈ broadcast(sellers)`, filtered to *delivered* orders. Vertices are the union of distinct seller IDs and distinct `customer_unique_id`s (not `customer_id` — CLAUDE.md §4); edges carry the item count and average delivery delay.""")

md("""### 2.1 Order-line base

Each row is one delivered order-item, with its customer and seller attached. Cached to `outputs/_cache/network_order_lines.parquet` via `@step`.""")

code('''from olist.pipeline.network import build_order_lines

order_lines = build_order_lines(spark)
print(f"order_lines rows: {order_lines.count():,}")
order_lines.limit(3).show(truncate=False)
''')


# ---------------------------------------------------------------------------
# 3. Data Preparation
# ---------------------------------------------------------------------------
md("""## 3. Data Preparation

Two transformations build the graph structure itself:

1. **Vertex set.** Distinct union of seller IDs and `customer_unique_id`s, tagged with a `type` column. Using `customer_unique_id` (not `customer_id`) ensures a returning customer is one vertex, not many — motifs and BFS depend on this.
2. **Edge set.** For every distinct `(customer, seller)` pair we emit one `purchase` edge (c→s) and one `serves` edge (s→c). Bidirectionality is required so PageRank flows both ways and BFS can reach *other sellers* via shared customers.""")

md("""### 3.1 Vertices — sellers ∪ unique customers""")

code('''from olist.pipeline.network import build_vertices

vertices = build_vertices(spark)
print(f"vertices total: {vertices.count():,}")
vertices.groupBy("type").agg(F.count("*").alias("n")).orderBy("type").show()
''')

md("""### 3.2 Edges — bidirectional `purchase` + `serves`""")

code('''from olist.pipeline.network import build_edges

edges = build_edges(spark)
print(f"edges total: {edges.count():,}")
edges.groupBy("edge_type").agg(F.count("*").alias("n")).orderBy("edge_type").show()
''')


# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------
md("""## 4. Modeling

Five GraphFrame algorithms, each its own `@step` so reruns on unchanged inputs skip the expensive compute.""")

md("""### 4.1 GraphFrame + per-seller degree stats

`GraphFrame(vertices, edges)` is cheap to reconstruct — only the algorithms below are cached.""")

code('''from olist.pipeline.network import build_graph_frame, seller_degree_stats

g = build_graph_frame(spark)
print("GraphFrame:", g)

seller_degrees = seller_degree_stats(spark)
print("\\nTop 5 sellers by purchase-only in-degree:")
seller_degrees.orderBy(F.col("in_degree_purchase_only").desc()).limit(5).show()
''')

md("""### 4.2 PageRank — seller centrality in the bidirectional graph

`resetProbability=0.15, maxIter=10`. Cached to `outputs/_cache/network_pagerank.parquet`. Projected to seller vertices only and renamed `pagerank_score`.""")

code('''from olist.pipeline.network import compute_pagerank

seller_pagerank = compute_pagerank(spark)
print(f"seller_pagerank rows: {seller_pagerank.count():,}")
''')

md("""### 4.3 Connected components — isolated-seller flag

Uses `algorithm="graphx"`. The default message-passing variant OOMs the JVM heap on ~100k vertices at 6g driver memory (see `decisions_log.md` 2026-04-22 NB3 entry). Sellers sitting in a component of size 1 are flagged `is_isolated = 1`.""")

code('''from olist.pipeline.network import compute_connected_components

cc_with_size = compute_connected_components(spark)
isolated_sellers = cc_with_size.filter(
    (F.col("type") == "seller") & (F.col("component_size") == 1)
)
print(f"isolated sellers: {isolated_sellers.count():,}")
print(f"distinct components: {cc_with_size.select('component').distinct().count():,}")
''')

md("""### 4.4 Motif — shared-customer seller pairs `(a)-[]->(c); (b)-[]->(c)`

Two sellers `a` and `b` sharing a customer `c` via two `serves` edges. Deduped with `a.id < b.id`. Captures the "product substitutability" relation — if `a` fails, `b` already serves many of `a`'s customers.""")

code('''from olist.pipeline.network import compute_shared_customer_motifs

shared_customer_pairs = compute_shared_customer_motifs(spark)
print(f"distinct seller-pairs sharing ≥1 customer: {shared_customer_pairs.count():,}")
''')

md("""### 4.5 BFS — backup seller for each top-10 PageRank seller

Per-seed BFS with `maxPathLength=3`; first hit other than self is the backup. Flagged `TOP10_PAGERANK_DRIVER` (10-row driver list) and `BFS_BACKUP_COLLECT` (≤1 row per BFS call) — both capped by construction.""")

code('''from olist.pipeline.network import compute_bfs_backups

backup_df = compute_bfs_backups(spark)
backup_df.show(truncate=False)
''')

md("""### 4.6 High-delay subgraph — PageRank of the late-shipping graph

Induced subgraph over edges with `avg_delay > 5`; PageRank there identifies sellers that are structurally central to *late* shipments — i.e. the contagion risk.""")

code('''from olist.pipeline.network import compute_delayed_subgraph_pagerank

seller_network_risk = compute_delayed_subgraph_pagerank(spark)
print(f"seller_network_risk rows: {seller_network_risk.count():,}")
''')


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
md("""## 5. Evaluation

Three diagnostic views: top-10 PageRank sellers (styled), top-20 shared-customer motifs (styled), and component-size distribution.""")

md("""### 5.1 Top-10 sellers by PageRank

These are the network's structural hubs. A failure at any of these cascades widely — they are the first candidates for proactive monitoring regardless of their demand/sentiment scores.""")

code('''from olist import viz

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 10 rows
top10_pagerank = (
    seller_pagerank.withColumnRenamed("id", "seller_id")
    .orderBy(F.col("pagerank_score").desc())
    .limit(10)
    .toPandas()
)
top10_pagerank["seller_id"] = top10_pagerank["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_pagerank,
    bar_cols=["pagerank_score"],
    fmt={"pagerank_score": "{:.4f}"},
    title="Top-10 sellers by PageRank (bidirectional graph)",
)
''')

md("""### 5.2 Top-20 seller pairs by shared customers

Substitutability signal: these pairs are the strongest candidate backup relationships in the marketplace. A high shared-customer count implies one seller can absorb the other's demand with minimal customer friction.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 20 rows
top20_motifs = (
    shared_customer_pairs.orderBy(F.col("n_shared_customers").desc())
    .limit(20)
    .toPandas()
)
top20_motifs["seller_a"] = top20_motifs["seller_a"].str.slice(0, 10) + "…"
top20_motifs["seller_b"] = top20_motifs["seller_b"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top20_motifs,
    bar_cols=["n_shared_customers"],
    fmt={"n_shared_customers": "{:d}"},
    title="Top-20 seller pairs by shared customers (motif `(a)->c<-(b)`)",
)
''')

md("""### 5.3 Component-size distribution

How fragmented is the marketplace? If almost every seller sits in one giant component, the graph is well-connected and isolation-risk is low; if many components of size 1 exist, those sellers have no structural redundancy.""")

code('''from olist.pipeline.network import component_size_histogram

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — 5-row aggregate
hist = component_size_histogram(spark)
hist.show()

bin_order = ["1 (isolated)", "2–4", "5–9", "10–99", "100+"]
hist_pd = hist.toPandas()
hist_pd["size_bin"] = hist_pd["size_bin"].astype("category").cat.set_categories(bin_order, ordered=True)
hist_pd = hist_pd.sort_values("size_bin")
viz.state_bar(
    hist_pd,
    value_col="n_components",
    label_col="size_bin",
    title="Component-size distribution (# components per size bin)",
    sort="asc",
    color_by_value=True,
)
''')


# ---------------------------------------------------------------------------
# 6. Deployment
# ---------------------------------------------------------------------------
md("""## 6. Deployment

The deployable artefact is `outputs/nb3_seller_network_scores.parquet` — one row per seller, six columns, consumed directly by the convergence layer (`olist.pipeline.convergence.build_seller_risk_index`).""")

md("""### 6.1 Per-seller network scores

Columns: `seller_id`, `pagerank_score`, `in_degree`, `is_isolated`, `backup_seller_id`, `network_risk_score` (high-delay-subgraph PageRank).""")

code('''from olist.pipeline.network import build_seller_network_scores

seller_network_scores = build_seller_network_scores(spark)
print(f"seller_network_scores rows: {seller_network_scores.count():,}")
seller_network_scores.groupBy("is_isolated").agg(F.count("*").alias("n")).orderBy("is_isolated").show()
''')

md("""### 6.2 Top-10 by delayed-subgraph PageRank — deployment-ready table

These are the "contagion hubs": sellers central to the part of the network where deliveries arrive late. An account manager should prioritise these for operational review — any service improvement here has network-wide spillover.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 10 rows
top10_risk = (
    seller_network_scores.orderBy(F.col("network_risk_score").desc()).limit(10).toPandas()
)
top10_risk["seller_id"] = top10_risk["seller_id"].str.slice(0, 10) + "…"
top10_risk["backup_seller_id"] = (
    top10_risk["backup_seller_id"].fillna("—").astype(str).str.slice(0, 10) + "…"
)
viz.styled_topn_table(
    top10_risk,
    bar_cols=["network_risk_score"],
    gradient_cols=["pagerank_score"],
    fmt={
        "pagerank_score": "{:.4f}",
        "network_risk_score": "{:.4f}",
        "in_degree": "{:d}",
        "is_isolated": "{:d}",
    },
    title="Top-10 sellers by delayed-subgraph PageRank (network contagion risk)",
)
''')

md("""### 6.3 Clean up""")

code('''ROOT_OUT = Path.cwd().parent / "outputs" if Path.cwd().name == "notebooks" else Path.cwd() / "outputs"
print("Notebook 3 outputs:")
for path in sorted(ROOT_OUT.glob("nb3_*.parquet")):
    print(" ", path.name)
spark.stop()
print("\\nSpark stopped.")
''')


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9.6"}
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    with OUT_NB.open("w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT_NB.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
