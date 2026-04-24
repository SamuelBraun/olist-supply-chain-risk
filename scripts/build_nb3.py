"""Emit notebooks/03_supply_network_graph.ipynb from cell definitions below.

Thin-wrapper edition: logic in src/olist/pipeline/network.py.
The convergence section has migrated to notebooks/00_main.ipynb per the
refactor plan; this notebook stops at the `nb3_seller_network_scores.parquet`
write.
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


md("""# NB3 — Supply-Network Graph

**Scope.** Produces one consumer parquet:
- `outputs/nb3_seller_network_scores.parquet` — per-seller graph metrics (PageRank, in-degree, isolation flag, BFS backup, high-delay PageRank).

**Rubric surface in this notebook.** GraphFrames (vertices = sellers ∪ `customer_unique_id`; edges = purchase + serves) · PageRank · connectedComponents (GraphX backend) · motif finding `(a)-[]->(c)<-[]-(b)` · BFS · induced high-delay subgraph.

**Convergence layer** (merging this parquet with NB1 + NB2 into `seller_risk_index.parquet`) lives in `00_main.ipynb` and `src/olist/pipeline/convergence.py`. This notebook stops at the network-scores write.

**Thin-wrapper notice.** Logic in `src/olist/pipeline/network.py`. Each GraphFrame algorithm is its own `@step`-cached function so reruns on unchanged inputs skip the expensive compute.""")

md("""## 1. Boot — `SparkSession` with the GraphFrames jar""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark

spark = get_spark("nb3-supply-network", with_graphframes=True)
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')

md("""## 2. Order-line base — delivered orders + customer_unique_id + seller_state

Uses `customer_unique_id` (not `customer_id`) per CLAUDE.md §4 so repeat-customer motifs work. `@step` caches the joined frame to `outputs/_cache/network_order_lines.parquet`.""")

code('''from olist.pipeline.network import build_order_lines
from pyspark.sql import functions as F

order_lines = build_order_lines(spark)
print("order_lines rows:", order_lines.count())
order_lines.limit(3).show(truncate=False)
''')

md("""## 3. Vertices — sellers ∪ unique customers

Distinct union of seller IDs and `customer_unique_id`s, each tagged with a `type` column.""")

code('''from olist.pipeline.network import build_vertices

vertices = build_vertices(spark)
print("vertices:", vertices.count())
vertices.groupBy("type").agg(F.count("*").alias("n")).show()
''')

md("""## 4. Edges — bidirectional `purchase` + `serves`

For each (customer_unique_id, seller_id) pair we emit one `purchase` edge (c→s) and one `serves` edge (s→c), with weight = order-item count and `avg_delay`. Bidirectionality lets PageRank flow both ways and BFS reach sellers via shared customers.""")

code('''from olist.pipeline.network import build_edges

edges = build_edges(spark)
print("edges:", edges.count())
edges.groupBy("edge_type").agg(F.count("*").alias("n")).show()
''')

md("""## 5. GraphFrame + degree analysis""")

code('''from olist.pipeline.network import build_graph_frame, seller_degree_stats

g = build_graph_frame(spark)
print("GraphFrame:", g)
print("\\nseller_degree_stats (top 5 by purchase-only in-degree):")
seller_degrees = seller_degree_stats(spark)
seller_degrees.orderBy(F.col("in_degree_purchase_only").desc()).limit(5).show()
''')

md("""## 6. PageRank — seller importance in the bidirectional graph

`resetProbability=0.15, maxIter=10`. Cached to `outputs/_cache/network_pagerank.parquet`.""")

code('''from olist.pipeline.network import compute_pagerank

seller_pagerank = compute_pagerank(spark)
print("top 5 sellers by PageRank:")
seller_pagerank.orderBy(F.col("pagerank_score").desc()).limit(5).show(truncate=False)
''')

md("""## 7. Connected components — flag isolated sellers

Uses the GraphX-backed algorithm; the default message-passing variant OOMs the JVM heap on ~100k vertices at 6g (see `decisions_log.md` 2026-04-22 NB3 entry).""")

code('''from olist.pipeline.network import compute_connected_components

cc_with_size = compute_connected_components(spark)
isolated_sellers = cc_with_size.filter(
    (F.col("type") == "seller") & (F.col("component_size") == 1)
)
print(f"isolated sellers: {isolated_sellers.count():,}")
print(f"distinct components: {cc_with_size.select('component').distinct().count():,}")
''')

md("""## 8. Motif — shared-customer seller pairs via `(a)-[e1]->(c); (b)-[e2]->(c)`

Two sellers `a` and `b` sharing a customer `c` via two `serves` edges. Deduped with `a.id < b.id`.""")

code('''from olist.pipeline.network import compute_shared_customer_motifs

shared_customer_pairs = compute_shared_customer_motifs(spark)
print("top shared-customer seller pairs:")
shared_customer_pairs.orderBy(F.col("n_shared_customers").desc()).limit(10).show(truncate=False)
''')

md("""## 9. BFS — backup seller for each top-10 PageRank seller

Per-seed BFS with `maxPathLength=3`; first hit other than self is the backup. Flagged `TOP10_PAGERANK_DRIVER` and `BFS_BACKUP_COLLECT` (both capped — 10 seeds, 1 row per BFS call).""")

code('''from olist.pipeline.network import compute_bfs_backups

backup_df = compute_bfs_backups(spark)
backup_df.show(truncate=False)
''')

md("""## 10. High-delay subgraph — re-run PageRank on late-shipping edges

Induced subgraph over edges with `avg_delay > 5`; PageRank there identifies sellers that are structurally central to *late shipments*.""")

code('''from olist.pipeline.network import compute_delayed_subgraph_pagerank

seller_network_risk = compute_delayed_subgraph_pagerank(spark)
print("top 5 sellers by delayed-subgraph PageRank:")
seller_network_risk.orderBy(F.col("network_risk_score").desc()).limit(5).show(truncate=False)
''')

md("""## 11. Assemble per-seller network scores → parquet (convergence input)

Columns: `seller_id`, `pagerank_score`, `in_degree`, `is_isolated`, `backup_seller_id`, `network_risk_score`. Written to `outputs/nb3_seller_network_scores.parquet`.""")

code('''from olist.pipeline.network import build_seller_network_scores

seller_network_scores = build_seller_network_scores(spark)
print("seller_network_scores rows:", seller_network_scores.count())
seller_network_scores.orderBy(F.col("pagerank_score").desc()).limit(5).show(truncate=False)
''')

md("""## 12. Clean up

Convergence into `seller_risk_index.parquet` lives in `00_main.ipynb` — it joins the three per-seller parquets and weights them (0.35 demand + 0.35 sentiment + 0.30 network).""")

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
