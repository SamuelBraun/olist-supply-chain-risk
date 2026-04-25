"""Emit notebooks/main.ipynb — the unified comprehensive deliverable.

Single-notebook architecture. Every transformation lives in
src/olist/{pipeline,data_foundation,viz}.py; this script orchestrates
them into a top-to-bottom narrative for both managers and DS reviewers.

Section layout:
  1. Project Introduction
  2. Data Foundation                 (NEW — schema diagram, shared keys, temporal coverage)
  3. Sub-Analysis 1 — Demand Forecasting
  4. Sub-Analysis 2 — Sentiment Analysis
  5. Sub-Analysis 3 — Network Analysis
  6. Cross-Analysis Synthesis
  7. Conclusions & Limitations

Sections 3–7 are added in subsequent commits; this commit lands §1 + §2 only.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "main.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ===========================================================================
# 0. Title
# ===========================================================================
md("""# Olist Supply-Chain Risk Intelligence

**Client:** Olist (Brazilian e-commerce marketplace) · **Consulting team:** BigDataCompany · **Audience:** Olist management + technical reviewers

This notebook is the *single comprehensive deliverable* for the project. It walks top-to-bottom through three PySpark sub-analyses — **demand forecasting**, **sentiment analysis**, and **supply-network graph** — that converge into one per-seller **Seller Risk Index**. Every section is written so a manager can skim the markdown and the headline charts; a data scientist can drop into any cell and inspect the methodology.

---

### How to read this notebook

- **Plain-language summaries** open every section and follow every chart (look for the *"What this means"* boxes).
- **Code cells** are kept short — heavy lifting lives in `src/olist/`. `inspect.getsource(...)` dumps the source of a function on demand so the reviewer can audit method without leaving the notebook.
- **Each sub-analysis** (§3, §4, §5) is a complete CRISP-DM-inspired data-science workflow: framing → EDA → cleaning → preprocessing → feature engineering → modelling → evaluation → interpretation, with a **Key Takeaways** box at the end.
- **Cross-analysis synthesis** (§6) connects the three sub-analyses into one decision-ready risk index plus an archetype map.
- **Conclusions** (§7) summarise findings, recommendations, limitations, and the big-data-safety log.""")


# ===========================================================================
# Boot
# ===========================================================================
md("""## 0. Boot — `SparkSession`

`spark.driver.memory = 6g`, `shuffle_partitions = 64`, time zone `America/Sao_Paulo`, GraphFrames jar `0.8.3-spark3.5-s_2.12` — all configured in `src/olist/spark_session.py`.""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark
from pyspark.sql import functions as F

spark = get_spark("olist-main", with_graphframes=True)
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')


# ===========================================================================
# 1. Project Introduction
# ===========================================================================
md("""## 1. Project Introduction

### 1.1 The consulting question

Olist is a marketplace platform: thousands of independent Brazilian sellers ship to millions of customers under one storefront. The core operational risk is **silent seller failure** — a seller starts shipping late, or accumulates negative reviews, or sits in a part of the network where their failure cascades widely, and Olist only finds out *after* customers leave.

> **The question we were hired to answer.** *Which sellers should Olist's account-management team intervene on this week to prevent quietly-developing supply-chain failures?*

### 1.2 Three sub-research-questions

To answer the operational question above, we decompose it into three independent data-science problems. Each is solved end-to-end, then fused into a single per-seller risk score.

| # | Sub-question | Method | Output signal |
|---|---|---|---|
| 1 | **Demand pressure & delivery risk.** Is this seller's order volume rising or falling, and are they shipping on time? | PySpark MLlib regressors (GBT vs RandomForest) under 3-fold CV on weekly volume features | `forecast_uplift_pct`, `avg_delay_days` |
| 2 | **Customer sentiment trend.** Are reviews getting more positive or negative for this seller, and does sentiment lead volume? | TF-IDF + Logistic Regression (Spark ML Pipeline) + PyTorch LSTM, plus 6-week rolling Window aggregation | `avg_sentiment_score`, `sentiment_trend_6wk` |
| 3 | **Network centrality & contagion.** Is this seller a structural hub whose failure would disrupt many customers? | GraphFrames PageRank + connected components + motif `(a)→c←(b)` + BFS + delayed-subgraph PageRank | `pagerank_score`, `network_risk_score` |

### 1.3 Methodology — why PySpark

The Olist sample dataset fits on a laptop (~100k orders, ~100k reviews), but the **analytical shape** is unambiguously big-data. A marketplace-peer platform (Mercado Libre, Shopee) operates at 10–1000× this scale, and any tool we build for Olist must run there too.

So every transformation lives in PySpark:
- **Distributed joins.** `orders ⋈ order_items ⋈ reviews ⋈ customers ⋈ sellers` is shuffle-heavy; pandas would swap to disk above ~10 M rows. Spark partitions the shuffle by key.
- **Distributed ML.** `pyspark.ml.Pipeline` + `CrossValidator` train regressors and classifiers across executors. Models survive a 100× scale-up unchanged.
- **Graph algorithms at scale.** GraphFrames runs PageRank, connected components, and motif-finding on a JVM-backed graph; `networkx` would be 10× slower on this graph and unusable at 1 M vertices.

Everywhere we *had* to step outside PySpark (PyTorch LSTM; matplotlib charts; tiny driver-side `collect()` for top-N tables) is **catalogued** in `docs/big_data_safety_log.md` with the production alternative and why the escape is acceptable at this size. §7 renders this log inline.

### 1.4 How to read this notebook (recap)

- **Manager view:** read the markdown headers, skip to the *"What this means"* boxes after each chart, and read the **Key Takeaways** at the end of each sub-analysis.
- **DS-review view:** every code cell is one to a handful of lines; methodology is in `src/olist/`; PySpark primitives are demonstrated visibly throughout (RDD chain in §3.1, SparkSQL queries in §3.4, ML Pipelines in §3.6 / §4.5, MLlib CV metrics, Window functions in §3.5 / §4.5, GraphFrames in §5.5–§5.6, EDA primitives in §3.2 / §4.2).""")


# ===========================================================================
# 2. Data Foundation
# ===========================================================================
md("""## 2. Data Foundation

Before we dive into the three sub-analyses, we make the *relational shape of the problem* visible. The Olist dataset is **nine interconnected CSVs** that share keys in non-obvious ways; getting the joins right (and using the right key — e.g. `customer_unique_id` not `customer_id` for "person") determines whether the downstream analyses are correct.

This section answers four questions:

1. **Which tables exist, what role do they play, and how big are they?**
2. **How are they connected?** (schema diagram + shared-key cardinality)
3. **Do their time ranges overlap?** (so cross-table time-series joins are valid)
4. **What did we drop, and why?** (a single styled cleaning audit)""")


md("""### 2.1 The nine source tables

Each table loaded with an explicit `StructType` from `src/olist/schemas.py` (no `inferSchema` — that would force an extra full-file pass per table, and break at scale). The table below is the *registry* — single source of truth for the rest of the project.""")

code('''from olist.data_foundation import table_overview, TABLE_REGISTRY
from olist import viz

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 9-row registry summary
overview = table_overview(spark)
overview.show(truncate=False)
overview_pd = overview.toPandas()
viz.styled_topn_table(
    overview_pd,
    bar_cols=["n_rows"],
    fmt={"n_rows": "{:,d}", "n_cols": "{:d}"},
    title="The nine Olist source tables",
)
''')

md("""**What this means.** Three transactional tables (`orders`, `order_items`, `order_payments`) carry the events; three dimensional tables (`customers`, `sellers`, `products`) describe the entities; one free-text table (`order_reviews`) carries customer voice; one geospatial table (`geolocation`) provides locations; one taxonomy (`category_translation`) maps Portuguese → English category names. The volume is dominated by `geolocation` (~1 M rows) — every other table is well under 200k rows, so joins on dimensional tables are broadcast-friendly.""")


md("""### 2.2 Schema diagram — how the tables connect

The diagram below shows the foreign-key relationships. Boxes are coloured by role; arrows point from the foreign-key holder to the referenced table; the label on each arrow names the join key.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — pure-matplotlib diagram, no data
viz.schema_diagram()
''')

md("""**What this means.** Three "hub" relationships drive the project:

1. **`order_items` is the fact table.** It links three dimensions (`orders`, `products`, `sellers`) into one row-per-order-line — every demand and network analysis joins through this table.
2. **`customers` is the bridge between people and orders.** A returning customer has *one* `customer_unique_id` but *many* `customer_id`s (one per order they placed). Using `customer_unique_id` for graph vertices is what makes the shared-customer motif in §5 surface real repeat-customer behaviour.
3. **Geolocation is many-to-one per zip prefix.** We aggregate to centroids once and broadcast — see §2.5 cleaning audit.""")


md("""### 2.3 Shared-key cardinality

The same key column lives in different tables with very different cardinalities. The chart below uses `approx_count_distinct` (HyperLogLog — big-data-safe) to count distinct values of each shared key per table.""")

code('''from olist.data_foundation import shared_key_cardinality

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 14-row aggregate (≤6 keys × ≤4 tables)
keys_df = shared_key_cardinality(spark)
keys_df.show(truncate=False)

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — small aggregate, log-scale bar chart
keys_pd = keys_df.toPandas()
viz.shared_key_grouped_bar(keys_pd)
''')

md("""**What this means.** Four observations from the chart:

- **`customer_id` and `customer_unique_id` differ by ≈3,000.** That's the count of repeat customers — same person, multiple orders.
- **`order_id` is well-defined.** It appears at the same cardinality in `orders`, `order_reviews`, and `order_payments` (allowing for some null reviews).
- **`order_items` carries multiple FKs at lower cardinality than `order_id`** — there are more `order_items` rows than `orders` (multi-item orders).
- **`zip_prefix` lives in three tables.** Customers and sellers each have one prefix per row; `geolocation` carries 19,015 distinct prefixes with multiple addresses each — the many-to-one we collapse.""")


md("""### 2.4 Temporal coverage

For cross-table time-series joins (especially the lead-indicator analysis in §4 that aligns weekly sentiment with weekly demand), we need the timestamp ranges to overlap meaningfully. The chart below shows the date range of every timestamp column in the schema.""")

code('''from olist.data_foundation import temporal_coverage

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 8-row min/max per timestamp column
temp_df = temporal_coverage(spark)
temp_df.show(truncate=False)

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — 8-row aggregate
temp_pd = temp_df.toPandas()
viz.temporal_overlap_chart(temp_pd)
''')

md("""**What this means.** All transactional + review timestamps fall within the same window (roughly Sep 2016 → Oct 2018, with sparse data at both edges). The `n` annotations on the bars show how dense each column is. The takeaway for downstream work: the lead-indicator analysis (§4.7) safely aligns sentiment with volume on a common weekly grid; the demand forecast (§3) can use the full time range without worrying about silent table-level coverage gaps.""")


md("""### 2.5 Cleaning audit

Every cleaning step we take across the three sub-analyses is documented here as a single styled table. Numbers come from real Spark counts — no hard-coded magic.""")

code('''from olist.data_foundation import cleaning_audit

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 4-row audit
audit = cleaning_audit(spark)
audit.show(truncate=False)

# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — 4-row styled table
audit_pd = audit.toPandas()
viz.styled_topn_table(
    audit_pd,
    bar_cols=["dropped_rows"],
    fmt={"input_rows": "{:,d}", "kept_rows": "{:,d}", "dropped_rows": "{:,d}"},
    title="Cleaning audit — every row drop in the project, with reason",
    hide_index=True,
)
''')

md("""**What this means.** Three lessons:

- **Demand cleaning is gentle.** Only ~3% of orders are dropped (`order_delivered_customer_date IS NULL`); the forecasting + delivery-delay analyses still see the bulk of the marketplace.
- **Sentiment cleaning is aggressive but principled.** Dropping neutral scores (`==3`) keeps the classifier bimodal; dropping NULL-comment rows is mandatory for the NLP pipeline. Both are documented above so the grader / reviewer can verify.
- **Geolocation aggregation is the single biggest reduction.** ~1 M raw rows → 19,015 zip-prefix centroids. We never join the raw table to anything else — only the broadcast centroids.""")


md("""### 2.6 Why this is a big-data problem (the 4 V's, condensed)

- **Volume.** Marketplace-peer platforms operate at 10–1000× this scale; the joins above (5-way `orders × order_items × customers × sellers × products`) are shuffle-heavy at production scale.
- **Velocity.** Orders, reviews, and deliveries arrive continuously; the weekly Window aggregations in §3 and §4 become Structured Streaming jobs at production velocity.
- **Variety.** Nine interconnected tables with five distinct *roles* (transactional / dimensional / free-text / geospatial / taxonomy) — Spark handles them via explicit `StructType` schemas without `inferSchema` passes.
- **Veracity.** Documented row-drops above; every cleaning decision recorded in `docs/decisions_log.md` with its row count.

Every function in `src/olist/pipeline/*.py` was written to work *identically* at 100× the current row count — no `collect`/`toPandas` on a non-aggregated DataFrame, all small lookups broadcast, hot DataFrames cached once, every `orderBy` paired with a `limit`. The complete catalogue of escape hatches (and why each one is acceptable at this scale) is in `docs/big_data_safety_log.md`, rendered inline in §7.""")


# ===========================================================================
# Sections 3-7 land in subsequent commits.
# ===========================================================================
md("""---

> **Sections §3 (Demand), §4 (Sentiment), §5 (Network), §6 (Cross-Analysis Synthesis), and §7 (Conclusions) are added in subsequent commits.**

This commit (commit 1 of 6) lands the project skeleton + the Data Foundation section. Sub-analyses arrive in commits 2–4; cross-analysis synthesis in commit 5; conclusions, README, and the consolidation of the old four-notebook submission into this single notebook in commit 6.""")

code('''spark.stop()
print("Spark stopped.")
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
