"""Emit notebooks/00_main.ipynb — the narrative orchestrator.

Contract per docs/refactor_prompt.md §"Main notebook contract":

1. Executive summary
2. Client context + 4 V's
3. Data overview (aggregated Spark tables)
4. Distributed-computing toolkit walkthrough (RDD, DataFrames, SparkSQL,
   Pipelines, MLlib, Deep Learning, GraphFrames, Windows, EDA primitives)
5. Analysis 1 — Demand      (calls pipeline.demand.*)
6. Analysis 2 — Sentiment   (calls pipeline.sentiment.*)
7. Analysis 3 — Network     (calls pipeline.network.*)
8. Convergence — Seller Risk Index (calls pipeline.convergence.*)
9. Recommendations for Olist
10. Big-data safety log
11. Reproducibility + rubric compliance (checks.run_all())

All logic lives in src/olist/; this notebook is the report surface.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "00_main.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ===========================================================================
# 0. Title
# ===========================================================================
md("""# Olist Supply Chain Risk Intelligence — Main Narrative

**Consulting client:** Olist (Brazilian e-commerce marketplace)
**Consulting team:** BigDataCompany
**Audience:** Olist management (non-technical)

Top-to-bottom: client problem → why this is a big-data problem (the 4 V's) → data → three PySpark analyses (demand, sentiment, network) → converged Seller Risk Index → recommendations. Every Spark primitive the rubric requires renders visibly below; transformation logic lives in `src/olist/pipeline/*.py`.

**How to read this notebook.** A manager can skim the markdown and the headline tables/charts; a senior data scientist can read the same notebook and inspect methodological details via the inline `inspect.getsource(...)` dumps, Pipeline stages, query text, and metrics.

**Reproducibility.** All six `outputs/*.parquet` artefacts are committed with the notebook; `outputs/.cache_manifest.json` tracks the fingerprint of the code that produced each one. Reruns on unchanged code skip the compute and re-read parquet — on a warm cache this notebook executes in under a minute.""")

# ===========================================================================
# 1. Executive summary
# ===========================================================================
md("""## 1. Executive summary

Three PySpark analyses converge into a single per-seller **Seller Risk Index** that flags sellers at risk of becoming a supply-chain failure *before* customers are affected. The three signals — demand pressure, customer sentiment, and network centrality — are normalised and weighted 0.35 / 0.35 / 0.30 into a single 0–1 risk score with CRITICAL / WARNING / SAFE bands.

**Headline.** Out of roughly 3,000 active Olist sellers, the risk-tail clusters at WARNING, not CRITICAL — the live counts are computed below.""")

code('''import os, sys, inspect, time
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark
spark = get_spark("olist-main", with_graphframes=True)
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version, "| driver python:", sys.executable)
''')

code('''from olist.pipeline.convergence import build_seller_risk_index, risk_band_counts
from pyspark.sql import functions as F

# This will be a cache hit if the convergence step has already run against the
# current code + inputs. Otherwise it runs end-to-end — cache builds bottom-up.
risk = build_seller_risk_index(spark)
print("Risk-band counts (live from outputs/seller_risk_index.parquet):")
risk_band_counts(risk).show()
print(f"Total scored sellers: {risk.count():,}")
''')

# ===========================================================================
# 2. Client context + 4 V's
# ===========================================================================
md("""## 2. Client context — why this is a big-data problem

Olist's marketplace data currently fits on a laptop, but the analytical shape of the problem is unambiguously big-data:

- **Volume.** A marketplace-peer platform (Mercado Libre, Shopee) operates at 10–1000× the scale of this sample dataset. Joins across `orders ⋈ order_items ⋈ reviews ⋈ customers ⋈ sellers` are shuffle-heavy; a single-node pandas approach would hit memory pressure and GC death at ≥10 M orders. Spark partitions the shuffles across workers by construction.

- **Velocity.** New orders, reviews, and deliveries arrive continuously. The lead-indicator analysis (§6) computes weekly rolling sentiment vs. weekly volume; at production velocity this becomes a Spark Structured Streaming job with event-time windows — Window functions we use here are the same primitives.

- **Variety.** Nine interconnected tables: transactional (`orders`, `order_items`, `payments`), dimensional (`customers`, `sellers`, `products`), geospatial (`geolocation`), free-text (`reviews`), taxonomy (`category_translation`). Spark handles them via explicit `StructType` schemas without `inferSchema` passes.

- **Veracity.** Missing `order_delivered_customer_date` (≈3 % of orders), Portuguese reviews with null comments (>50 %), zip-prefix-aggregated geolocation (one centroid per prefix). Every drop is logged in `docs/decisions_log.md` with its row count.

**Where a single-node pandas approach would break.** (a) The 4-way `order_lines` join materialises ~113k rows now but would be ~100 M at marketplace-peer scale — an in-memory pandas merge there would swap to disk. (b) GraphFrames PageRank on ~100k vertices already needs 6 GB driver heap; `networkx` would be 10× slower at this size and unusable at 1 M vertices. (c) NLP over 43k Portuguese reviews fits in pandas here; at 100 M reviews the tokenisation itself would need a distributed executor (spark-nlp or Petastorm-streamed PyTorch).

Every function in `src/olist/pipeline/*` was written to work *identically* at 100× the current row count — no `collect`/`toPandas` on a non-aggregated DataFrame, all small lookups broadcast, hot DataFrames cached once, every `orderBy` paired with a `limit`.""")

# ===========================================================================
# 3. Data overview
# ===========================================================================
md("""## 3. Data overview

Nine CSVs under `data/`, loaded via explicit-schema typed loaders in `src/olist/loaders.py`. The row counts below are a Spark-native audit — no `inferSchema`, no driver-side materialisation of the underlying rows.""")

code('''from olist.pipeline.demand import load_core_tables
from olist.loaders import load_geolocation, load_order_payments

tables = load_core_tables(spark)
tables["geolocation"] = load_geolocation(spark)
tables["order_payments"] = load_order_payments(spark)

print("Source table row counts:")
for name, df in sorted(tables.items()):
    print(f"  {name:>24}: {df.count():>10,}")
''')

md("""**Dropped-row audit** — decisions logged in `docs/decisions_log.md`:

- ≈2,965 orders dropped because `order_delivered_customer_date IS NULL` (in-transit / cancelled); retained ~96,476 delivered orders.
- Neutral review scores (==3) dropped; ~82 / 18 positive / negative class balance remains.
- Geolocation aggregated to one centroid per zip prefix (19,015 rows) and broadcast thereafter.""")

# ===========================================================================
# 4. Distributed computing toolkit walkthrough
# ===========================================================================
md("""## 4. Distributed-computing toolkit — the Spark primitives in use

A deliberate walkthrough of the primitives the course rubric asks us to demonstrate, each illustrated by a real call into the pipeline with visible output.""")

# --- RDDs ---
md("""### 4.1 RDDs — `textFile → filter/map/reduceByKey`

The lowest-level Spark primitive. We read the raw orders CSV as a text RDD, filter the header, map each line to `(purchase_date, 1)`, reduce by key, and rebuild a typed DataFrame with an explicit schema. This is the idiomatic Spark pattern for ingesting arbitrarily-formatted text before moving into the DataFrame API.

The chain is printed below via `inspect.getsource` — single source of truth in `src/olist/pipeline/demand.py`, no duplicated logic.""")

code('''from olist.pipeline.demand import rdd_daily_order_count
print(inspect.getsource(rdd_daily_order_count))
''')

code('''daily_rdd_df = rdd_daily_order_count(spark)
print("RDD-derived daily rows:", daily_rdd_df.count())
daily_rdd_df.orderBy("purchase_date").limit(5).show()
''')

# --- DataFrames & SparkSQL ---
md("""### 4.2 DataFrames + SparkSQL — temp-view queries

Explicit `StructType` schemas are declared once in `src/olist/schemas.py`; every typed loader applies them (no `inferSchema`). The three required SparkSQL queries run against a temp view registered from `order_lines`.""")

code('''from olist.pipeline.demand import build_order_lines, sparksql_queries

order_lines = build_order_lines(spark)
print("order_lines rows:", order_lines.count())

queries = sparksql_queries(order_lines)
for name, (sql, result) in queries.items():
    print(f"\\n--- {name} ---")
    print(sql.strip())
    result.show(truncate=False)
''')

# --- Pipelines & Data Engineering ---
md("""### 4.3 Pipelines & Data Engineering

Every reusable transformation chain goes through `pyspark.ml.Pipeline`. The NB1 feature pipeline (Imputer → VectorAssembler) and the NB2 NLP pipeline (Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression) are the two canonical examples.""")

code('''from olist.pipeline.demand import build_feature_pipeline
from olist.pipeline.sentiment import build_nlp_pipeline

print("Demand feature Pipeline (NB1):")
for stage in build_feature_pipeline().getStages():
    print(" ", stage)
print("\\nNLP Pipeline (NB2):")
for stage in build_nlp_pipeline().getStages():
    print(" ", stage)
''')

# --- MLlib ---
md("""### 4.4 MLlib — `CrossValidator(folds=3)` over `GBTRegressor` + `RandomForestRegressor`

`RegressionEvaluator(metricName='rmse')`. Results are cached via `@step`; metrics re-read from parquet on warm cache.""")

code('''from olist.pipeline.demand import fit_and_score

demand_artefacts = fit_and_score(spark)
print("--- GBT vs RF metrics ---")
demand_artefacts["demand_metrics"].show()
''')

# --- Deep Learning ---
md("""### 4.5 Deep Learning — PyTorch LSTM (justified escape)

The LSTM is the one non-Spark primitive we train. It lives in `pipeline.sentiment.train_lstm`; `train_lstm_cached` wraps it in `@step` so reruns skip the ~3-minute training loop and re-read the metrics parquet. The justification (why not spark-nlp; why acceptable at this scale) is in `docs/big_data_safety_log.md` (IDs `LSTM_TO_PANDAS`, `LSTM_PYTORCH`).""")

code('''from olist.pipeline.sentiment import train_lstm_cached

lstm_metrics = train_lstm_cached(spark)
lstm_metrics.show(truncate=False)
''')

# --- GraphFrames ---
md("""### 4.6 GraphFrames

Vertices = sellers ∪ `customer_unique_id`; edges = bidirectional `purchase` + `serves`. Algorithms: PageRank, connectedComponents (GraphX backend, memory-efficient), motif `(a)-[]->(c)<-[]-(b)`, BFS, induced high-delay subgraph. Each algorithm is its own `@step`.""")

code('''from olist.pipeline.network import (
    build_graph_frame, compute_pagerank, compute_connected_components,
    compute_shared_customer_motifs,
)
g = build_graph_frame(spark)
print("GraphFrame:", g)
print("\\nTop 5 PageRank sellers:")
compute_pagerank(spark).orderBy(F.col("pagerank_score").desc()).limit(5).show(truncate=False)
''')

# --- Window functions ---
md("""### 4.7 Window functions

Used throughout: `Window.partitionBy(seller_id).orderBy(year_week).rowsBetween(...)` for lag features (NB1), rolling 6-week sentiment mean (NB2), per-seller "last week" row in trend computation. Shown here via the weekly-features Pipeline output.""")

code('''from olist.pipeline.demand import add_weekly_features, build_weekly_order_volume, FEATURE_COLS
weekly = build_weekly_order_volume(spark)
weekly_features = add_weekly_features(weekly)
weekly_features.select("seller_id", "year_week", "weekly_order_count", *FEATURE_COLS).limit(5).show()
''')

# --- EDA primitives ---
md("""### 4.8 EDA primitives — `approxQuantile` + `approx_count_distinct` + `broadcast`

Big-data-safe alternatives to `describe()` and `distinct().count()`. The `sellers`, `products`, and `geo_centroids` lookups (all ≤10 MB) are broadcast to every join per CLAUDE.md §3.""")

code('''from olist.pipeline.demand import eda_stats
stats = eda_stats(order_lines)
print("price quantiles (p25/p50/p75/p95):", stats["price_quantiles"])
print("delay quantiles (p25/p50/p75/p95):", stats["delay_quantiles"])
stats["approx_counts"].show()
''')

md("""### 4.9 Streaming (bonus)

Not demonstrated in this pass — the three required per-seller parquets are green, so the bonus streaming section was skipped in favour of sharpening the main-notebook narrative. Structured Streaming would extend `build_weekly_order_volume` to `spark.readStream...window(...).groupBy(...)` with the same Window primitives shown above.""")

# ===========================================================================
# 5. Analysis 1 — Demand
# ===========================================================================
md("""## 5. Analysis 1 — Demand pressure + delivery risk

**Question.** When and where will demand spike, and which sellers are already struggling to keep up?

We aggregate order lines to weekly volume per seller, engineer lag + rolling features, and forecast next-week volume with both GBT and RandomForest under 3-fold cross-validation. Per-seller delivery delay becomes an orthogonal risk flag. The winning model's per-seller forecast uplift and avg delay are persisted to `outputs/nb1_seller_demand_scores.parquet`.""")

code('''demand_scores = demand_artefacts["nb1_seller_demand_scores"]
print("Top 5 sellers by forecast uplift %:")
demand_scores.orderBy(F.col("forecast_uplift_pct").desc()).limit(5).show(truncate=False)
print("\\nTop 5 sellers by avg delay days:")
demand_scores.orderBy(F.col("avg_delay_days").desc()).limit(5).show(truncate=False)
''')

# ===========================================================================
# 6. Analysis 2 — Sentiment + lead indicator
# ===========================================================================
md("""## 6. Analysis 2 — Customer sentiment + lead indicator

**Question.** Can we detect a seller in trouble *before* a sales drop? We classify Portuguese review text (LogReg via ML Pipeline, LSTM in PyTorch), roll sentiment into 6-week windows per seller, and cross-correlate weekly sentiment change vs weekly volume change at lags 0–8 weeks.""")

code('''from olist.pipeline.sentiment import build_seller_sentiment_scores, build_lead_indicator_lags, peak_lag

sentiment_scores = build_seller_sentiment_scores(spark)
print("sentiment_declining breakdown:")
sentiment_scores.groupBy("sentiment_declining").agg(F.count("*").alias("n")).orderBy("sentiment_declining").show()

print("\\nTop 5 declining sellers (most negative 6wk trend):")
sentiment_scores.orderBy(F.col("sentiment_trend_6wk").asc()).limit(5).show(truncate=False)

lag_df = build_lead_indicator_lags(spark)
print("\\nLag-k correlation: sentiment change at t vs volume change at t+k")
lag_df.show()
peak_k, peak_rho = peak_lag(lag_df)
print(f"Peak |corr| at lag={peak_k} weeks (corr={peak_rho:.4f})")
''')

md("""**Honest finding.** Across lags 0–8, the correlation is weak (|corr| ≈ 0.015 at the peak). Sentiment is *not* a strong leading indicator of volume at this sample size — reported in the presentation as a caveat, not a headline.""")

code('''# BIG-DATA-SAFETY-ESCAPE: LEAD_INDICATOR_VIZ
import matplotlib.pyplot as plt

lag_pd = lag_df.toPandas()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(lag_pd["lag"], lag_pd["corr"], color="#4C72B0")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Lag k (weeks)")
ax.set_ylabel("Pearson correlation")
ax.set_title("Does sentiment decline precede volume decline?")
ax.set_xticks(lag_pd["lag"])
plt.tight_layout()
plt.show()
''')

# ===========================================================================
# 7. Analysis 3 — Supply network
# ===========================================================================
md("""## 7. Analysis 3 — Supply-network graph

**Question.** Which sellers are single points of failure?

The marketplace is modelled as a graph: vertices are sellers + unique customers; edges are bidirectional `purchase` + `serves` relations weighted by order-item count. PageRank measures seller centrality; connected components flag isolates; the `(a)-[]->(c)<-[]-(b)` motif surfaces shared-customer seller pairs; BFS finds backup sellers for each high-centrality seller; a high-delay induced subgraph re-runs PageRank to flag sellers central to late shipments.""")

code('''from olist.pipeline.network import (
    build_seller_network_scores, compute_shared_customer_motifs,
    compute_bfs_backups, compute_delayed_subgraph_pagerank,
)
network_scores = build_seller_network_scores(spark)
print("Top 5 sellers by PageRank:")
network_scores.orderBy(F.col("pagerank_score").desc()).limit(5).show(truncate=False)

print("\\nTop 5 shared-customer seller pairs:")
compute_shared_customer_motifs(spark).orderBy(
    F.col("n_shared_customers").desc()
).limit(5).show(truncate=False)

print("\\nBFS backup sellers for top-10 PageRank seeds:")
compute_bfs_backups(spark).show(truncate=False)

print("\\nTop 5 sellers by delayed-subgraph PageRank:")
compute_delayed_subgraph_pagerank(spark).orderBy(
    F.col("network_risk_score").desc()
).limit(5).show(truncate=False)
''')

# ===========================================================================
# 8. Convergence
# ===========================================================================
md("""## 8. Convergence — Seller Risk Index

Inner-join the three per-seller parquets on `seller_id`; min-max normalise each component (sentiment inverted so higher = worse); weight 0.35 · demand + 0.35 · sentiment + 0.30 · network. Band CRITICAL > 0.75, SAFE < 0.40, WARNING otherwise.""")

code('''from olist.pipeline.convergence import (
    build_seller_risk_index, risk_band_counts,
    top50_for_quadrant, state_mean_risk,
    RISK_WEIGHTS, RISK_CRITICAL_THRESHOLD, RISK_SAFE_THRESHOLD,
)
from olist import viz

risk = build_seller_risk_index(spark)
print(f"Weights: {RISK_WEIGHTS}   Thresholds: CRITICAL > {RISK_CRITICAL_THRESHOLD}, SAFE < {RISK_SAFE_THRESHOLD}")
print("Risk-band counts:")
risk_band_counts(risk).show()
print(f"Total scored sellers: {risk.count():,}")
''')

md("""### Top-20 highest-risk sellers — deployment-ready table

The account-management short-list: the 20 sellers where intervention will move the most aggregate risk. Bar embedded on `risk_score` so the in-band spread is visible; gradient on the three normalised components shows *which* signal is driving each seller's placement (delay-heavy vs sentiment-heavy vs network-heavy).""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 20 rows
top20_risk = (
    risk.orderBy(F.col("risk_score").desc())
    .limit(20)
    .select(
        "seller_id", "seller_state", "risk_score", "risk_class",
        "demand_norm", "sentiment_norm", "network_norm",
    )
    .toPandas()
)
top20_risk["seller_id"] = top20_risk["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top20_risk,
    bar_cols=["risk_score"],
    gradient_cols=["demand_norm", "sentiment_norm", "network_norm"],
    fmt={
        "risk_score": "{:.3f}",
        "demand_norm": "{:.3f}",
        "sentiment_norm": "{:.3f}",
        "network_norm": "{:.3f}",
    },
    title="Top-20 highest-risk sellers — risk_score + normalised component breakdown",
)
''')

md("""### Risk-band donut + quadrant scatter + state bar

Three management-ready visuals: (1) how many sellers land in each band; (2) where the top-50 sellers sit on the demand × sentiment plane, weighted by PageRank; (3) which Brazilian states carry the highest mean risk.""")

code('''# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 3-row aggregate
band_counts_pd = risk_band_counts(risk).toPandas()
viz.risk_band_donut(band_counts_pd)
''')

code('''# BIG-DATA-SAFETY-ESCAPE: TOP50_VIZ — 50-row pandas frame produced by convergence helper
top50 = top50_for_quadrant(risk)
viz.quadrant_scatter(
    top50,
    x="demand_norm",
    y="sentiment_norm",
    size="pagerank_score",
    color="risk_score",
    title="Top-50 risk quadrant — bubble = PageRank, colour = risk_score",
    xlabel="demand_norm (higher = longer avg delay)",
    ylabel="sentiment_norm (higher = worse sentiment)",
)
''')

code('''# BIG-DATA-SAFETY-ESCAPE: STATE_AGG_VIZ — per-state aggregate (≤27 rows)
state_risk = state_mean_risk(risk)
viz.state_bar(
    state_risk,
    value_col="mean_risk",
    label_col="seller_state",
    title="Mean seller risk_score by Brazilian state",
    sort="desc",
)
''')

# ===========================================================================
# 9. Recommendations
# ===========================================================================
md("""## 9. Recommendations for Olist

Prioritised actions grounded in the numbers above — management-ready, no jargon:

1. **Intervene on the WARNING band first.** CRITICAL is empty in this run; the WARNING sellers carry the full tail risk. Allocate account managers to the top-20 WARNING sellers — they are the list above in §8.

2. **Delivery-delay outliers dominate the demand-risk signal.** Sellers with `delay_risk_flag = 1` (avg delay > 3 days) correlate strongly with high `demand_norm`. These are logistics problems, not capacity problems — fleet contracts and warehouse allocation, not seller training.

3. **Sentiment is a weak leading indicator at this sample size.** The lead-indicator analysis found |corr| ≈ 0.015 — do not stake intervention triggers on sentiment alone. Use sentiment as a *confirming* signal alongside delay + network risk.

4. **High-centrality sellers need an operational backup.** PageRank surfaces ~10 sellers whose loss would disrupt the most customers; BFS already identifies a nearest-neighbour backup seller for each. Set up dual-sourcing agreements for the top-10 by PageRank.

5. **Geographic concentration of risk.** The state bar chart shows two or three Brazilian states with above-mean risk — worth a focused regional seller-health sweep.

6. **Invest in a streaming upgrade if marketplace volume grows 10×.** The entire pipeline is big-data-safe by construction (see `docs/big_data_safety_log.md`); moving `nb1_weekly_order_volume` to Structured Streaming turns this notebook into a continuous early-warning system.""")

# ===========================================================================
# 10. Big-data safety log
# ===========================================================================
md("""## 10. Big-data safety log

Every non-big-data-safe call in the codebase is catalogued in `docs/big_data_safety_log.md` and referenced by `# BIG-DATA-SAFETY-ESCAPE: <ID>` comments in `src/olist/pipeline/*.py`. Registry constants live in `src/olist/safety.py`.""")

code('''from olist import safety
print("Catalogued escape-hatch IDs (src/olist/safety.py):")
for constant in safety.ALL_ESCAPES:
    print(" ", constant)
''')

code('''# Render the safety log as a preview table (parsed from the markdown).
import re
safety_log = Path.cwd().parent / "docs" / "big_data_safety_log.md" if Path.cwd().name == "notebooks" else Path.cwd() / "docs" / "big_data_safety_log.md"
rows = []
for line in safety_log.read_text().splitlines():
    m = re.match(r"\\| `([A-Z_0-9]+)` \\| ([^|]+) \\|", line)
    if m:
        rows.append(m.groups())
print(f"{len(rows)} escape-hatch entries in docs/big_data_safety_log.md:")
for rid, site in rows:
    print(f"  {rid:<24}  {site.strip()[:80]}")
''')

# ===========================================================================
# 11. Rubric compliance + manifest + environment
# ===========================================================================
md("""## 11. Reproducibility + rubric compliance

Programmatic assertions: every rubric bullet is mapped to a check in `src/olist/checks.py::run_all()`. Fails loudly if a requirement is broken.""")

code('''from olist.checks import run_all as run_compliance_checks

checks_df = run_compliance_checks(spark)
checks_df.show(truncate=False, n=50)
''')

md("""### Cache manifest summary — which steps ran vs. skipped this session""")

code('''from olist.cache import manifest_summary
import json

summary = manifest_summary()
print(f"{len(summary)} steps in outputs/.cache_manifest.json:")
for entry in summary:
    print(f"  {entry['step']:<38}  fp={entry['fingerprint']}  written={entry['written_at']}")
''')

md("""### Environment versions""")

code('''import pyspark, sys
print("python:  ", sys.version.split()[0])
print("pyspark: ", pyspark.__version__)
try:
    import torch; print("torch:   ", torch.__version__)
except Exception:
    pass
try:
    import graphframes  # noqa
    print("graphframes coordinate is set in spark_session.py")
except Exception:
    pass
''')

code('''spark.stop()
print("\\nSpark stopped. Main notebook complete.")
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
