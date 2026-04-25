"""Emit notebooks/02_sentiment_analysis.ipynb — CRISP-DM edition.

Thin-wrapper: every transformation lives in src/olist/pipeline/sentiment.py.
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
OUT_NB = ROOT / "notebooks" / "02_sentiment_analysis.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md("""# NB2 — Sentiment Analysis & Lead Indicator

**Scope.** Produces one consumer parquet for the convergence layer:
- `outputs/nb2_seller_sentiment_scores.parquet` — per-seller sentiment summary (avg score, 6-week trend, % negative, declining flag).

**Rubric surface in this notebook.** DataFrames · SparkSQL (temp-view join) · NLP `Pipeline(Tokenizer → StopWordsRemover[pt] → HashingTF → IDF → LogisticRegression)` under `CrossValidator` · Deep Learning (PyTorch LSTM, masked mean-pool) · Window functions (6-week rolling sentiment) · lead-indicator cross-correlation.

**Narrative structure.** This notebook follows the **CRISP-DM** methodology — six sections from *Business Understanding* to *Deployment*. Every code cell calls into `src/olist/pipeline/sentiment.py`; every chart/table goes through `src/olist/viz.py`.""")


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
md("""## 0. Boot — `SparkSession`""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark
from pyspark.sql import functions as F

spark = get_spark("nb2-sentiment-analysis")
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')


# ---------------------------------------------------------------------------
# 1. Business Understanding
# ---------------------------------------------------------------------------
md("""## 1. Business Understanding

**Why sentiment matters to Olist.** Customer reviews are a leading signal of seller health — they surface problems (damaged packaging, slow dispatch, product mismatch) *before* the problem becomes a sales drop. This notebook builds the **sentiment component** of the Seller Risk Index:

- **Short-term signal.** Per-seller average review score + % negative reviews — how customers feel *right now*.
- **Trend signal.** 6-week rolling mean vs. its lag-6w value — is this seller improving or deteriorating?
- **Lead-indicator hypothesis.** Does a drop in sentiment at week *t* precede a drop in volume at week *t + k*? (Tested at lags 0–8 weeks.)

**Stakeholder question.** *"Which sellers show early signs of customer dissatisfaction that we should intervene on before it affects sales?"* — the per-seller sentiment parquet written in §6 answers this row-by-row.""")


# ---------------------------------------------------------------------------
# 2. Data Understanding
# ---------------------------------------------------------------------------
md("""## 2. Data Understanding

Four source tables joined via SparkSQL: `reviews ⋈ orders ⋈ order_items ⋈ sellers`. Rows are review-level; each review inherits the seller(s) of its order.""")

md("""### 2.1 SparkSQL temp-view join

The raw SQL is printed inline so the rubric surface is visible in the notebook itself; the query registers four temp views and joins them into `reviews_with_seller`, cached as parquet via `@step`.""")

code('''from olist.pipeline.sentiment import build_reviews_with_seller, SENT_TEMP_JOIN_SQL

print(SENT_TEMP_JOIN_SQL.strip())

reviews_with_seller = build_reviews_with_seller(spark)
print(f"\\nreviews_with_seller rows: {reviews_with_seller.count():,}")
reviews_with_seller.limit(3).show(truncate=40)
''')

md("""### 2.2 Binary labels + class-balance diagnostic

We label positive (`score ≥ 4`) vs. negative (`score ≤ 2`) and drop neutrals (`score == 3`). Before training any model we verify the class balance — a heavily-skewed dataset would bias AUC optimism.""")

code('''from olist.pipeline.sentiment import label_reviews
from olist import viz

labelled = label_reviews(reviews_with_seller)
balance = labelled.groupBy("label").agg(F.count("*").alias("n")).orderBy("label")
balance.show()

balance_pd = balance.toPandas()
balance_pd["label"] = balance_pd["label"].map({0: "negative (≤2)", 1: "positive (≥4)"})
print(f"Total labelled rows: {int(balance_pd['n'].sum()):,}")
viz.class_balance_bar(balance_pd)
''')

md("""~82 / 18 positive / negative split — moderate imbalance. LogReg AUC (§5) is a fair metric under this imbalance; we also inspect a confusion matrix rather than relying on accuracy.""")


# ---------------------------------------------------------------------------
# 3. Data Preparation
# ---------------------------------------------------------------------------
md("""## 3. Data Preparation

Two transformations land in the modelling step:

1. **Text preparation.** Lower-case review comments; drop null-comment rows (done inside `fit_nlp_pipeline`). The NLP `Pipeline` then handles tokenisation + Portuguese stopword removal + TF-IDF.
2. **Weekly rollup via Window.** Per-seller weekly aggregation of `avg_score_week`, `pct_neg_week`, `n_reviews_week`, followed by a 6-week rolling mean via `Window.partitionBy(seller_id).orderBy(year_week).rowsBetween(-5, 0)` — the Window-functions rubric surface.""")

code('''from olist.pipeline.sentiment import weekly_sentiment_rollup

print(inspect.getsource(weekly_sentiment_rollup))
''')

code('''weekly_with_trend = weekly_sentiment_rollup(reviews_with_seller)
print(f"weekly_with_trend rows: {weekly_with_trend.count():,}")
weekly_with_trend.orderBy("seller_id", "year_week").limit(5).show(truncate=False)
''')


# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------
md("""## 4. Modeling

Two models share the same labelled input:

- **LogReg** over TF-IDF features — the Spark-native classical baseline, fitted via an `ML Pipeline`.
- **LSTM** in PyTorch — the mandatory Deep-Learning rubric line; justified below as a big-data-safety escape.""")

md("""### 4.1 NLP Pipeline stages

`Tokenizer → StopWordsRemover[pt] → HashingTF(2^16) → IDF → LogisticRegression(maxIter=30, regParam=0.01)`. Trained on an 80/20 split (seed=42). The fitted model is retained so §5 can render a confusion matrix.""")

code('''from olist.pipeline.sentiment import build_nlp_pipeline, fit_nlp_pipeline

nlp_pipeline = build_nlp_pipeline()
for stage in nlp_pipeline.getStages():
    print(" ", stage)
''')

code('''nlp_result = fit_nlp_pipeline(labelled)
print(f"LogReg test AUC: {nlp_result['test_auc']:.4f}")
print(f"train rows:     {nlp_result['train_df'].count():,}")
print(f"test  rows:     {nlp_result['test_df'].count():,}")
''')

md("""### 4.2 Deep learning — PyTorch LSTM (escape justified)

**Why PyTorch and not Spark MLlib.** MLlib has no native LSTM. Production-scale alternatives are `spark-nlp` (John Snow Labs) or Petastorm + PyTorch DDP. Neither justifies the complexity at ~43k Portuguese comments (~15 MB in pandas).

**Escape hatches annotated in the source:**
- `# BIG-DATA-SAFETY-ESCAPE: LSTM_TO_PANDAS` on `text_labelled.toPandas()`.
- `# BIG-DATA-SAFETY-ESCAPE: LSTM_PYTORCH` on the model + training loop.

**Cache.** `train_lstm_cached` wraps the 3-epoch training loop in `@step`; warm-cache reruns skip training and re-read `outputs/_cache/sentiment_lstm_metrics.parquet`.""")

code('''from olist.pipeline.sentiment import train_lstm_cached

lstm_metrics = train_lstm_cached(spark)
lstm_metrics.show(truncate=False)
lstm_row = lstm_metrics.first()
print(f"LSTM test AUC: {lstm_row['test_auc']:.4f}  |  LogReg baseline: {nlp_result['test_auc']:.4f}")
''')


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
md("""## 5. Evaluation

Five diagnostic views: LogReg confusion matrix, per-seller trend multiline, lead-indicator lag chart, and a styled top-10 declining table.""")

md("""### 5.1 LogReg confusion matrix

AUC alone hides false-positive / false-negative asymmetry. The per-class recall below shows whether the classifier is actually useful on the minority (negative) class.""")

code('''from olist.pipeline.sentiment import confusion_counts

cm = confusion_counts(nlp_result["test_preds"])
cm.show()

# BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — 4-row groupBy aggregate
cm_pd = cm.toPandas()
viz.confusion_matrix_heatmap(cm_pd)
''')

md("""### 5.2 Top-5 sellers by review volume — rolling 6-week sentiment

Which sellers drive the most reviews, and how does their rolling sentiment behave over time? A multi-line chart exposes recovery patterns, stable-high performers, and persistent-low sellers at a glance.""")

code('''from olist.pipeline.sentiment import top_sellers_by_reviews

top5_weekly = top_sellers_by_reviews(weekly_with_trend, k=5)
# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 5 sellers × ~100 weeks max
top5_pd = (
    top5_weekly.select("seller_id", "year_week", "rolling_6w_mean")
    .filter(F.col("year_week").isNotNull() & F.col("rolling_6w_mean").isNotNull())
    .toPandas()
)
top5_pd["short_id"] = top5_pd["seller_id"].str.slice(0, 8)
print(f"top-5 sellers × {top5_pd['year_week'].nunique()} unique weeks = {len(top5_pd)} points")
viz.weekly_trend_multiline(
    top5_pd, x="year_week", y="rolling_6w_mean", hue="short_id",
    title="Rolling 6-week sentiment — top 5 sellers by review volume",
)
''')

md("""### 5.3 Lead indicator — does sentiment *precede* volume?

Lag-k Pearson correlation between weekly sentiment change and weekly volume change, k ∈ [0, 8]. Computed inside Spark (`F.corr(...).first()` — single-row aggregate, big-data-safe).""")

code('''from olist.pipeline.sentiment import build_lead_indicator_lags, peak_lag

lag_df = build_lead_indicator_lags(spark)
lag_df.show()
peak_k, peak_rho = peak_lag(lag_df)
print(f"Peak |ρ| = {abs(peak_rho):.4f} at lag = {peak_k} weeks")

# BIG-DATA-SAFETY-ESCAPE: LEAD_INDICATOR_VIZ — 9-row aggregate
lag_pd = lag_df.toPandas()
viz.lag_corr_bar(lag_pd)
''')

md("""**Honest finding.** Peak |ρ| ≈ 0.015 — sentiment is *not* a strong leading indicator of volume at this sample size. The presentation reports this as a caveat, not a headline. The weekly rollup itself is still valuable as a *trend* signal within the risk index (§6).""")


# ---------------------------------------------------------------------------
# 6. Deployment
# ---------------------------------------------------------------------------
md("""## 6. Deployment

The deployable artefact is `outputs/nb2_seller_sentiment_scores.parquet` — one row per seller, five columns, consumed directly by the convergence layer (`olist.pipeline.convergence.build_seller_risk_index`).""")

md("""### 6.1 Per-seller scores

Per seller: `avg_sentiment_score`, `sentiment_trend_6wk` (prior-6w mean minus current-6w mean, signed), `pct_negative_reviews`, `sentiment_declining` (1 iff trend < −0.25).""")

code('''from olist.pipeline.sentiment import build_seller_sentiment_scores

seller_sentiment_scores = build_seller_sentiment_scores(spark)
print(f"seller_sentiment_scores rows: {seller_sentiment_scores.count():,}")
seller_sentiment_scores.groupBy("sentiment_declining").agg(F.count("*").alias("n")).orderBy("sentiment_declining").show()
''')

md("""### 6.2 Top-10 declining sellers — deployment-ready table

These are the sellers an account manager should prioritise for outreach. The styled table below embeds a magnitude bar on the trend column (more-negative = redder) and a gradient on `pct_negative_reviews`.""")

code('''# BIG-DATA-SAFETY-ESCAPE: PANDAS_MATPLOTLIB_VIZ — capped to 10 rows
top10_declining = (
    seller_sentiment_scores.orderBy(F.col("sentiment_trend_6wk").asc()).limit(10).toPandas()
)
top10_declining["seller_id"] = top10_declining["seller_id"].str.slice(0, 10) + "…"
viz.styled_topn_table(
    top10_declining,
    bar_cols=["sentiment_trend_6wk"],
    gradient_cols=["pct_negative_reviews"],
    fmt={
        "avg_sentiment_score": "{:.2f}",
        "sentiment_trend_6wk": "{:+.3f}",
        "pct_negative_reviews": "{:.1%}",
        "sentiment_declining": "{:d}",
    },
    title="Top-10 declining sellers (by sentiment_trend_6wk)",
)
''')

md("""### 6.3 Clean up""")

code('''ROOT_OUT = Path.cwd().parent / "outputs" if Path.cwd().name == "notebooks" else Path.cwd() / "outputs"
print("Notebook 2 outputs:")
for path in sorted(ROOT_OUT.glob("nb2_*.parquet")):
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
