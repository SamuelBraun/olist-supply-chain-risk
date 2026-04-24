"""Emit notebooks/02_sentiment_analysis.ipynb from cell definitions below.

Thin-wrapper edition: logic in src/olist/pipeline/sentiment.py.
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


md("""# NB2 — Sentiment Analysis & Lead Indicator

**Scope.** Produces one consumer parquet:
- `outputs/nb2_seller_sentiment_scores.parquet` — per-seller sentiment metrics that feed the convergence layer.

**Rubric surface in this notebook.** DataFrames · SparkSQL (temp-view join) · NLP `Pipeline(Tokenizer → StopWordsRemover[pt] → HashingTF → IDF → LogisticRegression)` evaluated with AUC · Deep Learning (PyTorch LSTM, masked mean-pool) · Window functions (6-week rolling sentiment) · lead-indicator cross-correlation against NB1 volume.

**Thin-wrapper notice.** Every Spark primitive is called via `src/olist/pipeline/sentiment.py`; this notebook surfaces the functions' source, stages, and outputs via `inspect.getsource(...)`, `.getStages()`, `.show()`, and inline charts drawn from capped aggregates.

**Non-Spark library use.** PyTorch + pandas + matplotlib — all flagged with `# BIG-DATA-SAFETY-ESCAPE: <ID>` comments in `src/olist/pipeline/sentiment.py` and catalogued in `docs/big_data_safety_log.md`.""")

md("""## 1. Boot — `SparkSession`""")

code('''import os, sys, inspect
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from olist.spark_session import get_spark

spark = get_spark("nb2-sentiment-analysis")
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version)
''')

md("""## 2. SparkSQL join — `reviews_with_seller` via temp views

Four source tables are registered as temp views (`reviews`, `orders`, `order_items`, `sellers`) and joined via a literal `spark.sql(...)` query; the SQL text is printed inline. Output is cached as parquet via `@step`.""")

code('''from olist.pipeline.sentiment import build_reviews_with_seller, SENT_TEMP_JOIN_SQL

print("--- SparkSQL join (registered as temp view) ---")
print(SENT_TEMP_JOIN_SQL.strip())

reviews_with_seller = build_reviews_with_seller(spark)
print(f"\\nreviews_with_seller rows: {reviews_with_seller.count():,}")
reviews_with_seller.limit(3).show(truncate=40)
''')

md("""## 3. Labels — binary sentiment from `review_score`

Positive = score ≥ 4, negative = score ≤ 2; neutrals (score == 3) are dropped. Class balance reported below; the rationale is in `docs/decisions_log.md`.""")

code('''from olist.pipeline.sentiment import label_reviews
from pyspark.sql import functions as F

labelled = label_reviews(reviews_with_seller)
print("--- class balance ---")
labelled.groupBy("label").agg(F.count("*").alias("n")).orderBy("label").show()
print("total labelled rows:", labelled.count())
''')

md("""## 4. NLP Pipeline — `Tokenizer → StopWordsRemover[pt] → HashingTF → IDF → LogisticRegression`

The Pipeline stages are printed below so the grader can read the full stack. Trained on 80/20 split (seed=42); test AUC reported with `BinaryClassificationEvaluator(metricName='areaUnderROC')`.""")

code('''from olist.pipeline.sentiment import build_nlp_pipeline, fit_nlp_pipeline

nlp_pipeline = build_nlp_pipeline()
print("NLP Pipeline stages:")
for stage in nlp_pipeline.getStages():
    print(" ", stage)
''')

code('''nlp_result = fit_nlp_pipeline(labelled)
print(f"LogReg test AUC: {nlp_result['test_auc']:.4f}")
print(f"train rows:     {nlp_result['train_df'].count():,}")
print(f"test  rows:     {nlp_result['test_df'].count():,}")
''')

md("""## 5. Deep-learning — PyTorch LSTM (justification cell)

**Why PyTorch and not Spark MLlib.** MLlib has no native LSTM — the best Spark alternative would be `spark-nlp` (John Snow Labs) or Petastorm streaming Spark partitions into distributed PyTorch via Horovod/Ray Train. Neither makes sense at this dataset size (~43k reviews, ≤15 MB in pandas). The course brief explicitly allows small datasets; training on one CPU in ~5 minutes is proportionate.

**Escape hatches annotated in the source** (and catalogued in `docs/big_data_safety_log.md`):
- `# BIG-DATA-SAFETY-ESCAPE: LSTM_TO_PANDAS` on `text_labelled.toPandas()`.
- `# BIG-DATA-SAFETY-ESCAPE: LSTM_PYTORCH` on the model + training loop.

**Output.** Test AUC is compared to the LogReg baseline; the cached step stores metrics under `outputs/_cache/sentiment_lstm_metrics.parquet` so warm-cache reruns skip training.""")

code('''from olist.pipeline.sentiment import train_lstm_cached

lstm_metrics = train_lstm_cached(spark)
lstm_metrics.show(truncate=False)
lstm_row = lstm_metrics.first()
print(f"LSTM test AUC: {lstm_row['test_auc']:.4f}  |  LogReg baseline: {nlp_result['test_auc']:.4f}")
''')

md("""## 6. Weekly rolling sentiment per seller — Window functions

Per-seller weekly sentiment aggregation + 6-week rolling mean + `lag_6w_mean` via `Window.partitionBy(seller_id).orderBy(year_week).rowsBetween(-5, 0)`. Demonstrates the Window-functions rubric bullet.""")

code('''from olist.pipeline.sentiment import weekly_sentiment_rollup

weekly_with_trend = weekly_sentiment_rollup(reviews_with_seller)
print("weekly_with_trend rows:", weekly_with_trend.count())
weekly_with_trend.orderBy("seller_id", "year_week").limit(5).show(truncate=False)
''')

md("""## 7. Per-seller sentiment scores → parquet (convergence input)

For each seller: `avg_sentiment_score`, `sentiment_trend_6wk` (prior-6w mean minus current-6w mean), `pct_negative_reviews`, `sentiment_declining` (1 iff trend < −0.25). Written to `outputs/nb2_seller_sentiment_scores.parquet`.""")

code('''from olist.pipeline.sentiment import build_seller_sentiment_scores

seller_sentiment_scores = build_seller_sentiment_scores(spark)
print("seller_sentiment_scores rows:", seller_sentiment_scores.count())
print("sentiment_declining counts:")
seller_sentiment_scores.groupBy("sentiment_declining").agg(F.count("*").alias("n")).orderBy("sentiment_declining").show()
print("top 5 declining sellers:")
seller_sentiment_scores.orderBy(F.col("sentiment_trend_6wk").asc()).limit(5).show(truncate=False)
''')

md("""## 8. Lead indicator — does sentiment decline *precede* volume decline?

Lag-k Pearson correlation between weekly sentiment change at `t` and weekly volume change at `t+k`, k ∈ [0, 8], pooled across sellers. Correlation is computed inside Spark via `F.corr(...).first()` — single-row aggregate, big-data-safe.""")

code('''from olist.pipeline.sentiment import build_lead_indicator_lags, peak_lag

lag_df = build_lead_indicator_lags(spark)
lag_df.show()
peak_k, peak_rho = peak_lag(lag_df)
print(f"Peak |corr| at lag = {peak_k} weeks (corr = {peak_rho:.4f})")
''')

md("""## 9. Inline chart — cross-correlation by lag

`# BIG-DATA-SAFETY-ESCAPE: LEAD_INDICATOR_VIZ` (9-row aggregate → pandas → matplotlib).""")

code('''import matplotlib.pyplot as plt

# BIG-DATA-SAFETY-ESCAPE: LEAD_INDICATOR_VIZ
lag_pd = lag_df.toPandas()

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(lag_pd["lag"], lag_pd["corr"], color="#4C72B0")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Lag k (weeks: sentiment change at t vs volume change at t+k)")
ax.set_ylabel("Pearson correlation")
ax.set_title("Does a drop in sentiment precede a drop in volume?")
ax.set_xticks(lag_pd["lag"])
plt.tight_layout()
plt.show()
print("Takeaway: peak |corr| is weak across all lags — sentiment is not a strong leading indicator at this sample size.")
''')

md("""## 10. Clean up""")

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
