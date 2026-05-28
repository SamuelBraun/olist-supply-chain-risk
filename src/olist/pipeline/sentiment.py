"""NB2 — sentiment analysis and lead-indicator against NB1 volume.

Public surface split into two kinds of functions:
* Spark-native primitives (temp-view join, NLP pipeline, weekly rollup via
  Window functions, lead-indicator correlation).
* Non-Spark escape-hatch primitives (PyTorch LSTM training) — flagged with
  `# BIG-DATA-SAFETY-ESCAPE: <ID>` comments referencing `olist.safety`.

Writes:
* `outputs/nb2_seller_sentiment_scores.parquet` — feeds convergence.

Caches:
* `outputs/_cache/sentiment_reviews_with_seller.parquet`
* `outputs/_cache/sentiment_weekly.parquet`
* `outputs/_cache/sentiment_lead_lags.parquet`
* `outputs/_cache/sentiment_lstm_metrics.parquet`
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ..cache import resolve_path, step
from ..loaders import load_order_items, load_order_reviews, load_orders, load_sellers
from ..safety import (  # noqa: F401 — referenced by annotation comments
    BFS_BACKUP_COLLECT,
    LEAD_INDICATOR_VIZ,
    LSTM_PYTORCH,
    LSTM_TO_PANDAS,
    SMALL_SUMMARY_COLLECT,
)

_SENT_CODE_DEPS = [
    "src/olist/pipeline/sentiment.py",
    "src/olist/loaders.py",
    "src/olist/schemas.py",
]

LSTM_VOCAB_SIZE = 20_000
LSTM_MAX_LEN = 128
LSTM_PAD_IDX = 0
LSTM_UNK_IDX = 1
LSTM_EMBED_DIM = 64
LSTM_HIDDEN_DIM = 64
LSTM_EPOCHS = 3
LSTM_BATCH_SIZE = 128
LSTM_SEED = 1394
NLP_SPLIT_SEED = 5067

_TOKEN_RE = re.compile(r"[a-záàâãéêíóôõúüç]+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Spark-native stages
# ---------------------------------------------------------------------------


@step(
    name="sentiment.reviews_with_seller",
    inputs=[
        "data/olist_order_reviews_dataset.csv",
        "data/olist_orders_dataset.csv",
        "data/olist_order_items_dataset.csv",
        "data/olist_sellers_dataset.csv",
    ],
    outputs=["outputs/_cache/sentiment_reviews_with_seller.parquet"],
    code_deps=_SENT_CODE_DEPS,
    version=1,
)
def build_reviews_with_seller(spark: SparkSession) -> DataFrame:
    """Register the four source tables as temp views and join via SparkSQL
    into one row per (review × seller). Satisfies the NB2 rubric line
    "≥1 SparkSQL join via temp view".
    """
    reviews_all = load_order_reviews(spark)
    orders = load_orders(spark)
    order_items = load_order_items(spark)
    sellers = load_sellers(spark)
    reviews_all.createOrReplaceTempView("reviews")
    orders.createOrReplaceTempView("orders")
    order_items.createOrReplaceTempView("order_items")
    sellers.createOrReplaceTempView("sellers")
    return spark.sql(
        """
        SELECT r.review_id,
               r.order_id,
               r.review_score,
               r.review_comment_message,
               r.review_creation_date,
               oi.seller_id,
               s.seller_state
        FROM reviews r
        JOIN orders      o  ON r.order_id = o.order_id
        JOIN order_items oi ON r.order_id = oi.order_id
        LEFT JOIN sellers s ON oi.seller_id = s.seller_id
        """
    )


SENT_TEMP_JOIN_SQL = """
SELECT r.review_id,
       r.order_id,
       r.review_score,
       r.review_comment_message,
       r.review_creation_date,
       oi.seller_id,
       s.seller_state
FROM reviews r
JOIN orders      o  ON r.order_id = o.order_id
JOIN order_items oi ON r.order_id = oi.order_id
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
"""


def label_reviews(reviews_with_seller: DataFrame) -> DataFrame:
    # >=4 -> pos, <=2 -> neg, drop neutrals. ~82/18 class balance after.
    return (
        reviews_with_seller.filter(F.col("review_score").isNotNull())
        .withColumn(
            "label",
            F.when(F.col("review_score") >= 4, 1).when(F.col("review_score") <= 2, 0),
        )
        .filter(F.col("label").isNotNull())
    )


def build_nlp_pipeline() -> Pipeline:
    """Tokenizer + PT stopwords + HashingTF + IDF + LR. Unfit."""
    pt_stopwords = StopWordsRemover.loadDefaultStopWords("portuguese")
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    stop_remover = StopWordsRemover(
        inputCol="tokens", outputCol="tokens_clean", stopWords=pt_stopwords
    )
    hashing_tf = HashingTF(inputCol="tokens_clean", outputCol="tf", numFeatures=2**16)
    idf = IDF(inputCol="tf", outputCol="features")
    lr = LogisticRegression(
        featuresCol="features", labelCol="label", maxIter=30, regParam=0.01
    )
    return Pipeline(stages=[tokenizer, stop_remover, hashing_tf, idf, lr])


def fit_nlp_pipeline(labelled: DataFrame) -> dict:
    """Filter to rows with non-null comments, fit the NLP pipeline under a
    3-fold ``CrossValidator`` over the LogisticRegression regulariser, and
    evaluate the best model's AUC on the held-out test split.

    Tunes `regParam ∈ {0.0, 0.01, 0.1}` and `elasticNetParam ∈ {0.0, 0.5}`
    (6 combinations × 3 folds = 18 sub-fits) with
    ``BinaryClassificationEvaluator(areaUnderROC)``. CV counts toward the
    rubric's `check_cv_models ≥ 3` assertion (alongside GBT + RF in demand).

    Returns a dict with `pipeline_model` (the best fit), `test_auc`,
    `best_params` (the chosen `regParam` / `elasticNetParam`), `cv_avg_metrics`
    (mean AUC per param combo), `train_df`, `test_df`, `test_preds`.
    """
    text_labelled = (
        labelled.filter(F.col("review_comment_message").isNotNull())
        .withColumn("text", F.lower(F.col("review_comment_message")))
        .select("text", "label")
    )
    train_df, test_df = text_labelled.randomSplit([0.8, 0.2], seed=NLP_SPLIT_SEED)

    nlp_pipeline = build_nlp_pipeline()
    lr_stage: LogisticRegression = nlp_pipeline.getStages()[-1]
    param_grid = (
        ParamGridBuilder()
        .addGrid(lr_stage.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr_stage.elasticNetParam, [0.0, 0.5])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    cv = CrossValidator(
        estimator=nlp_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=NLP_SPLIT_SEED,
        parallelism=2,
        collectSubModels=False,
    )
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel
    best_lr = best_model.stages[-1]
    best_params = {
        "regParam": float(best_lr.getRegParam()),
        "elasticNetParam": float(best_lr.getElasticNetParam()),
    }
    cv_avg_metrics = [
        {
            "regParam": float(pm[lr_stage.regParam]),
            "elasticNetParam": float(pm[lr_stage.elasticNetParam]),
            "cv_avg_auc": float(metric),
        }
        for pm, metric in zip(param_grid, cv_model.avgMetrics)
    ]
    preds = best_model.transform(test_df)
    auc = evaluator.evaluate(preds)
    return {
        "pipeline_model": best_model,
        "test_auc": float(auc),
        "best_params": best_params,
        "cv_avg_metrics": cv_avg_metrics,
        "train_df": train_df,
        "test_df": test_df,
        "test_preds": preds,
        "text_labelled": text_labelled,
    }


# ---------------------------------------------------------------------------
# LSTM (PyTorch) — mandatory Deep Learning rubric line
# ---------------------------------------------------------------------------


def _tokenize_portuguese(text: str, stopset: set[str]) -> list[str]:
    return [w for w in _TOKEN_RE.findall(str(text).lower()) if w not in stopset]


def _build_vocab(tokens_iter) -> dict[str, int]:
    counter: Counter = Counter()
    for toks in tokens_iter:
        counter.update(toks)
    most_common = [w for w, _ in counter.most_common(LSTM_VOCAB_SIZE - 2)]
    return {w: i + 2 for i, w in enumerate(most_common)}


def _encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(w, LSTM_UNK_IDX) for w in tokens][:LSTM_MAX_LEN]
    if len(ids) < LSTM_MAX_LEN:
        ids = ids + [LSTM_PAD_IDX] * (LSTM_MAX_LEN - len(ids))
    return ids


def train_lstm(text_labelled: DataFrame) -> dict:
    """Train a small LSTM on Portuguese review text for the mandatory deep-
    learning rubric line. Returns `{test_auc, train_loss_per_epoch, n_train,
    n_test, vocab_size}`.

    Non-Spark escape hatches in this function:
    * `text_labelled.toPandas()`    # BIG-DATA-SAFETY-ESCAPE: LSTM_TO_PANDAS
    * PyTorch model + training loop # BIG-DATA-SAFETY-ESCAPE: LSTM_PYTORCH
    Both are catalogued in `docs/big_data_safety_log.md`.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)

    # BIG-DATA-SAFETY-ESCAPE: LSTM_TO_PANDAS — see docs/big_data_safety_log.md
    text_pd = text_labelled.toPandas()
    pt_stopset = set(StopWordsRemover.loadDefaultStopWords("portuguese"))
    tokens_pd = text_pd["text"].map(
        lambda t: _tokenize_portuguese(t, pt_stopset)
    )
    vocab = _build_vocab(tokens_pd)

    encoded = np.array(
        [_encode(t, vocab) for t in tokens_pd], dtype=np.int64
    )
    labels = text_pd["label"].to_numpy(dtype=np.int64)

    perm = np.random.permutation(len(encoded))
    split = int(0.8 * len(encoded))
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = encoded[train_idx], labels[train_idx]
    X_test, y_test = encoded[test_idx], labels[test_idx]

    class ReviewsDS(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return (
                torch.from_numpy(self.X[i]),
                torch.tensor(self.y[i], dtype=torch.float32),
            )

    # BIG-DATA-SAFETY-ESCAPE: LSTM_PYTORCH — see docs/big_data_safety_log.md
    class LSTMSentiment(nn.Module):
        def __init__(
            self,
            vocab_size: int = LSTM_VOCAB_SIZE,
            embed_dim: int = LSTM_EMBED_DIM,
            hidden_dim: int = LSTM_HIDDEN_DIM,
        ):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=LSTM_PAD_IDX)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            mask = (x != LSTM_PAD_IDX).unsqueeze(-1).float()
            embedded = self.embed(x)
            out, _ = self.lstm(embedded)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.fc(pooled).squeeze(-1)

    device = "cpu"
    model = LSTMSentiment().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        ReviewsDS(X_train, y_train), batch_size=LSTM_BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        ReviewsDS(X_test, y_test), batch_size=LSTM_BATCH_SIZE * 2
    )

    losses: list[float] = []
    for epoch in range(LSTM_EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        mean_loss = running / len(X_train)
        losses.append(mean_loss)
        print(f"epoch {epoch + 1}  train_loss={mean_loss:.4f}")

    model.eval()
    scores_chunks, label_chunks = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            scores_chunks.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            label_chunks.append(yb.numpy())
    all_scores = np.concatenate(scores_chunks)
    all_labels = np.concatenate(label_chunks)
    auc = float(roc_auc_score(all_labels, all_scores))
    return {
        "test_auc": auc,
        "train_loss_per_epoch": losses,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "vocab_size": int(len(vocab) + 2),
    }


@step(
    name="sentiment.lstm_metrics",
    inputs=[
        "outputs/_cache/sentiment_reviews_with_seller.parquet",
        "src/olist/pipeline/sentiment.py",
    ],
    outputs=["outputs/_cache/sentiment_lstm_metrics.parquet"],
    code_deps=_SENT_CODE_DEPS,
    version=1,
)
def train_lstm_cached(spark: SparkSession) -> DataFrame:
    """Cacheable wrapper around `train_lstm`. Reads the cached reviews-with-
    seller parquet, re-derives the text-labelled frame, trains the LSTM, and
    writes a 1-row parquet: (test_auc, n_train, n_test, vocab_size, epochs_json).
    """
    import json

    reviews = spark.read.parquet(
        resolve_path("outputs/_cache/sentiment_reviews_with_seller.parquet")
    )
    labelled = label_reviews(reviews)
    text_labelled = (
        labelled.filter(F.col("review_comment_message").isNotNull())
        .withColumn("text", F.lower(F.col("review_comment_message")))
        .select("text", "label")
    )
    result = train_lstm(text_labelled)
    row = (
        float(result["test_auc"]),
        int(result["n_train"]),
        int(result["n_test"]),
        int(result["vocab_size"]),
        json.dumps(result["train_loss_per_epoch"]),
    )
    return spark.createDataFrame(
        [row],
        schema=(
            "test_auc double, n_train bigint, n_test bigint, "
            "vocab_size bigint, train_loss_json string"
        ),
    )


# ---------------------------------------------------------------------------
# Weekly rolling sentiment + per-seller scores
# ---------------------------------------------------------------------------


def weekly_sentiment_rollup(reviews_with_seller: DataFrame) -> DataFrame:
    """Weekly-aggregated sentiment per seller + 6-week rolling mean via
    Window functions. Demonstrates the "Window functions" rubric surface.
    """
    weekly_sentiment = (
        reviews_with_seller.filter(F.col("review_score").isNotNull())
        .withColumn(
            "year_week",
            F.concat(
                F.year("review_creation_date"),
                F.lit("-"),
                F.lpad(F.weekofyear("review_creation_date").cast("string"), 2, "0"),
            ),
        )
        .groupBy("seller_id", "year_week")
        .agg(
            F.avg("review_score").alias("avg_score_week"),
            F.avg(F.when(F.col("review_score") <= 2, 1.0).otherwise(0.0)).alias(
                "pct_neg_week"
            ),
            F.count("*").alias("n_reviews_week"),
        )
    )
    w_seller = Window.partitionBy("seller_id").orderBy("year_week")
    w_roll6 = w_seller.rowsBetween(-5, 0)
    return (
        weekly_sentiment.withColumn("week_num", F.row_number().over(w_seller))
        .withColumn("rolling_6w_mean", F.avg("avg_score_week").over(w_roll6))
        .withColumn("lag_6w_mean", F.lag("rolling_6w_mean", 6).over(w_seller))
    )


def confusion_counts(test_preds: DataFrame) -> DataFrame:
    # 4-row groupBy. materialisation only at render time.
    return (
        test_preds.groupBy("label", "prediction")
        .count()
        .withColumnRenamed("count", "n")
        .orderBy("label", "prediction")
    )


def top_sellers_by_reviews(
    weekly_with_trend: DataFrame,
    *,
    k: int = 5,
) -> DataFrame:
    """Top-k sellers by total review count, long-format for the multiline chart."""
    totals = (
        weekly_with_trend.groupBy("seller_id")
        .agg(F.sum("n_reviews_week").alias("total_reviews"))
        .orderBy(F.col("total_reviews").desc())
        .limit(k)
    )
    return weekly_with_trend.join(
        F.broadcast(totals.select("seller_id")), "seller_id", "inner"
    ).orderBy("seller_id", "year_week")


@step(
    name="sentiment.seller_scores",
    inputs=["outputs/_cache/sentiment_reviews_with_seller.parquet"],
    outputs=["outputs/nb2_seller_sentiment_scores.parquet"],
    code_deps=_SENT_CODE_DEPS,
    version=1,
)
def build_seller_sentiment_scores(spark: SparkSession) -> DataFrame:
    """Per-seller sentiment scores, the convergence-layer consumer artefact.

    Columns: seller_id, avg_sentiment_score, sentiment_trend_6wk,
    pct_negative_reviews, sentiment_declining.

    `sentiment_declining = 1` iff the 6-week-rolling mean dropped by more
    than 0.25 stars between the prior and current 6-week window.
    """
    reviews = spark.read.parquet(
        resolve_path("outputs/_cache/sentiment_reviews_with_seller.parquet")
    )
    weekly = weekly_sentiment_rollup(reviews)
    last_row_w = Window.partitionBy("seller_id").orderBy(F.col("week_num").desc())
    seller_trend = (
        weekly.withColumn("rk", F.row_number().over(last_row_w))
        .filter(F.col("rk") == 1)
        .select(
            "seller_id",
            F.col("rolling_6w_mean").alias("current_6w_mean"),
            F.col("lag_6w_mean").alias("prior_6w_mean"),
        )
        .withColumn(
            "sentiment_trend_6wk",
            F.when(
                F.col("prior_6w_mean").isNotNull(),
                F.col("current_6w_mean") - F.col("prior_6w_mean"),
            ).otherwise(F.lit(0.0)),
        )
    )
    seller_aggregates = (
        reviews.filter(F.col("review_score").isNotNull())
        .groupBy("seller_id")
        .agg(
            F.avg("review_score").alias("avg_sentiment_score"),
            F.avg(F.when(F.col("review_score") <= 2, 1.0).otherwise(0.0)).alias(
                "pct_negative_reviews"
            ),
        )
    )
    return (
        seller_aggregates.join(
            seller_trend.select("seller_id", "sentiment_trend_6wk"),
            "seller_id",
            "left",
        )
        .withColumn(
            "sentiment_trend_6wk",
            F.coalesce(F.col("sentiment_trend_6wk"), F.lit(0.0)),
        )
        .withColumn(
            "sentiment_declining",
            (F.col("sentiment_trend_6wk") < -0.25).cast("int"),
        )
        .select(
            "seller_id",
            "avg_sentiment_score",
            "sentiment_trend_6wk",
            "pct_negative_reviews",
            "sentiment_declining",
        )
    )


# ---------------------------------------------------------------------------
# Lead indicator: does sentiment decline precede volume decline?
# ---------------------------------------------------------------------------


@step(
    name="sentiment.lead_indicator_lags",
    inputs=[
        "outputs/_cache/sentiment_reviews_with_seller.parquet",
        "outputs/nb1_weekly_order_volume.parquet",
    ],
    outputs=["outputs/_cache/sentiment_lead_lags.parquet"],
    code_deps=_SENT_CODE_DEPS,
    version=1,
)
def build_lead_indicator_lags(spark: SparkSession) -> DataFrame:
    """For lags k ∈ [0, 8] weeks, compute Pearson correlation between weekly
    sentiment change at t and weekly volume change at t+k, pooled across
    sellers. Returns a 9-row DataFrame: (lag, corr, n_pairs).

    Correlation uses Spark's built-in `F.corr` — single-row aggregate → safe.
    The driver-side `.first()` is flagged as SMALL_SUMMARY_COLLECT.
    """
    reviews = spark.read.parquet(
        resolve_path("outputs/_cache/sentiment_reviews_with_seller.parquet")
    )
    weekly_sentiment = (
        reviews.filter(F.col("review_score").isNotNull())
        .withColumn(
            "year_week",
            F.concat(
                F.year("review_creation_date"),
                F.lit("-"),
                F.lpad(F.weekofyear("review_creation_date").cast("string"), 2, "0"),
            ),
        )
        .groupBy("seller_id", "year_week")
        .agg(F.avg("review_score").alias("avg_score_week"))
    )
    weekly_volume = spark.read.parquet(resolve_path("outputs/nb1_weekly_order_volume.parquet"))
    combined = (
        weekly_sentiment.join(weekly_volume, ["seller_id", "year_week"], "inner")
        .select("seller_id", "year_week", "avg_score_week", "weekly_order_count")
    )
    ws = Window.partitionBy("seller_id").orderBy("year_week")
    combined = (
        combined.withColumn(
            "d_sent",
            F.col("avg_score_week") - F.lag("avg_score_week", 1).over(ws),
        )
        .withColumn(
            "d_volume",
            F.col("weekly_order_count") - F.lag("weekly_order_count", 1).over(ws),
        )
        .filter(F.col("d_sent").isNotNull() & F.col("d_volume").isNotNull())
    )
    lag_rows = []
    for k in range(0, 9):
        paired = combined.withColumn(
            "d_vol_lead", F.lag(F.col("d_volume"), -k).over(ws)
        ).filter(F.col("d_vol_lead").isNotNull())
        # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — single-row agg
        rho = paired.agg(F.corr("d_sent", "d_vol_lead").alias("rho")).first()["rho"]
        n_pairs = paired.count()
        lag_rows.append((k, float(rho) if rho is not None else None, int(n_pairs)))
    return spark.createDataFrame(
        lag_rows, "lag int, corr double, n_pairs bigint"
    )


def peak_lag(lag_df: DataFrame) -> tuple[int, float]:
    """Return (lag_weeks, correlation) at peak |corr|. Safe collect — the
    input is 9 rows by construction.
    """
    # BIG-DATA-SAFETY-ESCAPE: LEAD_INDICATOR_VIZ — 9-row aggregate → pandas
    lag_pd = lag_df.toPandas()
    idx = lag_pd["corr"].abs().idxmax()
    return int(lag_pd.iloc[idx]["lag"]), float(lag_pd.iloc[idx]["corr"])
