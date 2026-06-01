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

import os
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
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
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

# Fast-iteration mode for CI / dev reruns. When OLIST_LIGHT=1 the heavy CV grid
# and LSTM epoch count shrink; the full grid/epochs run otherwise (nightly heavy
# run). Never touches labels, schemas, or any written parquet.
LIGHT = os.environ.get("OLIST_LIGHT") == "1"

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
    if LIGHT:
        # 2 combos (regParam only) × 2 folds — enough to exercise CV wiring fast.
        param_grid = (
            ParamGridBuilder()
            .addGrid(lr_stage.regParam, [0.01, 0.1])
            .build()
        )
        num_folds = 2
    else:
        param_grid = (
            ParamGridBuilder()
            .addGrid(lr_stage.regParam, [0.0, 0.01, 0.1])
            .addGrid(lr_stage.elasticNetParam, [0.0, 0.5])
            .build()
        )
        num_folds = 3
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    cv = CrossValidator(
        estimator=nlp_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
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
            # elasticNetParam is only in the grid in the full (non-LIGHT) run;
            # fall back to the stage default otherwise.
            "elasticNetParam": float(
                pm.get(lr_stage.elasticNetParam, lr_stage.getElasticNetParam())
            ),
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


def _patch_torchdistributor_ipv4_rendezvous() -> None:
    """Force TorchDistributor's local-mode rendezvous onto the IPv4 loopback.

    The stock ``_get_torchrun_args`` returns ``--standalone`` for local mode,
    which torchrun expands to ``--rdzv_backend=c10d --rdzv_endpoint=localhost:0``.
    On macOS that ``localhost`` resolves to the IPv6 loopback ``::1`` and the
    c10d TCPStore then stalls on a broken reverse lookup. We replace it with an
    explicit c10d endpoint on ``127.0.0.1:0`` (a free IPv4 port). Idempotent:
    a sentinel attribute guards against double-patching across reruns.
    """
    import os

    from pyspark.ml.torch import distributor as _dist_mod

    # macOS rendezvous fix (root cause). The elastic agent's
    # `next_rendezvous` creates a *shared* c10d TCPStore server bound to the
    # node's own resolved address (`self._this_node.addr`), not to our endpoint.
    # On this Mac that address routes through the IPv6 loopback `::1`, whose
    # broken reverse-DNS PTR (`…ip6.arpa`) makes the TCPStore hang ~300s and
    # fail. `TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1` is torch's documented opt-out:
    # it skips creating that shared store entirely (the worker then talks gloo
    # over MASTER_ADDR=127.0.0.1, set in `_lstm_train_distributed`). USE_LIBUV=0
    # is kept as harmless extra insurance against the libuv loopback path.
    # These are set in the driver and propagate to the agent subprocess via the
    # Popen env TorchDistributor inherits.
    os.environ["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
    os.environ["USE_LIBUV"] = "0"

    if getattr(_dist_mod.TorchDistributor, "_olist_ipv4_patched", False):
        return

    def _ipv4_torchrun_args(local_mode: bool, num_processes: int):
        if local_mode:
            args = [
                "--nnodes=1",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=127.0.0.1:0",
                "--rdzv_id=olist_lstm",
            ]
            return args, num_processes
        return _orig_get_torchrun_args(local_mode, num_processes)

    _orig_get_torchrun_args = _dist_mod.TorchDistributor._get_torchrun_args
    _dist_mod.TorchDistributor._get_torchrun_args = staticmethod(_ipv4_torchrun_args)
    _dist_mod.TorchDistributor._olist_ipv4_patched = True


def _lstm_train_distributed(npz_path: str, params: dict):
    """Self-contained training function launched by ``TorchDistributor`` in a
    worker subprocess (Week-9 lab pattern: every import + class def lives inside
    the function so it pickles to the worker, and the trained ``state_dict`` is
    returned to the driver). Reads the encoded train arrays from ``npz_path``.
    """
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Dataset

    # macOS rendezvous fix: TorchDistributor's default master address can resolve
    # to a link-local IPv6 host (…ip6.arpa) that the gloo TCPStore cannot reach,
    # so init_process_group hangs ~300s before failing. Pin the rendezvous to the
    # IPv4 loopback and bind gloo to the loopback interface. MASTER_PORT (set by
    # TorchDistributor) is left untouched.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    os.environ["USE_LIBUV"] = "0"  # macOS IPv6-loopback TCPStore fix (see patch helper)
    dist.init_process_group(backend="gloo")
    torch.manual_seed(params["seed"])
    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    pad_idx = params["pad_idx"]

    class _ReviewsDS(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return (
                torch.from_numpy(self.X[i]),
                torch.tensor(self.y[i], dtype=torch.float32),
            )

    class _LSTMSentiment(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(
                params["vocab_size"], params["embed_dim"], padding_idx=pad_idx
            )
            self.lstm = nn.LSTM(
                params["embed_dim"], params["hidden_dim"], batch_first=True
            )
            self.fc = nn.Linear(params["hidden_dim"], 1)

        def forward(self, x):
            mask = (x != pad_idx).unsqueeze(-1).float()
            out, _ = self.lstm(self.embed(x))
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.fc(pooled).squeeze(-1)

    model = _LSTMSentiment()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        _ReviewsDS(X_train, y_train), batch_size=params["batch_size"], shuffle=True
    )
    losses = []
    for _ in range(params["epochs"]):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        losses.append(running / len(X_train))
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    dist.destroy_process_group()
    return {"state_dict": state_dict, "losses": losses}


def train_lstm(text_labelled: DataFrame) -> dict:
    """Train the deep-learning model (mandatory rubric line) the way Week 9
    taught it — *inside Spark*, not on the bare driver:

    * Training runs through ``TorchDistributor(local_mode=True)``, which launches
      the self-contained ``_lstm_train_distributed`` worker and returns the
      trained ``state_dict`` to the driver (distributed-training launcher +
      state-dict round-trip).
    * Test scoring runs through ``predict_batch_udf`` — the model is loaded once
      per worker and the held-out set is scored as a distributed Spark batch
      job, so the reported AUC comes from distributed inference, not a driver loop.

    Returns `{test_auc, train_loss_per_epoch, n_train, n_test, vocab_size}`.

    Escape hatches (catalogued in `docs/big_data_safety_log.md`):
    * `text_labelled.toPandas()`     # BIG-DATA-SAFETY-ESCAPE: LSTM_TO_PANDAS
    * PyTorch model + training        # BIG-DATA-SAFETY-ESCAPE: LSTM_PYTORCH
    """
    import os
    import tempfile

    import numpy as np
    import torch
    from pyspark.ml.functions import predict_batch_udf
    from pyspark.ml.torch.distributor import TorchDistributor
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType,
        FloatType,
        IntegerType,
        StructField,
        StructType,
    )

    np.random.seed(LSTM_SEED)
    spark = text_labelled.sparkSession

    # BIG-DATA-SAFETY-ESCAPE: LSTM_TO_PANDAS — see docs/big_data_safety_log.md
    text_pd = text_labelled.toPandas()
    pt_stopset = set(StopWordsRemover.loadDefaultStopWords("portuguese"))
    tokens_pd = text_pd["text"].map(lambda t: _tokenize_portuguese(t, pt_stopset))
    vocab = _build_vocab(tokens_pd)

    encoded = np.array([_encode(t, vocab) for t in tokens_pd], dtype=np.int64)
    labels = text_pd["label"].to_numpy(dtype=np.int64)

    perm = np.random.permutation(len(encoded))
    split = int(0.8 * len(encoded))
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = encoded[train_idx], labels[train_idx]
    X_test, y_test = encoded[test_idx], labels[test_idx]

    # LIGHT: single epoch for fast reruns; full LSTM_EPOCHS in the heavy run.
    epochs = 1 if LIGHT else LSTM_EPOCHS
    params = {
        "seed": LSTM_SEED,
        "vocab_size": LSTM_VOCAB_SIZE,
        "embed_dim": LSTM_EMBED_DIM,
        "hidden_dim": LSTM_HIDDEN_DIM,
        "pad_idx": LSTM_PAD_IDX,
        "epochs": epochs,
        "batch_size": LSTM_BATCH_SIZE,
    }

    workdir = tempfile.mkdtemp(prefix="olist_lstm_")
    npz_path = os.path.join(workdir, "train.npz")
    np.savez(npz_path, X_train=X_train, y_train=y_train)

    # macOS rendezvous fix. TorchDistributor's local_mode passes torchrun
    # `--standalone`, which hardcodes the c10d rendezvous endpoint to
    # `localhost:0`. On macOS `localhost` resolves to the IPv6 loopback `::1`,
    # whose reverse-DNS PTR (`…ip6.arpa`) cannot be re-resolved, so the TCPStore
    # rendezvous hangs ~300s and then fails. We swap `--standalone` for an
    # explicit c10d rendezvous on the IPv4 literal `127.0.0.1:0` (free port),
    # which sidesteps IPv6 and the reverse lookup entirely. Scoped + idempotent.
    _patch_torchdistributor_ipv4_rendezvous()

    # BIG-DATA-SAFETY-ESCAPE: LSTM_PYTORCH — distributed training launcher
    result = TorchDistributor(
        num_processes=1, local_mode=True, use_gpu=False
    ).run(_lstm_train_distributed, npz_path, params)
    state_dict, losses = result["state_dict"], result["losses"]

    state_path = os.path.join(workdir, "lstm_state.pt")
    torch.save(state_dict, state_path)

    # Distributed inference on the held-out set via predict_batch_udf.
    test_schema = StructType([
        StructField("seq", ArrayType(IntegerType()), False),
        StructField("label", IntegerType(), False),
    ])
    test_rows = [
        (X_test[i].tolist(), int(y_test[i])) for i in range(len(X_test))
    ]
    test_sdf = spark.createDataFrame(test_rows, schema=test_schema)

    # predict_batch_udf factory — defined as a LOCAL closure so cloudpickle
    # ships it to executors by value (the `olist` package is not importable on
    # Spark workers; a module-level reference would raise ModuleNotFoundError).
    # local_mode only: workers share the driver's filesystem, so `state_path` is
    # readable; on a real cluster you'd broadcast the weights or use a shared FS.
    def _make_predict():
        import numpy as _np
        import torch as _torch
        import torch.nn as _nn

        _pad = params["pad_idx"]

        class _LSTM(_nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = _nn.Embedding(
                    params["vocab_size"], params["embed_dim"], padding_idx=_pad
                )
                self.lstm = _nn.LSTM(
                    params["embed_dim"], params["hidden_dim"], batch_first=True
                )
                self.fc = _nn.Linear(params["hidden_dim"], 1)

            def forward(self, x):
                mask = (x != _pad).unsqueeze(-1).float()
                out, _ = self.lstm(self.embed(x))
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                return self.fc(pooled).squeeze(-1)

        _model = _LSTM()
        _model.load_state_dict(_torch.load(state_path))
        _model.eval()

        def _predict(batch):
            with _torch.no_grad():
                x = _torch.from_numpy(_np.asarray(batch, dtype=_np.int64))
                return _torch.sigmoid(_model(x)).numpy().astype("float32")

        return _predict

    # input_tensor_shapes: the `seq` column is a fixed-length (max_len,) array,
    # so predict_batch_udf needs its per-row shape to reshape the flat batch.
    score_udf = predict_batch_udf(
        _make_predict,
        return_type=FloatType(),
        batch_size=256,
        input_tensor_shapes=[[LSTM_MAX_LEN]],
    )
    scored = test_sdf.withColumn("score", score_udf(F.col("seq")))
    # ≤9k-row test set; collect (score,label) pairs for the AUC metric only.
    # BIG-DATA-SAFETY-ESCAPE: SMALL_SUMMARY_COLLECT — held-out scores for AUC
    from sklearn.metrics import roc_auc_score

    scored_pd = scored.select("score", "label").toPandas()
    auc = float(roc_auc_score(scored_pd["label"], scored_pd["score"]))
    return {
        "test_auc": auc,
        "train_loss_per_epoch": [float(x) for x in losses],
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
    version=2,
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


def seller_sentiment_slopes(reviews_with_seller: DataFrame, *, min_weeks: int = 6) -> DataFrame:
    """Per-seller sentiment *trend slope* via grouped-map ``applyInPandas``
    (split-apply-combine): each seller's weekly-average-score series is fit with
    an ordinary-least-squares line and the slope (stars per week) is returned.

    `applyInPandas` is the scalable way to run a per-group computation that has
    no native Spark equivalent — the regression runs on the executors, one group
    at a time, never collecting to the driver. Sellers with `< min_weeks` weeks
    of history are dropped (slope undefined). Complements the window-based
    `sentiment_trend_6wk`: the slope uses a seller's whole history, not just the
    last two 6-week windows.

    Returns (seller_id, slope_per_week, n_weeks, mean_score).
    """
    weekly = (
        reviews_with_seller.filter(
            F.col("review_score").isNotNull() & F.col("seller_id").isNotNull()
        )
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

    out_schema = StructType([
        StructField("seller_id", StringType(), True),
        StructField("slope_per_week", DoubleType(), True),
        StructField("n_weeks", IntegerType(), True),
        StructField("mean_score", DoubleType(), True),
    ])

    def _ols_slope(pdf: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
        import numpy as np
        import pandas as pd

        pdf = pdf.sort_values("year_week")
        n = len(pdf)
        seller = pdf["seller_id"].iloc[0]
        y = pdf["avg_score_week"].to_numpy(dtype="float64")
        if n < min_weeks:
            return pd.DataFrame(
                [(seller, None, n, float(y.mean()))],
                columns=["seller_id", "slope_per_week", "n_weeks", "mean_score"],
            )
        x = np.arange(n, dtype="float64")
        slope = float(np.polyfit(x, y, 1)[0])
        return pd.DataFrame(
            [(seller, slope, n, float(y.mean()))],
            columns=["seller_id", "slope_per_week", "n_weeks", "mean_score"],
        )

    return (
        weekly.groupBy("seller_id")
        .applyInPandas(_ols_slope, schema=out_schema)
        .filter(F.col("slope_per_week").isNotNull())
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
