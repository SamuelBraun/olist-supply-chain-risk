# Lab coverage matrix — class material vs. our project

> **Update 2026-05-30 (post-gap-closure):** Four of the gaps below were closed in the same-day
> "class-coverage" pass — `applyInPandas` (§4.7.3), Spark–DL integration via `TorchDistributor` +
> `predict_batch_udf` (§4.6.2), silhouette (§6.3.1), and the Streaming bonus (§8, week-12 syllabus topic).
> `StringIndexer`/`OneHotEncoder` was the one deliberately skipped (avoids a 45-min demand-CV refit;
> defensible — our demand features are numeric by design). New `checks.py`: `check_spark_dl_integration`,
> `check_applyinpandas`, `check_streaming`. The table below reflects the *original* audit; the closures
> are noted inline.

Created 2026-05-30. Cross-references every technique taught in the weekly labs
(github.com/dhruv-pandit/bigDataAnalyticsIMSSpring, `labs/` weeks 1–10) against our
project (`src/olist/`, `notebooks/main.ipynb`). Verified by grep + this session's knowledge.

**Repo scope note:** the repo contains weeks 1–10 only. The syllabus (`plan.md`) lists
**week 11 (Advanced Graph + NLP)** and **week 12 (Spark Streaming)** — those lab folders are
**not in the repo**. Week 10's GraphFrames notebook is "Part 1" and explicitly defers PageRank /
BFS / labelPropagation / SCC / triangleCount to a "notebook_2" that isn't present.

## Verdict

We comprehensively cover the **core** of every lab week. The RDD chain, DataFrame verbs, Spark SQL +
temp views, window functions, broadcast joins, caching, explicit schemas, the feature + NLP Pipelines,
CrossValidator + ParamGridBuilder, GBT/RF/LR + AUC, TF-IDF, KMeans, the full GraphFrames suite, a
PyTorch deep-learning model, dataclasses + functional style — all present, and several go **beyond** the
labs (approxQuantile/approxCountDistinct, winsorisation, the co-customer graph projection, CrossValidator
which week-7 didn't even use, labelPropagation which week-10 deferred).

There are **a handful of techniques the labs taught that our project does not use.** Most are defensible
to skip; four are worth a decision for the "everything done in class" bar.

## Coverage table

| Week | Topic | Core covered? | Notable taught-but-absent |
| --- | --- | --- | --- |
| 1 | Python Functional Programming | ✅ | — (lambdas/map/filter/reduce in RDD chain; `@dataclass` ×2; comprehensions) |
| 2 | MapReduce from scratch | ✅ (as Spark) | hand-built MapReduce framework — conceptual prep, superseded by the RDD `reduceByKey` chain |
| 3 | Spark RDD intro | ✅ | advanced RDD methods (`aggregateByKey`, `combineByKey`, `mapPartitions`, `keyBy`) — breadth only; core RDD chain + lazy-eval covered |
| 4 | Spark DataFrames | ✅ | — (select/filter/withColumn/groupBy/agg/join/when/na all used) |
| 5 | Spark SQL | ✅ | `CTE (WITH)`, `CACHE TABLE` (we cache via parquet step-cache instead), `write.partitionBy`. `.explain()` ✅ now used |
| 6 | Advanced Spark (complex types + UDFs) | ⚠️ partial | **`pandas_udf` + `applyInPandas` (0 uses)**; nested struct/array + `explode`/`from_json` (N/A — flat data) |
| 7 | ML in Spark | ✅ | **`StringIndexer`/`OneHotEncoder` (0 uses — no categorical encoding)**; `Normalizer` (we use Imputer/manual scaling) |
| 8 | Pipelines | ✅ | **`ClusteringEvaluator`/silhouette (0 — elbow only)**; Pipeline `save/load` (0); `Word2Vec`/`PCA` (we use TF-IDF) |
| 9 | Deep Learning | ⚠️ partial | **`TorchDistributor` + `predict_batch_udf` (0) — the lab's actual emphasis**; we train PyTorch driver-side (justified safety escape) |
| 10 | GraphFrames | ✅✅ | `filterEdges/filterVertices` native API (we filter+rebuild), `triangleCount`/`aggregateMessages` (we have PageRank/CC/motif/BFS/labelPropagation) |
| 11* | Advanced Graph + NLP | n/a | lab not in repo; we already do labelPropagation + TF-IDF NLP |
| 12* | Spark Streaming | ❌ | not in repo, but on syllabus + brief bonus — see backlog #3 |

## Genuine gaps, prioritised (taught in class, absent in our project)

### Worth a decision (for the "everything done in class" bar)
1. **UDF family — `pandas_udf` (scalar) + `applyInPandas` (grouped-map)** [Week 6 marquee; Week 7 reuse].
   We use exactly one plain `F.udf` (the archetype-label UDF in `convergence.py`). The labs spent a whole
   week on the UDF trio, framing `applyInPandas` as the *scalable* split-apply-combine pattern (it runs on
   executors — consistent with our big-data-safety stance, unlike a row-wise Python UDF). Lowest-friction
   tick: one `pandas_udf` or `applyInPandas` in a defensible spot.
2. **Spark–DL integration: `TorchDistributor` / `predict_batch_udf`** [Week 9 — the lab's whole point].
   Week 9 was *not* "build a neural net" — it was "run PyTorch inside Spark at scale." We train the LSTM
   driver-side (`toPandas` → PyTorch, flagged `LSTM_TO_PANDAS`). Wrapping training in
   `TorchDistributor(local_mode=True)` and/or scoring via `predict_batch_udf` would demonstrate the taught
   pattern directly. Medium effort; high relevance since it's the DL lab's emphasis.
3. **`StringIndexer` / `OneHotEncoder`** [Weeks 7 & 8 — taught twice].
   Our ML features are all numeric (lag/rolling/calendar), so we never encode a categorical. Could one-hot
   `seller_state` (or product category) into the demand feature pipeline to show the technique.
4. **`ClusteringEvaluator` / silhouette** [Week 8]. We justify k=4 by WSSSE elbow only; the lab taught
   silhouette as the cluster-validation metric. Easy add (already backlog #6).

### Defensible to skip (note at oral if asked)
- **Hand-built MapReduce** (Wk2) — superseded by the Spark RDD chain; conceptual prep.
- **Complex types / `explode` / `from_json`** (Wk6) — our data is flat; nothing to nest/parse.
- **`Word2Vec` / `PCA`** (Wk8) — TF-IDF + LR is our NLP path; PCA not needed at 6 features.
- **Pipeline `save/load`** (Wk8) — we persist *results* as parquet via the step-cache, not fitted models.
- **`write.partitionBy` parquet / `CACHE TABLE`** (Wk5) — we repartition by key + cache via the parquet
  step-cache; equivalent intent.
- **`triangleCount` / `aggregateMessages` / `shortestPaths`** (Wk10 pt2) — we already run PageRank, CC,
  motif, BFS, labelPropagation; graph coverage is well past the bar.
- **Advanced RDD methods** (`aggregateByKey`/`combineByKey`/`mapPartitions`) (Wk3) — RDD bullet satisfied
  by the parse-and-count chain.

## Where we exceed the labs
approxQuantile / approxCountDistinct (Wk5 didn't cover), winsorisation, CrossValidator + ParamGridBuilder
(Wk7 tuned manually — CV came in Wk8), broadcast joins, the seller↔seller co-customer projection +
substitutability deficit + labelPropagation communities (Wk10 deferred these), p1/p99-clamped normalisation.
