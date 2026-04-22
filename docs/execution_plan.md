# Execution plan — 2026-04-20

Ground truth: parquets in `outputs/`. Plan below is the build order, not a recipe.

## 0. Env (done)
Python 3.9.6 venv at `.venv/`, PySpark 3.5.8, PyTorch 2.8, Java 11 at `/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home`, GraphFrames `0.8.3-spark3.5-s_2.12` resolves. First cell of every notebook must export `JAVA_HOME` before importing Spark — adding this to notebook boilerplate.

## 1. Shared pre-step — `outputs/geo_centroids.parquet`
One-off script (or NB1 prelude): aggregate `olist_geolocation_dataset` to one row per `zip_code_prefix` (mean lat/lon). Persist. Broadcasted from then on. Required by NB3 (state-level + distance features) and cheap to do once.

## 2. Sequencing
```
geo_centroids ─┐
               ├─► NB1 ─► nb1_weekly_order_volume.parquet ─┐
               │          nb1_seller_demand_scores.parquet │
               │                                            ├─► NB3 ─► convergence
               └─► NB2 ──────────► nb2_seller_sentiment_scores.parquet
                   (reads nb1_weekly_order_volume for lead-indicator section)
```
NB2 depends on NB1's `nb1_weekly_order_volume.parquet` only for its final lead-indicator section. So: start NB1 first; once `nb1_weekly_order_volume.parquet` lands, NB2 can launch in parallel as a subagent while NB1 finishes its forecasting + streaming-bonus sections. NB3 waits for both.

## 3. NB1 — `01_demand_forecasting.ipynb`
Sections I intend to write (in order):
1. Boot + JAVA_HOME + `get_spark()`.
2. RDD warm-up: `sc.textFile('data/olist_orders_dataset.csv')` → map/filter/reduceByKey daily-order count → DF with explicit schema. Rubric line #1.
3. Typed loads via `loaders.load_*`. Row-count logs. Filter non-delivered; log drops to `decisions_log.md`.
4. `geo_centroids` build + persist (if not already present).
5. Order-line join (orders ⋈ order_items ⋈ broadcast(sellers) ⋈ broadcast(products)) → `cache()`. One hot DF.
6. SparkSQL: register temp view `order_lines`; run ≥3 queries (top sellers by revenue; weekly volume per seller; late-delivery rate by state).
7. Weekly aggregation → write `nb1_weekly_order_volume.parquet` **early** so NB2 can start.
8. Feature Pipeline: lag-1, lag-4, rolling 4-week mean via Window, calendar features, `VectorAssembler`.
9. Models: `GBTRegressor` + `RandomForestRegressor` with `CrossValidator(folds=3)` + `RegressionEvaluator('rmse')`. Compare RMSE.
10. Per-seller forecast-uplift % (predicted next-4w vs trailing-4w mean) + avg delay days + delay_risk_flag → write `nb1_seller_demand_scores.parquet`.
11. **Bonus** streaming: `readStream` from a tiny file-source directory mimicking new orders; 10-min windowed count. Only if steps 1–10 green.
12. `unpersist()` before final writes.

## 4. NB2 — `02_sentiment_analysis.ipynb`
1. Boot, loads, filter reviews with non-null `review_comment_message` (log drops).
2. Label: `score ≥ 4` positive, `score ≤ 2` negative, drop 3s (or keep as neutral third class — decide by class balance, log choice).
3. SparkSQL join: temp view `reviews_with_seller` via orders ⋈ order_items ⋈ reviews. `cache()`.
4. NLP Pipeline: `Tokenizer → StopWordsRemover(stopwords=Portuguese) → HashingTF → IDF → LogisticRegression`. AUC via `BinaryClassificationEvaluator`.
5. PyTorch LSTM + justification markdown (per Hard Rule): why PyTorch, Spark alternative = `spark-nlp`, why acceptable (~40k reviews fits in RAM). Tokenize → integer-encode → `DataLoader` → 1-layer LSTM → sigmoid. Report test AUC, compare to LogReg baseline.
6. Weekly rolling sentiment per seller via Window (`rangeBetween` on week index). Write `nb2_seller_sentiment_scores.parquet` columns: `seller_id, avg_sentiment_score, sentiment_trend_6wk, pct_negative_reviews, sentiment_declining`.
7. Lead-indicator analysis: join `nb1_weekly_order_volume.parquet`; cross-correlate weekly sentiment slope vs. weekly volume slope at lags 0–8 weeks; report peak-correlation lag. Inline chart (small aggregated DF only).
8. `unpersist()` before writes.

## 5. NB3 — `03_supply_network_graph.ipynb`
1. Boot with `get_spark(with_graphframes=True)`.
2. Vertices: union of distinct `seller_id` and `customer_unique_id`; add a `type` column.
3. Edges: `order_items` ⋈ `orders` ⋈ `customers` → `(seller_id, customer_unique_id)` with edge weight = count.
4. `GraphFrame(v, e)`; degree analysis (in/out).
5. PageRank (`resetProbability=0.15, maxIter=10`). Write top-50 table.
6. Connected components → `is_isolated = (component size == 1)`.
7. Motif `(a)-[e1]->(c)<-[e2]-(b)` filter `a.id != b.id` to find seller pairs sharing customers.
8. BFS: for each top-10 PageRank seller, `bfs(from=that seller, to="type='seller' AND pagerank > threshold", maxPathLength=3)` → backup_seller_id (first hit other than self).
9. High-delay subgraph: edges where the underlying orders' avg delay > 5 days → re-run PageRank on that subgraph; store as `network_risk_score`.
10. Write `nb3_seller_network_scores.parquet`: `seller_id, pagerank_score, in_degree, is_isolated, backup_seller_id, network_risk_score`.

## 6. Convergence (final NB3 section)
Inner-join the 3 parquets on `seller_id`. Min-max normalise each of {demand_uplift_pct, sentiment_decline_flag_weighted, network_risk_score}; single-row aggregations flagged with a comment. `risk_score = 0.35·demand + 0.35·sentiment + 0.30·network`. Classify CRITICAL > 0.75, WARNING, SAFE < 0.40. Write `outputs/seller_risk_index.parquet`. Then `toPandas()` on top-50 only (flagged) for:
- Chart 1: demand × sentiment quadrant, bubble = pagerank.
- Chart 2: state bar chart of mean risk_score.

## 7. Dataset choices logged up-front
- Filter to delivered orders only (drop rows where `order_delivered_customer_date IS NULL`). Log row counts in notebook + `decisions_log.md`.
- Keep neutral (score=3) reviews as their own class **or** drop them — decide after checking class balance (logged).
- `customer_unique_id` is the person-level key everywhere.

## 8. Parallelisation
After NB1 step 7 (weekly volume parquet written), spawn NB2 as a subagent (`general-purpose` or `Explore` — likely general-purpose for write-heavy work). NB1 continues its forecasting + bonus sections. NB3 waits for both.

## 9. Risks / watch items
- Reviews are Portuguese — confirm `StopWordsRemover.loadDefaultStopWords('portuguese')` is available in PySpark 3.5 (it is).
- GraphFrames motif finder is JVM-heavy; may need `spark.driver.memory=4g` config bump for motifs.
- PyTorch LSTM on CPU: cap vocab at 20k + max seq length 128 to keep training <5 min.
- NB1 streaming bonus is *only* attempted after the three required parquets exist.

## 10. Deliverable check (per grading_checklist.md)
All six output parquets, all rubric lines hit, every cell re-run via `submission/build_zip.sh` before zipping. Presentation PDF is human-produced from these outputs.

---
**Awaiting approval before writing any notebook code.**
