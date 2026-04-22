# Grading checklist (mirrors the PDF rubric)

Tick each box only when the corresponding cell is committed and the notebook has been re-executed end-to-end. The final pass is run by `submission/build_zip.sh`.

## Mandatory Spark surface area
- [x] **RDDs** — NB1 has a `textFile` → `map`/`filter`/`reduceByKey` → DataFrame conversion via explicit schema. *(NB1 §2 — daily order-count RDD warm-up.)*
- [x] **DataFrames** — used across all three notebooks.
- [x] **SparkSQL** — NB1 ≥3 queries via `spark.sql()` on temp views; NB2 has at least one join via temp view. *(NB1 §7 top-sellers-by-revenue / weekly-volume preview / late-rate-by-state; NB2 §3 reviews_with_seller join.)*
- [x] **Pipelines & Data Engineering** — NB1 feature-engineering Pipeline; NB2 NLP Pipeline. *(NB1 §10 `Pipeline(Imputer, VectorAssembler)`; NB2 §5 `Pipeline(Tokenizer, StopWordsRemover[pt], HashingTF, IDF, LogisticRegression)`.)*
- [x] **MLlib** — NB1 GBTRegressor + RandomForestRegressor (CrossValidator, RegressionEvaluator); NB2 LogisticRegression. *(NB1 §11 — RF chosen, test RMSE 5.06 vs GBT 5.09.)*
- [x] **Deep Learning** — NB2 LSTM in PyTorch with the mandatory justification markdown cell. *(NB2 §6; masked mean-pool LSTM, test AUC 0.963 vs LogReg baseline 0.958.)*
- [x] **GraphFrames** — NB3 PageRank + connected components + motif finding + BFS. *(NB3 §7 PageRank, §8 connectedComponents(algorithm="graphx"), §9 shared-customer motif, §10 BFS for backup seller, §11 re-run PageRank on delayed subgraph.)*
- [x] **Code quality** — markdown above every code cell; descriptive names; no single-letter vars outside lambdas.
- [ ] **Streaming (bonus)** — NB1 final section if time permits. *(Skipped — three required parquets are all written; bonus not added.)*

## Big-data hygiene
- [x] No `collect()` / `toPandas()` on a non-aggregated DataFrame anywhere. *(Violations audit: NB2 §6 `text_labelled.toPandas()` on 43k rows — covered by the non-Spark-library justification cell per CLAUDE.md §2. All other toPandas/collect calls are ≤50 rows and flagged inline.)*
- [x] Every `orderBy()` is followed by `.limit()`.
- [x] `approxQuantile` / `approxCountDistinct` used for EDA stats. *(NB1 §8: price/delay quantiles + approx_count_distinct(seller_id, product_id).)*
- [x] Small lookups (`sellers`, `products`, `category_translation`, `geo_centroids`) joined with `broadcast()`. *(NB1 §6 + NB3 §3 use `broadcast(sellers)` / `broadcast(products)` / `broadcast(customers.select(...))`.)*
- [x] Hot DataFrames cached once and unpersisted before parquet write. *(NB1 `order_lines`, NB2 `reviews_with_seller`, NB3 `order_lines` + `vertices` + `edges`.)*
- [x] CSVs loaded via `loaders.*` (explicit schemas, no `inferSchema`).
- [x] Every non-Spark library use has its justification markdown cell. *(NB2 §6 PyTorch LSTM; NB3 §14 pandas/matplotlib top-50 charts with flags.)*

## Output parquets exist with the documented schema
- [x] `outputs/geo_centroids.parquet` — 19,015 rows × 4 cols.
- [x] `outputs/nb1_seller_demand_scores.parquet` — 2,970 × 5.
- [x] `outputs/nb1_weekly_order_volume.parquet` — 35,385 × 3.
- [x] `outputs/nb2_seller_sentiment_scores.parquet` — 3,090 × 5.
- [x] `outputs/nb3_seller_network_scores.parquet` — 2,970 × 6.
- [x] `outputs/seller_risk_index.parquet` — 2,967 × 11. Classes: 2,899 SAFE / 68 WARNING / 0 CRITICAL.

## Execution & submission
- [x] All 3 notebooks fully executed; every cell has visible output. *(Verified via `nbconvert --execute --inplace` on 2026-04-22.)*
- [ ] `presentation/presentation.pdf` exists (≤10 slides, NOVA IMS template). *(Human-produced from the notebook outputs and `docs/decisions_log.md`.)*
- [ ] `submission/build_zip.sh` produces a clean zip containing only the 3 notebooks + the PDF. *(Run once the PDF is exported.)*
