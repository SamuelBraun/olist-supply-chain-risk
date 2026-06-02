# Grading checklist (mirrors the course-brief rubric)

Single-notebook architecture: the one graded code artefact is `notebooks/main.ipynb`; all transformation logic lives in `src/olist/**`. Programmatic enforcement is `src/olist/checks.py::run_all(spark)`, rendered inline in `main.ipynb` §7.4. Current run: **24/24 checks pass**, every code cell has visible output, executed end-to-end via `bash run.sh`.

Tick each box only when the corresponding cell is committed and the notebook has been re-executed end-to-end.

## Mandatory Spark surface area (all enforced by `checks.run_all`)
- [x] **RDDs** — `pipeline.demand.rdd_daily_order_count` (textFile → filter/map → reduceByKey → typed DF), with `inspect.getsource` shown. NB §3.4. `check_rdd_chain`.
- [x] **DataFrames** — used throughout.
- [x] **SparkSQL** (≥4 temp-view queries) — three in NB §3.4 (top sellers by revenue / weekly-volume / late-rate-by-state) + the reviews-with-seller join in §4.2.1. `check_sparksql` counts 4.
- [x] **Pipelines & Data Engineering** (2 `Pipeline`s) — demand feature Pipeline (Imputer + VectorAssembler, §3.5); NLP Pipeline (Tokenizer → StopWordsRemover[pt] → HashingTF → IDF → LogisticRegression, §4.4). `check_pipelines == 2`.
- [x] **MLlib** (≥3 `CrossValidator`) — GBTRegressor + RandomForestRegressor under CV (§3.6, RMSE), LogisticRegression under CV (§4.6.1, AUC), KMeans archetypes (§6.3). `check_cv_models`.
- [x] **Deep Learning** — PyTorch LSTM (§4.6.2) with masked mean-pool + mandatory justification cell. `check_lstm_justification`.
- [x] **Spark–DL integration** — LSTM trained via `TorchDistributor(local_mode=True)`, scored via `predict_batch_udf` (§4.6.2). `check_spark_dl_integration`.
- [x] **UDF family** — grouped-map `applyInPandas` per-seller OLS slope (§4.7.3). `check_applyinpandas`.
- [x] **GraphFrames** — `GraphFrame(v,e)`, PageRank, `connectedComponents(graphx)`, motif, BFS, delayed-subgraph PageRank (§5.4–§5.6) + co-customer projection centrality/labelPropagation/backup map (§5.5.4) + transitive 2-hop substitutes (§5.5.5) + dense co-category+region projection (§5.5.6). `check_graphframe_ops`, `check_cocustomer_graph`, `check_two_hop_substitutes`, `check_catregion_graph`.
- [x] **Window functions with rolling** — demand lag/rolling features (§3.5) + 6-week sentiment rollup (§4.5). `check_window_used`.
- [x] **EDA primitives** — `approxQuantile` + `approx_count_distinct` (§3.2). `check_approx_eda`; winsorisation `check_outlier_treatment`.
- [x] **Unsupervised topic model** — `CountVectorizer → LDA(k=6)` over negative reviews → failure-mode mix (§4.7.5). `check_negative_topic_model`.
- [x] **Demand feature richness** — retail-calendar flags + per-seller covariates (§3.5). `check_demand_feature_richness`.
- [x] **Transformations vs actions** — lazy-eval callout (§3.4). `check_lazy_eval_documented`.
- [x] **Code quality** — every code block has a markdown preamble. `check_markdown_ratio`.
- [x] **Streaming (bonus)** — **present**: NB §8 file-source `readStream` → windowed weekly count → memory sink, `trigger(availableNow=True)`. `check_streaming`.

## Big-data safety (enforced by `check_safety_log_consistency` + scalability checks)
- [x] Every `collect()` / `toPandas()` / non-Spark-library call annotated `# BIG-DATA-SAFETY-ESCAPE: <ID>` referencing a constant in `src/olist/safety.py` (9 IDs).
- [x] Every ID in `safety.ALL_ESCAPES` has a row in `docs/big_data_safety_log.md`; rendered inline in §7.3.
- [x] No unbounded ordered driver pull. `check_no_unbounded_orderby` + `check_no_unguarded_collect` scan pipeline source **and** notebook cells.
- [x] No single-partition global sort: percentile rank/banding use `QuantileDiscretizer` / `approxQuantile`. `check_no_unpartitioned_window`.
- [x] Small lookups (`sellers`, `products`, `category_translation`, `geo_centroids`) joined with `broadcast()`.
- [x] Hot DataFrames cached once and unpersisted before parquet write.
- [x] CSVs loaded via `loaders.*` (explicit schemas, no `inferSchema`).
- [x] Every non-Spark library use has its justification markdown cell.

## Output parquets (enforced by `check_parquet_artefacts`)
- [x] `outputs/geo_centroids.parquet` — 19,015 zip-prefix centroids (broadcast lookup).
- [x] `outputs/nb1_weekly_order_volume.parquet` — `(seller_id, year_week, weekly_order_count)`, 35,385 rows.
- [x] `outputs/nb1_seller_demand_scores.parquet` — 1,630 sellers (those with ≥~5 weeks of history). RF RMSE **3.167** / GBT 3.239 → RF selected, beats all three naive baselines (3.60 / 3.45 / 4.89). Top features: `decay_wtd_8w` 0.275, `lag_1` 0.274, `rolling_4w_mean` 0.210, `lag_4` 0.125, `is_black_friday` 0.040.
- [x] `outputs/nb1_regional_demand_forecast.parquet` — honest negative result: model RMSE 109.3 > best naive 95.0 (state-aggregate volume ≈ random walk).
- [x] `outputs/nb2_seller_sentiment_scores.parquet` — LogReg ROC-AUC **0.9564** (PR-AUC 0.9753, negative-class recall 0.733), LSTM ROC-AUC **0.9619**. Lead-indicator |ρ| ≈ 0.015 (honest weak signal).
- [x] `outputs/nb2_seller_failure_modes.parquet` — LDA failure-mode mix ~83% delivery / 13% quality / 5% wrong (analysis layer, not in the risk score).
- [x] `outputs/nb3_seller_network_scores.parquet` — 2,970 sellers. Co-customer projection 7.6%; dense co-category+region coverage **72.9%**; `corr(supply_concentration_risk, in_degree) = −0.18`.
- [x] `outputs/seller_risk_index.parquet` — 1,630 sellers, bands **1,547 SAFE / 65 WARNING / 18 CRITICAL**, `escalate_no_backup = 13`.

## Architecture & reproducibility
- [x] `src/olist/cache.py` @step decorator + `outputs/.cache_manifest.json` tracked in git.
- [x] Pipeline logic lives in `src/olist/pipeline/{demand,sentiment,network,convergence,streaming}.py`; the notebook only calls in.
- [x] `notebooks/main.ipynb` — the single comprehensive deliverable (239 cells: 95 code / 144 markdown, all executed with output).
- [x] `src/olist/checks.py::run_all(spark)` — 24 assertions; all PASS this run.
- [x] Non-tutorial seeds (`GBT=7341`, `RF=2918`, `NLP=5067`, `LSTM=1394`, `KMeans=8825`, `LDA=6273`).
- [x] `docs/architecture.md` — module / cache / safety map.

## Execution & submission
- [x] Notebook fully executed; every code cell has visible, non-error output. Verified by `scripts/assert_notebook_outputs.py`, called from `submission/build_zip.sh`.
- [x] `bash run.sh` = warm-rerun entry point (executes `main.ipynb` end-to-end).
- [ ] `presentation/presentation.pdf` exists (≤10 slides, NOVA IMS template). *(Human-produced.)*
- [ ] `submission/build_zip.sh <GROUP>` produces `submission/olist_bigdata_group<N>.zip` (notebook + PDF). *(Run once the PDF is exported.)*
