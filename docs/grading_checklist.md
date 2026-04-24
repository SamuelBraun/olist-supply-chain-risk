# Grading checklist (mirrors the PDF rubric)

Tick each box only when the corresponding cell is committed and the notebook has been re-executed end-to-end. Programmatic enforcement lives in `src/olist/checks.py::run_all(spark)` which runs inline in `00_main.ipynb` §11.

## Mandatory Spark surface area (all enforced by `checks.run_all`)
- [x] **RDDs** — `pipeline.demand.rdd_daily_order_count` (textFile → filter/map → reduceByKey → typed DF). Rendered in NB1 §2 + `00_main` §4.1.
- [x] **DataFrames** — used across all four notebooks.
- [x] **SparkSQL** (≥4 temp-view queries) — three in NB1 (top sellers by revenue / weekly-volume preview / late-rate-by-state) + NB2 §2 reviews_with_seller join. Check counts 4.
- [x] **Pipelines & Data Engineering** (≥2 `Pipeline(stages=...)`) — NB1 feature Pipeline (Imputer + VectorAssembler); NB2 NLP Pipeline (Tokenizer → StopWordsRemover[pt] → HashingTF → IDF → LogisticRegression).
- [x] **MLlib** (≥3 `CrossValidator`) — NB1 GBTRegressor + RandomForestRegressor under CV (RMSE); NB2 LogisticRegression under CV (AUC).
- [x] **Deep Learning** — NB2 §5 PyTorch LSTM with masked mean-pool; justification markdown cell explicitly states "not Spark/MLlib" and "PyTorch on CPU is acceptable at this size".
- [x] **GraphFrames** (6/6 primitives) — `GraphFrame(v, e)` ctor, PageRank, connectedComponents (GraphX backend), motif `.find("(a)-[]->(c); (b)-[]->(c)")`, BFS, delayed-subgraph PageRank.
- [x] **Window functions with rolling** — `Window.partitionBy(seller_id).orderBy(year_week).rowsBetween(...)` for NB1 lag features and NB2 6-week sentiment rollup. 5 `Window.partitionBy` calls across pipeline modules.
- [x] **EDA primitives** — NB1 §8: `approxQuantile` on price + delay; `approx_count_distinct` on seller + product.
- [x] **Code quality** — `check_markdown_ratio` enforces that every code block has a preceding markdown preamble in all four notebooks.
- [ ] **Streaming (bonus)** — not demonstrated. The three required per-seller parquets are green; `00_main` §4.9 notes the streaming seam (`build_weekly_order_volume` → `spark.readStream`).

## Big-data safety (enforced by `check_safety_log_consistency`)
- [x] Every `collect()` / `toPandas()` / non-Spark-library call is annotated with `# BIG-DATA-SAFETY-ESCAPE: <ID>` referencing a constant in `src/olist/safety.py`.
- [x] Every ID in `safety.ALL_ESCAPES` has a row in `docs/big_data_safety_log.md` (call site, what it does, why it's an escape, production-scale alternative, why acceptable here).
- [x] Every `orderBy()` is followed by `.limit()`.
- [x] Small lookups (`sellers`, `products`, `category_translation`, `geo_centroids`) joined with `broadcast()`.
- [x] Hot DataFrames cached once and unpersisted before parquet write (demand `order_lines`, sentiment `reviews_with_seller`, network `order_lines` + `vertices` + `edges`).
- [x] CSVs loaded via `loaders.*` (explicit schemas, no `inferSchema`).
- [x] Every non-Spark library use has its justification markdown cell.

## Output parquets (enforced by `check_parquet_artefacts`)
- [x] `outputs/geo_centroids.parquet` — 19,015 × 4.
- [x] `outputs/nb1_seller_demand_scores.parquet` — 2,970 × 5.
- [x] `outputs/nb1_weekly_order_volume.parquet` — 35,385 × 3.
- [x] `outputs/nb2_seller_sentiment_scores.parquet` — 3,090 × 5 (LogReg AUC 0.9580, LSTM AUC 0.9630).
- [x] `outputs/nb3_seller_network_scores.parquet` — 2,970 × 6.
- [x] `outputs/seller_risk_index.parquet` — 2,967 × 11 (2,899 SAFE / 68 WARNING / 0 CRITICAL).

## Refactor milestones (2026-04-24)
- [x] `src/olist/cache.py` @step decorator + `outputs/.cache_manifest.json` tracked in git.
- [x] Pipeline logic lives in `src/olist/pipeline/{demand,sentiment,network,convergence}.py`; notebooks are thin wrappers.
- [x] `notebooks/00_main.ipynb` — primary narrative deliverable (52 cells, 26 md / 26 code, all executed).
- [x] `src/olist/checks.py::run_all(spark)` — 11 CheckResult assertions; all PASS.
- [x] `docs/architecture.md` — one-page module / cache / safety map.

## Execution & submission
- [x] All 4 notebooks fully executed; every code cell has visible, non-error output. Verified by `scripts/assert_notebook_outputs.py`, called from `submission/build_zip.sh`.
- [x] `bash run.sh` = warm-rerun entry point (executes 00_main end-to-end).
- [ ] `presentation/presentation.pdf` exists (≤10 slides, NOVA IMS template). *(Human-produced.)*
- [ ] `submission/build_zip.sh <GROUP>` produces `submission/olist_bigdata_group<N>.zip` with the 4 notebooks + the PDF. *(Run once the PDF is exported.)*
