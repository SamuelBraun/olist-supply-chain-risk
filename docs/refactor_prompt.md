# Prompt for Claude Code — Olist Project Refactor

Paste everything below the line into Claude Code from the project root. Claude Code has access to the repo; it must read `CLAUDE.md`, `docs/decisions_log.md`, `docs/grading_checklist.md`, the assignment brief under `docs/references/`, and the three existing executed notebooks **before writing any code**.

---

## Role and framing

You are a senior data engineer + data scientist pairing with me on the **Olist Supply Chain Risk Intelligence** project for the NOVA IMS *Big Data Analysis* course. The course brief positions us as the consulting team at "BigDataCompany" delivering a management-ready PySpark prototype to an imaginary client (Olist). Two things matter most to the grader, in this order:

1. **Correct and idiomatic use of PySpark** — explicit demonstrations of RDDs, DataFrames & SparkSQL, Pipelines & Data Engineering, Spark MLlib, Deep Learning, GraphFrames, and Code Quality. Streaming is bonus. The brief says our grade depends *primarily* on this.
2. **Big-data reasoning** — our dataset is small, but every choice must be defensible at scale. The brief explicitly says: *"you should indicate anywhere explicitly that you have not used big data safe functionality, and why."* We must do this systematically, not ad-hoc.

The analytical work is already done. Three executed notebooks produce six parquet outputs in `outputs/`, and the Seller Risk Index is computed. **This task is a structural refactor, not a rewrite of the analysis.** Numerical results, model choices, and decisions logged in `docs/decisions_log.md` must remain intact unless a rerun demonstrably changes them.

## Planning gate

Before touching any code, read in this order: `CLAUDE.md`, `docs/execution_plan.md`, `docs/decisions_log.md`, `docs/grading_checklist.md`, the assignment brief in `docs/references/`, and all three existing notebooks end-to-end. Then write `docs/refactor_plan.md` stating (a) what you understand the current state to be, (b) what you intend to change, (c) the order of commits, (d) any risks to executed-output preservation. **Stop after writing that plan and wait for my approval.** Only then proceed.

## Goals (non-negotiable)

1. **One main narrative notebook** — `notebooks/00_main.ipynb` — is the entry point for the whole project. Top-to-bottom it tells the end-to-end consulting story: client problem → why this is a big-data problem (the 4 V's) → data → three analyses (demand, sentiment, network) → converged Seller Risk Index → recommendations for Olist. Rich markdown, charts, tables. A non-technical manager reads it and gets the value story; a senior data scientist reads the same notebook and finds the methodological choices rigorous.

2. **Logic lives in `src/olist/`**, not in notebook cells. Notebook cells are short: import, call a function, display output, write commentary. No multi-dozen-line transformation chains inside notebooks. The brief confirms `.py` files are allowed for classes/objects, but makes clear that **only the `.ipynb` with visible outputs is graded**. So: logic in `src/`, but every Spark primitive the grader needs to see (the actual RDD chain, the actual SparkSQL query text, the Pipeline stages, the GraphFrame motif call, the CrossValidator setup, model metrics, Window function definitions) must **render visibly in the notebook** — via `show()`, printed DataFrame heads, printed Pipeline stages, printed query strings, etc. Functions in `src/` should expose these artefacts (return them or `.show()` them) rather than hide them.

3. **The three thematic notebooks (`01_…`, `02_…`, `03_…`) stay** — the rubric requires them as executed deliverables — but they become **thin section-runners** that import from `src/olist/` and call the same functions the main notebook calls. No duplicated logic between the main notebook and the thematic notebooks. If a function is called from both, it lives in `src/`.

4. **Idempotent step cache.** Every computational step writes its result to `outputs/` and will be **skipped on rerun** if its declared inputs and its source-code fingerprint are unchanged. I want to `git pull`, run the main notebook, and have it finish in seconds when nothing has changed; and have it selectively recompute only the steps whose inputs or code have changed. **Critical caveat:** a cached step still has to leave its *visible outputs* (tables, charts, printed query text) in the notebook — the rubric requires visible outputs. The cache skips the *computation*, not the *display*. The pattern is: functions return the parquet-backed DataFrame (reading it if the cache is warm, computing + writing it otherwise), and the notebook cell below calls `.show()` / `display()` / plots on the returned object. So cached reruns stay fast *and* the notebook still shows outputs after `nbconvert --execute`.

5. **All outputs committable to git.** Parquet files under a reasonable size threshold stay in `outputs/`. Larger or ephemeral artefacts (Spark checkpoints, `_gf_checkpoints/`, model temp dirs, raw CSVs, the `.venv`) go under `outputs/_cache/` or stay outside the repo and are gitignored. Every committed output is reproducible from code.

6. **Requirements compliance is continuous, not an afterthought.** The main notebook ends with a "Rubric compliance" section that programmatically asserts each hard rule from `CLAUDE.md §2` and each item from the assignment brief's required-demonstrations list and `docs/grading_checklist.md` — e.g. "RDD chain executed in NB1 section X", "≥3 SparkSQL queries via temp views registered", "`GBTRegressor` and `RandomForestRegressor` both fit under `CrossValidator(folds=3)`", "GraphFrame with motif pattern `(a)-[]->(c)<-[]-(b)` executed", "LSTM justification cell present", "Portuguese stopword list used". These assertions fail loudly if broken.

7. **Big-data-safety log.** Because the brief demands we flag every non-big-data-safe call, maintain `docs/big_data_safety_log.md` listing every intentional escape hatch: `toPandas()` on the top-50 viz frame, any `collect()` on single-row aggregates, the pandas/matplotlib use for charts, the PyTorch LSTM, sklearn (if used), etc. Each entry: the call site (file + function), why it's there, what the Spark-native alternative at production scale would be, and why the escape is acceptable at this dataset size. The main notebook surfaces a short table summarising this log.

## Target architecture

```
src/olist/
  __init__.py
  spark_session.py          # already exists — keep
  schemas.py                # already exists — keep
  loaders.py                # already exists — extend if needed
  transforms.py             # already exists — split as below, then delete if empty
  cache.py                  # NEW — step decorator + manifest
  viz.py                    # NEW — matplotlib/seaborn helpers; small-DF only
  safety.py                 # NEW — declares and logs big-data-safety escapes
  pipeline/
    __init__.py
    demand.py               # NB1 logic: RDD warm-up, SQL queries, feature pipeline, GBT+RF CV, demand scores
    sentiment.py            # NB2 logic: SQL join, TF-IDF+LR, LSTM, weekly sentiment, lead-indicator
    network.py              # NB3 logic: GraphFrame build, PageRank, CC, motifs, BFS backups, delayed subgraph
    convergence.py          # normalise + weight + risk bands + top-50 viz frames
  checks.py                 # NEW — one compliance assertion per rubric item

notebooks/
  00_main.ipynb                  # NEW — the narrative orchestrator (the primary artefact)
  01_demand_forecasting.ipynb    # refactored: thin wrapper around src/olist/pipeline/demand.py
  02_sentiment_analysis.ipynb    # refactored: thin wrapper around src/olist/pipeline/sentiment.py
  03_supply_network_graph.ipynb  # refactored: thin wrapper around src/olist/pipeline/network.py

outputs/                    # committed parquet outputs + .cache_manifest.json
  _cache/                   # gitignored: spark checkpoints, intermediate temps

docs/
  refactor_plan.md          # NEW — written before coding, approved before coding
  architecture.md           # NEW — one-page diagram + module responsibilities
  big_data_safety_log.md    # NEW — per-escape-hatch rationale
```

Every `pipeline/*.py` module exposes a small, pure-ish public surface, e.g.:

```python
def build_order_lines(spark, *, force: bool = False) -> DataFrame: ...
def fit_demand_models(order_lines: DataFrame, *, force: bool = False) -> dict: ...
def score_sellers_demand(spark, *, force: bool = False) -> DataFrame: ...
```

Each public function is wrapped by the step cache and is called identically from the main notebook and from the thematic notebook. Functions return Spark objects wherever sensible so the notebook can `.show()` them visibly.

## Step cache (contract)

Implement a minimal, explicit cache in `src/olist/cache.py`. No DVC, no MLflow — stay in-repo.

- `@step(name, inputs, outputs, code_deps, version=1)` decorator (or equivalent `run_step(...)` helper).
- `inputs`: list of upstream parquet paths or raw CSVs under `data/`.
- `outputs`: list of parquet paths under `outputs/`.
- `code_deps`: list of `src/olist/**` paths whose source is hashed.
- On call: compute a fingerprint = sha256 of (sorted input file sizes + mtimes, sorted code_deps source bytes, step version tag). Read `outputs/.cache_manifest.json`. If the stored fingerprint for `name` matches AND all `outputs` exist AND the manifest schema version matches, **skip** and return handles to the cached parquet. Otherwise run, write outputs, update manifest atomically (write-tempfile-then-rename).
- `force=True` bypasses skip for a single step. Env var `OLIST_FORCE_ALL=1` forces everything.
- The manifest `outputs/.cache_manifest.json` is committed to git.
- Every skip and every run emits one line: `[cache] skip demand.score_sellers (fingerprint abc123…)` or `[cache] run demand.score_sellers (4.2s, wrote 2970 rows → nb1_seller_demand_scores.parquet)`.

Do **not** cache Spark DataFrames — cache only the **materialised parquet outputs on disk**. Downstream steps always re-read from parquet; this keeps behaviour deterministic and avoids stale lineage.

## Main notebook (`notebooks/00_main.ipynb`) — contract

The main notebook is the dissertation-quality telling of the consulting engagement. Structure with H2 sections and rich markdown between cells. Required sections, roughly in order:

1. **Executive summary** — two paragraphs framed for Olist management, a bulleted topline, and the final risk-band counts pulled live from `seller_risk_index.parquet`.
2. **Client context and why this is a big-data problem** — 4 V's framing (volume, velocity, variety, veracity) applied concretely to Olist's marketplace data: shuffle-heavy joins across `orders × order_items × reviews × customers × sellers`, graph algorithms over what would scale to millions of edges, NLP over a Portuguese corpus, geospatial joins. Name the specific bottlenecks a single-node pandas approach would hit at production volume. Be precise, avoid hype.
3. **Data overview** — one small aggregated table per source, built via Spark (not pandas). Explicit schemas referenced. Null-handling decisions surfaced with dropped-row counts (reuse `decisions_log.md`).
4. **Distributed computing toolkit on display** — a deliberate walkthrough of the Spark primitives the brief enumerates, each illustrated by a real call into the pipeline with visible output:
   - **RDDs** — the NB1 `textFile → map → filter → reduceByKey` warm-up, result shown.
   - **DataFrames & SparkSQL** — explicit `StructType` loads, ≥3 SparkSQL queries via temp views with the query text shown.
   - **Pipelines & Data Engineering** — the feature-engineering `Pipeline` stages printed.
   - **Machine Learning (MLlib)** — GBT vs RF under `CrossValidator(folds=3)` with `RegressionEvaluator(rmse)`; TF-IDF + LogReg with AUC.
   - **Deep Learning** — PyTorch LSTM with the mandatory justification markdown cell.
   - **GraphFrames** — vertices/edges construction, PageRank, connected components, the motif pattern, BFS.
   - **Window functions** — weekly rolling sentiment.
   - **EDA primitives** — `approxQuantile`, `approxCountDistinct`, broadcast joins.
   - **Streaming (bonus)** — if and only if the core deliverable is green, a brief structured-streaming demo.
   Every primitive gets a markdown paragraph explaining **why it is the right choice at scale** and **what would go wrong without it**. This section is the "senior" signal and the rubric backbone.
5. **Analysis 1 — Demand** — calls `pipeline.demand.*`, shows the RDD section output, the three SparkSQL query results, RMSE table for GBT vs RF, feature-importance chart, per-state demand-uplift bar chart, top/bottom sellers table.
6. **Analysis 2 — Sentiment** — calls `pipeline.sentiment.*`, shows TF-IDF+LR AUC and confusion matrix, LSTM AUC with the mandatory justification cell, weekly rolling sentiment plot for representative sellers, lead-indicator cross-correlation plot with the honest "peak lag 7 weeks, |corr| ≈ 0.015" narrative.
7. **Analysis 3 — Network** — calls `pipeline.network.*`, shows degree distribution, top-10 PageRank table, connected-components summary with isolated-seller count, one motif example rendered as a small diagram, BFS backup-seller table, delayed-subgraph PageRank delta.
8. **Convergence — Seller Risk Index** — calls `pipeline.convergence.*`, shows normalisation step, weight rationale, risk-band histogram, the quadrant plot (demand × sentiment, bubble = pagerank), the state bar chart, the top-20 CRITICAL/WARNING sellers table.
9. **Recommendations for Olist** — 4–6 concrete management-level actions grounded in the numbers above. This is the material that maps to the 5-minute presentation.
10. **Big-data safety log** — short table rendered from `docs/big_data_safety_log.md`.
11. **Reproducibility and rubric compliance** — runs `checks.run_all()`, prints a pass/fail table, shows the cache manifest diff vs. HEAD, prints environment versions.

Every code cell has a markdown cell immediately above it explaining (a) what it does, (b) which Spark/ML concept it demonstrates, (c) any decision worth flagging. No orphaned code cells.

## Quality bar (senior data scientist)

- Every chart has a title, axis labels, a legend when needed, and a one-sentence takeaway printed below it in markdown.
- Every model reports the metric the rubric asks for and at least one additional sanity metric (RMSE **and** MAE; AUC **and** confusion matrix at a chosen threshold).
- Every non-Spark library (PyTorch, sklearn, pandas, matplotlib/seaborn) has the justification markdown cell mandated by `CLAUDE.md §2` **and** an entry in `docs/big_data_safety_log.md`.
- No `collect()` / `toPandas()` on non-aggregated DataFrames. Any legitimate small-DF `toPandas()` (e.g. top-50 for viz) has a `# BIG-DATA-SAFETY-ESCAPE: see safety.py::TOP50_VIZ` comment.
- No `orderBy()` without `limit()`. Enforced by a `checks.py` grep-style test that scans `src/olist/` and notebook code cells.
- Docstrings on every public function in `src/olist/`. Type hints where they help.
- Meaningful variable names — no single-letter vars except in lambdas. Readable, well-commented code per the brief's Code Quality criterion.
- No dead code, no commented-out blocks, no print-debug leftovers.

## Git hygiene and reruns

- `outputs/*.parquet` and `outputs/.cache_manifest.json` are tracked. `outputs/_cache/` and `.venv/` are gitignored.
- Add `./run.sh` that executes `notebooks/00_main.ipynb` with `jupyter nbconvert --execute --inplace` and exits non-zero on any failure — so CI or a grader can reproduce from a clean clone.
- Update `submission/build_zip.sh` to execute `00_main.ipynb` in addition to the three thematic notebooks and verify all four have visible outputs before zipping.

## Acceptance criteria — "done" means all of these

1. `docs/refactor_plan.md` was written and I approved it before any code changed.
2. `00_main.ipynb` + the three thematic notebooks execute top-to-bottom on a clean clone and all four have visible outputs in every code cell.
3. On a second run with no changes, every step reports `[cache] skip …`; total wall-clock < 60s; **visible outputs still render** (because cached functions still return DataFrames that the notebook displays).
4. Touching one file in `src/olist/pipeline/sentiment.py` and rerunning causes **only** sentiment-downstream steps (sentiment + convergence) to recompute.
5. No logic duplicated between the main notebook and the thematic notebooks — `grep -n` shows the same function names called from both.
6. `checks.run_all()` passes green. `docs/grading_checklist.md` is 18/18. Every bullet from the assignment brief's required-demonstrations list is mapped to a notebook section and a `checks.py` assertion.
7. Numerical results match the pre-refactor run within floating-point tolerance; any intentional change is logged in `docs/decisions_log.md` with justification.
8. `docs/architecture.md` has a module diagram and a one-paragraph description of the cache model.
9. `docs/big_data_safety_log.md` lists every non-big-data-safe call site with its rationale and Spark-native alternative.

## Hard rules (do not break)

- Do not modify `CLAUDE.md` or `docs/references/` without asking.
- Do not drop or re-run the three thematic notebooks in a way that loses their existing executed-output cells until the replacement is ready and verified.
- Do not commit `data/` CSVs, `.venv/`, or anything under `outputs/_cache/`.
- Do not introduce DVC, MLflow, Prefect, Dagster, Airflow, or any orchestration framework. The step cache is ~150 lines of code.
- Do not use `inferSchema=True`. Do not call `collect()` or `toPandas()` on non-aggregated DataFrames. Do not `orderBy()` without `limit()`.
- If a library install fails, a notebook fails twice after fixes, or a CSV schema doesn't match — stop and ask me.

## Working agreement

Plan first, then implement in this order, committing after each step with a clear message and appending to `docs/decisions_log.md` as you go:

1. `src/olist/cache.py` + a tiny unit test.
2. `src/olist/safety.py` and `docs/big_data_safety_log.md` with the current escape hatches catalogued.
3. Split existing logic into `pipeline/{demand,sentiment,network,convergence}.py` without changing behaviour; verify outputs are byte-identical parquet.
4. Rewrite the three thematic notebooks as thin wrappers; re-execute; confirm visible outputs and cached skips.
5. Build `00_main.ipynb` as the narrative; re-execute from a cold cache, then a warm cache; confirm both produce visible outputs and the warm run is < 60s.
6. Write `checks.py`, wire into the final notebook section, and update `docs/grading_checklist.md` and `submission/build_zip.sh`.

Log every non-trivial decision to `docs/decisions_log.md` as you go. If you are unsure whether a change is non-trivial, err on the side of logging.
