# Refactor plan — 2026-04-22

Status: **awaiting human approval. No code changes until approved.**

Companion to `docs/execution_plan.md` (build order) and `docs/decisions_log.md` (autonomous choices). This file covers the structural refactor requested in `docs/refactor_prompt.md`: logic into `src/olist/pipeline/*`, add a step cache, add a narrative `00_main.ipynb`, add a compliance layer. The analytical work itself is **not** re-derived.

---

## (a) Current state — what I understand to be true today

### Repository shape
```
.                              # NOT a git repo (env reports is_git=false) — see Risk R1
├── CLAUDE.md                  # already the refactor-era spec; §5 cache, §6 safety log present
├── README.md
├── pyproject.toml             # pyspark, torch, pandas, matplotlib, seaborn, jupyter, nbconvert
├── requirements.txt           # `-e .`
├── .gitignore                 # excludes outputs/ and *.parquet — CONFLICTS with §5 ("manifest committed")
├── data/                      # 120 MB, 9 CSVs, gitignored
├── src/olist/
│   ├── __init__.py
│   ├── spark_session.py       # 6g driver, shuffle=64, GraphFrames jar matrix
│   ├── schemas.py             # 9 explicit StructTypes + CSV_FILES registry
│   ├── loaders.py             # load_orders(), load_reviews(), …
│   └── transforms.py          # only geolocation_centroids() + delivery_delay_days() — thin
├── notebooks/
│   ├── 01_demand_forecasting.ipynb       # 29 cells, fully executed
│   ├── 02_sentiment_analysis.ipynb       # 25 cells, fully executed
│   ├── 03_supply_network_graph.ipynb     # 32 cells, fully executed (includes convergence)
│   └── _builders/build_nb{1,2,3}.py      # nbformat emitters — not graded, not in spec
├── outputs/                   # 1.2 MB total, 6 parquets present, gitignored
├── docs/
│   ├── decisions_log.md       # 6 entries through 2026-04-22
│   ├── execution_plan.md      # build-order plan, approved
│   ├── grading_checklist.md   # 18 bullets, 16 ticked (streaming bonus + pdf/zip open)
│   ├── refactor_prompt.md     # the prompt that produced this file
│   └── references/            # project_brief.pdf + storyline.rtf
├── presentation/              # pdf not yet exported
└── submission/build_zip.sh    # executes 3 notebooks, zips them + pdf
```

### Logic distribution today
`src/olist/` holds the SparkSession factory, schemas, typed loaders, and two shared transforms. **Every other transformation — the RDD warm-up, the 4-way order-line join, the SparkSQL queries, the feature Pipeline, the GBT/RF CV, the NLP Pipeline, the LSTM, the Window sentiment rollup, the GraphFrame build, PageRank, connected components, motif, BFS, delayed-subgraph, the convergence normalise/weight/band step — lives inline inside notebook code cells.** Cell sizes run up to 49 lines (e.g. NB1 cell 26 "forecast uplift + demand scores write", NB2 cell 14 "LSTM training loop"). This is what the refactor is about: those cell bodies must move to `src/olist/pipeline/*.py` while remaining visibly rendered in the notebook.

### Parquet outputs & numerical results currently in `outputs/`
Per `decisions_log.md` and `grading_checklist.md`:
- `geo_centroids.parquet` — 19,015 × 4
- `nb1_seller_demand_scores.parquet` — 2,970 × 5 (RandomForest winner, test RMSE 5.06 vs GBT 5.09)
- `nb1_weekly_order_volume.parquet` — 35,385 × 3
- `nb2_seller_sentiment_scores.parquet` — 3,090 × 5 (LogReg AUC 0.958, LSTM AUC 0.963; 491 flagged `sentiment_declining`)
- `nb3_seller_network_scores.parquet` — 2,970 × 6 (connected components via `algorithm="graphx"`, 0 isolated sellers)
- `seller_risk_index.parquet` — 2,967 × 11 (2,899 SAFE / 68 WARNING / 0 CRITICAL, max risk_score 0.71)

These numbers are the "intact unless rerun demonstrably changes them" baseline.

### What the spec adds that doesn't exist yet
Missing modules: `cache.py`, `safety.py`, `viz.py`, `checks.py`, `pipeline/{demand,sentiment,network,convergence,__init__}.py`.
Missing notebooks: `notebooks/00_main.ipynb`.
Missing docs: `architecture.md`, `big_data_safety_log.md`, `refactor_plan.md` (this file).
Missing scripts: `run.sh`.
`submission/build_zip.sh` executes three notebooks; must be extended to four.

---

## (b) What I intend to change

### New modules
- **`src/olist/cache.py`** (~150 LOC). `@step(name, inputs, outputs, code_deps, version=1)` decorator. Fingerprint = sha256 of (sorted input-file sizes+mtimes, sorted `code_deps` source bytes, `version`). On call: if fingerprint matches `outputs/.cache_manifest.json` entry AND all declared outputs exist → skip compute, return parquet-backed DataFrames via `spark.read.parquet(output_path)`. Else: run wrapped function, write parquet outputs, atomically update the manifest (tempfile-rename). Per-call `force=True` override + global `OLIST_FORCE_ALL=1` env var. One log line per call: `[cache] skip <name> (<fp>)` or `[cache] run <name> (<t>s, wrote <rows> → <path>)`. **Never caches Spark DataFrames across steps** — downstream always re-reads from parquet.
  - Unit test: `tests/test_cache.py` that exercises run → skip → force-rerun with a temp output dir and a trivial step function.

- **`src/olist/safety.py`**. Declares the constants that `# BIG-DATA-SAFETY-ESCAPE: <id>` comments reference, e.g. `TOP50_VIZ`, `RISK_NORM_AGG`, `LSTM_TO_PANDAS`, `SMALL_SUMMARY_COLLECT`. Small module; the real human-readable content lives in `docs/big_data_safety_log.md`.

- **`src/olist/viz.py`**. Matplotlib/seaborn helpers that take small pandas DFs (the ones produced from aggregated Spark results) and return figures: `risk_band_histogram`, `state_bar_chart`, `demand_sentiment_quadrant`, `feature_importance_bar`, `confusion_matrix_heatmap`, `lead_indicator_cross_corr`. Every helper adds title, axis labels, legend. Charts are the only surface where matplotlib is used in notebooks; all rendering goes through this module.

- **`src/olist/checks.py`**. One function per rubric bullet (RDDs, ≥3 SparkSQL temp-view queries, Pipeline stages printed, GBT+RF CV fitted, LSTM justification cell present, motif pattern `(a)-[]->(c)<-[]-(b)` called, Portuguese stopwords used, etc.). Each returns `CheckResult(name, passed, detail)`. `run_all()` aggregates and returns a DataFrame for inline display. Strategy for detecting notebook-only facts (e.g. "RDD chain executed"): parse the notebook JSON of the notebook that is supposed to demonstrate it (`01_demand_forecasting.ipynb` for the RDD check, `02_sentiment_analysis.ipynb` for LSTM justification) for the relevant markers. Parquet-existence checks re-read from `outputs/`. Hygiene checks (`no orderBy without limit`, `no bare inferSchema=True`) walk `src/olist/**/*.py` and notebook code cells via `ast`/regex.

- **`src/olist/pipeline/__init__.py`** + four modules, split along existing notebook boundaries:
  - **`demand.py`** — `rdd_daily_order_count_demo(spark)`, `build_order_lines(spark)`, `sparksql_queries(order_lines)` (returns dict of three named DFs + their query text strings), `write_weekly_order_volume(order_lines)`, `build_demand_feature_pipeline(weekly)`, `fit_demand_models(train, test)` (returns dict with `gbt_cv_model, rf_cv_model, gbt_rmse, rf_rmse, gbt_mae, rf_mae, best_model, best_name`), `score_sellers_demand(order_lines, best_model)` → writes `nb1_seller_demand_scores.parquet`.
  - **`sentiment.py`** — `build_reviews_with_seller(spark)`, `label_reviews(reviews_with_seller)`, `build_nlp_pipeline()` (returns unfit `Pipeline`), `fit_nlp_model(labelled)`, `train_lstm(text_labelled)` (returns test AUC + the trained `torch.nn.Module`), `weekly_sentiment_rollup(labelled)` → writes `nb2_seller_sentiment_scores.parquet`, `lead_indicator_cross_corr(weekly_sentiment, weekly_volume_parquet_path)`.
  - **`network.py`** — `build_graph_frame(spark)` (returns `GraphFrame` + the edge/vertex DFs for inspection), `compute_pagerank(gf)`, `compute_connected_components(gf)`, `compute_motifs(gf)`, `compute_bfs_backups(gf, top_pr_sellers)`, `compute_delayed_subgraph_pagerank(gf, order_lines)`, `write_seller_network_scores(...)` → `nb3_seller_network_scores.parquet`.
  - **`convergence.py`** — `converge_risk(spark)` reads the three per-seller parquets, normalises (single-row agg, flagged), weights 0.35/0.35/0.30, bands CRITICAL/WARNING/SAFE, writes `seller_risk_index.parquet`. Produces small pandas frames for the two viz charts (top-50 + per-state mean risk_score).

Each public function in `pipeline/*` is wrapped by `@step(...)` so its parquet output is cached by fingerprint. Functions return Spark objects wherever reasonable (or the cached parquet re-read into a DataFrame) so notebook cells can call `.show()`/`display()` on them — preserving visible outputs after cached skips.

### Notebooks

- **`notebooks/00_main.ipynb`** (new, primary deliverable) — narrative orchestrator, structure per spec §"Main notebook contract":
  1. Executive summary (risk-band counts pulled live from `seller_risk_index.parquet`).
  2. Client context + 4 V's.
  3. Data overview (aggregated Spark-native tables, dropped-row audit).
  4. Distributed-computing toolkit walkthrough (RDD, DataFrames, SparkSQL, Pipelines, MLlib, Deep Learning, GraphFrames, Windows, EDA primitives, Streaming bonus if core is green).
  5. Analysis 1 — Demand (calls `pipeline.demand.*`).
  6. Analysis 2 — Sentiment (calls `pipeline.sentiment.*`).
  7. Analysis 3 — Network (calls `pipeline.network.*`).
  8. Convergence — Seller Risk Index (calls `pipeline.convergence.*`).
  9. Recommendations for Olist.
  10. Big-data safety log (table rendered from `docs/big_data_safety_log.md`).
  11. Reproducibility & rubric compliance (`checks.run_all()` printed as pass/fail table + cache-manifest summary + environment versions).

- **`notebooks/01_…`, `02_…`, `03_…`** — rewritten as thin wrappers over `src/olist/pipeline/*`. Each cell is markdown + a 1–5-line function call + a `.show()` or `display()` or plot rendering. No duplicated logic between thematic notebooks and the main notebook — verified by a grep in `checks.py`.

### Infra / scripts
- **`run.sh`** (new) — `set -euo pipefail; export JAVA_HOME=...; .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/00_main.ipynb`. Exits non-zero on any cell failure.
- **`submission/build_zip.sh`** — add `notebooks/00_main.ipynb` to the executed-and-zipped list; add a post-execution check that every code cell has non-empty `outputs` (walk the JSON).
- **`.gitignore`** — remove blanket `outputs/` + `*.parquet`; replace with precise `outputs/_cache/` and `outputs/_gf_checkpoints/`. Commit `outputs/*.parquet` + `outputs/.cache_manifest.json`.
- **`notebooks/_builders/`** — remove after thin-wrapper notebooks are written directly as `.ipynb` files. The nbformat emitter approach is superseded by "logic in `src/olist/pipeline/`, thin ipynb committed directly".

### New docs
- **`docs/architecture.md`** — one-page module diagram + cache-model paragraph. Final artefact, not code.
- **`docs/big_data_safety_log.md`** — per-escape-hatch table: call site, why, Spark-native alternative at production scale, why acceptable at this dataset size. Initially populated with the escapes already present in the three notebooks (`toPandas()` on 43k reviews for LSTM, `toPandas()` on top-50 for charts, small `collect()` on single-row aggregates, pandas/matplotlib for charting, PyTorch for LSTM). Future escapes are appended here before the offending code lands.

---

## (c) Order of commits

This matches spec §"Working agreement" and is incremental enough that any step can be reverted without losing the currently-committed executed outputs.

| # | Commit message | Changes | Risk if reverted |
|---|---|---|---|
| 1 | `chore(cache): add step decorator + manifest + unit test` | New: `src/olist/cache.py`, `tests/test_cache.py`, `pytest` added to pyproject dev-deps. No existing code touched. | Zero — additive. |
| 2 | `chore(safety): add safety constants + big-data-safety log` | New: `src/olist/safety.py`, `docs/big_data_safety_log.md`. Catalogues escapes already present; no notebook edits. | Zero — additive. |
| 3 | `refactor(pipeline): extract demand logic to src/olist/pipeline/demand.py` | Move NB1 inline transformations to `pipeline/demand.py`. Notebook left untouched; parquet outputs not regenerated. | Zero — notebook still runs inline code; new module is dormant. |
| 4 | `refactor(pipeline): extract sentiment logic to pipeline/sentiment.py` | Same for NB2. | Zero. |
| 5 | `refactor(pipeline): extract network + convergence logic to pipeline/{network,convergence}.py` | Same for NB3. | Zero. |
| 6 | `refactor(nb1): rewrite 01_demand_forecasting as thin pipeline wrapper` | Replace `.ipynb` with cell-per-public-function wrapper, execute, verify visible outputs, verify parquets byte-equivalent (or numerically within tolerance; see Risk R2). | **Non-zero — see R2.** Keep the previous `.ipynb` as `notebooks/.backup_01_demand_forecasting.ipynb` until step 9 completes. |
| 7 | `refactor(nb2): rewrite 02_sentiment as thin wrapper` | Same. Backup kept. | R2. |
| 8 | `refactor(nb3): rewrite 03_supply_network as thin wrapper + move convergence to main` | Same, **minus** the convergence section — that migrates to `00_main.ipynb`. Backup kept. | R2. |
| 9 | `feat(main): add notebooks/00_main.ipynb — narrative orchestrator` | Cold-cache rerun verified to produce all six parquets + visible outputs. Warm-cache rerun verified < 60 s. | Zero if cold-run matched baseline within tolerance. |
| 10 | `feat(checks): add checks.run_all() + wire into main notebook` | `src/olist/checks.py` + the compliance section of `00_main.ipynb`. | Zero. |
| 11 | `chore(submission): extend build_zip + add run.sh + architecture.md` | `run.sh`, updated `build_zip.sh`, `.gitignore` flip, `docs/architecture.md`. Remove `notebooks/_builders/` and the three `.backup_*.ipynb` files. | Zero once prior steps green. |
| 12 | `chore(checklist): mark refactor items complete in grading_checklist.md` | Doc-only. | Zero. |

Each commit appends to `docs/decisions_log.md` with a short entry (one heading + 2–4 lines, per the existing format).

---

## (d) Risks to executed-output preservation

### R1 — Repo is not a git repo
The environment reports `is_git_repository=false`. The spec presumes a repo ("commit after each step", "git pull and run", "outputs committed"). **Before step 1 I need confirmation that I should either (i) `git init` the project now, or (ii) proceed with file-level changes only and skip the commit discipline.** My default assumption is (i) — initialise, with `.gitignore` already in place and the twelve commits as described. This is the only question that genuinely needs a human answer before I start.

### R2 — Thin-wrapper notebooks may produce slightly different numerical outputs
The refactor only *moves* logic; arithmetic is unchanged. But:
- **GBT / RandomForest `CrossValidator` with `parallelism=2`** picks a fold order that is stable given a fixed seed but can vary across Spark versions. Seed is already 42 in the existing notebook. Mitigation: lift the exact seeded construction into `pipeline.demand.fit_demand_models`; do not "clean up" to `parallelism=1`.
- **LSTM training** is CPU-non-deterministic even with `torch.manual_seed` (parallel matmul ordering, DataLoader with multiple workers). Current reported AUC 0.9630 could shift by ±0.005 on rerun. Mitigation: `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(False)` to keep speed, single-worker DataLoader; accept ±0.005 AUC tolerance. Log if observed delta exceeds that.
- **GraphFrames `pageRank(resetProbability=0.15, maxIter=10)`** is deterministic for a fixed edge set. Safe. `connectedComponents(algorithm="graphx")` is deterministic. Safe.
- **Float-representation parquet diffs** — column-level ULP differences between a rerun and the current parquets will show even when the logical values match. "Byte-identical" is therefore not the right bar; "numerically within tolerance" is. Concretely: after step 6/7/8, run a small script `scripts/verify_parquet_equiv.py` that reads old vs new parquet per-column and asserts `rtol=1e-9` on floats, exact match on keys/flags. Log any real differences to `decisions_log.md`.

Mitigation pattern for steps 6–8: write the new parquets to `outputs/_candidate/*.parquet`, diff against `outputs/*.parquet`, only promote if within tolerance. Keep the backup `.ipynb` files around until after step 9 verifies end-to-end.

### R3 — Warm-cache `< 60 s` budget and visible outputs
The spec's non-negotiable says cached reruns must still render visible outputs. Pattern: every cached function returns either (a) the parquet-backed DataFrame re-read from disk, or (b) a small pre-serialised summary (e.g. RMSE dict) stored alongside parquet as a JSON sidecar in `outputs/_cache/_sidecars/<step>.json`. Notebook cells call `.show()` / `display()` / `viz.*` on the returned object. The bulk of NB3's `graphframes`-related .show calls (PageRank top-10, CC summary, motif count, BFS table) can all render off the per-step parquets without re-constructing the GraphFrame. The only expensive reconstruct-on-warm-cache operation is getting a `GraphFrame` handle for the primitive-demonstration section in the main notebook — that itself is a cached step (`build_graph_frame`) and reading vertices + edges parquets and passing them to `GraphFrame(v, e)` is seconds, not minutes. Budget should hold.

### R4 — Notebook JSON `"outputs": []` lint
The grader's hard rule: notebooks without outputs will not be graded. `build_zip.sh` needs a post-execute walk that fails loudly if any code cell has an empty `outputs` array. Planned: add a `scripts/assert_notebook_outputs.py` called from `build_zip.sh`.

### R5 — Keeping thematic notebooks and the main notebook in sync
If a function signature changes in `pipeline/*`, four notebooks break, not one. Acceptable because: (a) signatures are stable after step 5, (b) `checks.py` grep-asserts that the same function names are called from both surfaces, (c) logic duplication is explicitly forbidden by spec. Risk is real but scoped.

---

## Stop-and-ask gate

Before proceeding past this file, I need an answer to **R1 only**: should I `git init` the project and proceed with the twelve-commit sequence, or do the refactor as file edits without version control?

Everything else in this plan I will decide myself and log to `docs/decisions_log.md` as I go, per the existing working agreement.
