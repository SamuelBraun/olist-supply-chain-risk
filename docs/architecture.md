# Architecture

One-page map of the codebase and its execution model. Companion to `CLAUDE.md` (the contract) and `docs/decisions_log.md` (the "why X" record).

## Module layout

```
src/olist/
├── spark_session.py     SparkSession factory: driver memory 6g, GraphFrames jar pinned
├── schemas.py           Explicit StructType per CSV (no inferSchema)
├── loaders.py           load_orders(), load_reviews(), ... apply the above schemas
├── transforms.py        Reused leaf helpers: geolocation_centroids, delivery_delay_days
├── cache.py             @step decorator + fingerprint manifest (see §Caching)
├── safety.py            String constants — catalogues big-data-safety escape IDs
├── checks.py            Rubric compliance: CheckResult + run_all(spark)
├── pipeline/
│   ├── demand.py        RDD warm-up, 4-way join, SparkSQL, feature Pipeline, GBT+RF CV, per-seller scores
│   ├── sentiment.py     SparkSQL join, labelling, NLP Pipeline, PyTorch LSTM, Window rolling, lead indicator
│   ├── network.py       GraphFrame build, PageRank, CC (GraphX), motif, BFS, delayed subgraph, per-seller scores
│   └── convergence.py   Min-max normalise → weighted risk → CRITICAL/WARNING/SAFE bands, viz frames
```

**Principle.** Logic lives in `src/olist/`. Notebooks are report surfaces: one-to-five-line calls into `pipeline.*` followed by `.show()` / `display()` / `plt.show()`. This makes reruns cheap (the cache skips compute) and keeps the graded artefacts focused on narrative.

## Notebook flow

```
00_main.ipynb            primary narrative (exec summary → walkthrough → 3 analyses → convergence → recs → checks)
01_demand_forecasting    thin wrapper — calls pipeline.demand.*
02_sentiment_analysis    thin wrapper — calls pipeline.sentiment.*
03_supply_network_graph  thin wrapper — calls pipeline.network.*   (convergence lives in 00_main)
```

`checks.run_all(spark)` at the end of `00_main` returns a Spark DataFrame asserting every rubric bullet (RDDs, SparkSQL ≥4, Pipeline ≥2, CrossValidator ≥3, LSTM justification, GraphFrame ops ≥6, Window rolling, approx EDA, markdown preambles, safety-log consistency, 6 parquets present).

## Caching

`src/olist/cache.py` implements a ~200-line fingerprint-based step cache (CLAUDE.md §5). Every public function in `pipeline/*` is wrapped with `@step(name, inputs, outputs, code_deps, version)`:

- **Fingerprint** = sha256(sorted input file sizes+mtimes, sorted `code_deps` source bytes, `version`).
- **Skip** if the manifest entry's fingerprint matches AND every declared output parquet exists.
- **Run** otherwise: execute the function, write parquet outputs, atomically update `outputs/.cache_manifest.json` via tempfile-rename.
- **Log line** per call: `[cache] skip <name> (<fp8>…)` or `[cache] run <name> (<t>s, wrote <rows> → <path>)`.
- **Override** with per-call `force=True` or global `OLIST_FORCE_ALL=1`.

The cache never holds Spark DataFrames across steps — every downstream step re-reads its inputs from parquet. Deterministic lineage, no stale state.

**Committed vs. gitignored:**

| Path | Tracked? |
|---|---|
| `outputs/*.parquet` | yes — ~1.4 MB total, ground truth for "git pull and run" |
| `outputs/.cache_manifest.json` | yes — fingerprint state survives clones |
| `outputs/_cache/` | no — intermediate parquets rebuilt on rerun |
| `outputs/_gf_checkpoints/` | no — GraphFrames checkpoints, huge and ephemeral |
| `data/` | no — 120 MB of raw CSVs |

## Big-data safety

Every `collect()` / `toPandas()` / non-Spark-library use is:

1. **Tagged in code** — `# BIG-DATA-SAFETY-ESCAPE: <ID>` comment at the call site.
2. **Registered** — `<ID>` is a constant in `src/olist/safety.py` (with a module docstring explaining it).
3. **Catalogued** — one row in `docs/big_data_safety_log.md` documents (a) call site, (b) what it does, (c) why it counts as an escape, (d) Spark-native production alternative, (e) why it is acceptable at ~100k rows.

`checks.check_safety_log_consistency()` enforces: every ID declared in `safety.ALL_ESCAPES` has a row in the log AND an annotated call site somewhere in `pipeline/*.py` or the notebooks.

## How to run

- **Warm rerun** (every step cache-hit): `bash run.sh` — executes `notebooks/00_main.ipynb` in under a minute.
- **Cold rerun** (fingerprints invalidated by code changes): same command, 10–15 minutes.
- **Force everything to recompute**: `OLIST_FORCE_ALL=1 bash run.sh`.
- **Build the submission zip**: `bash submission/build_zip.sh <GROUP_NUMBER>` — re-executes all four notebooks, asserts non-empty outputs via `scripts/assert_notebook_outputs.py`, zips the deliverables.

## Extension points

- **New analysis** → add a new module under `pipeline/`, wire through the step cache, add a thematic `.ipynb` thin wrapper (or embed directly in `00_main`), tick a new row in `docs/grading_checklist.md`.
- **New big-data-safety escape** → declare the constant in `safety.py`, row in `big_data_safety_log.md`, annotate the call site. `checks.py` will start enforcing it automatically.
- **Streaming (bonus rubric item)** — NB1's `build_weekly_order_volume` is the natural seam: swap `spark.read.parquet(...)` for `spark.readStream.csv(...).withWatermark(...).groupBy(window(...))` and the downstream Window-function code is unchanged.
