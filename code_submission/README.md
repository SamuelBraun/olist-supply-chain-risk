# Olist Supply-Chain Risk Intelligence — code submission

NOVA IMS, Big Data Analysis 2025/26. Group project: Lukas Belser, Samuel Braun, Margarida Quintino, Jan Thier.

The deliverable is **one notebook**, executed end-to-end with visible outputs in every cell: [`notebooks/main.ipynb`](notebooks/main.ipynb). The management-facing PDF presentation is submitted alongside this folder.

## Layout

```
code_submission/
├── notebooks/main.ipynb          ← THE graded artefact (executed, all outputs visible)
├── src/olist/                    ← all transformation logic; the notebook calls into it
│   ├── pipeline/{demand,sentiment,network,convergence,streaming}.py
│   ├── spark_session.py · schemas.py · loaders.py · transforms.py
│   ├── cache.py · safety.py · checks.py · viz.py · data_foundation.py
├── data/                         ← 9 source CSVs
├── outputs/                      ← committed parquet artefacts + cache manifest
├── docs/big_data_safety_log.md   ← every non-Spark-safe call, catalogued
├── pyproject.toml · requirements.txt · run.sh
```

The 9 source CSVs are the Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0, ~100k orders 2016–2018).

## Setup

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt   # installs the local `olist` package + deps
```

Java 11 is required for PySpark. On macOS the runner sets `JAVA_HOME` automatically if Homebrew `openjdk@11` is present.

## Run

```bash
bash run.sh
```

Executes `notebooks/main.ipynb` end-to-end. A warm-cache run completes in under a minute; cold (cache empty) takes 10–15 minutes. To force a full recompute:

```bash
OLIST_FORCE_ALL=1 bash run.sh
```

## What's in the notebook

| Section | Contents |
|---|---|
| §1–2 | Project introduction, 9-table data foundation, schema diagram, shared-key cardinality, temporal coverage, cleaning audit, 4 V's framing |
| §3 | **Demand forecasting** — RDD warm-up, 4 SparkSQL queries, EDA primitives (`approxQuantile`, `approx_count_distinct`), Window-based lag/rolling/decay features, retail-calendar + per-seller covariates, ML Pipeline (Imputer + VectorAssembler), GBT + RF under k-fold CV *and* rolling-origin time-series selection, naive baselines, regional state-level forecast |
| §4 | **Sentiment analysis** — SparkSQL temp-view join, NLP Pipeline (Tokenizer → StopWordsRemover[pt] → HashingTF → IDF → LogReg under CV), PyTorch LSTM trained via `TorchDistributor` and scored via `predict_batch_udf`, `applyInPandas` grouped-map for per-seller OLS slopes, LDA topic model over negative reviews (failure-mode mix per seller), Window-based weekly rollup, lead-indicator cross-correlation |
| §5 | **Supply-network graph** — bipartite `GraphFrame(v, e)`, PageRank, `connectedComponents(algorithm="graphx")`, shared-customer motif, BFS backups, delayed-subgraph PageRank, co-customer projection (PageRank + label propagation + direct backup map), transitive 2-hop substitutes, dense co-category+region substitution graph (PageRank + label propagation + per-seller substitute count) |
| §6 | **Cross-analysis** — percentile-rank normalisation via `QuantileDiscretizer` (distributable), weighted composite (0.35 demand / 0.35 sentiment / 0.30 network where network = 0.5 contagion + 0.5 supply-concentration), percentile banding (CRITICAL top 1%, WARNING next 4%, SAFE bottom 95%), correlation heatmap, K-Means archetypes with elbow + silhouette justification |
| §7 | **Conclusions** — recommendations by archetype, honest limitations, inline big-data-safety log, **24/24 programmatic rubric checks** via `olist.checks.run_all`, cache manifest, environment versions |
| §8 | **Bonus** — Structured Streaming twin of the weekly volume aggregation (file source, `availableNow` trigger, memory sink) |

## Constraints we hold ourselves to

- Every `collect()` / `toPandas()` / non-Spark-library call is tagged `# BIG-DATA-SAFETY-ESCAPE: <ID>` and catalogued in `docs/big_data_safety_log.md` with its production-scale alternative. The 9 declared IDs are enforced by `checks.check_safety_log_consistency` and three programmatic scans (`check_no_unguarded_collect`, `check_no_unbounded_orderby`, `check_no_unpartitioned_window`).
- Logic lives in `src/olist/`; the notebook is a report surface.
- Every code cell has visible output. `run.sh` re-executes the notebook end-to-end; the rubric-check cell prints all 24 checks pass.

## Notebook output ground truth (current run)

- 1,630 sellers scored. **18 CRITICAL, 65 WARNING, 1,547 SAFE** (percentile-banded). Weekly watchlist = 83 sellers.
- Demand: RF beats GBT on a rolling-origin holdout (test RMSE 3.17 vs 3.24). Both beat the rolling-4w naive baseline (3.45).
- Sentiment: LogReg AUC 0.9641, LSTM AUC 0.9614 (trained distributed via `TorchDistributor`, scored via `predict_batch_udf`). LDA reveals ~97% of negative reviews complain about late/non-delivery; 63 sellers have a meaningfully non-delivery complaint profile.
- Network: dense co-category+region projection covers 2,221 of 2,970 sellers (74.8%) vs the sparse co-customer projection (7.6%). 1,618 sellers are single points of failure with no substitute anywhere.
