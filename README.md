# Olist Supply-Chain Risk Intelligence

A PySpark pipeline that turns nine distributed e-commerce tables into a weekly seller
watchlist: which marketplace sellers are about to fail on delivery, found before the
complaints arrive rather than after. Course project for Big Data Analysis, M.Sc. Data
Science and Advanced Analytics, NOVA IMS, 2025/2026.

**The operational result: a Seller Risk Index that narrows 1,630 active sellers to
83 calls per week.** That is the difference between a team reacting to complaints and
a team with a prioritised list on Monday morning.

## Data

The Brazilian E-Commerce Public Dataset by Olist (Kaggle): roughly **1.5M rows** across
nine sources — 99,441 orders, 112,650 line items, 104,162 reviews, 103,886 payments and
about 1M raw geolocation points, spanning September 2016 to August 2018.

The CSVs are **not** included here (CC BY-NC-SA 4.0). Download them from Kaggle and
place them under `data/` before running.

## What it does

| Stage | Approach |
|---|---|
| Data foundation | Nine-table schema, shared-key cardinality, temporal coverage, cleaning audit, framed against the four V's |
| Demand forecasting | RDD warm-up, SparkSQL, window-based lag / rolling / decay features, retail-calendar and per-seller covariates, ML Pipeline with Imputer and VectorAssembler, GBT and Random Forest under both k-fold and rolling-origin time-series validation, against naive baselines |
| Sentiment | Portuguese free-text complaints — 43k of the 104k reviews carry text |
| Network | Supply-network graph across sellers, products and customers |
| Streaming | Structured Streaming layer for live signals alongside weekly batch scoring |

Each Spark tool is matched deliberately to the challenge it fits, rather than using one
API throughout. Every call that is not Spark-safe is catalogued in
[docs/big_data_safety_log.md](docs/big_data_safety_log.md) — distributed code fails
quietly when work silently collapses onto the driver, so the audit is part of the
deliverable.

## Layout

```
notebooks/main.ipynb    The graded artefact — executed end to end, outputs visible
src/olist/              All transformation logic; the notebook calls into it
  pipeline/             demand · sentiment · network · convergence · streaming
  spark_session.py · schemas.py · loaders.py · transforms.py
  cache.py · safety.py · checks.py · viz.py · data_foundation.py
docs/                   Big-data safety log
presentation.html       Management-facing presentation
report/report.pdf       Project report
```

## Reproduce

```bash
python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Java 11 is required for PySpark; place the nine Olist CSVs under data/
bash run.sh                 # warm cache: under a minute. cold: 10-15 minutes
OLIST_FORCE_ALL=1 bash run.sh   # force a full recompute
```

## Team

Group project (4 people): Lukas Belser, Margarida Quintino, Jan Thier, Samuel Braun.
