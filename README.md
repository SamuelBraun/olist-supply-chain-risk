# Olist Supply-Chain Risk Intelligence

NOVA IMS — *Big Data Analysis* group project. A PySpark big-data project that fuses three independent analyses — **demand forecasting**, **sentiment analysis**, and **supply-network graph analytics** — into a single per-seller **Seller Risk Index** for Olist's account-management team.

The complete deliverable is one notebook: [`notebooks/main.ipynb`](notebooks/main.ipynb). It executes top-to-bottom with all outputs visible; both managers and DS reviewers can read it.

---

## What's in this repo

```
.
├── CLAUDE.md                       ← AI-assistant contract (do not delete)
├── README.md                       ← you are here
├── notebooks/
│   ├── main.ipynb                  ← THE comprehensive deliverable (executed, all outputs)
│   └── _archive/                   ← previous 4-notebook version (kept for git history)
├── src/olist/                      ← all transformation logic — single source of truth
│   ├── spark_session.py            ← SparkSession factory (driver memory 6g, GraphFrames jar)
│   ├── schemas.py                  ← explicit StructType per CSV
│   ├── loaders.py                  ← typed loaders (no inferSchema)
│   ├── transforms.py               ← delivery-delay + geolocation centroids
│   ├── cache.py                    ← @step decorator + fingerprint-based parquet cache
│   ├── safety.py                   ← registry of big-data-safety escape IDs
│   ├── checks.py                   ← programmatic rubric-compliance assertions
│   ├── viz.py                      ← 15 reusable matplotlib + pandas-styler helpers
│   ├── data_foundation.py          ← schema metadata + shared-key + temporal helpers
│   └── pipeline/
│       ├── demand.py               ← main §3 — demand forecasting
│       ├── sentiment.py            ← main §4 — sentiment analysis
│       ├── network.py              ← main §5 — supply-network graph
│       └── convergence.py          ← main §6 — Seller Risk Index + archetypes
├── data/                           ← 9 source CSVs (gitignored)
├── outputs/                        ← 6 committed parquet artefacts + .cache_manifest.json
│   └── _cache/                     ← gitignored intermediate parquets + GraphX checkpoints
├── docs/                           ← decisions log, big-data-safety log, architecture doc
├── scripts/
│   ├── build_main.py               ← regenerates notebooks/main.ipynb from cell definitions
│   ├── assert_notebook_outputs.py  ← CI-style "every cell has output" check
│   ├── diff_parquets.py            ← numerical-equivalence guard for refactor PRs
│   └── _archive/                   ← previous 4-notebook build scripts
├── submission/build_zip.sh         ← builds the Moodle zip
├── run.sh                          ← one-shot: re-execute main.ipynb end-to-end
└── requirements.txt                ← pinned Python deps
```

## Setup

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Java 11 is required for PySpark (`java -version`). On macOS the harness sets `JAVA_HOME` automatically if it finds Homebrew `openjdk@11`.

## Running

**Interactively:**
```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home \
  .venv/bin/jupyter notebook notebooks/main.ipynb
```
Then *Kernel → Restart & Run All*. Cache hits skip the expensive steps; warm-cache runs complete in under a minute, cold runs take ~10–15 minutes.

**Headless from the command line:**
```bash
./run.sh
```

**Force a complete recompute (ignores the cache):**
```bash
OLIST_FORCE_ALL=1 ./run.sh
```

## Submission

```bash
./submission/build_zip.sh <GROUP_NUMBER>
```

Re-executes `notebooks/main.ipynb`, asserts every code cell has visible output, and zips the notebook + `presentation/presentation.pdf` into `submission/olist_bigdata_group<N>.zip`. That zip is the only graded artefact.

## What's where

- **The whole story** — [`notebooks/main.ipynb`](notebooks/main.ipynb): top-to-bottom narrative with §1 Project Introduction → §2 Data Foundation → §3 Demand → §4 Sentiment → §5 Network → §6 Cross-Analysis Synthesis → §7 Conclusions.
- **Decisions made along the way** — [`docs/decisions_log.md`](docs/decisions_log.md): append-only record of every analytical choice with rationale and impact.
- **Why each non-Spark call is acceptable** — [`docs/big_data_safety_log.md`](docs/big_data_safety_log.md): every `toPandas()` / `collect()` / non-Spark library call catalogued with its production-scale alternative.
- **Architecture notes** — [`docs/architecture.md`](docs/architecture.md): engineer-tone overview of the package layout and step cache.
- **Programmatic rubric checks** — `from olist.checks import run_all`: returns a Spark DataFrame with one row per requirement; rendered inline in §7.4 of the notebook.

## Constraints we hold ourselves to

- All Spark code must be safe at big-data scale (no `collect`/`toPandas` on large DFs, no `orderBy` without `limit`, broadcast small lookups, cache hot DFs once).
- Every notebook code cell has a markdown cell above it explaining what it does.
- Every non-Spark library use (PyTorch LSTM, matplotlib, sklearn) requires a markdown justification cell + a `# BIG-DATA-SAFETY-ESCAPE: <ID>` annotation + an entry in `docs/big_data_safety_log.md`.
- Logic lives in `src/olist/`; the notebook is a report surface — function changes happen in `src/`, not in cells.
- After any change, regenerate the notebook with `.venv/bin/python scripts/build_main.py` and re-execute it with `./run.sh`.

Full rules in [`CLAUDE.md`](./CLAUDE.md).

## Deadline

**5 June 2026, 23:59 (Lisbon).** Hard.
