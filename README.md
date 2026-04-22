# Olist Supply Chain Risk Intelligence

NOVA IMS — *Big Data Analysis* group project. PySpark-powered early-warning system that flags Olist sellers about to become a supply-chain risk **before** customers are affected.

Three notebooks, three angles, one converged seller-level Risk Index:

| Notebook | Question | Spark surface |
|---|---|---|
| `01_demand_forecasting.ipynb` | Where will demand spike, and which sellers are already late? | RDDs · SparkSQL · Pipeline · MLlib (GBT, RandomForest) · *Streaming (bonus)* |
| `02_sentiment_analysis.ipynb` | Is sentiment falling before sales drop? | NLP Pipeline · MLlib (LogReg) · LSTM (PyTorch) · Window functions |
| `03_supply_network_graph.ipynb` | Which sellers are single points of failure? | GraphFrames (PageRank, motifs, BFS, connected components) |

The convergence section at the end of NB3 joins the three per-seller scores into `outputs/seller_risk_index.parquet` and produces the charts that go on the deck.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # editable-installs the local `olist` package + deps
```

Java 8 or 11 is required for PySpark (`java -version`).

## Running

Open the notebooks in order — each writes parquet outputs that the next one reads.

```python
from olist.spark_session import get_spark
from olist.loaders import load_orders

spark = get_spark("nb1", with_graphframes=False)
orders = load_orders(spark)
```

Notebook 3 needs the GraphFrames jar — pass `with_graphframes=True` to `get_spark()`.

## Layout

```
.
├── CLAUDE.md                     ← AI-assistant contract (do not delete)
├── README.md                     ← you are here
├── pyproject.toml                ← canonical deps; installs `olist` package
├── requirements.txt              ← `-e .` shortcut
├── .gitignore
├── data/                         ← 9 raw Olist CSVs (gitignored)
├── src/olist/                    ← shared package: SparkSession, schemas, loaders, transforms
├── notebooks/                    ← the three deliverable notebooks
├── outputs/                      ← parquet outputs (gitignored)
├── docs/
│   ├── decisions_log.md          ← autonomous choices, for the oral defence
│   ├── grading_checklist.md      ← rubric mirror, ticked before submission
│   └── references/               ← original brief PDF + storyline RTF
├── presentation/                 ← deck PDF + slide-to-output mapping
└── submission/build_zip.sh       ← re-executes notebooks then zips the Moodle bundle
```

## Submission

```bash
bash submission/build_zip.sh <GROUP_NUMBER>
```

Re-executes all three notebooks in place, then zips notebooks + `presentation/presentation.pdf` into `submission/olist_bigdata_group<N>.zip`. That zip is the only thing graded.

## Constraints we hold ourselves to

- All Spark code must be safe at big-data scale (no `collect`/`toPandas` on large DFs, no `orderBy` without `limit`, broadcast small lookups, cache hot DFs once).
- Every notebook cell has a markdown cell above it explaining what it does.
- Any non-Spark library (PyTorch, sklearn, pandas) requires a markdown cell justifying *why* and naming the Spark-native production alternative.

Full rules in [`CLAUDE.md`](./CLAUDE.md).

## Deadline

**5 June 2026, 23:59 (Lisbon).** Hard.
