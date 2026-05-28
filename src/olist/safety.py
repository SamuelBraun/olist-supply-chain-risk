"""Big-data-safety escape-hatch registry.

Every `collect()` / `toPandas()` / non-Spark library use in the codebase
must be annotated with a `# BIG-DATA-SAFETY-ESCAPE: <ID>` comment whose
`<ID>` is one of the constants below, and must have a corresponding
entry in `docs/big_data_safety_log.md` describing the call site, the
Spark-native alternative at production scale, and why the escape is
acceptable at this dataset size.

The log file is the human-readable surface; this module exists so that
(a) the IDs are greppable from code, and (b) `src/olist/checks.py` can
enforce that every ID in code is catalogued in the log, and vice versa.
"""

from __future__ import annotations

#: Top-50-by-risk-score DataFrame pulled to pandas for the convergence-layer
#: quadrant + state-bar charts (NB3 / 00_main convergence section).
TOP50_VIZ = "TOP50_VIZ"

#: Per-state aggregate DataFrame (<30 rows after groupBy seller_state) pulled
#: to pandas for the state-bar chart.
STATE_AGG_VIZ = "STATE_AGG_VIZ"

#: Single-row aggregates (min/max/mean collected as Python scalars) used to
#: normalise per-seller demand/sentiment/network scores before weighting.
RISK_NORM_AGG = "RISK_NORM_AGG"

#: LSTM training-set DataFrame (text + binary label) pulled to pandas so
#: PyTorch can iterate over it via a DataLoader.
LSTM_TO_PANDAS = "LSTM_TO_PANDAS"

#: PyTorch is not a native Spark library. Used for the mandatory Deep
#: Learning rubric item; Spark-native alternative at production scale
#: would be `spark-nlp` or Petastorm + distributed training.
LSTM_PYTORCH = "LSTM_PYTORCH"

#: Top-10 PageRank seller IDs collected as a Python list to drive per-seller
#: BFS calls. The list is ≤10 rows by construction.
TOP10_PAGERANK_DRIVER = "TOP10_PAGERANK_DRIVER"

#: BFS path-result `.limit(1).collect()` inside the per-seller backup search
#: loop. Each call returns 0 or 1 row.
BFS_BACKUP_COLLECT = "BFS_BACKUP_COLLECT"

#: Lead-indicator cross-correlation summary (9 rows × 3 cols) pulled to pandas
#: for the lag-vs-correlation chart.
LEAD_INDICATOR_VIZ = "LEAD_INDICATOR_VIZ"

#: pandas + Plotly (with Kaleido for static PNG) used for charts from pre-
#: aggregated / pre-capped Spark frames. Spark has no native chart surface;
#: Plotly's payload is a JSON description of the figure (independent of
#: upstream Spark dataset size), but the upstream `toPandas()` still
#: requires the aggregate to fit on the driver — hence the escape.
PLOTLY_STATIC_VIZ = "PLOTLY_STATIC_VIZ"

#: `row_counts.first()` / `.collect()` on a one-row aggregate DataFrame —
#: the standard Spark idiom for extracting a scalar from a driver-side reduce.
SMALL_SUMMARY_COLLECT = "SMALL_SUMMARY_COLLECT"


#: The canonical list of escape-hatch IDs. `checks.py` asserts that this
#: set exactly matches the IDs documented in `docs/big_data_safety_log.md`.
ALL_ESCAPES: tuple[str, ...] = (
    TOP50_VIZ,
    STATE_AGG_VIZ,
    RISK_NORM_AGG,
    LSTM_TO_PANDAS,
    LSTM_PYTORCH,
    TOP10_PAGERANK_DRIVER,
    BFS_BACKUP_COLLECT,
    LEAD_INDICATOR_VIZ,
    PLOTLY_STATIC_VIZ,
    SMALL_SUMMARY_COLLECT,
)
