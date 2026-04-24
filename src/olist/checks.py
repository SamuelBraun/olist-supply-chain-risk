"""Rubric compliance checks — `run_all(spark)` produces a Spark DataFrame
summarising pass/fail per bullet from the course brief.

Mapping (mirrors CLAUDE.md §8 + docs/grading_checklist.md):

    RDDs                        → check_rdd_chain
    DataFrames & SparkSQL       → check_sparksql
    Pipelines & Data Engineering→ check_pipelines
    MLlib                       → check_cv_models
    Deep Learning               → check_lstm_justification
    GraphFrames                 → check_graphframe_ops
    Window functions            → check_window_used
    EDA primitives              → check_approx_eda
    Code Quality & Docs         → check_markdown_ratio
    Big-data safety disclosure  → check_safety_log_consistency
    Output parquets exist       → check_parquet_artefacts

Each returns `CheckResult(name, passed, detail)`. `run_all()` aggregates
and returns a 3-column Spark DataFrame suitable for `.show(truncate=False)`
inline in `00_main.ipynb` §11.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pyspark.sql import DataFrame, SparkSession

from . import safety
from .cache import project_root
from .pipeline import convergence, demand, network, sentiment


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _src(mod) -> str:
    return inspect.getsource(mod)


def _nb_cells(nb_name: str) -> tuple[list[str], list[str]]:
    path = project_root() / "notebooks" / nb_name
    with path.open() as f:
        n = json.load(f)
    code = ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"]
    md = ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "markdown"]
    return code, md


# ---------------------------------------------------------------------------
# Individual checks — each ~6 lines
# ---------------------------------------------------------------------------


def check_rdd_chain() -> CheckResult:
    src = _src(demand)
    has_text = bool(re.search(r"sparkContext\.textFile|\.textFile\(", src))
    has_reduce = ".reduceByKey" in src
    has_schema_conv = "createDataFrame" in src
    ok = has_text and has_reduce and has_schema_conv
    return CheckResult(
        "RDDs — textFile → filter/map/reduceByKey → DF",
        ok,
        f"textFile={has_text}, reduceByKey={has_reduce}, createDataFrame={has_schema_conv} (demand.py)",
    )


def check_sparksql() -> CheckResult:
    src = _src(demand) + "\n" + _src(sentiment)
    count = len(re.findall(r"spark\.sql\(", src))
    ok = count >= 4
    return CheckResult(
        "SparkSQL temp-view queries (≥4)",
        ok,
        f"{count} spark.sql(...) calls in demand.py + sentiment.py",
    )


def check_pipelines() -> CheckResult:
    src = _src(demand) + "\n" + _src(sentiment)
    count = len(re.findall(r"\bPipeline\(\s*stages", src))
    ok = count >= 2
    return CheckResult(
        "Pipeline(stages=...) constructors (≥2)",
        ok,
        f"{count} Pipeline constructors (feature + NLP)",
    )


def check_cv_models() -> CheckResult:
    src = _src(demand) + "\n" + _src(sentiment)
    count = len(re.findall(r"\bCrossValidator\(", src))
    ok = count >= 3
    return CheckResult(
        "MLlib CrossValidator-wrapped models (≥3)",
        ok,
        f"{count} CrossValidator instantiations (GBT + RF + LR)",
    )


def check_lstm_justification() -> CheckResult:
    _, md = _nb_cells("02_sentiment_analysis.ipynb")
    blob = "\n".join(md).lower()
    ok = (
        "pytorch" in blob
        and ("justif" in blob or "why" in blob)
        and ("not spark" in blob or "mllib has no native" in blob or "no native lstm" in blob)
    )
    return CheckResult(
        "LSTM justification cell present (NB2)",
        ok,
        "PyTorch + justification + Spark-alternative language found"
        if ok
        else "Missing justification markdown in NB2",
    )


def check_graphframe_ops() -> CheckResult:
    src = _src(network)
    patterns = {
        "pageRank": r"\.pageRank\(",
        "connectedComponents": r"\.connectedComponents\(",
        "motif .find": r"\.find\(\s*['\"]",
        "bfs": r"\.bfs\(",
        "GraphFrame ctor": r"GraphFrame\(",
        "delayed subgraph": r"delayed_edges|delayed_vertex",
    }
    hits = [name for name, pat in patterns.items() if re.search(pat, src)]
    ok = len(hits) >= 6
    return CheckResult(
        "GraphFrame operations (≥6)",
        ok,
        f"{len(hits)}/{len(patterns)} found: {hits}",
    )


def check_window_used() -> CheckResult:
    src = _src(demand) + "\n" + _src(sentiment)
    partition = len(re.findall(r"Window\.partitionBy", src))
    rolling = "rowsBetween" in src or "rangeBetween" in src
    ok = partition >= 1 and rolling
    return CheckResult(
        "Window functions with rowsBetween/rangeBetween",
        ok,
        f"Window.partitionBy={partition}, rolling={rolling}",
    )


def check_approx_eda() -> CheckResult:
    src = _src(demand)
    has_quantile = "approxQuantile" in src
    has_count = "approx_count_distinct" in src or "approxCountDistinct" in src
    ok = has_quantile and has_count
    return CheckResult(
        "EDA primitives (approxQuantile + approx_count_distinct)",
        ok,
        f"approxQuantile={has_quantile}, approx_count_distinct={has_count}",
    )


def check_markdown_ratio() -> CheckResult:
    """CLAUDE.md §3: markdown above every code cell. Accept runs of code
    cells inside one section as long as the run starts after a markdown
    cell — otherwise the contract ("describe what it does") is broken.
    """
    per_nb = []
    orphan_total = 0
    for name in (
        "00_main.ipynb",
        "01_demand_forecasting.ipynb",
        "02_sentiment_analysis.ipynb",
        "03_supply_network_graph.ipynb",
    ):
        path = project_root() / "notebooks" / name
        with path.open() as f:
            cells = json.load(f)["cells"]
        orphan_blocks = 0
        in_block = False
        for i, c in enumerate(cells):
            if c["cell_type"] == "code":
                if not in_block:
                    in_block = True
                    preamble_is_md = i > 0 and cells[i - 1]["cell_type"] == "markdown"
                    if not preamble_is_md:
                        orphan_blocks += 1
            else:
                in_block = False
        per_nb.append(f"{name.split('_')[0]}={'OK' if orphan_blocks == 0 else f'{orphan_blocks} orphan'}")
        orphan_total += orphan_blocks
    ok = orphan_total == 0
    return CheckResult(
        "Every code-block has a markdown preamble",
        ok,
        ", ".join(per_nb),
    )


def check_safety_log_consistency() -> CheckResult:
    """Every ID in `safety.ALL_ESCAPES` must appear in
    `docs/big_data_safety_log.md` and be tagged as
    `# BIG-DATA-SAFETY-ESCAPE: <ID>` somewhere in `src/olist/pipeline/*.py`
    or in a notebook (matplotlib-rendering IDs live in notebooks).
    """
    root = project_root()
    log_text = (root / "docs" / "big_data_safety_log.md").read_text()
    pipeline_src = _src(demand) + _src(sentiment) + _src(network) + _src(convergence)
    notebook_src = "\n".join(
        (root / "notebooks" / nb).read_text()
        for nb in (
            "00_main.ipynb",
            "01_demand_forecasting.ipynb",
            "02_sentiment_analysis.ipynb",
            "03_supply_network_graph.ipynb",
        )
        if (root / "notebooks" / nb).exists()
    )
    annotations = pipeline_src + "\n" + notebook_src
    missing_in_log = [c for c in safety.ALL_ESCAPES if c not in log_text]
    missing_in_code = [
        c for c in safety.ALL_ESCAPES
        if f"BIG-DATA-SAFETY-ESCAPE: {c}" not in annotations
    ]
    ok = not missing_in_log and not missing_in_code
    detail = (
        f"{len(safety.ALL_ESCAPES)} IDs"
        + (f"; missing in log: {missing_in_log}" if missing_in_log else "")
        + (f"; missing in code/nb: {missing_in_code}" if missing_in_code else "")
    )
    return CheckResult(
        "Safety log ↔ safety.py ↔ annotations consistent",
        ok,
        detail,
    )


def check_parquet_artefacts() -> CheckResult:
    expected = [
        "outputs/geo_centroids.parquet",
        "outputs/nb1_seller_demand_scores.parquet",
        "outputs/nb1_weekly_order_volume.parquet",
        "outputs/nb2_seller_sentiment_scores.parquet",
        "outputs/nb3_seller_network_scores.parquet",
        "outputs/seller_risk_index.parquet",
    ]
    root = project_root()
    missing = [p for p in expected if not (root / p).is_dir()]
    ok = not missing
    return CheckResult(
        f"All {len(expected)} output parquets present",
        ok,
        "All present" if ok else f"Missing: {missing}",
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_rdd_chain,
    check_sparksql,
    check_pipelines,
    check_cv_models,
    check_lstm_justification,
    check_graphframe_ops,
    check_window_used,
    check_approx_eda,
    check_markdown_ratio,
    check_safety_log_consistency,
    check_parquet_artefacts,
)


def run_all(spark: SparkSession) -> DataFrame:
    """Run every compliance check; return a Spark DataFrame with columns
    (check, passed, detail). Suitable for `.show(truncate=False, n=50)`
    inline in the main notebook's reproducibility section.
    """
    rows = [(r.name, r.passed, r.detail) for r in (c() for c in CHECKS)]
    return spark.createDataFrame(
        rows, "check string, passed boolean, detail string"
    )
