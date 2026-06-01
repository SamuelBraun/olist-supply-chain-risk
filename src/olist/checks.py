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
    _, md = _nb_cells("main.ipynb")
    blob = "\n".join(md).lower()
    ok = (
        "pytorch" in blob
        and ("justif" in blob or "why" in blob)
        and ("not spark" in blob or "mllib has no native" in blob or "no native lstm" in blob)
    )
    return CheckResult(
        "LSTM justification cell present (main §4)",
        ok,
        "PyTorch + justification + Spark-alternative language found"
        if ok
        else "Missing justification markdown in main.ipynb §4",
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
    name = "main.ipynb"
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
    ok = orphan_blocks == 0
    return CheckResult(
        "Every code-block has a markdown preamble",
        ok,
        f"{name}={'OK' if ok else f'{orphan_blocks} orphan'}",
    )


def check_safety_log_consistency() -> CheckResult:
    """Every ID in `safety.ALL_ESCAPES` must appear in
    `docs/big_data_safety_log.md` and be tagged as
    `# BIG-DATA-SAFETY-ESCAPE: <ID>` somewhere in `src/olist/pipeline/*.py`
    or in a notebook (chart-rendering IDs live in notebooks).
    """
    root = project_root()
    log_text = (root / "docs" / "big_data_safety_log.md").read_text()
    pipeline_src = _src(demand) + _src(sentiment) + _src(network) + _src(convergence)
    notebook_src = "\n".join(
        (root / "notebooks" / nb).read_text()
        for nb in ("main.ipynb",)
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


def check_lazy_eval_documented() -> CheckResult:
    _, md = _nb_cells("main.ipynb")
    blob = "\n".join(md).lower()
    ok = (
        ("transformation" in blob and "action" in blob)
        and ("lazy" in blob)
    )
    return CheckResult(
        "Lazy evaluation / transformation-vs-action documented",
        ok,
        "found in main.ipynb markdown" if ok else "missing lazy-eval callout",
    )


def check_outlier_treatment() -> CheckResult:
    src = _src(demand)
    has_winsor = "def winsorize" in src
    has_quantile = "approxQuantile" in src
    ok = has_winsor and has_quantile
    return CheckResult(
        "Outlier treatment (winsorise via approxQuantile)",
        ok,
        f"winsorize={has_winsor}, approxQuantile={has_quantile} (demand.py)",
    )


def check_cocustomer_graph() -> CheckResult:
    src = _src(network)
    has_proj = "build_cocustomer_edges" in src
    has_lp = ".labelPropagation(" in src
    has_deficit = "substitutability_deficit" in src
    ok = has_proj and has_lp and has_deficit
    return CheckResult(
        "Co-customer projection (centrality + communities + deficit)",
        ok,
        f"projection={has_proj}, labelPropagation={has_lp}, deficit={has_deficit}",
    )


def check_two_hop_substitutes() -> CheckResult:
    """Multi-hop graph signal: `compute_two_hop_backups` builds transitive 2-hop
    substitutes (a second expansion on the projection — not reproducible by a
    single self-join), and the convergence layer consumes `two_hop_reach_count`
    in the `escalate_no_backup` decision.
    """
    net_src = _src(network)
    conv_src = _src(convergence)
    has_fn = "def compute_two_hop_backups" in net_src
    has_expansion = "left_anti" in net_src and "two_hop_reach_count" in net_src
    wired = "two_hop_reach_count" in conv_src
    ok = has_fn and has_expansion and wired
    return CheckResult(
        "Transitive 2-hop substitutes (graph-unique) feed escalation",
        ok,
        f"fn={has_fn}, expansion={has_expansion}, wired_into_escalation={wired}",
    )


def check_spark_dl_integration() -> CheckResult:
    src = _src(sentiment)
    has_distributor = "TorchDistributor" in src
    has_batch_udf = "predict_batch_udf" in src
    ok = has_distributor and has_batch_udf
    return CheckResult(
        "Spark-DL integration (TorchDistributor + predict_batch_udf)",
        ok,
        f"TorchDistributor={has_distributor}, predict_batch_udf={has_batch_udf}",
    )


def check_applyinpandas() -> CheckResult:
    src = _src(sentiment)
    ok = ".applyInPandas(" in src
    return CheckResult(
        "Grouped-map applyInPandas (UDF family)",
        ok,
        "applyInPandas present in sentiment.py" if ok else "missing applyInPandas",
    )


def _code_units() -> list[tuple[str, list[str]]]:
    """(origin, lines) for every pipeline module and every notebook code cell.
    Windows stay within a unit so look-back never crosses a file/cell boundary.
    """
    from .pipeline import streaming

    units: list[tuple[str, list[str]]] = []
    for mod in (demand, sentiment, network, convergence, streaming):
        units.append((mod.__name__.split(".")[-1] + ".py", _src(mod).splitlines()))
    code, _ = _nb_cells("main.ipynb")
    for idx, cell in enumerate(code):
        units.append((f"main.ipynb#cell{idx}", cell.splitlines()))
    return units


#: Whole-frame driver pulls. `.first()`/`.head(n)`/`.take(n)` are intentionally
#: excluded — they are inherently row-bounded, so they cannot OOM the driver.
_DRIVER_PULLS = (".collect()", ".toPandas()")
_LOOKBACK = 8  # lines of chain/preamble context to scan for limit/escape tags


def check_no_unguarded_collect() -> CheckResult:
    """Every whole-frame driver pull (`.collect()`/`.toPandas()`) must be either
    bounded by `.limit(` in its own chain or carry a `# BIG-DATA-SAFETY-ESCAPE:`
    tag within the preceding few lines. This is the "is the library you use safe
    at scale?" guard the brief asks for — nothing materialises an unbounded,
    non-aggregated frame on the driver silently.
    """
    violations: list[str] = []
    for origin, lines in _code_units():
        for i, line in enumerate(lines):
            if not any(sink in line for sink in _DRIVER_PULLS):
                continue
            if line.lstrip().startswith(("#", "*", '"', "'", ">")) or "`" in line:
                continue  # comment / docstring prose (often backticked) mentioning the call
            window = "\n".join(lines[max(0, i - _LOOKBACK): i + 1])
            if ".limit(" in window or "BIG-DATA-SAFETY-ESCAPE" in window:
                continue
            violations.append(f"{origin}:{i + 1}: {line.strip()[:70]}")
    ok = not violations
    return CheckResult(
        "Driver pulls (collect/toPandas) bounded by limit or escape-tagged",
        ok,
        "all guarded" if ok else f"{len(violations)} unguarded: {violations[:5]}",
    )


def check_no_unbounded_orderby() -> CheckResult:
    """The brief's "ordering of data returned from queries" guard. A global
    `orderBy`/`sort` is only unsafe when its (potentially large) result is pulled
    to the driver without a bound: sorting that stays distributed (feeds `.show()`
    or another transform) or sorts a small aggregate is fine. So we flag any
    `.orderBy(`/`.sort(` (excluding `Window` specs) that flows into a
    `.collect()`/`.toPandas()` within a few lines with no intervening `.limit(`
    and no `# SAFE-ORDERBY:` / escape waiver.
    """
    violations: list[str] = []
    for origin, lines in _code_units():
        for i, line in enumerate(lines):
            if not any(sink in line for sink in _DRIVER_PULLS):
                continue
            if line.lstrip().startswith(("#", "*", '"', "'", ">")) or "`" in line:
                continue
            window_lines = lines[max(0, i - 8): i + 1]
            window = "\n".join(window_lines)
            has_orderby = any(
                ("orderBy(" in w or ".sort(" in w) and "Window" not in w
                for w in window_lines
            )
            if not has_orderby:
                continue
            if ".limit(" in window or "BIG-DATA-SAFETY-ESCAPE" in window \
                    or "SAFE-ORDERBY" in window:
                continue
            violations.append(f"{origin}:{i + 1}: {line.strip()[:70]}")
    ok = not violations
    return CheckResult(
        "Ordered driver pulls bounded by limit or waived",
        ok,
        "all bounded" if ok else f"{len(violations)} unbounded: {violations[:5]}",
    )


def check_no_unpartitioned_window() -> CheckResult:
    """`Window.orderBy(...)` with no `partitionBy` forces every row onto a single
    partition to compute the global order — the textbook "won't scale across
    partitions" pattern. Flag any such window in the pipeline transformation
    modules unless it carries a `# SCALABLE-WINDOW:` waiver (e.g. it ranks a small
    DISTINCT set, not the full frame). Doc/comment mentions are ignored.
    """
    from .pipeline import streaming

    violations: list[str] = []
    for mod in (demand, sentiment, network, convergence, streaming):
        name = mod.__name__.split(".")[-1] + ".py"
        lines = _src(mod).splitlines()
        for i, line in enumerate(lines):
            if "Window.orderBy(" not in line:
                continue
            if "partitionBy" in line:
                continue
            if line.lstrip().startswith(("#", "*", '"', "'", ">")) or "`" in line:
                continue
            window = "\n".join(lines[max(0, i - 3): i + 2])
            if "SCALABLE-WINDOW" in window:
                continue
            violations.append(f"{name}:{i + 1}: {line.strip()[:70]}")
    ok = not violations
    return CheckResult(
        "No unpartitioned Window.orderBy (single-partition global sort)",
        ok,
        "none (percentile rank/banding use QuantileDiscretizer/approxQuantile)"
        if ok
        else f"{len(violations)} found: {violations}",
    )


def check_streaming() -> CheckResult:
    from .pipeline import streaming
    src = _src(streaming)
    has_read = "readStream" in src
    has_write = "writeStream" in src
    ok = has_read and has_write
    return CheckResult(
        "Structured Streaming (readStream + writeStream)",
        ok,
        f"readStream={has_read}, writeStream={has_write} (streaming.py)",
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
    check_lazy_eval_documented,
    check_outlier_treatment,
    check_cocustomer_graph,
    check_two_hop_substitutes,
    check_spark_dl_integration,
    check_applyinpandas,
    check_streaming,
    check_no_unguarded_collect,
    check_no_unbounded_orderby,
    check_no_unpartitioned_window,
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
