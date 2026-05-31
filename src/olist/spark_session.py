"""Single SparkSession factory shared by all three notebooks.

The GraphFrames jar coordinate must match the running Spark + Scala version.
The default below targets PySpark 3.5.x. If `pyspark.__version__` differs, the
factory swaps in a known-good coordinate; otherwise it raises so the mismatch
is loud, not silent.
"""

from __future__ import annotations

import os
import sys

import pyspark
from pyspark.sql import SparkSession

# Pin Spark's Python workers to the *same* interpreter as the driver (the venv).
# Without this, workers fall back to `/usr/bin/python3`, which lacks venv-only
# deps like torch — fine until an executor-side UDF imports one (e.g. the
# predict_batch_udf LSTM scorer in pipeline.sentiment). Unconditional (not
# setdefault): a stale `PYSPARK_PYTHON` exported in a shell profile / CI runner
# would otherwise silently reintroduce the wrong interpreter. Set before the
# JVM/py4j gateway launches.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Map Spark major.minor → GraphFrames jar coordinate that ships for it.
_GRAPHFRAMES_BY_SPARK = {
    "3.2": "graphframes:graphframes:0.8.2-spark3.2-s_2.12",
    "3.3": "graphframes:graphframes:0.8.3-spark3.3-s_2.12",
    "3.4": "graphframes:graphframes:0.8.3-spark3.4-s_2.12",
    "3.5": "graphframes:graphframes:0.8.3-spark3.5-s_2.12",
}


def _graphframes_coordinate() -> str:
    major_minor = ".".join(pyspark.__version__.split(".")[:2])
    if major_minor not in _GRAPHFRAMES_BY_SPARK:
        raise RuntimeError(
            f"No GraphFrames coordinate registered for PySpark {pyspark.__version__}. "
            f"Add a mapping in spark_session.py."
        )
    return _GRAPHFRAMES_BY_SPARK[major_minor]


def get_spark(
    app_name: str = "Olist",
    *,
    with_graphframes: bool = False,
    shuffle_partitions: int = 64,
    driver_memory: str = "6g",
) -> SparkSession:
    """Return a configured SparkSession.

    `with_graphframes=True` adds the version-matched jar via spark.jars.packages.
    Notebooks 1 and 2 leave it False; notebook 3 sets it True.
    `driver_memory` bumps -Xmx for the JVM. 6g is the default because the NB3
    GraphFrames workload (PageRank + ConnectedComponents on ~100k vertices)
    OOMs at 2g. Must be set before the JVM launches, i.e., before getOrCreate.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "America/Sao_Paulo")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "2g")
    )
    if with_graphframes:
        builder = builder.config("spark.jars.packages", _graphframes_coordinate())
    return builder.getOrCreate()
