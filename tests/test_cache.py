"""Cache contract tests.

Runnable three ways, all use bare `assert` so no pytest dependency is
required:
    python -m tests.test_cache            # runs all tests
    python tests/test_cache.py            # same
    pytest tests/test_cache.py            # if pytest is installed

Covers:
* fingerprint stability across calls when nothing changed
* fingerprint change when code_deps source bytes change
* fingerprint change when an input file's mtime/size changes
* run → skip round-trip through an actual SparkSession
* `force=True` bypasses the skip
* `OLIST_FORCE_ALL=1` bypasses globally

The test points OLIST_ROOT at a tempdir so the real outputs/ manifest is
never touched.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _reload_cache(root: Path):
    """Reload src/olist/cache with OLIST_ROOT pinned to `root`."""
    os.environ["OLIST_ROOT"] = str(root)
    if "olist.cache" in sys.modules:
        importlib.reload(sys.modules["olist.cache"])
    else:
        importlib.import_module("olist.cache")
    return sys.modules["olist.cache"]


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_fingerprint_stable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = _reload_cache(root)
        _write(root / "data/a.csv", "x,y\n1,2\n")
        _write(root / "src/olist/pipeline/demo.py", "def f():\n    return 1\n")
        inputs = [root / "data/a.csv"]
        deps = [root / "src/olist/pipeline/demo.py"]
        fp1 = cache._compute_fingerprint(inputs, deps, version=1)
        fp2 = cache._compute_fingerprint(inputs, deps, version=1)
        assert fp1 == fp2, "fingerprint must be stable for identical inputs/deps"


def test_fingerprint_changes_with_code_deps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = _reload_cache(root)
        _write(root / "data/a.csv", "x,y\n1,2\n")
        dep = root / "src/olist/pipeline/demo.py"
        _write(dep, "def f():\n    return 1\n")
        fp1 = cache._compute_fingerprint(
            [root / "data/a.csv"], [dep], version=1
        )
        _write(dep, "def f():\n    return 2\n")  # body changed
        fp2 = cache._compute_fingerprint(
            [root / "data/a.csv"], [dep], version=1
        )
        assert fp1 != fp2, "fingerprint must change when code_deps bytes change"


def test_fingerprint_changes_with_input_mtime() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = _reload_cache(root)
        inp = root / "data/a.csv"
        _write(inp, "x,y\n1,2\n")
        dep = root / "src/olist/pipeline/demo.py"
        _write(dep, "def f():\n    return 1\n")
        fp1 = cache._compute_fingerprint([inp], [dep], version=1)
        # Mtime resolution is nanoseconds on macOS; sleep briefly then rewrite.
        time.sleep(0.01)
        _write(inp, "x,y\n1,2\n3,4\n")  # content + mtime + size change
        fp2 = cache._compute_fingerprint([inp], [dep], version=1)
        assert fp1 != fp2, "fingerprint must change when input file changes"


def test_fingerprint_changes_with_version_tag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = _reload_cache(root)
        inp = root / "data/a.csv"
        _write(inp, "x,y\n1,2\n")
        dep = root / "src/olist/pipeline/demo.py"
        _write(dep, "def f():\n    return 1\n")
        fp1 = cache._compute_fingerprint([inp], [dep], version=1)
        fp2 = cache._compute_fingerprint([inp], [dep], version=2)
        assert fp1 != fp2, "fingerprint must change when version tag changes"


def test_manifest_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = _reload_cache(root)
        assert cache._read_manifest() == {"version": 1, "entries": {}}
        payload = {
            "version": 1,
            "entries": {
                "demand.build_order_lines": {
                    "fingerprint": "a" * 64,
                    "written_at": "2026-04-22T12:00:00",
                    "outputs": ["outputs/order_lines.parquet"],
                    "version": 1,
                }
            },
        }
        cache._write_manifest(payload)
        assert cache._read_manifest() == payload
        # Tempfile-rename atomicity: no leftover .tmp file.
        tmp_files = list((root / "outputs").glob("*.tmp"))
        assert tmp_files == [], f"stray tmp files: {tmp_files}"


def test_run_then_skip_then_force() -> None:
    """Full wrapper exercise with a real SparkSession."""
    # Import lazily — Spark is expensive to start.
    from pyspark.sql import SparkSession

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = _reload_cache(root)

        dep_file = root / "src/olist/pipeline/demo.py"
        _write(dep_file, "# v1\n")
        input_file = root / "data/raw.txt"
        _write(input_file, "first\n")

        spark = (
            SparkSession.builder.master("local[1]")
            .appName("olist-cache-test")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        try:
            compute_calls = {"count": 0}

            @cache.step(
                name="demo.build",
                inputs=["data/raw.txt"],
                outputs=["outputs/demo.parquet"],
                code_deps=["src/olist/pipeline/demo.py"],
                version=1,
            )
            def build(spark_arg):
                compute_calls["count"] += 1
                return spark_arg.createDataFrame(
                    [(1, "a"), (2, "b")], schema=["id", "tag"]
                )

            # First call: cold miss → run.
            df1 = build(spark)
            assert compute_calls["count"] == 1
            assert df1.count() == 2
            assert (root / "outputs/demo.parquet").exists()
            manifest = cache._read_manifest()
            assert "demo.build" in manifest["entries"]

            # Second call: warm hit → skip, no recompute.
            df2 = build(spark)
            assert compute_calls["count"] == 1, "second call must skip compute"
            assert df2.count() == 2

            # force=True: recompute even on hit.
            df3 = build(spark, force=True)
            assert compute_calls["count"] == 2
            assert df3.count() == 2

            # OLIST_FORCE_ALL=1: recompute globally.
            os.environ["OLIST_FORCE_ALL"] = "1"
            try:
                df4 = build(spark)
                assert compute_calls["count"] == 3
                assert df4.count() == 2
            finally:
                os.environ.pop("OLIST_FORCE_ALL", None)

            # Changing code_deps bytes: cache miss on next call.
            _write(dep_file, "# v2 — changed\n")
            df5 = build(spark)
            assert compute_calls["count"] == 4, (
                "changing code_deps bytes must invalidate the cache"
            )
            assert df5.count() == 2
        finally:
            spark.stop()


TESTS = [
    test_fingerprint_stable,
    test_fingerprint_changes_with_code_deps,
    test_fingerprint_changes_with_input_mtime,
    test_fingerprint_changes_with_version_tag,
    test_manifest_roundtrip,
    test_run_then_skip_then_force,
]


def _run_all() -> int:
    failures = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface any error type
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
