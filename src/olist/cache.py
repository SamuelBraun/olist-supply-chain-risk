"""Fingerprint-based step cache for the Olist pipeline.

Contract (from CLAUDE.md §5):
* `@step(name, inputs, outputs, code_deps, version=1)` wraps every public
  pipeline function.
* Fingerprint = sha256 of sorted input file sizes+mtimes, sorted `code_deps`
  source bytes, and the `version` tag.
* On call: if the manifest's stored fingerprint for `name` matches AND every
  declared output exists → skip compute, return a parquet-backed DataFrame
  (or dict of DataFrames) re-read from disk. Otherwise run the wrapped
  function, write its return value as parquet(s), and atomically rewrite
  the manifest.
* `force=True` per-call bypass; `OLIST_FORCE_ALL=1` forces everything.
* One log line per call: `[cache] skip <name> (<fp8>…)` or
  `[cache] run <name> (<t>s, wrote <rows> → <path>)`.

The cache never caches Spark DataFrames across steps. Downstream steps
always re-read from parquet — deterministic lineage, no stale state.

`OLIST_ROOT` env var overrides the repo root (used by the unit test so
that real pipeline paths are never touched).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable

from pyspark.sql import DataFrame, SparkSession

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = 1


def _root() -> Path:
    override = os.environ.get("OLIST_ROOT")
    return Path(override) if override else _DEFAULT_ROOT


def _manifest_path() -> Path:
    return _root() / "outputs" / ".cache_manifest.json"


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _root() / path


def _read_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {"version": MANIFEST_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"version": MANIFEST_VERSION, "entries": {}}
    if data.get("version") != MANIFEST_VERSION:
        return {"version": MANIFEST_VERSION, "entries": {}}
    return data


def _write_manifest(data: dict) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _file_signature(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _directory_signature(path: Path) -> str:
    parts = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            parts.append(f"{child.relative_to(path)}:{_file_signature(child)}")
    return "|".join(parts)


def _path_signature(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_dir():
        return _directory_signature(path)
    return _file_signature(path)


def _compute_fingerprint(
    inputs: list[Path], code_deps: list[Path], version: int
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"manifest_version={MANIFEST_VERSION}\n".encode())
    hasher.update(f"step_version={version}\n".encode())
    for path in sorted(inputs, key=lambda p: str(p)):
        hasher.update(f"INPUT {path}\n".encode())
        hasher.update((_path_signature(path) + "\n").encode())
    for path in sorted(code_deps, key=lambda p: str(p)):
        hasher.update(f"CODE {path}\n".encode())
        if path.exists() and path.is_file():
            hasher.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
            hasher.update(b"\n")
        else:
            hasher.update(b"MISSING\n")
    return hasher.hexdigest()


def _active_spark(args: tuple) -> SparkSession:
    for arg in args:
        if isinstance(arg, SparkSession):
            return arg
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "Cache skip requires an active SparkSession. "
            "Pass `spark` as the first argument to the wrapped function."
        )
    return spark


def _read_outputs(spark: SparkSession, outputs: list[Path]):
    if len(outputs) == 1:
        return spark.read.parquet(str(outputs[0]))
    return {p.stem: spark.read.parquet(str(p)) for p in outputs}


def _count_rows(obj) -> int:
    if isinstance(obj, DataFrame):
        return obj.count()
    if isinstance(obj, dict):
        return sum(df.count() for df in obj.values() if isinstance(df, DataFrame))
    return 0


def step(
    name: str,
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    code_deps: Iterable[str | Path],
    version: int = 1,
):
    """Wrap a pipeline function with fingerprint-based parquet caching.

    The wrapped function must take the SparkSession as its first argument
    (or one of its positional args) and return either:
    * a single `DataFrame` (when len(outputs) == 1), or
    * a `dict[str, DataFrame]` keyed by `Path(output).stem` (when > 1).
    """
    inputs_list = list(inputs)
    outputs_list = list(outputs)
    code_deps_list = list(code_deps)

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, force: bool = False, **kwargs):
            input_paths = [_resolve(p) for p in inputs_list]
            output_paths = [_resolve(p) for p in outputs_list]
            code_dep_paths = [_resolve(p) for p in code_deps_list]

            spark = _active_spark(args)
            force_all = os.environ.get("OLIST_FORCE_ALL") == "1"
            fingerprint = _compute_fingerprint(input_paths, code_dep_paths, version)
            manifest = _read_manifest()
            entry = manifest["entries"].get(name)

            hit = (
                not force
                and not force_all
                and entry is not None
                and entry.get("fingerprint") == fingerprint
                and all(p.exists() for p in output_paths)
            )
            if hit:
                print(f"[cache] skip {name} ({fingerprint[:8]}…)")
                return _read_outputs(spark, output_paths)

            for path in output_paths:
                path.parent.mkdir(parents=True, exist_ok=True)

            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if len(output_paths) == 1:
                if not isinstance(result, DataFrame):
                    raise TypeError(
                        f"step {name}: expected a DataFrame, got {type(result).__name__}"
                    )
                result.write.mode("overwrite").parquet(str(output_paths[0]))
            else:
                if not isinstance(result, dict):
                    raise TypeError(
                        f"step {name}: expected dict[str, DataFrame] for "
                        f"{len(output_paths)} outputs, got {type(result).__name__}"
                    )
                for path in output_paths:
                    df = result.get(path.stem)
                    if not isinstance(df, DataFrame):
                        raise TypeError(
                            f"step {name}: missing or non-DataFrame value for "
                            f"output key {path.stem!r}"
                        )
                    df.write.mode("overwrite").parquet(str(path))

            returned = _read_outputs(spark, output_paths)
            rows = _count_rows(returned)
            elapsed = time.perf_counter() - start

            manifest["entries"][name] = {
                "fingerprint": fingerprint,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "outputs": [str(p.relative_to(_root())) for p in output_paths],
                "version": version,
            }
            _write_manifest(manifest)

            primary = output_paths[0].relative_to(_root())
            print(
                f"[cache] run {name} ({elapsed:.1f}s, wrote {rows} rows → {primary})"
            )
            return returned

        return wrapper

    return decorator


def manifest_summary() -> list[dict]:
    """Return the manifest entries as a sorted list of dicts for display."""
    manifest = _read_manifest()
    rows = []
    for step_name, entry in sorted(manifest["entries"].items()):
        rows.append(
            {
                "step": step_name,
                "fingerprint": entry["fingerprint"][:12] + "…",
                "written_at": entry["written_at"],
                "outputs": ", ".join(entry["outputs"]),
            }
        )
    return rows
