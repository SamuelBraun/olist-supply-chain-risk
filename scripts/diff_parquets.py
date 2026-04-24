"""Numerically diff two parquet directories.

Used after a pipeline rerun to verify Risk R2 (numerical equivalence within
tolerance) per docs/refactor_plan.md.

Usage:
    .venv/bin/python scripts/diff_parquets.py <baseline_dir> <candidate_dir> [--rtol 1e-6] [--atol 1e-9] [--key seller_id]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _read(dir_: Path) -> pd.DataFrame:
    parts = sorted(dir_.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet found under {dir_}")
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    return df


def diff(
    baseline: Path,
    candidate: Path,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    key: str | None = None,
) -> int:
    base = _read(baseline)
    cand = _read(candidate)

    print(f"baseline:  rows={len(base):>7,}  cols={list(base.columns)}")
    print(f"candidate: rows={len(cand):>7,}  cols={list(cand.columns)}")
    if set(base.columns) != set(cand.columns):
        print("SCHEMA MISMATCH")
        return 1
    if len(base) != len(cand):
        print(f"ROW-COUNT MISMATCH: {len(base)} vs {len(cand)}")
        return 1

    # Align rows by key if provided.
    if key is None:
        # Try common keys.
        for candidate_key in ("seller_id", "seller_id_a", "lag"):
            if candidate_key in base.columns:
                key = candidate_key
                break

    if key is not None and key in base.columns:
        base = base.sort_values(key).reset_index(drop=True)
        cand = cand.sort_values(key).reset_index(drop=True)
    else:
        base = base.sort_values(list(base.columns)).reset_index(drop=True)
        cand = cand.sort_values(list(cand.columns)).reset_index(drop=True)

    numeric_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
    other_cols = [c for c in base.columns if c not in numeric_cols]

    print(f"\nKey: {key!r}")
    print(f"Numeric cols: {numeric_cols}")
    print(f"Other cols  : {other_cols}")

    diffs: list[str] = []

    for col in other_cols:
        mismatches = (base[col].fillna("__NA__") != cand[col].fillna("__NA__")).sum()
        if mismatches:
            diffs.append(f"  {col!r}: {mismatches} row mismatches")

    for col in numeric_cols:
        b = base[col].astype("float64").to_numpy()
        c = cand[col].astype("float64").to_numpy()
        both_nan = pd.isna(b) & pd.isna(c)
        diff = abs(b - c)
        tol = atol + rtol * abs(b)
        bad = (~both_nan) & ((diff > tol) | (pd.isna(b) ^ pd.isna(c)))
        n_bad = bad.sum()
        if n_bad:
            max_diff = diff[bad].max() if n_bad else 0.0
            max_rel = (diff[bad] / (abs(b[bad]) + 1e-30)).max() if n_bad else 0.0
            diffs.append(
                f"  {col!r}: {n_bad:,}/{len(b):,} rows outside tol; "
                f"max_abs_diff={max_diff:.6g}, max_rel_diff={max_rel:.3g}"
            )

    if not diffs:
        print("\n[EQUIV] baseline and candidate match within tolerance.")
        return 0
    print("\n[DIFF] differences found:")
    for line in diffs:
        print(line)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--key", default=None)
    args = parser.parse_args()
    return diff(
        Path(args.baseline),
        Path(args.candidate),
        rtol=args.rtol,
        atol=args.atol,
        key=args.key,
    )


if __name__ == "__main__":
    sys.exit(main())
