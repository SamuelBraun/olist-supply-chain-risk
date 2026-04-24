"""Assert every code cell in the given notebooks has non-empty outputs.

Usage: python scripts/assert_notebook_outputs.py notebooks/*.ipynb

The course brief states that notebooks without outputs will not be graded,
so the submission bundle refuses to build if any cell is unpopulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(paths: list[str]) -> int:
    failures: list[str] = []
    for p in paths:
        nb_path = Path(p)
        with nb_path.open() as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if not "".join(cell.get("source", [])).strip():
                continue  # blank cells are skipped by Jupyter
            if not cell.get("outputs"):
                failures.append(f"{nb_path}: code cell {i} has empty outputs")
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    failures.append(
                        f"{nb_path}: code cell {i} is an error "
                        f"({out.get('ename')}: {(out.get('evalue') or '')[:100]})"
                    )
    if failures:
        print("ASSERT FAILED:")
        for line in failures:
            print(" ", line)
        return 1
    print(f"OK: every code cell in {len(paths)} notebook(s) has non-empty, non-error outputs.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/assert_notebook_outputs.py NOTEBOOK [NOTEBOOK ...]")
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
