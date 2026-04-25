#!/usr/bin/env bash
# Re-execute notebooks/main.ipynb in place, assert every code cell has
# non-empty outputs, then zip the notebook + presentation PDF into a
# single Moodle-ready submission file.
#
# Usage: bash submission/build_zip.sh [GROUP_NUMBER]
#   GROUP_NUMBER defaults to "X" — pass your Moodle group number.
#
# Run from the repo root.

set -euo pipefail

GROUP_NUMBER="${1:-X}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure Java is discoverable for PySpark. Prefer Homebrew openjdk@11
# (matches the dev environment); fall back to whatever JAVA_HOME is set.
if [[ -z "${JAVA_HOME:-}" ]] && [[ -d /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home ]]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
fi

# Prefer the project venv if it exists, so nbconvert uses the pinned deps.
if [[ -x "$ROOT_DIR/.venv/bin/jupyter" ]]; then
    JUPYTER="$ROOT_DIR/.venv/bin/jupyter"
else
    JUPYTER="jupyter"
fi

NOTEBOOK="notebooks/main.ipynb"
PRESENTATION="presentation/presentation.pdf"
ZIP_PATH="submission/olist_bigdata_group${GROUP_NUMBER}.zip"

# 1. Sanity check inputs
[[ -f "$NOTEBOOK" ]] || { echo "MISSING: $NOTEBOOK"; exit 1; }
[[ -f "$PRESENTATION" ]] || { echo "MISSING: $PRESENTATION (export the pptx as PDF first)"; exit 1; }

# 2. Re-execute the notebook in place so all cell outputs are fresh
echo "Executing $NOTEBOOK ..."
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 "$NOTEBOOK"

# GraphFrames leaves a checkpoint dir inside outputs/ — not part of the deliverable.
rm -rf "$ROOT_DIR/outputs/_gf_checkpoints"

# 3. Assert every code cell has non-empty, non-error outputs.
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/assert_notebook_outputs.py" "$NOTEBOOK"

# 4. Build the zip — only the graded artefacts
rm -f "$ZIP_PATH"
zip -j "$ZIP_PATH" "$NOTEBOOK" "$PRESENTATION"

echo
echo "Built: $ZIP_PATH"
unzip -l "$ZIP_PATH"
