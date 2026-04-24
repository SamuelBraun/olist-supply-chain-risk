#!/usr/bin/env bash
# Execute notebooks/00_main.ipynb end-to-end (narrative orchestrator).
# On a warm cache this runs in under a minute; cold, 10-15 minutes.
#
# Usage: bash run.sh
#
# Run from the repo root. Exits non-zero on any cell failure.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${JAVA_HOME:-}" ]] && [[ -d /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home ]]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
fi

if [[ -x "$ROOT_DIR/.venv/bin/jupyter" ]]; then
    JUPYTER="$ROOT_DIR/.venv/bin/jupyter"
else
    JUPYTER="jupyter"
fi

echo "Executing notebooks/00_main.ipynb (JAVA_HOME=$JAVA_HOME)..."
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebooks/00_main.ipynb

echo
echo "00_main.ipynb executed end-to-end."
