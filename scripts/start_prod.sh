#!/usr/bin/env bash
# Runs the pipeline and tees all output to <run_dir>/out.log.
set -euo pipefail

ARTIFACT_URI="${ARTIFACT_URI:-/artifacts}"

export RUN_ID="${RUN_ID:-$(uv run python -c 'from analysis.run import generate_run_id; print(generate_run_id())')}"

LOG_DIR="${ARTIFACT_URI}/runs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

echo ""
echo "🚀  Run ID: ${RUN_ID}"
echo "📁  Artifacts → ${LOG_DIR}"
echo ""

uv run python workflows/pipeline.py 2>&1 | tee "${LOG_DIR}/out.log"
