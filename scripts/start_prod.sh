#!/usr/bin/env bash
# Starts the progress dashboard in the background, then runs the pipeline.
# The dashboard stays up until the pipeline exits (or is killed).
set -euo pipefail

DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
ARTIFACT_URI="${ARTIFACT_URI:-/artifacts}"

# Generate a run ID here so both the pipeline and the dashboard point at the
# same progress.jsonl from the start.
export RUN_ID="${RUN_ID:-$(uv run python -c 'from analysis.run import generate_run_id; print(generate_run_id())'")}"
export PROGRESS_PATH="${ARTIFACT_URI}/runs/${RUN_ID}/progress.jsonl"

echo ""
echo "🚀  Run ID: ${RUN_ID}"
echo "📊  Progress dashboard → http://localhost:${DASHBOARD_PORT}"
echo ""

uv run uvicorn server.app:app --host 0.0.0.0 --port "${DASHBOARD_PORT}" &
DASHBOARD_PID=$!

# Kill the dashboard when this script exits for any reason
trap 'kill "${DASHBOARD_PID}" 2>/dev/null || true' EXIT

echo "🧪  Running pipeline (headless)…"
uv run python workflows/pipeline.py

echo ""
echo "✅  Pipeline complete."
