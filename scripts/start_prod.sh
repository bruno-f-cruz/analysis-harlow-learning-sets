#!/usr/bin/env bash
# Starts the progress dashboard in the background, then runs the pipeline.
# The dashboard stays up until the pipeline exits (or is killed).
set -euo pipefail

DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

echo ""
echo "📊  Progress dashboard → http://localhost:${DASHBOARD_PORT}"
echo ""

uv run uvicorn server.app:app --host 0.0.0.0 --port "${DASHBOARD_PORT}" &
DASHBOARD_PID=$!

# Kill the dashboard when this script exits for any reason
trap 'kill "${DASHBOARD_PID}" 2>/dev/null || true' EXIT

echo "🚀  Running pipeline (headless)…"
uv run python workflows/pipeline.py

echo ""
echo "✅  Pipeline complete."
