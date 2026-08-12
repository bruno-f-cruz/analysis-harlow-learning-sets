"""Minimal progress dashboard (design spec section 17-18).

State is derived entirely from ``progress.jsonl`` on each request — no
in-memory run state — so a container restart doesn't lose the dashboard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from analysis.progress import read_status


def _progress_path() -> Path:
    return Path(os.environ.get("PROGRESS_PATH", "artifacts/runs/current/progress.jsonl"))


def _read_events(path: Path, limit: int = 100) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated/malformed trailing line (e.g. the writer was
            # killed mid-``fh.write``) should not crash the endpoint —
            # skip it and keep the well-formed events around it.
            continue
    return events


_DASHBOARD_HTML = """
<!doctype html>
<html><head><title>Analysis Progress</title></head>
<body>
  <h1>Analysis Progress</h1>
  <pre id="status">loading...</pre>
  <h2>Recent events</h2>
  <pre id="events">loading...</pre>
  <script>
    async function refresh() {
      const status = await (await fetch('/api/status')).json();
      document.getElementById('status').textContent = JSON.stringify(status, null, 2);
      const events = await (await fetch('/api/events')).json();
      document.getElementById('events').textContent = JSON.stringify(events, null, 2);
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body></html>
"""


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def root():
        return _DASHBOARD_HTML

    @app.get("/api/status")
    def status():
        return read_status(_progress_path())

    @app.get("/api/events")
    def events():
        return _read_events(_progress_path())

    return app


app = create_app()
