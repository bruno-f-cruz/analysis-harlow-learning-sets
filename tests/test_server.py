from fastapi.testclient import TestClient

from analysis.progress import ProgressWriter
from server.app import create_app


def test_api_status_reflects_progress_file(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.jsonl"
    monkeypatch.setenv("PROGRESS_PATH", str(progress_path))
    writer = ProgressWriter(progress_path, run_id="abc123")
    writer.started(stage="run")
    writer.started(stage="preprocess", session="001", total_sessions=1)
    writer.completed(stage="preprocess", session="001")

    client = TestClient(create_app())
    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "abc123"
    assert body["completed"] == 1


def test_api_events_returns_recent_events(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.jsonl"
    monkeypatch.setenv("PROGRESS_PATH", str(progress_path))
    writer = ProgressWriter(progress_path, run_id="abc123")
    writer.started(stage="run")
    writer.log("hello")

    client = TestClient(create_app())
    response = client.get("/api/events")

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 2
    assert events[-1]["message"] == "hello"


def test_root_serves_html_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("PROGRESS_PATH", str(tmp_path / "progress.jsonl"))
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
