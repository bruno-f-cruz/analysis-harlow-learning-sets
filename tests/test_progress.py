import json

from analysis.progress import ProgressWriter


def test_started_writes_jsonl_event(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    writer.started(stage="preprocess", session="001")

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["run_id"] == "abc123"
    assert event["stage"] == "preprocess"
    assert event["session"] == "001"
    assert event["status"] == "started"
    assert "timestamp" in event


def test_completed_includes_elapsed_seconds(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    writer.completed(stage="preprocess", session="001", elapsed_seconds=16.2)

    event = json.loads(path.read_text().splitlines()[0])
    assert event["status"] == "completed"
    assert event["elapsed_seconds"] == 16.2


def test_events_append_across_writer_instances(tmp_path):
    path = tmp_path / "progress.jsonl"
    ProgressWriter(path, run_id="abc123").started(stage="run")
    ProgressWriter(path, run_id="abc123").completed(stage="run")

    lines = path.read_text().splitlines()
    assert len(lines) == 2


def test_error_records_message(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    writer.error(stage="preprocess", message="boom")

    event = json.loads(path.read_text().splitlines()[0])
    assert event["status"] == "error"
    assert event["message"] == "boom"


def test_log_writes_informational_event(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    writer.log("loaded 12 sessions")

    event = json.loads(path.read_text().splitlines()[0])
    assert event["status"] == "info"
    assert event["message"] == "loaded 12 sessions"
