import json

import pytest

from analysis.progress import ProgressWriter, read_status


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

    writer.error("boom", stage="preprocess")

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


def test_failed_records_status(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    writer.failed(stage="preprocess", reason="disk full")

    event = json.loads(path.read_text().splitlines()[0])
    assert event["status"] == "failed"
    assert event["reason"] == "disk full"


def test_warning_records_message(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    writer.warning("low disk space", stage="preprocess")

    event = json.loads(path.read_text().splitlines()[0])
    assert event["status"] == "warning"
    assert event["message"] == "low disk space"
    assert event["stage"] == "preprocess"


@pytest.mark.parametrize("reserved_key", ["timestamp", "run_id", "status"])
def test_reserved_key_in_fields_raises_on_started(tmp_path, reserved_key):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    with pytest.raises(ValueError):
        writer.started(**{reserved_key: "spoofed"})

    assert not path.exists()


@pytest.mark.parametrize("reserved_key", ["timestamp", "run_id", "status"])
def test_reserved_key_in_fields_raises_on_completed(tmp_path, reserved_key):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    with pytest.raises(ValueError):
        writer.completed(**{reserved_key: "spoofed"})

    assert not path.exists()


@pytest.mark.parametrize("reserved_key", ["timestamp", "run_id", "status"])
def test_reserved_key_in_fields_raises_on_failed(tmp_path, reserved_key):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    with pytest.raises(ValueError):
        writer.failed(**{reserved_key: "spoofed"})

    assert not path.exists()


@pytest.mark.parametrize("reserved_key", ["timestamp", "run_id", "status", "message"])
def test_reserved_key_in_fields_raises_on_error(tmp_path, reserved_key):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    with pytest.raises(ValueError):
        writer.error("boom", **{reserved_key: "spoofed"})

    assert not path.exists()


@pytest.mark.parametrize("reserved_key", ["timestamp", "run_id", "status", "message"])
def test_reserved_key_in_fields_raises_on_warning(tmp_path, reserved_key):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    with pytest.raises(ValueError):
        writer.warning("careful", **{reserved_key: "spoofed"})

    assert not path.exists()


@pytest.mark.parametrize("reserved_key", ["timestamp", "run_id", "status", "message"])
def test_reserved_key_in_fields_raises_on_log(tmp_path, reserved_key):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")

    with pytest.raises(ValueError):
        writer.log("note", **{reserved_key: "spoofed"})

    assert not path.exists()


def test_read_status_reconstructs_from_event_log(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")
    writer.started(stage="run")
    writer.started(stage="preprocess", session="001", total_sessions=2)
    writer.completed(stage="preprocess", session="001")
    writer.started(stage="preprocess", session="002", total_sessions=2)

    status = read_status(path)

    assert status["run_id"] == "abc123"
    assert status["status"] == "running"
    assert status["stage"] == "preprocess"
    assert status["current_session"] == "002"
    assert status["completed"] == 1
    assert status["total"] == 2
    assert status["progress"] == 0.5


def test_read_status_reports_completed_run(tmp_path):
    path = tmp_path / "progress.jsonl"
    writer = ProgressWriter(path, run_id="abc123")
    writer.started(stage="run")
    writer.completed(stage="run")

    status = read_status(path)
    assert status["status"] == "completed"


def test_read_status_missing_file_returns_unknown(tmp_path):
    status = read_status(tmp_path / "does-not-exist.jsonl")
    assert status["status"] == "unknown"
