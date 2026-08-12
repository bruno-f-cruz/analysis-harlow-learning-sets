"""Unit tests for the ``scripts/sync_and_process.py`` ops script.

Loaded dynamically via ``importlib`` (see ``tests/test_attach_datasets.py``
for why) since it lives outside the ``analysis`` package. Both real steps
(``sync_uris_to_local``, ``process_all``) are stubbed out here — this only
exercises the wiring: which locations get synced, and that the processed
dataset is always force-rebuilt.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parent.parent / "scripts" / "sync_and_process.py"
_spec = importlib.util.spec_from_file_location("sync_and_process", _MODULE_PATH)
sync_and_process = importlib.util.module_from_spec(_spec)
sys.modules["sync_and_process"] = sync_and_process
_spec.loader.exec_module(sync_and_process)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "raw_sessions.json"
    path.write_text(json.dumps({
        "version": 1,
        "attached_datasets": [
            {"id": "a1", "mount": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
            {"id": "a2", "mount": "841312_2026-06-04_20-19-36", "location": "s3://aind-open-data/841312_2026-06-04_20-19-36"},
        ],
    }))
    monkeypatch.setattr(sync_and_process, "RAW_MANIFEST_PATH", path)
    monkeypatch.setattr(sync_and_process, "DATA_ROOT", tmp_path / "data")
    return path


def test_syncs_every_location_from_the_manifest(manifest, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sync_and_process,
        "sync_uris_to_local",
        lambda uris, output_root, **kw: captured.update(uris=uris, output_root=output_root),
    )
    monkeypatch.setattr(sync_and_process, "process_all", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])

    sync_and_process.main()

    assert captured["uris"] == [
        "s3://aind-open-data/841299_2026-06-05_19-13-19",
        "s3://aind-open-data/841312_2026-06-04_20-19-36",
    ]
    assert captured["output_root"] == sync_and_process.DATA_ROOT


def test_always_force_rebuilds_the_processed_dataset(manifest, monkeypatch):
    captured = {}
    monkeypatch.setattr(sync_and_process, "sync_uris_to_local", lambda *a, **kw: None)
    monkeypatch.setattr(sync_and_process, "process_all", lambda data_root, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])

    sync_and_process.main()

    assert captured["force"] is True


def test_passes_through_exclude_processors_and_upload(manifest, monkeypatch):
    captured = {}
    monkeypatch.setattr(sync_and_process, "sync_uris_to_local", lambda *a, **kw: None)
    monkeypatch.setattr(sync_and_process, "process_all", lambda data_root, **kw: captured.update(kw))
    monkeypatch.setattr(
        sys, "argv", ["sync_and_process.py", "--exclude-processors", "licks", "position_velocity", "--upload"]
    )

    sync_and_process.main()

    assert captured["exclude_processors"] == frozenset({"licks", "position_velocity"})
    assert captured["upload"] is True


def test_raises_when_manifest_is_empty(tmp_path, monkeypatch):
    path = tmp_path / "raw_sessions.json"
    path.write_text(json.dumps({"version": 1, "attached_datasets": []}))
    monkeypatch.setattr(sync_and_process, "RAW_MANIFEST_PATH", path)
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])

    with pytest.raises(SystemExit):
        sync_and_process.main()


def test_raises_when_manifest_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_and_process, "RAW_MANIFEST_PATH", tmp_path / "does-not-exist.json")
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])

    with pytest.raises(SystemExit):
        sync_and_process.main()
