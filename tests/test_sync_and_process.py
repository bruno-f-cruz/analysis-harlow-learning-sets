"""Unit tests for the ``scripts/sync_and_process.py`` ops script.

Loaded dynamically via ``importlib`` (see ``tests/test_attach_datasets.py``
for why) since it lives outside the ``analysis`` package. ``aws_sync`` itself
is tested directly (first section); for ``main()``'s wiring, ``aws_sync``,
``process_sessions``, and ``aggregate`` are stubbed out -- exercising which
locations get synced, which local session dirs get handed to
``process_sessions``, and how the aggregator/upload flag map through.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aind_behavior_vr_foraging_packaging.export_pipeline import (
    AggregationRule,
    Aggregator,
)

_MODULE_PATH = Path(__file__).parent.parent / "scripts" / "sync_and_process.py"
_spec = importlib.util.spec_from_file_location("sync_and_process", _MODULE_PATH)
sync_and_process = importlib.util.module_from_spec(_spec)
sys.modules["sync_and_process"] = sync_and_process
_spec.loader.exec_module(sync_and_process)


@pytest.fixture(autouse=True)
def _stub_aws_cli_check(monkeypatch):
    monkeypatch.setattr(sync_and_process, "check_aws_cli_exists", lambda: None)


def test_aws_sync_builds_the_expected_command(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: captured.update(cmd=cmd, kw=kw)
    )

    sync_and_process.aws_sync("s3://aind-open-data/some-session", "/local/dest")

    assert captured["cmd"] == [
        "aws",
        "s3",
        "sync",
        "s3://aind-open-data/some-session",
        "/local/dest",
        "--exclude",
        "Behavior-Videos/*",
        "--no-progress",
        "--only-show-errors",
    ]
    assert captured["kw"] == {"check": True}


def test_aws_sync_no_sign_request_appends_the_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: captured.update(cmd=cmd))

    sync_and_process.aws_sync(
        "s3://aind-open-data/some-session", "/local/dest", no_sign_request=True
    )

    assert captured["cmd"][-1] == "--no-sign-request"


def test_aws_sync_propagates_failure(monkeypatch):
    def _raise(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", _raise)

    with pytest.raises(subprocess.CalledProcessError):
        sync_and_process.aws_sync("s3://aind-open-data/some-session", "/local/dest")


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "raw_sessions.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "attached_datasets": [
                    {
                        "id": "a1",
                        "mount": "841299_2026-06-05_19-13-19",
                        "location": "s3://aind-open-data/841299_2026-06-05_19-13-19",
                    },
                    {
                        "id": "a2",
                        "mount": "841312_2026-06-04_20-19-36",
                        "location": "s3://aind-open-data/841312_2026-06-04_20-19-36",
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(sync_and_process, "RAW_MANIFEST_PATH", path)
    monkeypatch.setattr(sync_and_process, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(
        sync_and_process, "PROCESSED_DIR", tmp_path / "data" / "processed"
    )
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])
    return path


@pytest.fixture
def stubs(monkeypatch):
    """Stub the three real steps and record every call."""
    sync_calls = []

    def _fake_aws_sync(src, dst, **kw):
        sync_calls.append({"src": src, "dst": dst, **kw})
        if not str(dst).startswith("s3://"):
            Path(dst).mkdir(parents=True, exist_ok=True)

    process_calls = []
    aggregate_calls = []
    monkeypatch.setattr(sync_and_process, "aws_sync", _fake_aws_sync)
    monkeypatch.setattr(
        sync_and_process,
        "process_sessions",
        lambda *a, **kw: process_calls.append((a, kw)),
    )
    monkeypatch.setattr(
        sync_and_process, "aggregate", lambda *a, **kw: aggregate_calls.append((a, kw))
    )
    return {"sync": sync_calls, "process": process_calls, "aggregate": aggregate_calls}


def test_syncs_every_raw_session_unsigned(manifest, stubs):
    sync_and_process.main()

    raw_syncs = [c for c in stubs["sync"] if c["src"].startswith("s3://aind-open-data")]
    assert {c["src"] for c in raw_syncs} == {
        "s3://aind-open-data/841299_2026-06-05_19-13-19",
        "s3://aind-open-data/841312_2026-06-04_20-19-36",
    }
    assert {c["dst"] for c in raw_syncs} == {
        str(sync_and_process.RAW_DIR / "841299_2026-06-05_19-13-19"),
        str(sync_and_process.RAW_DIR / "841312_2026-06-04_20-19-36"),
    }
    assert all(c["no_sign_request"] is True for c in raw_syncs)


def test_skips_a_session_whose_local_dir_already_exists(manifest, stubs):
    existing = sync_and_process.RAW_DIR / "841299_2026-06-05_19-13-19"
    existing.mkdir(parents=True)

    sync_and_process.main()

    raw_syncs = [c for c in stubs["sync"] if c["src"].startswith("s3://aind-open-data")]
    assert {c["dst"] for c in raw_syncs} == {
        str(sync_and_process.RAW_DIR / "841312_2026-06-04_20-19-36"),
    }


def test_force_sync_flag_resyncs_existing_dirs_too(manifest, stubs, monkeypatch):
    existing = sync_and_process.RAW_DIR / "841299_2026-06-05_19-13-19"
    existing.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py", "--force-sync"])

    sync_and_process.main()

    raw_syncs = [c for c in stubs["sync"] if c["src"].startswith("s3://aind-open-data")]
    assert {c["dst"] for c in raw_syncs} == {
        str(sync_and_process.RAW_DIR / "841299_2026-06-05_19-13-19"),
        str(sync_and_process.RAW_DIR / "841312_2026-06-04_20-19-36"),
    }


def test_processes_the_locally_synced_session_dirs(manifest, stubs):
    sync_and_process.main()

    ((args, kwargs),) = stubs["process"]
    dataset_paths, output_dir = args
    assert sorted(dataset_paths) == [
        sync_and_process.RAW_DIR / "841299_2026-06-05_19-13-19",
        sync_and_process.RAW_DIR / "841312_2026-06-04_20-19-36",
    ]
    assert output_dir == sync_and_process.PROCESSED_DIR
    assert kwargs == {
        "exclude_processors": ["sniffing"],
        "write_nwb": False,
        "clean": False,
    }


def test_aggregates_sites_and_session_without_cleanup(manifest, stubs):
    sync_and_process.main()

    ((args, kwargs),) = stubs["aggregate"]
    sessions_dir, output_dir, aggregator = args
    assert sessions_dir == sync_and_process.PROCESSED_DIR / "sessions"
    assert output_dir == sync_and_process.PROCESSED_DIR
    assert aggregator == Aggregator(
        rules=[
            AggregationRule("sites", cleanup=False),
            AggregationRule("session", cleanup=False),
        ]
    )
    assert kwargs == {}


def test_no_upload_by_default(manifest, stubs):
    sync_and_process.main()

    assert not any(c["dst"].startswith("s3://aind-scratch-data") for c in stubs["sync"])


def test_upload_flag_syncs_processed_dir_to_the_scratch_bucket(
    manifest, stubs, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py", "--upload"])

    sync_and_process.main()

    upload_calls = [
        c for c in stubs["sync"] if c["dst"].startswith("s3://aind-scratch-data")
    ]
    assert len(upload_calls) == 1
    assert upload_calls[0]["src"] == str(sync_and_process.PROCESSED_DIR)
    assert (
        upload_calls[0]["dst"]
        == "s3://aind-scratch-data/vr-foraging/harlow-experiments/harlow-experiment"
    )
    assert "no_sign_request" not in upload_calls[0]


def test_raises_when_manifest_is_empty(tmp_path, monkeypatch):
    path = tmp_path / "raw_sessions.json"
    path.write_text(json.dumps({"version": 1, "attached_datasets": []}))
    monkeypatch.setattr(sync_and_process, "RAW_MANIFEST_PATH", path)
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])

    with pytest.raises(SystemExit):
        sync_and_process.main()


def test_raises_when_manifest_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sync_and_process, "RAW_MANIFEST_PATH", tmp_path / "does-not-exist.json"
    )
    monkeypatch.setattr(sys, "argv", ["sync_and_process.py"])

    with pytest.raises(SystemExit):
        sync_and_process.main()
