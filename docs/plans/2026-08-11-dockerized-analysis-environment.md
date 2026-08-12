# Dockerized Reproducible Analysis Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrofit the existing `analysis-harlow-learning-sets` research code (currently flat scripts: `data_loading.py`, `process_sessions.py`, `helpers.py`, `viz_helpers.py`, `demo_marimo.py`) into the reproducible, dockerized, provenance-tracked architecture described in the design spec — without rewriting the working analysis logic.

**Architecture:** Reusable logic moves into `src/analysis/` (thin wrappers around existing `data_loading.py`/`process_sessions.py`/`helpers.py` functions, not rewrites). `workflows/pipeline.py` becomes the marimo composition layer (evolved from `demo_marimo.py`). A small `progress.py` writes JSONL events consumed by a FastAPI dashboard (`server/app.py`). Every run gets an immutable `run_id`, a `manifest.json`, `selection.json`, and `inputs.json` under `artifacts/runs/<run_id>/`, written through an `artifact_store` abstraction that is local-filesystem in dev and S3 in production. Docker + `compose.yaml` + `.devcontainer/` make this runnable identically on a laptop, in Codespaces, and on an EC2 VM.

**Tech Stack:** Python 3.13, `uv`, marimo, DuckDB, boto3/`aind-data-access-api` (already in use), FastAPI + uvicorn, pytest, ruff, Docker/Docker Compose.

**Existing code this plan reuses (not replaces):**
- `data_loading.py` → source for `src/analysis/sessions.py` (catalog query/resolution) and `src/analysis/io.py` (S3 sync/download)
- `process_sessions.py` → source for `src/analysis/preprocessing.py` (trial/session table building)
- `helpers.py` / `viz_helpers.py` → source for `src/analysis/features.py` / `src/analysis/analysis.py`
- `demo_marimo.py` → evolves into `workflows/pipeline.py`

**Naming note (discovered during Task 10.1's implementation, applied retroactively throughout this doc):** the workflow notebook is named `workflows/pipeline.py`, not `workflows/analysis.py` as earlier drafts of this plan called it. Naming it `analysis.py` causes a genuine, verified runtime break: running a script prepends the script's own directory to `sys.path`, so `workflows/analysis.py` would be resolved as the module `analysis` before the real installed `src/analysis` package — breaking every `from analysis.X import Y` in the notebook with `ModuleNotFoundError: No module named 'analysis.io'; 'analysis' is not a package`. Confirmed independently via both a minimal runtime repro and `marimo check --strict`, which flags every affected cell with `error[self-import]`. All references below use the corrected `workflows/pipeline.py` name.

---

## Phase 1 — Repository Skeleton

### Task 1.1: Create the target directory structure

**Files:**
- Create: `src/analysis/__init__.py`
- Create: `workflows/` (dir)
- Create: `server/__init__.py`
- Create: `configs/` (dir)
- Create: `scripts/` (dir)
- Create: `tests/__init__.py`
- Create: `artifacts/.gitkeep`
- Create: `data/.gitkeep` (data/ already exists and is gitignored — confirm `.gitkeep` isn't itself ignored)

**Step 1:** Create empty package files/dirs listed above.

**Step 2:** Confirm `.gitignore` currently ignores `data/` contents but check whether `artifacts/` needs a new ignore rule (everything under `artifacts/runs/*` except keep the dir tracked via `.gitkeep`). Add to `.gitignore`:

```gitignore
# Run artifacts (local dev only — production artifacts live in S3)
/artifacts/runs/*
!/artifacts/.gitkeep
```

**Step 3:** Commit.

```bash
git add src workflows server configs scripts tests artifacts/.gitkeep .gitignore
git commit -m "chore: scaffold src/workflows/server/tests directory layout"
```

---

## Phase 2 — Python Environment

### Task 2.1: Add missing dev/runtime dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

The project already has `marimo[sql]`, `duckdb`, `pandas`, `pydantic`, `ruff`. Missing per spec section 4: `pytest`, `debugpy`, `fastapi`, `uvicorn`.

**Step 1:** Add a `[dependency-groups]` (or `[project.optional-dependencies]`) `dev` group and runtime deps:

```toml
dependencies = [
    "aind-behavior-vr-foraging-packaging == 0.0.3",
    "aind-behavior-vr-foraging[data] >= 1.1",
    "jupyter>=1.1.1",
    "matplotlib>=3.10.7",
    "numpy>=2.3.3",
    "pandas>=2.3.3",
    "pydantic",
    "ruff",
    "tqdm>=4.67.1",
    "scikit-learn>=1.8.0",
    "aind-data-access-api>=1.9.2",
    "pyarrow>=24.0.0",
    "marimo[sql]>=0.23.15",
    "duckdb>=1.5.5",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "boto3>=1.35",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "debugpy>=1.8",
]
```

(`boto3` is already an implicit dependency of `data_loading.py` via `import boto3` — pin it explicitly since it wasn't previously declared.)

**Step 2:** Regenerate the lockfile.

Run: `uv lock`
Expected: `uv.lock` updates, exits 0.

**Step 3:** Verify the environment still syncs.

Run: `uv sync --locked`
Expected: exits 0, no "would remove/add" drift on a second run.

**Step 4:** Commit.

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add fastapi/uvicorn/pytest/debugpy/boto3 to dependencies"
```

---

## Phase 3 — Progress Event System (`src/analysis/progress.py`)

This is pure logic with no I/O dependencies beyond a file path — build it TDD first since everything downstream (workflow, server) depends on it.

### Task 3.1: Write failing tests for the progress writer

**Files:**
- Create: `tests/test_progress.py`

```python
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
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis'` (package not installed/importable yet).

**Step 3: Make `analysis` importable**

**Files:**
- Modify: `pyproject.toml` — add:

```toml
[tool.uv]
package = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/analysis"]
```

(Correction from Task 3.1's implementation: the table is `[tool.hatch.build.targets.wheel]`, not `[tool.hatchling...]` — the latter isn't a real hatchling config key and fails the build with "Unable to determine which files to ship." Verified by reproducing the failure directly.)

Run: `uv sync --locked`
Expected: exits 0; `analysis` is now installed in editable mode.

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProgressWriter'` (module doesn't exist yet — this is the expected TDD red).

**Step 4: Implement `ProgressWriter`**

**Files:**
- Create: `src/analysis/progress.py`

```python
"""Append-only JSONL progress event writer.

Events are the machine-readable record of workflow state (see design spec
section 16). Each call appends exactly one JSON object, newline-terminated,
to the run's ``progress.jsonl``. The writer holds no in-memory state — the
event log on disk *is* the state, so a restarted process (or the progress
server) can always reconstruct current status by reading the file (section 18).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProgressWriter:
    path: Path
    run_id: str

    def _write(self, status: str, **fields: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "status": status,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def started(self, **fields: Any) -> None:
        self._write("started", **fields)

    def completed(self, **fields: Any) -> None:
        self._write("completed", **fields)

    def failed(self, **fields: Any) -> None:
        self._write("failed", **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._write("error", message=message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._write("warning", message=message, **fields)

    def log(self, message: str, **fields: Any) -> None:
        self._write("info", message=message, **fields)
```

**Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_progress.py -v`
Expected: PASS (5 passed).

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/analysis/progress.py tests/test_progress.py
git commit -m "feat: add JSONL progress event writer"
```

### Task 3.2: Write the status-reconstruction reader

The FastAPI server (Phase 8) needs to derive `{status, progress, stage, current_session}` from `progress.jsonl` alone — no separate state store (spec section 18).

**Files:**
- Modify: `tests/test_progress.py` (append)
- Modify: `src/analysis/progress.py`

**Step 1: Write the failing test**

```python
from analysis.progress import ProgressWriter, read_status


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
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_status'`.

**Step 3: Implement `read_status`**

Append to `src/analysis/progress.py`:

```python
def read_status(path: Path) -> dict[str, Any]:
    """Reconstruct current run status by replaying ``progress.jsonl``.

    No state is kept anywhere except this file, so a fresh process (e.g. the
    progress server after a container restart) gets an identical answer.
    """
    if not path.exists():
        return {"status": "unknown"}

    run_id = None
    run_status = "unknown"
    stage = None
    current_session = None
    session_started: set[str] = set()
    session_completed: set[str] = set()
    total_sessions = 0
    recent_error: str | None = None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            run_id = event.get("run_id", run_id)
            if event.get("stage") == "run":
                if event["status"] == "started":
                    run_status = "running"
                elif event["status"] == "completed":
                    run_status = "completed"
                elif event["status"] == "failed":
                    run_status = "failed"
                continue

            if event.get("stage") is not None:
                stage = event["stage"]

            session = event.get("session")
            if session is not None:
                if "total_sessions" in event:
                    total_sessions = event["total_sessions"]
                if event["status"] == "started":
                    session_started.add(session)
                    current_session = session
                elif event["status"] == "completed":
                    session_completed.add(session)

            if event["status"] == "error":
                recent_error = event.get("message")

    completed = len(session_completed)
    total = total_sessions or len(session_started)
    progress = (completed / total) if total else 0.0

    result = {
        "run_id": run_id,
        "status": run_status,
        "progress": progress,
        "completed": completed,
        "total": total,
        "stage": stage,
        "current_session": current_session,
    }
    if recent_error:
        result["error"] = recent_error
    return result
```

**Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_progress.py -v`
Expected: PASS (8 passed).

**Step 5: Commit**

```bash
git add src/analysis/progress.py tests/test_progress.py
git commit -m "feat: reconstruct run status from progress.jsonl event log"
```

---

## Phase 4 — Artifact Store Abstraction (`src/analysis/artifacts.py`)

### Task 4.1: Local-filesystem backend with TDD

**Files:**
- Create: `tests/test_artifacts.py`
- Create: `src/analysis/artifacts.py`

**Step 1: Write the failing tests**

```python
import json

import pandas as pd
import pytest

from analysis.artifacts import LocalArtifactStore


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(root=tmp_path / "runs" / "run-001")


def test_write_json_round_trips(store):
    store.write_json("manifest.json", {"run_id": "run-001"})
    assert json.loads((store.root / "manifest.json").read_text()) == {"run_id": "run-001"}


def test_write_text_creates_parent_dirs(store):
    store.write_text("logs/application.log", "hello\n")
    assert (store.root / "logs" / "application.log").read_text() == "hello\n"


def test_write_parquet_round_trips(store):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    store.write_parquet("results/table.parquet", df)
    result = pd.read_parquet(store.root / "results" / "table.parquet")
    pd.testing.assert_frame_equal(result, df)


def test_uri_returns_local_path_string(store):
    store.write_json("manifest.json", {})
    assert store.uri("manifest.json") == str(store.root / "manifest.json")
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.artifacts'`.

**Step 3: Implement `LocalArtifactStore` and the abstract interface**

```python
"""Artifact storage abstraction (design spec section 15).

Analysis code writes through ``ArtifactStore`` without knowing whether the
backend is a local directory (dev) or S3 (production). Construct the right
backend once via ``artifact_store_for_uri`` and pass it down.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactStore(ABC):
    @abstractmethod
    def write_json(self, relative_path: str, data: Any) -> None: ...

    @abstractmethod
    def write_text(self, relative_path: str, text: str) -> None: ...

    @abstractmethod
    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def uri(self, relative_path: str) -> str: ...


class LocalArtifactStore(ArtifactStore):
    """Writes under ``<root>/<relative_path>`` on the local filesystem."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        full = self.root / relative_path
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def write_json(self, relative_path: str, data: Any) -> None:
        self._resolve(relative_path).write_text(json.dumps(data, indent=2, default=str))

    def write_text(self, relative_path: str, text: str) -> None:
        self._resolve(relative_path).write_text(text)

    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._resolve(relative_path))

    def uri(self, relative_path: str) -> str:
        return str(self.root / relative_path)
```

**Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: PASS (4 passed).

**Step 5: Commit**

```bash
git add src/analysis/artifacts.py tests/test_artifacts.py
git commit -m "feat: add local-filesystem artifact store"
```

### Task 4.2: S3-backed artifact store + factory

**Files:**
- Modify: `tests/test_artifacts.py` (append, using `moto` — add as a dev dependency first)
- Modify: `src/analysis/artifacts.py`
- Modify: `pyproject.toml` (add `moto[s3]` to the `dev` group)

**Step 1: Add `moto` for S3 mocking**

```toml
dev = [
    "pytest>=8.3",
    "debugpy>=1.8",
    "moto[s3]>=5.0",
]
```

Run: `uv lock && uv sync --locked`

**Step 2: Write the failing tests**

```python
import boto3
import moto

from analysis.artifacts import S3ArtifactStore, artifact_store_for_uri, LocalArtifactStore


@moto.mock_aws
def test_s3_store_write_json_round_trips():
    boto3.client("s3", region_name="us-west-2").create_bucket(Bucket="test-bucket")
    store = S3ArtifactStore(bucket="test-bucket", prefix="runs/run-001")

    store.write_json("manifest.json", {"run_id": "run-001"})

    body = boto3.client("s3", region_name="us-west-2").get_object(
        Bucket="test-bucket", Key="runs/run-001/manifest.json"
    )["Body"].read()
    assert body == b'{\n  "run_id": "run-001"\n}'


@moto.mock_aws
def test_s3_store_uri_is_s3_scheme():
    boto3.client("s3", region_name="us-west-2").create_bucket(Bucket="test-bucket")
    store = S3ArtifactStore(bucket="test-bucket", prefix="runs/run-001")
    assert store.uri("manifest.json") == "s3://test-bucket/runs/run-001/manifest.json"


def test_artifact_store_for_uri_local(tmp_path):
    store = artifact_store_for_uri(str(tmp_path / "runs" / "run-001"))
    assert isinstance(store, LocalArtifactStore)


def test_artifact_store_for_uri_s3():
    store = artifact_store_for_uri("s3://my-bucket/runs/run-001")
    assert isinstance(store, S3ArtifactStore)
    assert store.bucket == "my-bucket"
    assert store.prefix == "runs/run-001"
```

**Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL — `ImportError: cannot import name 'S3ArtifactStore'`.

**Step 4: Implement**

Append to `src/analysis/artifacts.py`:

```python
from urllib.parse import urlparse

import boto3


class S3ArtifactStore(ArtifactStore):
    """Writes under ``s3://<bucket>/<prefix>/<relative_path>``."""

    def __init__(self, bucket: str, prefix: str, client=None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client or boto3.client("s3")

    def _key(self, relative_path: str) -> str:
        return f"{self.prefix}/{relative_path}"

    def write_json(self, relative_path: str, data: Any) -> None:
        body = json.dumps(data, indent=2, default=str)
        self._client.put_object(Bucket=self.bucket, Key=self._key(relative_path), Body=body)

    def write_text(self, relative_path: str, text: str) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._key(relative_path), Body=text)

    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> None:
        import io

        buf = io.BytesIO()
        df.to_parquet(buf)
        self._client.put_object(Bucket=self.bucket, Key=self._key(relative_path), Body=buf.getvalue())

    def uri(self, relative_path: str) -> str:
        return f"s3://{self.bucket}/{self._key(relative_path)}"


def artifact_store_for_uri(uri: str) -> ArtifactStore:
    """Build the right ``ArtifactStore`` for a local path or an ``s3://`` URI."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return S3ArtifactStore(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    return LocalArtifactStore(root=uri)
```

**Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: PASS (8 passed).

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/analysis/artifacts.py tests/test_artifacts.py
git commit -m "feat: add S3 artifact store and artifact_store_for_uri factory"
```

---

## Phase 5 — Session Selection (`src/analysis/sessions.py`)

Wraps existing `data_loading.py` query functions; adds deterministic resolution + `selection.json` (spec section 9).

### Task 5.1: Move `data_loading.py` into the package

**Files:**
- Create: `src/analysis/io.py` (S3 sync/download functions from `data_loading.py`: `check_aws_cli_exists`, `sync_open_data_sessions`, `sync_sessions_by_subject_and_date`, `sync_s3_catalog_records_to_local`, `_sync_uris_to_local`, `extract_s3_locations`, `download_s3_asset`)
- Create: `src/analysis/sessions.py` (catalog query functions from `data_loading.py`: `query_records_by_subject_and_date`, `list_open_data_sessions`, plus new resolution/selection logic below)
- Delete: `data_loading.py` (superseded)
- Modify: `demo_marimo.py` and any other importer of `data_loading` to import from `analysis.io` / `analysis.sessions` instead

**Step 1:** Move the functions verbatim (no behavior change) — split by concern as listed above. Keep docstrings and type hints intact.

**Step 2:** Update imports across the repo.

Run: `rg "data_loading" --files-with-matches` (or `grep -rl data_loading .`) — fix every hit.

**Step 3:** Sanity-check nothing else references the old module.

Run: `uv run python -c "import analysis.io, analysis.sessions"`
Expected: exits 0, no import errors.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: split data_loading.py into analysis.io and analysis.sessions"
```

### Task 5.2: Map DocDB records into attachment entries (TDD)

**Revised per user feedback (see Task 5.3):** rather than resolving a declarative query against a live catalog *inside every run*, this project pins an explicit, git-tracked `data_assets.json` (repo root) — analogous to Code Ocean's `attached_datasets` — and only refreshes it via an explicit, separate command. This task builds the pure mapping function that turns raw DocDB records into that file's entry shape; Task 5.3 builds the file and the refresh script around it.

**Files:**
- Create: `tests/test_sessions.py`
- Modify: `src/analysis/sessions.py`

**Step 1: Write the failing test**

Uses fake DocDB-record dicts (matching the projection `data_loading.py`'s query functions already use: `name`, `location`, plus Mongo's implicit `_id`) rather than a live DocDB connection, since this logic should be testable without network access.

```python
from analysis.sessions import build_attached_dataset_entries


RECORDS = [
    {"_id": "9e2b1c3a", "name": "841312_2026-06-04_20-19-36", "location": "s3://aind-open-data/841312_2026-06-04_20-19-36"},
    {"_id": "c16d7200", "name": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
]


def test_build_attached_dataset_entries_maps_id_mount_location():
    entries = build_attached_dataset_entries(RECORDS)
    assert entries[0] == {
        "id": "c16d7200",
        "mount": "841299_2026-06-05_19-13-19",
        "location": "s3://aind-open-data/841299_2026-06-05_19-13-19",
    }


def test_build_attached_dataset_entries_sorted_by_mount_for_stable_diffs():
    entries = build_attached_dataset_entries(RECORDS)
    assert [e["mount"] for e in entries] == [
        "841299_2026-06-05_19-13-19",
        "841312_2026-06-04_20-19-36",
    ]


def test_build_attached_dataset_entries_empty_input():
    assert build_attached_dataset_entries([]) == []
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_sessions.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_attached_dataset_entries'`.

**Step 3: Implement**

Append to `src/analysis/sessions.py`:

```python
from typing import Any, Mapping, Sequence


def build_attached_dataset_entries(
    records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Map raw DocDB session records into the ``data_assets.json`` entry shape
    (``id``/``mount``/``location``), sorted by mount name for stable git diffs.

    ``mount`` mirrors the local session directory naming (``<subject>_<date>_<time>``,
    i.e. the record's ``name``) so the attachment file reads like a Code Ocean
    ``attached_datasets`` list — a human can tell what's attached at a glance.
    """
    entries = [
        {"id": str(record["_id"]), "mount": record["name"], "location": record["location"]}
        for record in records
    ]
    return sorted(entries, key=lambda entry: entry["mount"])
```

**Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_sessions.py -v`
Expected: PASS (3 passed).

**Step 5: Commit**

```bash
git add src/analysis/sessions.py tests/test_sessions.py
git commit -m "feat: map DocDB records into data_assets.json entry shape"
```

### Task 5.3: `data_assets.json` manifest + self-contained `attach_datasets.py` refresh script

This is the Code-Ocean-style piece: a durable, git-tracked, top-level file declaring exactly which sessions this repo currently targets, plus a standalone script (own inline dependencies via PEP 723, run with `uv run` — **not** part of the project's `uv.lock`/`.venv`) that queries DocDB and refreshes it. Routine analysis runs then only ever read this file — they never need DocDB access, only S3 read access to each entry's pinned `location`. Refreshing the manifest is a deliberate, explicit action, run whenever the user actually wants to change what's attached — never implicit or run-time.

**Files:**
- Create: `data_assets.json` (repo root, git-tracked)
- Create: `attach_datasets.py` (repo root, PEP 723 self-contained script)
- Modify: `src/analysis/sessions.py` — add `load_attached_datasets()`
- Modify: `tests/test_sessions.py`

**Step 1:** Seed the manifest:

```json
{
  "version": 1,
  "attached_datasets": []
}
```

**Step 2: Write the failing test for the reader**

```python
import json

from analysis.sessions import load_attached_datasets


def test_load_attached_datasets_reads_manifest(tmp_path):
    path = tmp_path / "data_assets.json"
    path.write_text(json.dumps({
        "version": 1,
        "attached_datasets": [
            {"id": "c16d7200", "mount": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
        ],
    }))

    entries = load_attached_datasets(path)

    assert entries == [
        {"id": "c16d7200", "mount": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
    ]


def test_load_attached_datasets_missing_file_returns_empty_list(tmp_path):
    assert load_attached_datasets(tmp_path / "does-not-exist.json") == []
```

**Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_sessions.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_attached_datasets'`.

**Step 4: Implement**

Append to `src/analysis/sessions.py`:

```python
import json
from pathlib import Path


def load_attached_datasets(path: Path | str = "data_assets.json") -> list[dict[str, Any]]:
    """Read the repo's ``data_assets.json`` — the durable declaration of which
    sessions this analysis currently targets (refreshed via ``attach_datasets.py``,
    not by any live query at run time).
    """
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("attached_datasets", [])
```

**Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_sessions.py -v`
Expected: PASS (5 passed).

**Step 6: Write `attach_datasets.py`**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "aind-data-access-api>=1.9.2",
# ]
# ///
"""Query the AIND metadata DocDB and refresh data_assets.json.

Self-contained: the ``# /// script`` block above is PEP 723 inline metadata —
``uv run attach_datasets.py`` builds its own throwaway environment from it and
does NOT touch this repo's uv.lock/.venv. That keeps this ops command cheap to
run (no need to sync the full analysis environment) and usable even before
`uv sync` has ever been run on the project.

Usage
-----
::

    uv run attach_datasets.py --subject-ids 841299 841312 --start-date 2026-06-01
    uv run attach_datasets.py --subject-ids 841299 --start-date 2026-06-01 --prune

By default, newly matched sessions are ADDED to the existing attached_datasets
list — existing entries are kept even if they'd no longer match the query,
since detaching a dataset should be a deliberate choice. Pass --prune to
instead replace the whole list with exactly what this query returns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aind_data_access_api.document_db import MetadataDbClient

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
MANIFEST_PATH = Path(__file__).parent / "data_assets.json"
_PROJECTION = {"name": 1, "location": 1, "subject.subject_id": 1}


def query_sessions(subject_ids: list[str], start_date: str) -> list[dict[str, Any]]:
    client = MetadataDbClient(host=API_GATEWAY_HOST, database="metadata_index", collection="data_assets")
    query = {
        "subject.subject_id": {"$in": subject_ids},
        "session.session_start_time": {"$gte": start_date},
    }
    return client.retrieve_docdb_records(filter_query=query, projection=_PROJECTION)


def build_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [
        {"id": str(r["_id"]), "mount": r["name"], "location": r["location"]} for r in records
    ]
    return sorted(entries, key=lambda e: e["mount"])


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "attached_datasets": []}


def merge(existing: list[dict], fresh: list[dict], prune: bool) -> list[dict]:
    if prune:
        return sorted(fresh, key=lambda e: e["mount"])
    by_id = {e["id"]: e for e in existing}
    by_id.update({e["id"]: e for e in fresh})
    return sorted(by_id.values(), key=lambda e: e["mount"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-ids", nargs="+", required=True)
    parser.add_argument("--start-date", required=True, help="ISO date, e.g. 2026-06-01")
    parser.add_argument("--prune", action="store_true", help="Replace the list instead of merging")
    args = parser.parse_args()

    records = query_sessions(args.subject_ids, args.start_date)
    fresh_entries = build_entries(records)

    manifest = load_manifest(MANIFEST_PATH)
    manifest["attached_datasets"] = merge(manifest["attached_datasets"], fresh_entries, args.prune)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"data_assets.json now has {len(manifest['attached_datasets'])} attached datasets "
        f"({len(fresh_entries)} matched this query).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

Note: `build_entries`/`merge` here intentionally duplicate the shape of `analysis.sessions.build_attached_dataset_entries` rather than importing it — this script is self-contained by design (its own PEP 723 dependency set, no dependency on the `analysis` package or this repo's environment). If that duplication becomes annoying to keep in sync, revisit — but don't reach for a shared import by default here.

**Step 7: Manual verification** (requires real DocDB/AWS network access — not part of the automated test suite):

Run: `uv run attach_datasets.py --subject-ids 841299 --start-date 2026-06-01`
Expected: exits 0; `data_assets.json` gains entries for matching sessions; stderr prints a count. Re-running with identical args is idempotent — no duplicate entries, no changed content.

**Step 8: Commit**

```bash
git add data_assets.json attach_datasets.py src/analysis/sessions.py tests/test_sessions.py
git commit -m "feat: add data_assets.json manifest and self-contained attach_datasets.py refresh script"
```

---

## Phase 6 — Input Provenance (`src/analysis/io.py`)

### Task 6.1: `inputs.json` manifest generation (TDD)

**Files:**
- Create: `tests/test_inputs_manifest.py`
- Modify: `src/analysis/io.py`

**Correction (found while reviewing `data_loading.py` more closely):** this project's input data lives entirely in `aind-open-data`, a **public bucket accessed with unsigned/anonymous requests** — `data_loading.py` already does this everywhere (`Config(signature_version=UNSIGNED)`, `aws s3 sync --no-sign-request`). That means reading inputs needs **no AWS credentials at all**, on any machine. `build_inputs_manifest` must default to an unsigned client, not a signed `boto3.client("s3")` — otherwise it breaks for anyone without AWS credentials configured, which is the common case here. (Writing *artifacts* to a private S3 bucket in production is the one place real credentials still matter — see Phase 15/16.)

Also: each `data_assets.json` entry's `location` is a **session-level folder prefix** (e.g. `s3://aind-open-data/841299_2026-06-05_19-13-19`), not a single object key — AIND sessions are many files under one prefix. So this lists every object under each prefix rather than `head_object`-ing a single key, matching spec section 11's "for datasets consisting of many files, generate an `inputs.json` manifest containing the resolved object list."

**Step 1: Write the failing test**

```python
import boto3
import moto

from analysis.io import build_inputs_manifest


@moto.mock_aws
def test_build_inputs_manifest_lists_every_object_under_session_prefix():
    client = boto3.client("s3", region_name="us-west-2")
    client.create_bucket(Bucket="test-bucket")
    client.put_object(Bucket="test-bucket", Key="session-001/data.parquet", Body=b"hello world")
    client.put_object(Bucket="test-bucket", Key="session-001/metadata.json", Body=b"{}")

    manifest = build_inputs_manifest(["s3://test-bucket/session-001"], client=client)

    assert {e["uri"] for e in manifest} == {
        "s3://test-bucket/session-001/data.parquet",
        "s3://test-bucket/session-001/metadata.json",
    }
    entry = next(e for e in manifest if e["uri"].endswith("data.parquet"))
    assert entry["size"] == 11
    assert entry["session"] == "session-001"
    assert "etag" in entry


@moto.mock_aws
def test_build_inputs_manifest_handles_multiple_session_prefixes():
    client = boto3.client("s3", region_name="us-west-2")
    client.create_bucket(Bucket="test-bucket")
    client.put_object(Bucket="test-bucket", Key="session-001/a.parquet", Body=b"aa")
    client.put_object(Bucket="test-bucket", Key="session-002/b.parquet", Body=b"bbb")

    manifest = build_inputs_manifest(
        ["s3://test-bucket/session-001", "s3://test-bucket/session-002"], client=client
    )
    assert {e["session"] for e in manifest} == {"session-001", "session-002"}
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_inputs_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_inputs_manifest'`.

**Step 3: Implement**

Append to `src/analysis/io.py` (`Config`/`UNSIGNED` are already imported there from the Task 5.1 move — `data_loading.py` used them for the same reason):

```python
from urllib.parse import urlparse


def build_inputs_manifest(locations: list[str], client=None) -> list[dict[str, Any]]:
    """List every object under each session's S3 prefix and record its size/etag
    (spec section 11). Defaults to anonymous/unsigned access, matching the rest
    of this module — input data is the public ``aind-open-data`` bucket, so no
    AWS credentials are needed to build this manifest.

    ``locations`` are session-level prefixes (as stored in each
    ``data_assets.json`` attachment's ``location`` field), not single object
    keys. Written to ``inputs.json`` before or at the start of processing so a
    run's exact inputs are pinned even if the underlying objects later change.
    """
    client = client or boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest = []
    for location in locations:
        parsed = urlparse(location)
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                manifest.append(
                    {
                        "uri": f"s3://{bucket}/{obj['Key']}",
                        "session": prefix.rstrip("/"),
                        "size": obj["Size"],
                        "etag": obj["ETag"].strip('"'),
                    }
                )
    return manifest
```

(No `version_id` field for now — `list_objects_v2` doesn't return it; that would need `list_object_versions`, which only matters if the bucket has versioning enabled. `aind-open-data` doesn't, so this is a deliberate simplification, not an oversight — revisit only if a future bucket needs it.)

**Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_inputs_manifest.py -v`
Expected: PASS (2 passed).

**Step 5: Commit**

```bash
git add src/analysis/io.py tests/test_inputs_manifest.py
git commit -m "feat: build inputs.json provenance manifest from S3 object heads"
```

---

## Phase 7 — Run Manifest (`src/analysis/artifacts.py` or new `src/analysis/run.py`)

### Task 7.1: `run_id` generation + `manifest.json` (TDD)

**Files:**
- Create: `tests/test_manifest.py`
- Create: `src/analysis/run.py`

**Step 1: Write the failing test**

```python
import re

from analysis.run import generate_run_id, build_manifest


def test_generate_run_id_matches_expected_format():
    run_id = generate_run_id(now="2026-08-11T18:55:00", suffix="a81f42c")
    assert run_id == "20260811T185500-a81f42c"


def test_build_manifest_contains_required_fields():
    manifest = build_manifest(
        run_id="20260811T185500-a81f42c",
        started_at="2026-08-11T18:55:00Z",
        completed_at="2026-08-11T19:02:00Z",
        status="completed",
        git_commit="d5b3367",
        container_image="analysis:latest",
        python_version="3.13.0",
    )
    required = {
        "run_id", "started_at", "completed_at", "status", "git_commit",
        "container_image", "python_version", "config", "inputs", "selection",
    }
    assert required.issubset(manifest.keys())
    assert manifest["config"] == "config.yaml"
    assert manifest["inputs"] == "inputs.json"
    assert manifest["selection"] == "selection.json"
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.run'`.

**Step 3: Implement**

```python
"""Run identity and provenance manifest (design spec sections 12-13)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def generate_run_id(now: str | None = None, suffix: str | None = None) -> str:
    """``<UTC timestamp>-<short random/git suffix>``, e.g. ``20260811T185500-a81f42c``."""
    if now is None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    timestamp = datetime.fromisoformat(now).strftime("%Y%m%dT%H%M%S")
    if suffix is None:
        import secrets

        suffix = secrets.token_hex(4)
    return f"{timestamp}-{suffix}"


def build_manifest(
    *,
    run_id: str,
    started_at: str,
    completed_at: str | None,
    status: str,
    git_commit: str | None,
    container_image: str | None,
    python_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "git_commit": git_commit,
        "container_image": container_image,
        "python_version": python_version,
        "config": "config.yaml",
        "inputs": "inputs.json",
        "selection": "selection.json",
    }
    if extra:
        manifest.update(extra)
    return manifest
```

**Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: PASS (2 passed).

**Step 5: Commit**

```bash
git add src/analysis/run.py tests/test_manifest.py
git commit -m "feat: add run_id generation and manifest.json builder"
```

### Task 7.2: Capture git/host provenance helpers

**Files:**
- Modify: `src/analysis/run.py`

**Step 1:** Add helpers that shell out to `git` and read the environment, with graceful fallback (e.g. inside a container without `.git`, or a dirty tree):

```python
import platform
import socket
import subprocess


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_is_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def host_info() -> dict[str, str]:
    return {"hostname": socket.gethostname(), "python_version": platform.python_version()}
```

No test required for these (thin wrappers over subprocess/platform — covered indirectly by the e2e test in Phase 12). Manual check:

Run: `uv run python -c "from analysis.run import git_commit, git_is_dirty, host_info; print(git_commit(), git_is_dirty(), host_info())"`
Expected: prints current commit hash, `True` (since this branch has uncommitted plan-writing changes at some points), and host info — no traceback.

**Step 2: Commit**

```bash
git add src/analysis/run.py
git commit -m "feat: add git/host provenance helpers for manifest"
```

---

## Phase 8 — Configuration (`configs/default.yaml`)

### Task 8.1: Config file + loader with env override

**Files:**
- Create: `configs/default.yaml`
- Create: `tests/test_config.py`
- Create: `src/analysis/config.py`

**Note on `DATASET_URI` (revised per user feedback, twice now):** there is no single input dataset root to configure, and as of Phase 5.3 there's also no live selection query to configure — which sessions this analysis targets is pinned in the git-tracked `data_assets.json` at the repo root (refreshed explicitly via `uv run attach_datasets.py`, not read from `configs/default.yaml`). So `configs/default.yaml` carries none of the selection-query fields from earlier drafts of this plan. `DATASET_URI` is kept only as an optional override for the *local raw-data root* that `process_all()` reads from (i.e. where `sync_sessions_by_subject_and_date` already lands files, or a local fixture path in tests) — a separate concern from which sessions are attached.

**Step 1:** Write `configs/default.yaml`:

```yaml
data_root: "./data"                  # overridden by $DATASET_URI; local dir process_all() reads from
artifact_uri: "./artifacts"          # overridden by $ARTIFACT_URI; s3://bucket/runs/ in production
aws_region: "us-west-2"              # overridden by $AWS_REGION

# Which sessions are analyzed is NOT configured here — see data_assets.json
# (repo root) and attach_datasets.py.

processing:
  with_position_velocity: false
  with_licks: false
```

**Step 2: Write the failing test**

```python
import os

from analysis.config import load_config


def test_load_config_reads_yaml_defaults(tmp_path):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("data_root: ./data\nartifact_uri: ./artifacts\n")

    config = load_config(config_path)

    assert config["data_root"] == "./data"
    assert config["artifact_uri"] == "./artifacts"


def test_load_config_env_vars_override_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("data_root: ./data\nartifact_uri: ./artifacts\n")
    monkeypatch.setenv("DATASET_URI", "/mnt/override-data")

    config = load_config(config_path)

    assert config["data_root"] == "/mnt/override-data"
    assert config["artifact_uri"] == "./artifacts"  # unset env var doesn't clobber yaml
```

**Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.config'`.

**Step 4: Implement**

```python
"""Config loading (design spec section 14): YAML defaults, env-var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ENV_OVERRIDES = {
    "DATASET_URI": "data_root",
    "ARTIFACT_URI": "artifact_uri",
    "AWS_REGION": "aws_region",
}


def load_config(path: Path | str) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text()) or {}
    for env_var, config_key in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            config[config_key] = os.environ[env_var]
    return config
```

**Step 5:** Add `pyyaml` to dependencies (`pyproject.toml`), `uv lock && uv sync --locked`.

**Step 6: Run to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

**Step 7: Commit**

```bash
git add configs/default.yaml src/analysis/config.py tests/test_config.py pyproject.toml uv.lock
git commit -m "feat: add YAML config with env-var overrides"
```

---

## Phase 9 — Reusable Preprocessing/Features/Analysis Modules

This phase relocates existing logic; it is refactor-only (no new behavior). Per user direction, no integration/e2e fixture dataset is being built for this — verify the move by import + a manual run against real local `data/`, not a synthetic-data pytest.

### Task 9.1: Move `process_sessions.py` logic into `src/analysis/preprocessing.py`

**Files:**
- Create: `src/analysis/preprocessing.py` (move `_build_sessions`, `_find_logs_dir`, `_session_dirs`, `process_all`, `_packaging_version` verbatim)
- Modify: `process_sessions.py` → becomes a thin CLI shim:

```python
"""CLI entrypoint — logic lives in analysis.preprocessing."""

from analysis.preprocessing import main

if __name__ == "__main__":
    main()
```

(Add a `main()` to `preprocessing.py` that wraps the existing `argparse` block from the bottom of the old `process_sessions.py`.)

**Step 1:** Move the code verbatim into `src/analysis/preprocessing.py`, update `process_sessions.py` to the shim above.

**Step 2:** Verify the move didn't break anything by re-running the real pipeline against actual local `data/` (already-synced sessions from a prior run):

Run: `uv run python -m analysis.preprocessing` (or `uv run python process_sessions.py`, whichever the shim exposes)
Expected: exits 0; `data/processed/trials.parquet` and `data/processed/sessions.parquet` are written/updated exactly as before the move (spot-check row counts against a pre-move run if unsure).

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: move process_sessions.py logic into analysis.preprocessing"
```

### Task 9.2: Move `helpers.py` / `viz_helpers.py` into `src/analysis/features.py` / `src/analysis/plotting.py`

**Decision (confirmed with user):** the GLM fitting / bias / counterfactual *analysis itself* stays where it already lives — as marimo cells in the notebook (which becomes `workflows/pipeline.py` in Phase 10, replacing the spec's generic "reusable `analysis.py` module" idea). This phase only relocates the lower-level, non-notebook-specific utilities `helpers.py`/`viz_helpers.py` currently hold (data prep, feature/design-matrix construction, plot-styling helpers like `a_lot_of_style`) so the notebook can still import them from `src/analysis/`. **Do not** try to extract the GLM/bias/counterfactual cell logic itself into a library module — that composition is meant to stay interactive and visible in the notebook.

**Files:**
- Create: `src/analysis/features.py` — feature/design-matrix construction and data-prep helpers from `helpers.py` that are generic (not GLM-fitting-specific plotting glue)
- Create: `src/analysis/plotting.py` — plot-styling helpers from `viz_helpers.py` (`a_lot_of_style` etc.)
- Delete: `helpers.py`, `viz_helpers.py`
- Modify: `demo_marimo.py` imports (temporary — this file gets renamed to `workflows/pipeline.py` in Phase 10, so update imports here and they carry over with the `git mv`)

This is a mechanical split of what's left after leaving analysis/plotting-composition cells alone. Do it function-by-function, running `uv run python -c "import analysis.features, analysis.plotting"` after each batch to catch import errors early, rather than one giant diff.

**Step 1:** Grep every `def` in `helpers.py`/`viz_helpers.py`. For each, decide: is this called from *multiple* notebook cells / is it pure data transformation with no plotting (→ `features.py`), is it purely about matplotlib styling (→ `plotting.py`), or is it actually part of one specific analysis's composition and only used once (→ leave inline in the corresponding notebook cell, don't move it at all). When in doubt, leave it in the notebook — this phase should shrink `helpers.py`/`viz_helpers.py`, not necessarily eliminate all notebook-cell logic.

**Step 2:** Update `demo_marimo.py`'s imports (`from helpers import ...` / `from viz_helpers import ...` → `from analysis.features import ...` / `from analysis.plotting import ...`).

**Step 3:** Run the notebook headlessly to confirm nothing broke:

Run: `uv run python -m marimo run demo_marimo.py --headless` (or `uv run marimo export html demo_marimo.py -o /tmp/check.html` if `run --headless` isn't suitable — pick whichever marimo subcommand actually executes all cells; verify with `marimo --help` first since exact flag names vary by version)
Expected: exits 0, no exceptions.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: split helpers.py/viz_helpers.py into analysis.features/analysis.plotting"
```

---

## Phase 10 — Workflow (`workflows/pipeline.py`)

### Task 10.1: `demo_marimo.py` becomes `workflows/pipeline.py`

**Decision (confirmed with user):** the marimo notebook *is* the analysis — it already builds the GLM/bias/counterfactual plots as cells, so there is no separate library-level `analyze()` to call. This task is a `git mv` (preserving history) plus adding provenance/progress/artifact-writing cells around the existing analysis cells — it is **not** a rewrite of the analysis logic.

**Files:**
- Rename: `demo_marimo.py` → `workflows/pipeline.py` (`git mv`, not copy — keep blame/history)
- Modify: `workflows/pipeline.py` — add setup/provenance cells before the existing analysis cells, and a save/manifest cell after

**Step 1:** Move the file.

```bash
git mv demo_marimo.py workflows/pipeline.py
```

**Step 2:** Add new cells at the top of the notebook (before the existing `imports_marimo`/`imports_data_loading`/`sync_raw_data`/`load_and_prepare_trials` cells) for config, run identity, and progress — leave every existing analysis cell (`choice_by_block_position_pooled`, `history_glm_per_session`, `bias_by_odor_identity`, `counterfactual_matrix`, etc.) untouched:

```python
@app.cell
def imports_provenance():
    from analysis.config import load_config
    from analysis.artifacts import artifact_store_for_uri
    from analysis.progress import ProgressWriter
    from analysis.run import generate_run_id, build_manifest, git_commit, git_is_dirty, host_info
    from analysis.sessions import load_attached_datasets
    from analysis.io import build_inputs_manifest
    import os
    import sys
    from pathlib import Path
    return (
        load_config, artifact_store_for_uri, ProgressWriter, generate_run_id,
        build_manifest, git_commit, git_is_dirty, host_info, load_attached_datasets,
        build_inputs_manifest, os, sys, Path,
    )


@app.cell
def setup(load_config, generate_run_id, artifact_store_for_uri, ProgressWriter, os, Path):
    config = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    run_id = os.environ.get("RUN_ID") or generate_run_id()
    store = artifact_store_for_uri(f"{config['artifact_uri']}/runs/{run_id}")
    progress_path = Path(store.uri("progress.jsonl"))
    progress = ProgressWriter(progress_path, run_id=run_id)
    progress.started(stage="run")
    return config, run_id, store, progress, progress_path


@app.cell
def selection(load_attached_datasets, build_inputs_manifest, store, progress):
    # No live DocDB query here — data_assets.json (repo root) is the pinned,
    # git-tracked source of truth for which sessions this run analyzes.
    # Refresh it separately with `uv run attach_datasets.py ...` when needed.
    attached = load_attached_datasets(Path(__file__).parent.parent / "data_assets.json")
    store.write_json("selection.json", {"attached_datasets": attached})

    inputs = build_inputs_manifest([entry["location"] for entry in attached])
    store.write_json("inputs.json", inputs)
    progress.log(f"resolved {len(attached)} attached sessions from data_assets.json")
    return attached, inputs
```

This `selection` cell replaces `demo_marimo.py`'s existing `sync_raw_data` cell's *role* in deciding which sessions to use — `sync_raw_data` still does the actual S3→local sync/download (via `sync_sessions_by_subject_and_date` or similar from `analysis.io`), it just now iterates over `attached`'s `mount`/`location` values instead of whatever it previously used to pick sessions. Wire that cell to consume `attached` from this cell rather than re-deriving the session list itself.

Finally, add a closing cell after the last analysis cell that writes the run manifest and marks the run complete:

```python
@app.cell
def finalize(store, progress, run_id, config, build_manifest, git_commit, git_is_dirty, host_info):
    manifest = build_manifest(
        run_id=run_id,
        started_at=progress_started_at,  # capture from the `started` event timestamp, or track separately
        completed_at=None,  # fill with current UTC time
        status="completed",
        git_commit=git_commit(),
        container_image=os.environ.get("CONTAINER_IMAGE"),
        python_version=host_info()["python_version"],
        # NOT `**host_info()` here — host_info()'s "python_version" key collides
        # with the reserved key already passed above via the explicit
        # `python_version=` argument, and Task 7.1's `_reject_reserved` guard on
        # `extra` raises ValueError on any such collision (caught during Task 7.2's
        # code quality review, before this cell was ever implemented). Pull out
        # only the non-reserved field(s) instead:
        extra={"git_dirty": git_is_dirty(), "hostname": host_info()["hostname"]},
    )
    store.write_json("manifest.json", manifest)
    progress.completed(stage="run")
    return (manifest,)
```

(Fill in the exact timestamp-capture detail during implementation — either have `setup` return a `started_at` string or read it back out of `progress.jsonl`.)

**Step 3:** Confirm marimo edit-mode launch works.

Run: `uv run marimo edit workflows/pipeline.py --host 0.0.0.0 --port 2718`
Expected: server starts, notebook opens, no import errors in the first cell.

**Step 4:** Confirm non-interactive execution works (spec section 8 requirement).

Run: `uv run python workflows/pipeline.py`
Expected: exits 0; `artifacts/runs/<run_id>/manifest.json` exists afterward.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: move demo_marimo.py to workflows/pipeline.py, add provenance/progress cells"
```

### Task 10.2: `scripts/run.py` non-interactive entrypoint

**Files:**
- Create: `scripts/run.py`

Per spec this can just be the documented equivalent of `python workflows/pipeline.py` — keep it a one-line shim so there's a single source of truth:

```python
"""Non-interactive pipeline entrypoint. Equivalent to `python workflows/pipeline.py`."""

import runpy

if __name__ == "__main__":
    runpy.run_path("workflows/pipeline.py", run_name="__main__")
```

Run: `uv run python scripts/run.py`
Expected: same result as Task 10.1 Step 3.

Commit:

```bash
git add scripts/run.py
git commit -m "feat: add scripts/run.py non-interactive entrypoint"
```

---

## Phase 11 — Progress Web Server (`server/app.py`)

### Task 11.1: FastAPI app with `/api/status` and `/api/events` (TDD)

**Files:**
- Create: `tests/test_server.py`
- Create: `server/app.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.app'`.

**Step 3: Implement**

```python
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
    return [json.loads(line) for line in lines[-limit:]]


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
```

**Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (3 passed).

**Step 5:** Add `httpx` to dev deps if `TestClient` requires it (FastAPI's `TestClient` needs `httpx` installed): `uv lock && uv sync --locked`.

**Step 6: Manual verification of live server**

Run: `PROGRESS_PATH=artifacts/runs/current/progress.jsonl uv run uvicorn server.app:app --host 0.0.0.0 --port 8080`
Expected: server starts; `curl http://localhost:8080/api/status` returns JSON.

**Step 7: Commit**

```bash
git add server/app.py tests/test_server.py pyproject.toml uv.lock
git commit -m "feat: add FastAPI progress dashboard with /api/status and /api/events"
```

---

## Phase 12 — Test Scaffolding (no integration/e2e tests)

Per user direction: no synthetic fixture dataset, no full-pipeline e2e test, no `docker compose run --rm analysis` correctness test. This phase only leaves `tests/conftest.py` populated with small, reusable pytest fixtures so future unit tests (and anyone who later decides to add integration coverage) have somewhere to plug in — it adds no dataset and no end-to-end test itself.

### Task 12.1: `tests/conftest.py` with reusable fixtures

**Files:**
- Create: `tests/conftest.py`

**Step 1:** Populate with fixtures the existing unit test files (Phases 3-8) can already benefit from, generalizing the ad hoc `tmp_path`/`moto` setup that's currently duplicated inline in `test_artifacts.py` / `test_inputs_manifest.py` / `test_server.py`:

```python
"""Shared pytest fixtures. No integration/e2e fixtures live here by design —
see docs/plans/2026-08-11-dockerized-analysis-environment.md Phase 12."""

from pathlib import Path

import boto3
import moto
import pytest


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """A throwaway <root>/runs/<run_id>-style directory for artifact-store tests."""
    return tmp_path / "runs" / "test-run-001"


@pytest.fixture
def s3_bucket():
    """A mocked S3 bucket (moto) for artifact/inputs-manifest tests. Yields the bucket name."""
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-west-2")
        client.create_bucket(Bucket="test-bucket")
        yield "test-bucket"


@pytest.fixture
def sample_docdb_records() -> list[dict]:
    """Fake DocDB session records, shaped like real query results
    (`name`/`location`/implicit `_id`), for analysis.sessions tests."""
    return [
        {"_id": "9e2b1c3a", "name": "841312_2026-06-04_20-19-36", "location": "s3://aind-open-data/841312_2026-06-04_20-19-36"},
        {"_id": "c16d7200", "name": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
    ]
```

**Step 2:** Optionally fold these into the existing tests from earlier phases where it removes duplication (e.g. `test_sessions.py`'s inline `RECORDS` constant → `sample_docdb_records` fixture) — this is a nice-to-have cleanup pass, not required for the scaffolding itself to be "ready to use."

**Step 3:** Confirm the whole suite still passes with the new file present.

Run: `uv run pytest -v`
Expected: all tests pass (no new tests were added that require anything not already built).

**Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared pytest fixtures (no integration/e2e dataset)"
```

---

## Phase 13 — Docker

### Task 13.1: `Dockerfile`

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Step 1:** Write `.dockerignore`:

```text
.git
.venv
__pycache__
*.pyc
.mypy_cache
.ruff_cache
__marimo__
scratch
data
artifacts/runs
.pytest_cache
```

**Step 2:** Write `Dockerfile` (single image per spec section 5 — dev and runtime use the same image for now):

```dockerfile
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .
RUN uv sync --locked

ENV PATH="/workspace/.venv/bin:$PATH"

EXPOSE 2718 8080

CMD ["sleep", "infinity"]
```

**Step 3:** Build it.

Run: `docker build -t analysis-harlow:dev .`
Expected: exits 0.

**Step 4:** Verify the environment inside the image.

Run: `docker run --rm analysis-harlow:dev uv run pytest tests/ -v`
Expected: all tests pass inside the container (this is the real proof the lockfile-based build is self-sufficient).

**Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile for dev/runtime image"
```

---

## Phase 14 — Docker Compose

### Task 14.1: `compose.yaml`

**Files:**
- Create: `compose.yaml`

```yaml
services:
  dev:
    build:
      context: .
    volumes:
      - .:/workspace
      - ./artifacts:/artifacts
    ports:
      - "2718:2718"
      - "8080:8080"
    working_dir: /workspace
    environment:
      - DATASET_URI=${DATASET_URI:-}
      - ARTIFACT_URI=${ARTIFACT_URI:-/artifacts}
      - AWS_REGION=${AWS_REGION:-us-west-2}
      # Reading input session data needs NO AWS credentials at all — it's the
      # public aind-open-data bucket, accessed unsigned (see analysis.io).
      # Credentials only matter if ARTIFACT_URI points at a private S3 bucket
      # for writing outputs; if so they come from the host's normal credential
      # chain — never hardcode keys here. Uncomment locally if needed and not
      # using SSO/instance roles:
      # - AWS_ACCESS_KEY_ID
      # - AWS_SECRET_ACCESS_KEY
    stdin_open: true
    tty: true

  analysis:
    build:
      context: .
    volumes:
      - .:/workspace
      - ./artifacts:/artifacts
    environment:
      - DATASET_URI=${DATASET_URI:-}
      - ARTIFACT_URI=${ARTIFACT_URI:-/artifacts}
      - AWS_REGION=${AWS_REGION:-us-west-2}
    working_dir: /workspace
    command: ["uv", "run", "python", "workflows/pipeline.py"]
    profiles: ["run"]
```

(`analysis` service under a `run` profile satisfies spec section 24's `docker compose run --rm analysis` e2e requirement without making it start automatically on `docker compose up`.)

**Step 2:** Verify `up` keeps the dev container alive without running the pipeline.

Run: `docker compose up -d dev && docker compose ps`
Expected: `dev` shows `running`, no analysis executed.

**Step 3:** Verify the one-shot analysis run.

Run: `docker compose run --rm analysis`
Expected: exits 0; `./artifacts/runs/<run_id>/manifest.json` appears on the host (bind mount).

**Step 4:** Tear down.

Run: `docker compose down`

**Step 5: Commit**

```bash
git add compose.yaml
git commit -m "feat: add compose.yaml with dev and one-shot analysis services"
```

---

## Phase 15 — VS Code / Codespaces Devcontainer

### Task 15.1: `.devcontainer/devcontainer.json`

**Files:**
- Create: `.devcontainer/devcontainer.json`

```json
{
  "name": "analysis-harlow-learning-sets",
  "dockerComposeFile": "../compose.yaml",
  "service": "dev",
  "workspaceFolder": "/workspace",
  "shutdownAction": "stopCompose",
  "forwardPorts": [2718, 8080],
  "portsAttributes": {
    "2718": { "label": "marimo", "onAutoForward": "notify" },
    "8080": { "label": "progress dashboard", "onAutoForward": "notify" }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.debugpy",
        "charliermarsh.ruff"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/workspace/.venv/bin/python",
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": ["tests"]
      }
    }
  },
  "postCreateCommand": "uv sync --locked"
}
```

**Step 2:** Verify locally (VS Code, not Codespaces, is enough to check the config parses and the container attaches).

Manual check: open the repo in VS Code → "Reopen in Container" → confirm the terminal is inside `/workspace` with `.venv` active, and `python.testing` discovers `tests/`.
Expected: no devcontainer build errors; `uv run pytest` and the VS Code Test Explorer both find tests.

**Step 3:** Confirm a breakpoint in `src/analysis/features.py` is hit from both a plain script run and marimo (spec section 7 requirement) — use the VS Code "Python: Debug using launch.json" or "Attach" to a debugpy session; document exact steps in the README (Phase 16).

**Step 4: Commit**

```bash
git add .devcontainer/devcontainer.json
git commit -m "feat: add devcontainer.json for VS Code and Codespaces"
```

---

## Phase 16 — Documentation

### Task 16.1: Rewrite `README.md`

**Files:**
- Modify: `README.md`

Keep the existing "Experiment" section (Harlow's learning sets description) verbatim; add sections:

1. **Quickstart (local)** — clone → `docker compose up -d dev` → attach VS Code or `docker compose exec dev bash` → `uv run marimo edit workflows/pipeline.py --host 0.0.0.0 --port 2718` → open `localhost:2718`.
2. **Running the analysis non-interactively** — `docker compose run --rm analysis` or `uv run python workflows/pipeline.py` (document both container and bare-metal `uv run` paths).
3. **Progress dashboard** — `uv run uvicorn server.app:app --port 8080` (or via compose `dev` service) → `localhost:8080`.
4. **Codespaces** — open in Codespaces, ports 2718/8080 auto-forward, same commands.
5. **Which sessions get analyzed — `data_assets.json`** — the repo root has a git-tracked `data_assets.json` (Code-Ocean-style `attached_datasets` list of `{id, mount, location}`) that pins exactly which sessions this analysis targets. To change it, run `uv run attach_datasets.py --subject-ids <ids> --start-date <date>` (a self-contained script — its dependencies are declared inline via PEP 723 and installed into a throwaway env by `uv run`, independent of this repo's own `uv.lock`) and commit the diff. `workflows/pipeline.py` only ever reads this file — it never queries DocDB itself.
6. **Configuration** — `configs/default.yaml` holds `data_root`, `artifact_uri`, `aws_region`, and processing flags; env var overrides (`DATASET_URI` for the local raw-data root — *not* related to session selection, which lives in `data_assets.json` above — `ARTIFACT_URI`, `AWS_REGION`, `RUN_ID`).
7. **AWS credentials — you probably don't need any.** Input session data lives in `aind-open-data`, a public S3 bucket accessed anonymously (unsigned requests) — reading inputs works identically with zero AWS setup on a laptop, in Codespaces, or on EC2. Credentials only come into play if `ARTIFACT_URI` is pointed at a private S3 bucket for writing run outputs in production; in that case, use the normal AWS SDK credential chain (local `~/.aws/config`, or an IAM instance role on EC2) — never keys in the repo.
8. **Run artifacts & provenance** — where `artifacts/runs/<run_id>/` lives, what `manifest.json`/`selection.json`/`inputs.json` mean, how to reproduce a past run.
9. **EC2 deployment** — `git clone && docker compose up -d`; if writing artifacts to S3, attach an IAM instance role scoped to that bucket (input reads need no role at all); recommend SSH tunnel/VPN for port access rather than public exposure (spec section 22) — do not implement auth, just document the tunnel command: `ssh -L 2718:localhost:2718 -L 8080:localhost:8080 user@ec2-host`.
10. **Testing** — `uv run pytest` (unit tests only — no integration/e2e suite by design; see Phase 12).

**Step 2:** Commit.

```bash
git add README.md
git commit -m "docs: document docker/codespaces/EC2 workflows and provenance model"
```

---

## Phase 17 — Final Verification Pass

### Task 17.1: Full acceptance walkthrough (manual, per spec section 28)

Run through and confirm each of the three flows end-to-end; fix anything that breaks before considering this plan done:

1. **Local:** `git clone` (fresh dir) → open in VS Code → "Reopen in Container" → `uv run marimo edit workflows/pipeline.py --host 0.0.0.0 --port 2718` → sessions come from the repo's committed `data_assets.json` → run → `localhost:8080` shows progress → check `./artifacts/runs/<run_id>/`.
2. **Re-run reproducibility:** re-run without touching `data_assets.json` → confirm a new `run_id` but identical `selection.json`/`inputs.json` content (same attached sessions, read from the same static file — not re-queried against DocDB) and no mutation of the prior run's directory.
3. **Container disposability:** `docker compose down -v` then `docker compose up -d dev` again → confirm `./artifacts/runs/` (host bind mount) still has every prior run untouched.
4. **EC2 (or an EC2-shaped local simulation if no test instance is available):** confirm `docker compose up -d` alone (no VS Code) can run `docker compose run --rm analysis` successfully using only an IAM-role-equivalent credential source.

**Step 2:** Run the full test suite one last time.

Run: `uv run pytest -v`
Expected: all tests pass.

Run: `uv run ruff check .`
Expected: no lint errors (fix any introduced during the refactor phases).

**Step 3:** Final commit / open PR per `finishing-a-development-branch` skill once everything above is green.

---

## Decisions Resolved With User

1. **Notebook structure:** `demo_marimo.py` is renamed (`git mv`) to `workflows/pipeline.py`, not duplicated. The GLM/bias/counterfactual analysis logic stays as marimo cells in that notebook; only lower-level, non-analysis utilities move into `src/analysis/features.py`/`plotting.py`. See Phase 9.2/10.1.
2. **Test scope:** no synthetic fixture dataset and no integration/e2e test — `tests/conftest.py` is left populated with small reusable unit-test fixtures only, ready for future tests to use. See Phase 12.
3. **Input dataset location / session selection:** no static `DATASET_URI` bucket, and no live per-run DocDB query either. Which sessions this analysis targets is pinned in a git-tracked `data_assets.json` at the repo root (Code-Ocean-style `attached_datasets` list of `{id, mount, location}`), refreshed only via the explicit, self-contained `uv run attach_datasets.py` command. Routine runs read `data_assets.json`, never DocDB. See Phase 5.3.

## Still Open

None blocking. Per user direction: assume everything is unauthenticated for now — this plan implements and exercises only the local-filesystem path end-to-end (`LocalArtifactStore`, `ARTIFACT_URI=./artifacts`); no real production bucket name or IAM role is needed to complete it. `S3ArtifactStore` (Phase 4.2) is still built and unit-tested against moto since the abstraction is part of the design, it's just not wired to a real bucket yet. The commented-out `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` lines stay in `compose.yaml` (Phase 14) as an escape hatch for whoever eventually needs them, but nothing in this plan requires uncommenting them. Revisit `ARTIFACT_URI`/IAM details only when actual EC2 deployment with S3-backed artifacts happens.
