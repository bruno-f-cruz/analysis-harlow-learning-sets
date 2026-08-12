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

#: Keys owned by :meth:`ProgressWriter._write` (and, for the message-taking
#: methods, by their ``message`` parameter). Callers may not supply these
#: via ``**fields`` — doing so raises ``ValueError`` instead of silently
#: overwriting the writer's own values or raising an opaque ``TypeError``
#: from a duplicate-keyword-argument collision.
_RESERVED_FIELD_KEYS = {"timestamp", "run_id", "status"}


@dataclass
class ProgressWriter:
    path: Path
    run_id: str

    def _write(self, status: str, **fields: Any) -> None:
        """Append one JSON event line to the progress log.

        ``timestamp``, ``run_id``, and ``status`` are reserved for this
        method's own values; callers must not pass them via ``fields``
        (enforced by the public methods below before they call this one).
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "status": status,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    @staticmethod
    def _reject_reserved(fields: dict[str, Any], *extra_reserved: str) -> None:
        reserved = (_RESERVED_FIELD_KEYS | set(extra_reserved)) & fields.keys()
        if reserved:
            raise ValueError(
                f"reserved field name(s) {sorted(reserved)} may not be passed via **fields"
            )

    def started(self, **fields: Any) -> None:
        self._reject_reserved(fields)
        self._write("started", **fields)

    def completed(self, **fields: Any) -> None:
        self._reject_reserved(fields)
        self._write("completed", **fields)

    def failed(self, **fields: Any) -> None:
        self._reject_reserved(fields)
        self._write("failed", **fields)

    def error(self, message: str, /, **fields: Any) -> None:
        self._reject_reserved(fields, "message")
        self._write("error", message=message, **fields)

    def warning(self, message: str, /, **fields: Any) -> None:
        self._reject_reserved(fields, "message")
        self._write("warning", message=message, **fields)

    def log(self, message: str, /, **fields: Any) -> None:
        self._reject_reserved(fields, "message")
        self._write("info", message=message, **fields)


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
