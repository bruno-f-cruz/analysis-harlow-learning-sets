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
