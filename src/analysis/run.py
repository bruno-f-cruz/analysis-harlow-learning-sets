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
