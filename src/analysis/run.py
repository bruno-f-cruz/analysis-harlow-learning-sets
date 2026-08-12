"""Run identity and provenance manifest (design spec sections 12-13)."""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any

#: Keys owned by :func:`build_manifest`'s own fixed/required fields. Callers
#: may not supply these via ``extra`` — doing so raises ``ValueError`` instead
#: of silently overwriting the manifest's own values.
_RESERVED_MANIFEST_KEYS = {
    "run_id",
    "started_at",
    "completed_at",
    "status",
    "git_commit",
    "container_image",
    "python_version",
    "config",
    "inputs",
    "selection",
}


def _reject_reserved(extra: dict[str, Any]) -> None:
    reserved = _RESERVED_MANIFEST_KEYS & extra.keys()
    if reserved:
        raise ValueError(
            f"reserved manifest field name(s) {sorted(reserved)} may not be passed via 'extra'"
        )


def generate_run_id(now: str | None = None, suffix: str | None = None) -> str:
    """``<UTC timestamp>-<short random/git suffix>``, e.g. ``20260811T185500-a81f42c``."""
    if now is None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        timestamp = datetime.fromisoformat(now).strftime("%Y%m%dT%H%M%S")
    except ValueError as exc:
        raise ValueError(f"invalid 'now' timestamp {now!r}: {exc}") from exc
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
    """Build the run's provenance manifest (design spec sections 12-13).

    Returns a dict with the manifest's fixed/required fields: ``run_id``,
    ``started_at``, ``completed_at``, ``status``, ``git_commit``,
    ``container_image``, ``python_version``, ``config``, ``inputs``, and
    ``selection``.

    ``extra``, if given, is merged into the result to add caller-supplied
    fields. Callers must not use ``extra`` to overwrite any of the fixed
    fields above — doing so raises ``ValueError`` instead of silently
    clobbering the manifest's own values.
    """
    if extra:
        _reject_reserved(extra)
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


def git_commit() -> str | None:
    """Current git commit hash, or ``None`` if unavailable (e.g. no ``.git``)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_is_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes, or ``None`` if unavailable."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def host_info() -> dict[str, str]:
    """Hostname and Python version for the current environment."""
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }
