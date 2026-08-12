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
list, and existing entries are preserved even if they no longer match the
query (so a session that ages out of your date range stays attached until
you explicitly prune). Caveat: if you hand-remove an entry for a session that
still matches your query criteria, re-running this script with the same
query will silently re-add it — there is currently no way to permanently
exclude a still-matching session short of changing your query or using
--prune with different criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest atomically so a crash mid-write can't corrupt it.

    Writes to a temp file in the same directory, then replaces the target
    via ``os.replace`` (atomic on POSIX and Windows), so ``path`` always
    either holds the old complete content or the new complete content.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


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
    write_manifest(MANIFEST_PATH, manifest)

    print(
        f"data_assets.json now has {len(manifest['attached_datasets'])} attached datasets "
        f"({len(fresh_entries)} matched this query).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
