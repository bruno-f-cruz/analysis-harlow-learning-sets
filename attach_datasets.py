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
