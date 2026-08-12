#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "aind-data-access-api>=1.9.2",
# ]
# ///
"""Query the AIND metadata DocDB and refresh raw_sessions.json.

``raw_sessions.json`` pins which raw sessions feed the processed dataset
(see ``src/analysis/preprocessing.py``) -- it's not what
``workflows/pipeline.py`` reads (that's ``data_assets.json``, which just
points at the processed dataset). To regenerate: sync these locations to
local disk, then run ``uv run python process_sessions.py --force --upload``.

Uses ``version="v2"`` since these sessions are on the newer aind-data-schema
layout, where the default ``"v1"`` returns nothing and the timestamp field
moved from ``session.session_start_time`` to
``acquisition.acquisition_start_time``. ``data_description.data_level: "raw"``
excludes derived/processed assets.

Self-contained PEP 723 script -- ``uv run attach_datasets.py`` builds its own
env, independent of this repo's uv.lock/.venv.

``SUBJECT_IDS``/``START_DATE`` below are hard-coded rather than CLI flags --
edit and re-run to change what's attached.

Usage
-----
::

    uv run attach_datasets.py
    uv run attach_datasets.py --prune   # replace the list instead of merging

New matches are added to the existing list; stale entries are kept until
`--prune`. Caveat: re-running with the same query silently re-adds a
hand-removed entry that still matches.
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
MANIFEST_PATH = Path(__file__).parent / "raw_sessions.json"
_PROJECTION = {"name": 1, "location": 1, "subject.subject_id": 1}

# Hard-coded selection criteria — same animals/cutoff date the notebook's
# now-dead `sync_raw_data` cell used to declare (see workflows/pipeline.py
# git history). Edit these directly to change which sessions get attached.
SUBJECT_IDS = ["841312", "841299", "866063", "864846", "864845"]
START_DATE = "2026-06-01"


def query_sessions(subject_ids: list[str], start_date: str) -> list[dict[str, Any]]:
    client = MetadataDbClient(
        host=API_GATEWAY_HOST, database="metadata_index", collection="data_assets", version="v2"
    )
    query = {
        "subject.subject_id": {"$in": subject_ids},
        "acquisition.acquisition_start_time": {"$gte": start_date},
        "data_description.data_level": "raw",
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
    parser.add_argument("--prune", action="store_true", help="Replace the list instead of merging")
    args = parser.parse_args()

    records = query_sessions(SUBJECT_IDS, START_DATE)
    fresh_entries = build_entries(records)

    manifest = load_manifest(MANIFEST_PATH)
    manifest["attached_datasets"] = merge(manifest["attached_datasets"], fresh_entries, args.prune)
    write_manifest(MANIFEST_PATH, manifest)

    print(
        f"raw_sessions.json now has {len(manifest['attached_datasets'])} attached datasets "
        f"({len(fresh_entries)} matched this query).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
