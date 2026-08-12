"""Sync raw sessions from raw_sessions.json, then regenerate the processed dataset.

Combines two previously-separate steps into one command: download every
session listed in the raw-data manifest (``raw_sessions.json``, refreshed via
``scripts/attach_datasets.py``) to ``data/``, then rebuild ``data/processed/``
from scratch (always -- that's the point of this script) via the same logic
``scripts/process_sessions.py`` uses.

Usage
-----
::

    uv run python scripts/sync_and_process.py
    uv run python scripts/sync_and_process.py --upload
    uv run python scripts/sync_and_process.py --exclude-processors licks position_velocity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.io import _sync_uris_to_local as sync_uris_to_local
from analysis.preprocessing import process_all
from analysis.sessions import load_attached_datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_MANIFEST_PATH = REPO_ROOT / "raw_sessions.json"
DATA_ROOT = REPO_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-processors",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Extra processor output names to skip (sniffing is always excluded)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also sync the rebuilt data/processed/ to the scratch bucket",
    )
    args = parser.parse_args()

    entries = load_attached_datasets(RAW_MANIFEST_PATH)
    if not entries:
        raise SystemExit(
            f"No raw sessions found in {RAW_MANIFEST_PATH} -- run "
            "scripts/attach_datasets.py first."
        )

    sync_uris_to_local(
        [entry["location"] for entry in entries], DATA_ROOT, no_sign_request=True, confirm=False
    )
    process_all(
        DATA_ROOT,
        force=True,  # this script's whole point is to replace the current processed dataset
        exclude_processors=frozenset(args.exclude_processors),
        upload=args.upload,
    )


if __name__ == "__main__":
    main()
