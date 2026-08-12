"""Process all sessions in the ``data/`` directory into aggregated Parquet
outputs at ``data/processed/``.

Thin driver over ``aind_behavior_vr_foraging_packaging``'s own two-phase
export pipeline: Phase 1 (``process_sessions``) runs every processor per raw
session dir; Phase 2 (``aggregate``) concatenates into flat multi-session
outputs. Sniffing is always excluded; everything else runs by default --
pass ``--exclude-processors`` to drop more (e.g. the slow ``position_velocity``/
``licks`` tables).

Usage
-----
::

    uv run python process_sessions.py
    uv run python process_sessions.py --force
    uv run python process_sessions.py --exclude-processors licks position_velocity
    uv run python process_sessions.py --upload   # sync data/processed/ to the scratch bucket

Tables written (``data/processed/``)
-------------------------------------
sites.parquet     one row per site (== one trial); all sessions concatenated
session.parquet   one row per session

Every other table stays per-session under ``data/processed/sessions/{session_id}/``
-- only ``sites`` is flattened, since that's all the notebook reads.

Scratch bucket
--------------
``--upload`` syncs ``data/processed/`` to
``s3://aind-scratch-data/vr-foraging/harlow-experiments/harlow-experiment/``.
Reads there are public/unsigned; writing needs real AWS credentials (standard
SDK credential chain).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import boto3

from aind_behavior_vr_foraging_packaging.export_pipeline import (
    DEFAULT_AGGREGATOR,
    aggregate,
    process_sessions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

SCRATCH_BUCKET = "aind-scratch-data"
SCRATCH_PREFIX = "vr-foraging/harlow-experiments/harlow-experiment"

#: Never run, regardless of --exclude-processors (spec: this analysis doesn't use sniffing).
_ALWAYS_EXCLUDED = frozenset({"sniffing"})


def _session_dirs(data_root: Path) -> list[Path]:
    """Return subdirectories of *data_root* that look like session folders
    (``<subject_id>_<YYYY-MM-DD>_<HH-MM-SS>``).
    """
    pattern = re.compile(r"^\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    return sorted(p for p in data_root.iterdir() if p.is_dir() and pattern.match(p.name))


def upload_to_scratch(
    processed_dir: Path, bucket: str = SCRATCH_BUCKET, prefix: str = SCRATCH_PREFIX
) -> None:
    """Upload every file under *processed_dir* to ``s3://{bucket}/{prefix}/``,
    preserving its relative path.
    """
    client = boto3.client("s3")
    prefix = prefix.strip("/")
    files = sorted(p for p in processed_dir.rglob("*") if p.is_file())
    for path in files:
        key = f"{prefix}/{path.relative_to(processed_dir).as_posix()}"
        _log.info("Uploading %s -> s3://%s/%s", path, bucket, key)
        client.upload_file(str(path), bucket, key)
    _log.info("Uploaded %d file(s) to s3://%s/%s", len(files), bucket, prefix)


def process_all(
    data_root: Path,
    force: bool = False,
    exclude_processors: frozenset[str] | set[str] = frozenset(),
    upload: bool = False,
) -> None:
    """Process every session under *data_root* and write aggregated Parquet outputs.

    Skips Phase 1/2 entirely when ``data_root/processed`` already exists and
    *force* is ``False`` (all-or-nothing cache).
    """
    processed_dir = data_root / "processed"

    if processed_dir.exists() and not force:
        _log.info("Found existing output at %s. Pass --force to recompute.", processed_dir)
    else:
        session_dirs = _session_dirs(data_root)
        if not session_dirs:
            _log.error("No session directories found under %s", data_root)
            return
        _log.info("Found %d session(s) under %s", len(session_dirs), data_root)

        exclude = sorted(_ALWAYS_EXCLUDED | frozenset(exclude_processors))
        process_sessions(session_dirs, processed_dir, exclude_processors=exclude, clean=True)
        aggregate(processed_dir / "sessions", processed_dir, DEFAULT_AGGREGATOR)
        _log.info("Done. Output in %s", processed_dir)

    if upload:
        upload_to_scratch(processed_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process all sessions -> data/processed/*.parquet"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        # preprocessing.py lives at <repo_root>/src/analysis/preprocessing.py, so
        # three .parent hops back up gets us to <repo_root>/data.
        default=Path(__file__).resolve().parent.parent.parent / "data",
        help="Root directory containing session folders (default: data/ at the repo root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even when data/processed/ already exists",
    )
    parser.add_argument(
        "--exclude-processors",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Extra processor output names to skip, e.g. licks position_velocity "
        "(sniffing is always excluded regardless of this flag)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Also sync data/processed/ to s3://{SCRATCH_BUCKET}/{SCRATCH_PREFIX}/",
    )
    args = parser.parse_args()

    process_all(
        args.data_dir,
        force=args.force,
        exclude_processors=frozenset(args.exclude_processors),
        upload=args.upload,
    )


if __name__ == "__main__":
    main()
