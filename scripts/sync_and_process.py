"""Sync raw sessions from raw_sessions.json and (re)build the processed dataset.

1. Download every raw session listed in ``raw_sessions.json`` (refreshed via
   ``scripts/attach_datasets.py``) to ``data/raw/``. A session whose local
   directory already exists is assumed complete and skipped entirely (no
   ``aws s3 sync`` call at all -- that's what makes a no-op run near-instant);
   pass ``--force-sync`` to actually re-run ``aws s3 sync`` against every
   session regardless. Sequential, not concurrent: measured slower here, not
   faster -- see git history.
2. Run every processor except ``sniffing`` on each session
   (``aind_behavior_vr_foraging_packaging.export_pipeline.process_sessions``),
   then aggregate ``sites``/``session`` into flat, all-sessions Parquet files
   at ``data/processed/`` (``export_pipeline.aggregate``). Only sessions
   without a ``data/processed/sessions/{session_id}/`` directory are
   processed, then everything is re-aggregated; the whole step is skipped
   only when the export already covers every raw session. Pass
   ``--force-process`` to reprocess all of them anyway.
3. With ``--upload``, sync ``data/processed/`` to the scratch bucket
   ``data_assets.json`` points at.

Usage
-----
::

    uv run python scripts/sync_and_process.py
    uv run python scripts/sync_and_process.py --upload
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from aind_behavior_vr_foraging_packaging.pipeline import (
    aggregate,
    process_sessions,
)
from tqdm import tqdm

from analysis.sessions import load_attached_datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_MANIFEST_PATH = REPO_ROOT / "raw_sessions.json"
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SCRATCH_BUCKET = "aind-scratch-data"
SCRATCH_PREFIX = "vr-foraging/harlow-experiments/harlow-experiment"

#: This analysis doesn't use sniffing.
EXCLUDE_PROCESSORS = ["sniffing"]

logger = logging.getLogger(__name__)


def check_aws_cli_exists() -> None:
    """Check if AWS CLI is installed and available in PATH."""
    try:
        subprocess.run(
            ["aws", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "AWS CLI is not installed or not found in PATH. Please install it to proceed."
        )


def aws_sync(src: str, dst: str, *, no_sign_request: bool = False) -> None:
    """Run ``aws s3 sync`` from *src* to *dst*; raises ``CalledProcessError`` on failure.

    ``aws s3 sync`` only transfers objects that are new or changed, so calling
    this repeatedly (e.g. for sessions already downloaded) is idempotent.
    Behavior videos are always excluded -- this analysis doesn't use them.

    ``--no-progress``/``--only-show-errors`` matter for more than tidy output:
    without them the CLI's live progress renderer dominates the runtime even
    on a no-op sync (measured ~5x slower on an already-fully-synced session).

    Parameters
    ----------
    no_sign_request:
        When ``True``, pass ``--no-sign-request`` so anonymous access is used.
        Required for public buckets (e.g. ``aind-open-data``) when no AWS
        credentials with access to the bucket are configured.
    """
    cmd = [
        "aws",
        "s3",
        "sync",
        src,
        dst,
        "--exclude",
        "Behavior-Videos/*",
        "--no-progress",
        "--only-show-errors",
    ]
    if no_sign_request:
        cmd.append("--no-sign-request")

    logger.info("  $ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Also sync data/processed/ to s3://{SCRATCH_BUCKET}/{SCRATCH_PREFIX}",
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Re-run aws s3 sync even for sessions whose local directory already "
        "exists (by default, an existing directory is assumed complete and skipped).",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Reprocess every raw session (by default, only sessions missing "
        "from data/processed/sessions/ are processed).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    check_aws_cli_exists()  # once per run, not once per session -- ~0.9s per subprocess spawn

    entries = load_attached_datasets(RAW_MANIFEST_PATH)
    if not entries:
        raise SystemExit(
            f"No raw sessions found in {RAW_MANIFEST_PATH} -- run "
            "scripts/attach_datasets.py first."
        )

    for entry in tqdm(entries, desc="Syncing raw sessions", unit="session"):
        dest = RAW_DIR / entry["mount"]
        if dest.exists() and not args.force_sync:
            continue
        aws_sync(entry["location"], str(dest), no_sign_request=True)

    all_sessions = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    sessions_dir = PROCESSED_DIR / "sessions"
    already_processed = (
        {p.name for p in sessions_dir.iterdir() if p.is_dir()}
        if sessions_dir.exists()
        else set()
    )
    pending = [p for p in all_sessions if p.name not in already_processed]

    # An existing session.parquet is not on its own proof the export is current:
    # a later sync can add raw sessions that predate it. Only a processed
    # directory covering every raw session earns the skip.
    if (
        (PROCESSED_DIR / "session.parquet").exists()
        and not pending
        and not args.force_process
    ):
        logger.info(
            "%s already covers all %d raw sessions -- skipping "
            "process_sessions/aggregate (pass --force-process to rebuild).",
            PROCESSED_DIR / "session.parquet",
            len(all_sessions),
        )
    else:
        to_process = all_sessions if args.force_process else pending
        logger.info(
            "Processing %d of %d raw session(s) (%d already processed).",
            len(to_process),
            len(all_sessions),
            len(already_processed),
        )
        process_sessions(
            to_process,
            PROCESSED_DIR,
            exclude_processors=EXCLUDE_PROCESSORS,
            write_nwb=False,
            clean=False,
            max_workers=4,
        )
        # Always re-aggregate over the full sessions/ tree, including sessions
        # processed by earlier runs -- otherwise the flat parquets would only
        # describe this run's increment.
        logger.info("Aggregating %s -> %s", sessions_dir, PROCESSED_DIR)
        aggregate(
            sessions_dir,
            PROCESSED_DIR,
        )

    if args.upload:
        aws_sync(str(PROCESSED_DIR), f"s3://{SCRATCH_BUCKET}/{SCRATCH_PREFIX}")


if __name__ == "__main__":
    main()
