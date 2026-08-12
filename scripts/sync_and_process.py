"""Sync raw sessions from raw_sessions.json and (re)build the processed dataset.

1. Download every raw session listed in ``raw_sessions.json`` (refreshed via
   ``scripts/attach_datasets.py``) to ``data/raw/`` via ``aws s3 sync``
   (``MAX_CONCURRENT_SYNCS`` at a time) -- idempotent, so already-downloaded
   sessions are skipped.
2. Run every processor except ``sniffing`` on each session
   (``aind_behavior_vr_foraging_packaging.export_pipeline.process_sessions``),
   then aggregate ``sites``/``session`` into flat, all-sessions Parquet files
   at ``data/processed/`` (``export_pipeline.aggregate``).
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aind_behavior_vr_foraging_packaging.export_pipeline import (
    AggregationRule,
    Aggregator,
    aggregate,
    process_sessions,
)

from analysis.sessions import load_attached_datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_MANIFEST_PATH = REPO_ROOT / "raw_sessions.json"
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SCRATCH_BUCKET = "aind-scratch-data"
SCRATCH_PREFIX = "vr-foraging/harlow-experiments/harlow-experiment"

#: This analysis doesn't use sniffing.
EXCLUDE_PROCESSORS = ["sniffing"]

#: Each aws_sync call pays fixed CLI-startup + S3-listing overhead (~2s even
#: for a no-op sync) that's otherwise fully additive across sessions; running
#: a handful concurrently lets that overhead overlap instead of stacking.
MAX_CONCURRENT_SYNCS = 4


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
        "aws", "s3", "sync", src, dst,
        "--exclude", "Behavior-Videos/*",
        "--no-progress",
        "--only-show-errors",
    ]
    if no_sign_request:
        cmd.append("--no-sign-request")

    logging.info("  $ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Also sync data/processed/ to s3://{SCRATCH_BUCKET}/{SCRATCH_PREFIX}",
    )
    args = parser.parse_args()

    check_aws_cli_exists()  # once per run, not once per session -- ~0.9s per subprocess spawn

    entries = load_attached_datasets(RAW_MANIFEST_PATH)
    if not entries:
        raise SystemExit(
            f"No raw sessions found in {RAW_MANIFEST_PATH} -- run "
            "scripts/attach_datasets.py first."
        )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SYNCS) as executor:
        futures = [
            executor.submit(
                aws_sync, entry["location"], str(RAW_DIR / entry["mount"]), no_sign_request=True
            )
            for entry in entries
        ]
        for future in futures:
            future.result()  # re-raise on the first sync failure

    to_process = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    process_sessions(
        to_process,
        PROCESSED_DIR,
        exclude_processors=EXCLUDE_PROCESSORS,
        write_nwb=False,
        clean=False,
    )
    aggregate(
        PROCESSED_DIR / "sessions",
        PROCESSED_DIR,
        Aggregator(
            rules=[
                AggregationRule("sites", cleanup=False),
                AggregationRule("session", cleanup=False),
            ]
        ),
    )

    if args.upload:
        aws_sync(str(PROCESSED_DIR), f"s3://{SCRATCH_BUCKET}/{SCRATCH_PREFIX}")


if __name__ == "__main__":
    main()
