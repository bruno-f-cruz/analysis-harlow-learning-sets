import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from aind_data_access_api.document_db import MetadataDbClient
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.prompt import Confirm
from rich.table import Table

MAX_CONCURRENT_DOWNLOADS = 4
API_GATEWAY_HOST = "api.allenneuraldynamics.org"

_DEFAULT_PROJECTION = {
    "name": 1,
    "created": 1,
    "location": 1,
    "subject.subject_id": 1,
    "subject.date_of_birth": 1,
}


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
        raise RuntimeError("AWS CLI is not installed or not found in PATH. Please install it to proceed.")



def query_records_by_subject_and_date(
    subject_ids: List[str],
    start_date: str,
    projection: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Query docdb for session records matching the given subject IDs and start date.

    Parameters
    ----------
    subject_ids:
        List of subject (animal) IDs to filter on.
    start_date:
        ISO-format date string (``"YYYY-MM-DD"``). Only sessions whose
        ``session.session_start_time`` is on or after this date are returned.
    projection:
        Optional MongoDB projection dict. Defaults to the standard fields used
        throughout this module.

    Returns
    -------
    List[Dict[str, Any]]
        Raw records returned from the DocumentDB.
    """
    if projection is None:
        projection = _DEFAULT_PROJECTION

    filter_query = {
        "subject.subject_id": {"$in": subject_ids},
        "session.session_start_time": {"$gte": start_date},
    }

    docdb_api_client = MetadataDbClient(host=API_GATEWAY_HOST)
    records = docdb_api_client.retrieve_docdb_records(
        filter_query=filter_query,
        projection=projection,
    )

    logging.info(
        "Found %d records for subjects %s from %s onward",
        len(records),
        subject_ids,
        start_date,
    )
    logging.debug("Records: %s", json.dumps(records, indent=4, sort_keys=True, default=str))
    return records


def sync_sessions_by_subject_and_date(
    subject_ids: List[str],
    start_date: str,
    output_root: Path,
) -> None:
    """Download all sessions matching the given subject IDs and start date.

    Queries the AIND DocumentDB for sessions whose
    ``session.session_start_time`` is on or after *start_date* and whose
    ``subject.subject_id`` is in *subject_ids*, then syncs the matching S3
    assets to *output_root*.

    Parameters
    ----------
    subject_ids:
        List of subject (animal) IDs to filter on, e.g. ``["841312", "841299"]``.
    start_date:
        ISO-format date string (``"YYYY-MM-DD"``), e.g. ``"2026-06-01"``.
    output_root:
        Local directory where the S3 assets will be synced.
    """
    records = query_records_by_subject_and_date(
        subject_ids=subject_ids,
        start_date=start_date,
    )

    if not records:
        logging.warning(
            "No records found for subjects %s from %s onward. Nothing to download.",
            subject_ids,
            start_date,
        )
        return

    sync_s3_catalog_records_to_local(records, output_root)


def sync_s3_catalog_records_to_local(records: Dict[str, Any], output_root: Path) -> None:
    check_aws_cli_exists()
    uris: list[str] = []
    for record in records:
        s3_uris = extract_s3_locations(record)
        if not s3_uris:
            raise ValueError(f"No S3 URIs found in record: {record}")
        uris.extend(s3_uris)

    console = Console()

    message = f"About to download {len(uris)} sessions. Proceed?"

    if not Confirm.ask(message, default=False):
        logging.info("Download cancelled by user.")
        return

    statuses: dict[str, str] = {uri: "Not started" for uri in uris}

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    )
    overall_task = progress.add_task("Overall", total=len(uris))

    def render_status_table() -> Table:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Asset")
        table.add_column("Status")

        for uri in uris:
            status = statuses.get(uri, "Not started")
            if status == "Not started":
                style = "dim"
            elif status == "Downloading":
                style = "cyan"
            elif status == "Done":
                style = "green"
            elif status == "Error":
                style = "red"
            else:
                style = "white"

            table.add_row(uri, f"[{style}]{status}[/{style}]")

        return table

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        futures: dict[object, str] = {}

        for uri in uris:
            statuses[uri] = "Downloading"
            future = executor.submit(download_s3_asset, uri, output_root)
            futures[future] = uri

        with Live(
            Group(progress, render_status_table()),
            console=console,
            refresh_per_second=4,
        ) as live:
            for future in as_completed(futures):
                uri = futures[future]
                try:
                    future.result()
                    statuses[uri] = "Done"
                except subprocess.CalledProcessError as exc:
                    statuses[uri] = "Error"
                    logging.error("Failed to download %s: %s", uri, exc)
                except Exception as exc:
                    statuses[uri] = "Error"
                    logging.exception("Unexpected error downloading %s: %s", uri, exc)

                progress.update(overall_task, advance=1)
                live.update(Group(progress, render_status_table()))


def extract_s3_locations(record: Dict[str, Any]) -> List[str]:
    """Best-effort extraction of S3 locations from a record's `location` field."""

    locations: List[str] = []
    raw_location = record.get("location")

    def handle_one(item: Any) -> None:
        if isinstance(item, str) and item.startswith("s3://"):
            locations.append(item)
            return

        if isinstance(item, dict):
            uri = None
            if "s3_uri" in item and isinstance(item["s3_uri"], str):
                uri = item["s3_uri"]
            elif "bucket" in item and "key" in item:
                bucket = item["bucket"]
                key = item["key"]
                if isinstance(bucket, str) and isinstance(key, str):
                    uri = f"s3://{bucket}/{key}"

            if uri and isinstance(uri, str) and uri.startswith("s3://"):
                locations.append(uri)

    if isinstance(raw_location, list):
        for sub in raw_location:
            if isinstance(sub, list):
                for subsub in sub:
                    handle_one(subsub)
            else:
                handle_one(sub)
    else:
        handle_one(raw_location)

    return locations


def download_s3_asset(s3_uri: str, output_root: Path) -> None:
    """Download a folder-like S3 asset using aws s3 sync."""

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        logging.warning("Skipping non-s3 URI: %s", s3_uri)
        return

    prefix_name = Path(parsed.path).name or parsed.netloc
    dest_dir = output_root / prefix_name
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aws",
        "s3",
        "sync",
        s3_uri,
        str(dest_dir),
        "--exclude",
        "Behavior-Videos/*",
        "--no-progress",
        "--only-show-errors",
    ]

    logging.info("Starting download: %s -> %s", s3_uri, dest_dir)
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )



