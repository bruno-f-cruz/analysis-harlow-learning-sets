import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.prompt import Confirm
from rich.table import Table

MAX_CONCURRENT_DOWNLOADS = 4


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


def _sync_uris_to_local(
    uris: List[str],
    output_root: Path,
    no_sign_request: bool = False,
    confirm: bool = True,
) -> None:
    """Idempotently sync a list of S3 asset URIs to *output_root*.

    Re-running is safe: ``aws s3 sync`` skips objects already present locally.

    Parameters
    ----------
    confirm:
        When ``True`` (default), prompt interactively before downloading. Set to
        ``False`` to skip the prompt (the ``input()`` prompt is unreliable inside
        Jupyter / VS Code notebooks).
    """
    check_aws_cli_exists()

    console = Console()

    if confirm:
        message = f"About to download {len(uris)} sessions. Proceed?"
        if not Confirm.ask(message, default=False):
            logging.info("Download cancelled by user.")
            return
    else:
        logging.info("Downloading %d sessions to %s", len(uris), output_root)

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
            future = executor.submit(
                download_s3_asset, uri, output_root, no_sign_request
            )
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


def download_s3_asset(
    s3_uri: str, output_root: Path, no_sign_request: bool = False
) -> None:
    """Download a folder-like S3 asset using aws s3 sync.

    ``aws s3 sync`` only transfers objects that are new or changed relative to
    *output_root*, so calling this repeatedly is idempotent.

    Parameters
    ----------
    no_sign_request:
        When ``True``, pass ``--no-sign-request`` so anonymous access is used.
        Required for public buckets (e.g. ``aind-open-data``) when no AWS
        credentials with access to the bucket are configured.
    """

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
    if no_sign_request:
        cmd.append("--no-sign-request")

    logging.info("Starting download: %s -> %s", s3_uri, dest_dir)
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_inputs_manifest(locations: List[str], client: Any = None) -> List[Dict[str, Any]]:
    """List every object under each session's S3 prefix and record its size/etag
    (spec section 11). Defaults to anonymous/unsigned access, matching the rest
    of this module — input data is the public ``aind-open-data`` bucket, so no
    AWS credentials are needed to build this manifest.

    ``locations`` are session-level prefixes (as stored in each
    ``data_assets.json`` attachment's ``location`` field), not single object
    keys. Written to ``inputs.json`` before or at the start of processing so a
    run's exact inputs are pinned even if the underlying objects later change.
    """
    client = client or boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest: List[Dict[str, Any]] = []
    for location in locations:
        parsed = urlparse(location)
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                manifest.append(
                    {
                        "uri": f"s3://{bucket}/{obj['Key']}",
                        "session": prefix.rstrip("/"),
                        "size": obj["Size"],
                        "etag": obj["ETag"].strip('"'),
                    }
                )
    return manifest
