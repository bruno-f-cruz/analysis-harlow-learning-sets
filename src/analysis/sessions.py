import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlparse

import boto3
import pandas as pd
import polars as pl
from botocore import UNSIGNED
from botocore.config import Config


def build_attached_dataset_entries(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Map raw DocDB session records into the ``data_assets.json`` entry shape
    (``id``/``mount``/``location``), sorted by mount name for stable git diffs.

    ``mount`` mirrors the local session directory naming (``<subject>_<date>_<time>``,
    i.e. the record's ``name``) so the attachment file reads like a Code Ocean
    ``attached_datasets`` list — a human can tell what's attached at a glance.
    """
    entries = [
        {
            "id": str(record["_id"]),
            "mount": record["name"],
            "location": record["location"],
        }
        for record in records
    ]
    return sorted(entries, key=lambda entry: entry["mount"])


def load_attached_datasets(
    path: Path | str = "data_assets.json",
) -> List[Dict[str, Any]]:
    """Read the repo's ``data_assets.json`` — the durable declaration of what
    this analysis reads, not resolved via any live query at run time.
    """
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("attached_datasets", [])


def build_inputs_manifest(
    locations: List[str], client: Any = None
) -> List[Dict[str, Any]]:
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


def load_processed_table(dataset_dir: str, table: str) -> pd.DataFrame:
    """Read one processed-dataset table (e.g. ``"sites"``, ``"session"``) from
    *dataset_dir*.

    *dataset_dir* may be a local path or an ``s3://`` URI — this is the one
    place that distinction is handled. Callers (the analysis) just get a
    DataFrame back; they don't need to know or care whether the data came
    from local disk or S3, signed or unsigned. S3 reads are public/unsigned,
    matching every other read in this repo.
    """
    path = f"{dataset_dir}/{table}.parquet"
    kwargs = (
        {"storage_options": {"skip_signature": "true"}}
        if dataset_dir.startswith("s3://")
        else {}
    )
    return pl.scan_parquet(path, **kwargs).collect().to_pandas()
