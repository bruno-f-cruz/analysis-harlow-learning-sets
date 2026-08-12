import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import boto3
from aind_data_access_api.document_db import MetadataDbClient
from botocore import UNSIGNED
from botocore.config import Config

API_GATEWAY_HOST = "api.allenneuraldynamics.org"

_DEFAULT_PROJECTION = {
    "name": 1,
    "created": 1,
    "location": 1,
    "subject.subject_id": 1,
    "subject.date_of_birth": 1,
}

OPEN_DATA_BUCKET = "aind-open-data"


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
    logging.debug(
        "Records: %s", json.dumps(records, indent=4, sort_keys=True, default=str)
    )
    return records


def list_open_data_sessions(
    subject_ids: List[str],
    start_date: str,
    bucket: str = OPEN_DATA_BUCKET,
    include_derived: bool = True,
) -> List[str]:
    """List S3 asset URIs for the given subjects with session date >= start_date.

    Raw AIND assets are stored at ``s3://<bucket>/<subject_id>_<YYYY-MM-DD>_<HH-MM-SS>``
    (derived assets append ``_processed_<...>``). These very recent acquisitions
    are not always indexed in DocumentDB yet, so this lists the public bucket
    directly via anonymous (unsigned) S3 access.

    Parameters
    ----------
    subject_ids:
        Subject (animal) IDs, e.g. ``["841312", "841299"]``.
    start_date:
        ISO date string (``"YYYY-MM-DD"``). Sessions whose date is on or after
        this are included. Compared lexicographically (valid for ISO dates).
    include_derived:
        When ``False``, ``*_processed_*`` derived assets are skipped and only
        raw session folders are returned.

    Returns
    -------
    List[str]
        Sorted list of ``s3://`` URIs matching the filter.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    uris: List[str] = []
    for subject_id in subject_ids:
        prefix = f"{subject_id}_"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for common_prefix in page.get("CommonPrefixes", []):
                name = common_prefix["Prefix"].rstrip("/")
                if not include_derived and "_processed_" in name:
                    continue
                # Folder name is "<subject_id>_<YYYY-MM-DD>_<...>".
                session_date = name[len(prefix) : len(prefix) + 10]
                if session_date < start_date:
                    continue
                uris.append(f"s3://{bucket}/{name}")

    logging.info(
        "Found %d S3 assets for subjects %s from %s onward",
        len(uris),
        subject_ids,
        start_date,
    )
    return sorted(uris)


def build_attached_dataset_entries(
    records: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Map raw DocDB session records into the ``data_assets.json`` entry shape
    (``id``/``mount``/``location``), sorted by mount name for stable git diffs.

    ``mount`` mirrors the local session directory naming (``<subject>_<date>_<time>``,
    i.e. the record's ``name``) so the attachment file reads like a Code Ocean
    ``attached_datasets`` list — a human can tell what's attached at a glance.
    """
    entries = [
        {"id": str(record["_id"]), "mount": record["name"], "location": record["location"]}
        for record in records
    ]
    return sorted(entries, key=lambda entry: entry["mount"])


def load_attached_datasets(path: Path | str = "data_assets.json") -> List[Dict[str, Any]]:
    """Read the repo's ``data_assets.json`` — the durable declaration of which
    sessions this analysis currently targets (refreshed via ``attach_datasets.py``,
    not by any live query at run time).
    """
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("attached_datasets", [])
