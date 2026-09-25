import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence
from urllib.parse import urlparse

import boto3
import pandas as pd
import polars as pl
from botocore import UNSIGNED
from botocore.config import Config

_S3_STORAGE_OPTIONS = {
    "skip_signature": "true",
    "aws_region": "us-west-2",
    # Per-animal loads issue dozens of large GETs; the default of 2 retries
    # turns a single dropped connection into a failed run.
    "max_retries": 10,
}


def load_attached_datasets(
    path: Path | str,
) -> List[Dict[str, Any]]:
    """Read the ``attached_datasets`` list from one of the repo's manifests:
    ``data_assets.json`` (where the shared processed dataset lives) or
    ``raw_sessions.json`` (which sessions of it this analysis uses, refreshed
    via ``scripts/attach_datasets.py``). Neither is resolved via any live
    query at run time.
    """
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("attached_datasets", [])


def build_inputs_manifest(
    locations: List[str], client: Any = None
) -> List[Dict[str, Any]]:
    """List every object under each S3 prefix and record its size/etag
    (spec section 11). Defaults to anonymous/unsigned access, matching the rest
    of this module -- no AWS credentials are needed to build this manifest.

    ``locations`` are prefixes, not single object keys. Written to
    ``inputs.json`` before or at the start of processing so a run's exact
    inputs are pinned even if the underlying objects later change.
    """
    client = client or boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest: List[Dict[str, Any]] = []
    for location in locations:
        parsed = urlparse(location)
        bucket, prefix = parsed.netloc, parsed.path.strip("/") + "/"
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


#: Files next to the dataset's tables that record what generated it.
DATASET_PROVENANCE_FILES = ("data_description.json", "processing.json")


def fetch_dataset_provenance(uri: str, client: Any = None) -> Dict[str, bytes]:
    """Raw bytes of each of :data:`DATASET_PROVENANCE_FILES` in the dataset at
    *uri*, keyed by file name -- saved verbatim with each run so it's known
    what built the tables it read. Raises if either file is missing.
    """
    if not uri.startswith("s3://"):
        return {
            name: (Path(uri) / name).read_bytes() for name in DATASET_PROVENANCE_FILES
        }
    client = client or boto3.client("s3", config=Config(signature_version=UNSIGNED))
    parsed = urlparse(uri)
    bucket, prefix = parsed.netloc, parsed.path.strip("/")
    return {
        name: client.get_object(Bucket=bucket, Key=f"{prefix}/{name}")["Body"].read()
        for name in DATASET_PROVENANCE_FILES
    }


def _scan_table(uri: str, table: str) -> pl.LazyFrame:
    kwargs = {"storage_options": _S3_STORAGE_OPTIONS} if uri.startswith("s3://") else {}
    return pl.scan_parquet(f"{uri}/{table}.parquet", **kwargs)


def _raise_if_missing(wanted: Sequence[str], found: Sequence[str], where: str) -> None:
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(set(wanted))} requested session(s) not found in "
            f"{where}: {missing}"
        )


#: Per-session tables that live only in each session's own processed asset
#: (``session.source_s3_location``), not in the aggregated dataset.
SessionTable = Literal["licks", "position_velocity", "sniffing", "software_events"]

#: Serialized-config JSON columns of the ``session`` table. They're ~98% of its
#: bytes and the analysis never reads them, so they're skipped unless asked for.
SESSION_CONFIG_COLUMNS = (
    "session",
    "rig",
    "task_logic",
    "trainer_state",
    "session_migrated",
    "rig_migrated",
    "task_logic_migrated",
)


@dataclass(frozen=True)
class Dataset:
    """The selected sessions of the shared processed VR-foraging dataset.

    ``session`` and ``sites`` are loaded eagerly (restricted to the selected
    sessions); the large per-session tables (licks, position, sniffing, ...)
    are loaded on demand via the ``load_*`` methods.
    """

    uri: str
    session: pd.DataFrame
    sites: pd.DataFrame

    @classmethod
    def load(
        cls, uri: str, session_ids: Sequence[str], *, include_config: bool = False
    ) -> "Dataset":
        """Load ``session``/``sites`` from the dataset at *uri*, restricted to
        *session_ids*.

        The filter is pushed down into the Parquet scan, so row groups that
        can't hold a requested session are never downloaded. The session
        table's :data:`SESSION_CONFIG_COLUMNS` are skipped (never downloaded)
        unless *include_config*. Raises ``ValueError`` if any requested
        session is absent from either table -- the analysis must never
        silently run on a subset.
        """
        ids = sorted(set(session_ids))
        if not ids:
            raise ValueError("No session ids requested.")
        in_ids = pl.col("session_id").is_in(ids)

        session = _scan_table(uri, "session")
        if not include_config:
            session = session.drop(SESSION_CONFIG_COLUMNS, strict=False)
        session = session.filter(in_ids).collect()
        _raise_if_missing(
            ids, session["session_id"].to_list(), f"{uri}/session.parquet"
        )
        sites = _scan_table(uri, "sites").filter(in_ids).collect()
        _raise_if_missing(
            ids, sites["session_id"].unique().to_list(), f"{uri}/sites.parquet"
        )
        return cls(uri=uri, session=session.to_pandas(), sites=sites.to_pandas())

    @classmethod
    def from_manifests(
        cls,
        data_assets: Path | str,
        raw_sessions: Path | str,
        *,
        include_config: bool = False,
    ) -> "Dataset":
        """Load the dataset ``data_assets.json`` points at, restricted to the
        sessions listed in ``raw_sessions.json`` (each entry's ``mount``)."""
        return cls.load(
            dataset_uri(data_assets),
            [entry["mount"] for entry in load_attached_datasets(raw_sessions)],
            include_config=include_config,
        )

    def load_table(
        self, table: SessionTable, session_ids: Sequence[str]
    ) -> pd.DataFrame:
        """Load *table* for each of *session_ids* and stack them.

        Each session's ``{table}.parquet`` lives under its
        ``source_s3_location``. A ``session_id`` column (categorical, since
        these tables run to ~1M rows per session) is added so the result can
        be joined back to ``session``/``sites``. Raises if a session isn't in
        this dataset; a missing file raises from the Parquet reader.
        """
        ids = sorted(set(session_ids))
        if not ids:
            raise ValueError(f"No sessions to load {table!r} for.")
        rows = self.session[self.session["session_id"].isin(ids)]
        _raise_if_missing(ids, rows["session_id"].tolist(), "this Dataset")
        session_id_dtype = pl.Enum(ids)
        frames = [
            _scan_table(location, table).with_columns(
                pl.lit(session_id, dtype=session_id_dtype).alias("session_id")
            )
            for session_id, location in zip(
                rows["session_id"], rows["source_s3_location"]
            )
        ]
        return pl.concat(frames).collect().to_pandas()

    def subject_session_ids(self, subject_id: str) -> List[str]:
        """*subject_id*'s session ids, chronologically. Raises if there are none."""
        ids = sorted(
            self.session.loc[
                self.session["subject_id"] == str(subject_id), "session_id"
            ]
        )
        if not ids:
            raise ValueError(
                f"Subject {subject_id!r} has no sessions in this Dataset; "
                f"available: {sorted(self.session['subject_id'].unique())}"
            )
        return ids

    def load_licks(self, subject_id: str) -> pd.DataFrame:
        """All of *subject_id*'s ``licks``."""
        return self.load_table("licks", self.subject_session_ids(subject_id))

    def load_position(self, subject_id: str) -> pd.DataFrame:
        """All of *subject_id*'s ``position_velocity``."""
        return self.load_table(
            "position_velocity", self.subject_session_ids(subject_id)
        )

    def load_sniffing(self, subject_id: str) -> pd.DataFrame:
        """All of *subject_id*'s ``sniffing``."""
        return self.load_table("sniffing", self.subject_session_ids(subject_id))


def dataset_uri(data_assets: Path | str) -> str:
    """The processed dataset's location from ``data_assets.json``, which must
    attach exactly one dataset."""
    attached = load_attached_datasets(data_assets)
    if len(attached) != 1:
        raise ValueError(
            f"{data_assets} must attach exactly one dataset, found {len(attached)}."
        )
    return attached[0]["location"]
