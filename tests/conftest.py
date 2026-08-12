"""Shared pytest fixtures. No integration/e2e fixtures live here by design —
see docs/plans/2026-08-11-dockerized-analysis-environment.md Phase 12."""

from pathlib import Path

import boto3
import moto
import pytest


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """A throwaway <root>/runs/<run_id>-style directory for artifact-store tests."""
    return tmp_path / "runs" / "test-run-001"


@pytest.fixture
def s3_bucket():
    """A mocked S3 bucket (moto) for artifact/inputs-manifest tests. Yields the bucket name."""
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-west-2")
        client.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        yield "test-bucket"


@pytest.fixture
def sample_docdb_records() -> list[dict]:
    """Fake DocDB session records, shaped like real query results
    (`name`/`location`/implicit `_id`), for analysis.sessions tests."""
    return [
        {
            "_id": "9e2b1c3a",
            "name": "841312_2026-06-04_20-19-36",
            "location": "s3://aind-open-data/841312_2026-06-04_20-19-36",
        },
        {
            "_id": "c16d7200",
            "name": "841299_2026-06-05_19-13-19",
            "location": "s3://aind-open-data/841299_2026-06-05_19-13-19",
        },
    ]
