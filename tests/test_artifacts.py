import json

import boto3
import moto
import pandas as pd
import pytest

from analysis.artifacts import LocalArtifactStore, S3ArtifactStore, artifact_store_for_uri


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(root=tmp_path / "runs" / "run-001")


def test_write_json_round_trips(store):
    store.write_json("manifest.json", {"run_id": "run-001"})
    assert json.loads((store.root / "manifest.json").read_text()) == {"run_id": "run-001"}


def test_write_text_creates_parent_dirs(store):
    store.write_text("logs/application.log", "hello\n")
    assert (store.root / "logs" / "application.log").read_text() == "hello\n"


def test_write_parquet_round_trips(store):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    store.write_parquet("results/table.parquet", df)
    result = pd.read_parquet(store.root / "results" / "table.parquet")
    pd.testing.assert_frame_equal(result, df)


def test_uri_returns_local_path_string(store):
    assert store.uri("manifest.json") == str(store.root / "manifest.json")


def test_write_json_overwrites_existing_file(store):
    store.write_json("manifest.json", {"run_id": "first"})
    store.write_json("manifest.json", {"run_id": "second"})
    assert json.loads((store.root / "manifest.json").read_text()) == {"run_id": "second"}


def test_write_json_rejects_absolute_path(store, tmp_path):
    absolute_path = str(tmp_path / "outside" / "escape.json")
    with pytest.raises(ValueError):
        store.write_json(absolute_path, {})


def test_write_json_rejects_path_that_escapes_root(store):
    with pytest.raises(ValueError):
        store.write_json("../escape.json", {})


@moto.mock_aws
def test_s3_store_write_json_round_trips():
    boto3.client("s3", region_name="us-west-2").create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )
    store = S3ArtifactStore(bucket="test-bucket", prefix="runs/run-001")

    store.write_json("manifest.json", {"run_id": "run-001"})

    body = boto3.client("s3", region_name="us-west-2").get_object(
        Bucket="test-bucket", Key="runs/run-001/manifest.json"
    )["Body"].read()
    assert body == b'{\n  "run_id": "run-001"\n}'


@moto.mock_aws
def test_s3_store_uri_is_s3_scheme():
    boto3.client("s3", region_name="us-west-2").create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )
    store = S3ArtifactStore(bucket="test-bucket", prefix="runs/run-001")
    assert store.uri("manifest.json") == "s3://test-bucket/runs/run-001/manifest.json"


def test_artifact_store_for_uri_local(tmp_path):
    store = artifact_store_for_uri(str(tmp_path / "runs" / "run-001"))
    assert isinstance(store, LocalArtifactStore)


def test_artifact_store_for_uri_s3():
    store = artifact_store_for_uri("s3://my-bucket/runs/run-001")
    assert isinstance(store, S3ArtifactStore)
    assert store.bucket == "my-bucket"
    assert store.prefix == "runs/run-001"
