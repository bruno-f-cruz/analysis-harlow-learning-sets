import boto3
import moto

from analysis.io import build_inputs_manifest


@moto.mock_aws
def test_build_inputs_manifest_lists_every_object_under_session_prefix():
    client = boto3.client("s3", region_name="us-west-2")
    client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    client.put_object(Bucket="test-bucket", Key="session-001/data.parquet", Body=b"hello world")
    client.put_object(Bucket="test-bucket", Key="session-001/metadata.json", Body=b"{}")

    manifest = build_inputs_manifest(["s3://test-bucket/session-001"], client=client)

    assert {e["uri"] for e in manifest} == {
        "s3://test-bucket/session-001/data.parquet",
        "s3://test-bucket/session-001/metadata.json",
    }
    entry = next(e for e in manifest if e["uri"].endswith("data.parquet"))
    assert entry["size"] == 11
    assert entry["session"] == "session-001"
    assert "etag" in entry


@moto.mock_aws
def test_build_inputs_manifest_handles_multiple_session_prefixes():
    client = boto3.client("s3", region_name="us-west-2")
    client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    client.put_object(Bucket="test-bucket", Key="session-001/a.parquet", Body=b"aa")
    client.put_object(Bucket="test-bucket", Key="session-002/b.parquet", Body=b"bbb")

    manifest = build_inputs_manifest(
        ["s3://test-bucket/session-001", "s3://test-bucket/session-002"], client=client
    )
    assert {e["session"] for e in manifest} == {"session-001", "session-002"}
