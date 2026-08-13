"""Unit tests for ``analysis.sessions.load_processed_table``.

This is the one place the "how do we get the data" concern (local disk vs.
S3, signed vs. unsigned) is handled -- the analysis notebook just calls this
and gets a pandas DataFrame back.
"""

import pandas as pd
import pytest

from analysis.sessions import load_processed_table


def test_reads_a_local_table_as_a_pandas_dataframe(tmp_path):
    df = pd.DataFrame({"session_id": ["a", "b"], "has_choice": [True, False]})
    df.to_parquet(tmp_path / "sites.parquet", index=False)

    result = load_processed_table(str(tmp_path), "sites")

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df)


def test_s3_path_uses_unsigned_storage_options(monkeypatch):
    captured = {}

    class _FakeLazyFrame:
        def collect(self):
            return self

        def to_pandas(self):
            return pd.DataFrame()

    def fake_scan_parquet(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return _FakeLazyFrame()

    monkeypatch.setattr("analysis.sessions.pl.scan_parquet", fake_scan_parquet)

    load_processed_table("s3://aind-scratch-data/some/dataset", "session")

    assert captured["path"] == "s3://aind-scratch-data/some/dataset/session.parquet"
    assert captured["kwargs"] == {"storage_options": {"skip_signature": "true"}}


def test_local_path_passes_no_storage_options(monkeypatch, tmp_path):
    captured = {}

    class _FakeLazyFrame:
        def collect(self):
            return self

        def to_pandas(self):
            return pd.DataFrame()

    def fake_scan_parquet(path, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeLazyFrame()

    monkeypatch.setattr("analysis.sessions.pl.scan_parquet", fake_scan_parquet)

    load_processed_table(str(tmp_path), "sites")

    assert captured["kwargs"] == {}


@pytest.mark.parametrize("table", ["sites", "session"])
def test_builds_the_expected_parquet_filename(monkeypatch, table):
    captured = {}

    class _FakeLazyFrame:
        def collect(self):
            return self

        def to_pandas(self):
            return pd.DataFrame()

    def fake_scan_parquet(path, **kwargs):
        captured["path"] = path
        return _FakeLazyFrame()

    monkeypatch.setattr("analysis.sessions.pl.scan_parquet", fake_scan_parquet)

    load_processed_table("/some/dir", table)

    assert captured["path"] == f"/some/dir/{table}.parquet"
