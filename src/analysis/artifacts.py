"""Artifact storage abstraction (design spec section 15).

Analysis code writes through ``ArtifactStore`` without knowing whether the
backend is a local directory (dev) or S3 (production). Construct the right
backend once via ``artifact_store_for_uri`` and pass it down.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import urlparse

import boto3
import pandas as pd


class ArtifactStore(ABC):
    @abstractmethod
    def write_json(self, relative_path: str, data: Any) -> None: ...

    @abstractmethod
    def write_text(self, relative_path: str, text: str) -> None: ...

    @abstractmethod
    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def uri(self, relative_path: str) -> str: ...


class LocalArtifactStore(ArtifactStore):
    """Writes under ``<root>/<relative_path>`` on the local filesystem."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        if PurePath(relative_path).is_absolute():
            raise ValueError(
                f"relative_path must be relative to root, got absolute path: {relative_path!r}"
            )
        full = self.root / relative_path
        root_norm = os.path.normpath(str(self.root))
        full_norm = os.path.normpath(str(full))
        try:
            common = os.path.commonpath([root_norm, full_norm])
        except ValueError:
            # Raised e.g. when the two paths are on different drives, which
            # also means relative_path escaped root.
            common = None
        if common != root_norm:
            raise ValueError(
                f"relative_path escapes root {self.root!r}: {relative_path!r}"
            )
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def write_json(self, relative_path: str, data: Any) -> None:
        self._resolve(relative_path).write_text(json.dumps(data, indent=2, default=str))

    def write_text(self, relative_path: str, text: str) -> None:
        self._resolve(relative_path).write_text(text)

    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._resolve(relative_path))

    def uri(self, relative_path: str) -> str:
        return str(self.root / relative_path)


class S3ArtifactStore(ArtifactStore):
    """Writes under ``s3://<bucket>/<prefix>/<relative_path>``."""

    def __init__(self, bucket: str, prefix: str, client=None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client or boto3.client("s3")

    def _key(self, relative_path: str) -> str:
        return f"{self.prefix}/{relative_path}"

    def write_json(self, relative_path: str, data: Any) -> None:
        body = json.dumps(data, indent=2, default=str)
        self._client.put_object(
            Bucket=self.bucket, Key=self._key(relative_path), Body=body
        )

    def write_text(self, relative_path: str, text: str) -> None:
        self._client.put_object(
            Bucket=self.bucket, Key=self._key(relative_path), Body=text
        )

    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> None:
        import io

        buf = io.BytesIO()
        df.to_parquet(buf)
        self._client.put_object(
            Bucket=self.bucket, Key=self._key(relative_path), Body=buf.getvalue()
        )

    def uri(self, relative_path: str) -> str:
        return f"s3://{self.bucket}/{self._key(relative_path)}"


def artifact_store_for_uri(uri: str) -> ArtifactStore:
    """Build the right ``ArtifactStore`` for a local path or an ``s3://`` URI."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return S3ArtifactStore(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    return LocalArtifactStore(root=uri)
