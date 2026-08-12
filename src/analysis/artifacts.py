"""Artifact storage abstraction (design spec section 15).

Analysis code writes through ``ArtifactStore`` without knowing whether the
backend is a local directory (dev) or S3 (production). Construct the right
backend once via ``artifact_store_for_uri`` and pass it down.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
        full = self.root / relative_path
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
