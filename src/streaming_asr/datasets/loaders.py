from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from streaming_asr.exceptions import BackendNotAvailableError


class DatasetBackend(ABC):
    @abstractmethod
    def load(self, config: dict[str, Any]) -> Any:
        """Load a dataset according to config; Phase 0 adapters do not download data."""


class LocalDatasetBackend(DatasetBackend):
    def load(self, config: dict[str, Any]) -> Path:
        path = Path(config.get("path", "data"))
        if not path.exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {path}")
        return path


class KaggleDatasetBackend(DatasetBackend):
    def load(self, config: dict[str, Any]) -> Path:
        path = Path(config.get("path", "/kaggle/input"))
        if not str(path).startswith("/kaggle/input"):
            raise ValueError("Kaggle dataset inputs must be under /kaggle/input")
        return path


class HuggingFaceDatasetBackend(DatasetBackend):
    def load(self, config: dict[str, Any]) -> Any:
        raise BackendNotAvailableError("Hugging Face dataset loading is reserved for a later phase; no download was attempted")


def get_dataset_backend(name: str) -> DatasetBackend:
    backends = {"local": LocalDatasetBackend, "kaggle": KaggleDatasetBackend, "huggingface": HuggingFaceDatasetBackend}
    try:
        return backends[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset backend: {name}") from exc

