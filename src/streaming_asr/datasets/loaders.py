from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

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
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise BackendNotAvailableError("LibriSpeech loading requires the optional 'datasets' package") from exc
        dataset_name = config.get("dataset", "openslr/librispeech_asr")
        subset = config.get("subset", "clean")
        split = config.get("split", "test")
        if split == "test-clean":
            subset, split = "clean", "test"
        return load_dataset(dataset_name, subset, split=split, streaming=bool(config.get("streaming", False)))


class LibriSpeechExample:
    def __init__(self, sample_id: str, audio: Any, reference: str, speaker_id: str | None = None):
        self.sample_id = sample_id
        self.audio = audio
        self.reference = reference
        self.speaker_id = speaker_id


def iter_librispeech(config: dict[str, Any]) -> Iterator[LibriSpeechExample]:
    """Yield normalized LibriSpeech examples without using speaker identity."""
    dataset = get_dataset_backend(config.get("backend", "huggingface")).load(config)
    if isinstance(dataset, Path):
        from .manifests import read_manifest
        records = read_manifest(dataset) if dataset.is_file() else read_manifest(dataset / "manifest.jsonl")
        for index, record in enumerate(records):
            yield LibriSpeechExample(str(index), record.audio_path, record.transcript)
        return
    max_samples = config.get("max_samples")
    for index, row in enumerate(dataset):
        audio = row.get("audio", {})
        if isinstance(audio, dict):
            audio = {"array": audio.get("array"), "sampling_rate": audio.get("sampling_rate")}
        yield LibriSpeechExample(
            sample_id=str(row.get("id", row.get("file", index))),
            audio=audio,
            reference=str(row.get("text", row.get("sentence", ""))),
            speaker_id=str(row["speaker_id"]) if row.get("speaker_id") is not None else None,
        )
        if max_samples is not None and index + 1 >= int(max_samples):
            break


def get_dataset_backend(name: str) -> DatasetBackend:
    backends = {"local": LocalDatasetBackend, "kaggle": KaggleDatasetBackend, "huggingface": HuggingFaceDatasetBackend}
    try:
        return backends[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset backend: {name}") from exc
