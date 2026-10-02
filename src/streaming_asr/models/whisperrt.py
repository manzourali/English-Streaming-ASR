from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from streaming_asr.exceptions import BackendNotAvailableError


class ModelBackend(ABC):
    @abstractmethod
    def load(self, model_config: dict[str, Any]) -> Any: ...


class HuggingFaceModelBackend(ModelBackend):
    def load(self, model_config: dict[str, Any]) -> Any:
        raise BackendNotAvailableError("WhisperRT loading is reserved for Phase 1; no model was downloaded")


class LocalModelBackend(ModelBackend):
    def load(self, model_config: dict[str, Any]) -> Path:
        path = Path(model_config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        return path


KaggleModelBackend = LocalModelBackend


def get_model_backend(name: str) -> ModelBackend:
    backends = {"huggingface": HuggingFaceModelBackend, "local": LocalModelBackend, "kaggle": KaggleModelBackend}
    try:
        return backends[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown model backend: {name}") from exc

