from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

import numpy as np

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.exceptions import BackendNotAvailableError, StreamingStateError


class ModelBackend(ABC):
    @abstractmethod
    def load(self, model_config: dict[str, Any]) -> Any: ...


class HuggingFaceModelBackend(ModelBackend):
    def load(self, model_config: dict[str, Any]) -> Any:
        """Load a WhisperRT checkpoint through the project's configured Hub repo.

        The upstream package's loader accepts a local checkpoint path. We resolve
        that path with huggingface_hub first, then call the official loader.
        """
        try:
            import torch
            import whisper_rt
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise BackendNotAvailableError(
                "WhisperRT requires torch, huggingface_hub, and the official WhisperRT package"
            ) from exc
        checkpoint = model_config.get("checkpoint")
        if checkpoint is None:
            repo_id = model_config.get("repo_id", model_config.get("name"))
            filename = model_config.get("filename")
            if not repo_id or not filename:
                raise BackendNotAvailableError(
                    "Hugging Face WhisperRT loading requires model.repo_id/model.filename "
                    "or model.checkpoint"
                )
            checkpoint = hf_hub_download(repo_id=repo_id, filename=filename, revision=model_config.get("revision"))
        device = model_config.get("device", "cpu")
        return whisper_rt.load_streaming_model(
            name=str(checkpoint),
            gran=int(model_config["chunk_ms"]),
            multilingual=bool(model_config.get("multilingual", False)),
            varying_chunk_size=bool(model_config.get("varying_chunk_size", False)),
            device=device,
        )


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


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    chunk_index: int
    start_time: float
    end_time: float
    is_final: bool = False
    processing_time: float | None = None


class WhisperRTStreamingASR:
    """Adapter for the verified ``whisper_rt`` causal streaming API.

    The adapter feeds each incoming chunk through the official
    ``SpectrogramStream`` and ``StreamingWhisper.decode`` path. It never calls
    an offline Whisper implementation and never reloads the model per chunk.
    """
    def __init__(self, config: dict[str, Any], model: Any | None = None):
        self.config = config
        self.model = model
        self.device = config.get("device", "cpu")
        self._spectrogram = None
        self._options = None
        self._started = False
        self._last_update: TranscriptUpdate | None = None
        self._load_time: float | None = None
        self._processing_times: list[float] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WhisperRTStreamingASR":
        model_config = dict(config)
        model_config["chunk_ms"] = config.get("chunk_ms", 300)
        model_config["device"] = resolve_device(config.get("device", "auto"))
        validate_dtype(model_config["dtype"] if "dtype" in model_config else "auto", model_config["device"])
        started = time.perf_counter()
        model = get_model_backend(config.get("backend", "huggingface")).load(model_config)
        instance = cls(config, model=model)
        instance._load_time = time.perf_counter() - started
        return instance

    @property
    def load_time(self) -> float | None:
        return self._load_time

    @property
    def actual_dtype(self) -> str | None:
        if self.model is None:
            return None
        try:
            return str(next(self.model.parameters()).dtype).replace("torch.", "")
        except (StopIteration, AttributeError):
            return None

    @property
    def processing_times(self) -> list[float]:
        return list(self._processing_times)

    def start(self) -> None:
        if self.model is None:
            raise BackendNotAvailableError("WhisperRT model is not loaded")
        try:
            import torch
            from whisper_rt.audio import SpectrogramStream
            from whisper_rt.streaming_decoding import DecodingOptions
        except ImportError as exc:
            raise BackendNotAvailableError("WhisperRT runtime dependencies are not installed") from exc
        self.model.reset(use_stream=True)
        self.model.eval()
        self._spectrogram = SpectrogramStream(n_mels=getattr(self.model.dims, "n_mels", 80))
        gran = int(getattr(self.model.encoder, "gran", int(self.config.get("chunk_ms", 300)) // 20))
        self._options = DecodingOptions(
            language=self.config.get("language", "en"),
            gran=gran,
            single_frame_mel=True,
            without_timestamps=True,
            beam_size=self.config.get("beam_size", 5),
            temperature=self.config.get("temperature", 0.0),
            stream_decode=True,
            use_kv_cache=bool(self.config.get("sa_kv_cache", False)),
            use_ca_kv_cache=bool(self.config.get("ca_kv_cache", False)),
        )
        self._started = True
        self._last_update = None
        self._processing_times.clear()

    def process(self, audio_chunk: AudioChunk) -> TranscriptUpdate:
        if not self._started or self._spectrogram is None or self._options is None:
            raise StreamingStateError("WhisperRT stream must be started before processing chunks")
        if audio_chunk.sample_rate != 16000:
            raise ValueError("WhisperRT requires 16 kHz audio")
        try:
            import torch
        except ImportError as exc:
            raise BackendNotAvailableError("WhisperRT requires PyTorch") from exc
        started = time.perf_counter()
        frame = torch.from_numpy(np.asarray(audio_chunk.samples, dtype=np.float32))
        if self.device.startswith("cuda") and frame.device.type == "cpu":
            frame = frame.pin_memory().to(self.device, non_blocking=True)
        mel = self._spectrogram.calc_mel_with_new_frame(frame)
        result = self.model.decode(mel.squeeze(0), self._options)
        elapsed = time.perf_counter() - started
        self._processing_times.append(elapsed)
        update = TranscriptUpdate(
            text=str(getattr(result, "text", "")).strip(),
            chunk_index=audio_chunk.index,
            start_time=audio_chunk.start_time,
            end_time=audio_chunk.end_time,
            processing_time=elapsed,
        )
        self._last_update = update
        return update

    def finalize(self) -> TranscriptUpdate | None:
        if not self._started:
            return None
        self._started = False
        if self._last_update is None:
            return None
        self._last_update = TranscriptUpdate(**{**self._last_update.__dict__, "is_final": True})
        return self._last_update


class TranscriptAccumulator:
    """Keep the latest rolling hypothesis without duplicating partial text."""
    def __init__(self):
        self.latest: TranscriptUpdate | None = None

    def update(self, update: TranscriptUpdate) -> TranscriptUpdate:
        self.latest = update
        return update

    def finalize(self) -> TranscriptUpdate | None:
        if self.latest is None:
            return None
        self.latest = TranscriptUpdate(**{**self.latest.__dict__, "is_final": True})
        return self.latest

    @property
    def text(self) -> str:
        return self.latest.text if self.latest else ""


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda")
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise BackendNotAvailableError("CUDA was requested but PyTorch is not installed")
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise BackendNotAvailableError("CUDA was requested but is unavailable")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested


def validate_dtype(dtype: str, device: str) -> None:
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("model.dtype must be one of: auto, float32, float16, bfloat16")
    if device == "cpu" and dtype == "float16":
        raise ValueError("float16 is not supported for the Phase 1 CPU path; use float32 or auto")
