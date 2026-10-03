"""Phase 7 structured, low-latency adapter for an external SURT 2.0 decoder.

SURT 2.0 is a genuine streaming multi-talker architecture, but this repository
does not vendor its rapidly changing recipe/runtime.  The adapter therefore
requires an explicitly configured decoder factory.  It owns only generic
audio-window, state, and hypothesis semantics and never falls back to Whisper.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.exceptions import BackendNotAvailableError, StreamingStateError


@dataclass(frozen=True)
class MultiTalkerStream:
    """A model output stream, not a real-world speaker identity."""
    stream_id: int
    text: str
    new_text: str
    start_time: float
    end_time: float
    is_final: bool = False
    confidence: float | None = None


@dataclass(frozen=True)
class MultiTalkerTranscriptUpdate:
    timestamp: float
    streams: tuple[MultiTalkerStream, ...]
    chunk_index: int
    window_start_time: float
    window_end_time: float
    is_final: bool = False
    processing_time: float | None = None
    streaming_mode: str = "low_latency_windowed"


class StreamingMultiTalkerASR(ABC):
    """Stable branch-facing interface for multi-talker ASR implementations."""
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> MultiTalkerTranscriptUpdate: ...

    @abstractmethod
    def finalize(self) -> MultiTalkerTranscriptUpdate | None: ...


def _append_without_duplication(previous: str, current: str) -> tuple[str, str]:
    """Merge rolling hypotheses while exposing only newly committed suffix text."""
    previous_words, current_words = previous.split(), current.split()
    common = 0
    for left, right in zip(previous_words, current_words):
        if left != right:
            break
        common += 1
    # Emit a suffix only when the old rolling hypothesis is retained in full.
    # A non-prefix revision replaces the displayed hypothesis but emits no new
    # text, avoiding duplicate transcript content from unstable partials.
    new_words = current_words[common:] if common == len(previous_words) else []
    return " ".join(current_words), " ".join(new_words)


class SURT2WindowedASR(StreamingMultiTalkerASR):
    """External SURT-compatible decoder with bounded causal rolling windows.

    This adapter is intentionally classified ``low_latency_windowed`` because
    the externally supplied decoder may or may not expose SURT's native state.
    It calls the decoder after each configured hop with at most
    ``context_duration_ms + chunk_duration_ms`` samples. A decoder may return
    strings, mappings with ``text``/``confidence``, or a mapping with
    ``streams``. Stream indices are model output channels, not speaker IDs.
    """
    streaming_mode = "low_latency_windowed"

    def __init__(self, decoder: Callable[[np.ndarray, int], Any], *, sample_rate: int = 16000, chunk_duration_ms: int = 320, context_duration_ms: int = 1280, lookahead_ms: int = 0, hop_duration_ms: int | None = None) -> None:
        if sample_rate <= 0 or chunk_duration_ms <= 0 or context_duration_ms < 0 or lookahead_ms < 0:
            raise ValueError("invalid multi-talker streaming configuration")
        if hop_duration_ms is not None and hop_duration_ms <= 0:
            raise ValueError("hop_duration_ms must be positive")
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.context_duration_ms = context_duration_ms
        self.lookahead_ms = lookahead_ms
        self.hop_duration_ms = hop_duration_ms or chunk_duration_ms
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0.0
        self._last_decode_end = 0.0
        self._last_audio_end = 0.0
        self._latest: dict[int, str] = {}
        self._last_update: MultiTalkerTranscriptUpdate | None = None
        self._started = False
        self._finalized = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SURT2WindowedASR":
        factory_path = config.get("decoder_factory")
        if not factory_path:
            raise BackendNotAvailableError(
                "SURT 2.0 runtime is not bundled. Configure model.decoder_factory "
                "as 'package.module:factory' for a verified external SURT recipe; "
                "this adapter will not substitute WhisperRT."
            )
        decoder = _load_decoder_factory(str(factory_path), dict(config))
        streaming = dict(config.get("streaming", {}))
        return cls(decoder, sample_rate=int(config.get("sample_rate", 16000)), chunk_duration_ms=int(streaming.get("chunk_duration_ms", config.get("chunk_duration_ms", 320))), context_duration_ms=int(streaming.get("context_duration_ms", 1280)), lookahead_ms=int(streaming.get("lookahead_ms", 0)), hop_duration_ms=int(streaming.get("hop_duration_ms", streaming.get("chunk_duration_ms", 320))))

    def start(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = self._last_decode_end = self._last_audio_end = 0.0
        self._latest.clear(); self._last_update = None
        self._started, self._finalized = True, False

    def process(self, audio_chunk: AudioChunk) -> MultiTalkerTranscriptUpdate:
        if not self._started or self._finalized:
            raise StreamingStateError("Multi-talker stream must be started and not finalized")
        if audio_chunk.sample_rate != self.sample_rate:
            raise ValueError(f"SURT adapter requires {self.sample_rate} Hz audio")
        if self._buffer.size == 0:
            self._buffer_start = audio_chunk.start_time
        self._buffer = np.concatenate((self._buffer, np.asarray(audio_chunk.samples, dtype=np.float32)))
        self._last_audio_end = audio_chunk.end_time
        hop_seconds = self.hop_duration_ms / 1000.0
        if audio_chunk.end_time - self._last_decode_end + 1e-9 < hop_seconds:
            return self._empty_update(audio_chunk)
        return self._decode(audio_chunk, final=False)

    def finalize(self) -> MultiTalkerTranscriptUpdate | None:
        if not self._started or self._finalized:
            return None
        self._finalized = True
        if self._buffer.size == 0 or self._last_update is None:
            return self._last_update
        if self._last_decode_end >= self._last_audio_end:
            streams = tuple(MultiTalkerStream(**{**stream.__dict__, "is_final": True}) for stream in self._last_update.streams)
            self._last_update = MultiTalkerTranscriptUpdate(**{**self._last_update.__dict__, "streams": streams, "is_final": True})
            return self._last_update
        final_chunk = AudioChunk(np.empty(0, dtype=np.float32), self.sample_rate, self._last_update.window_end_time, self._last_update.window_end_time, self._last_update.chunk_index)
        return self._decode(final_chunk, final=True)

    def _decode(self, chunk: AudioChunk, *, final: bool) -> MultiTalkerTranscriptUpdate:
        max_samples = int(round((self.context_duration_ms + self.chunk_duration_ms + self.lookahead_ms) * self.sample_rate / 1000))
        window = self._buffer[-max_samples:] if max_samples else self._buffer
        window_start = max(self._buffer_start, chunk.end_time - len(window) / self.sample_rate)
        started = perf_counter()
        raw = self.decoder(np.asarray(window, dtype=np.float32), self.sample_rate)
        elapsed = perf_counter() - started
        streams = self._normalise_streams(raw, window_start, chunk.end_time, final)
        self._last_decode_end = chunk.end_time
        update = MultiTalkerTranscriptUpdate(chunk.end_time, tuple(streams), chunk.index, window_start, chunk.end_time, final, elapsed, self.streaming_mode)
        self._last_update = update
        return update

    def _empty_update(self, chunk: AudioChunk) -> MultiTalkerTranscriptUpdate:
        return MultiTalkerTranscriptUpdate(chunk.end_time, (), chunk.index, chunk.start_time, chunk.end_time, False, 0.0, self.streaming_mode)

    def _normalise_streams(self, raw: Any, start: float, end: float, final: bool) -> list[MultiTalkerStream]:
        values = raw.get("streams", raw) if isinstance(raw, Mapping) else raw
        if isinstance(values, Mapping) and "text" in values:
            values = [values]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Iterable) or isinstance(values, (bytes, bytearray, Mapping)):
            raise ValueError("SURT decoder output must be a string, stream list, or {'streams': [...]} mapping")
        streams = []
        for index, value in enumerate(values):
            if isinstance(value, str):
                item: Mapping[str, Any] = {"text": value}
            elif isinstance(value, Mapping):
                item = value
            else:
                raise ValueError("Each SURT output stream must be a string or mapping")
            stream_id = int(item.get("stream_id", index))
            text = str(item.get("text", "")).strip()
            full, new = _append_without_duplication(self._latest.get(stream_id, ""), text)
            self._latest[stream_id] = full
            confidence = item.get("confidence")
            streams.append(MultiTalkerStream(stream_id, full, new, float(item.get("start", start)), float(item.get("end", end)), bool(item.get("is_final", final)), float(confidence) if confidence is not None else None))
        return streams


def _load_decoder_factory(path: str, config: dict[str, Any]) -> Callable[[np.ndarray, int], Any]:
    try:
        module_name, attribute = path.split(":", 1)
        factory = getattr(import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise BackendNotAvailableError(f"Unable to load configured SURT decoder_factory {path!r}") from exc
    decoder = factory(config)
    if not callable(decoder):
        raise BackendNotAvailableError("Configured SURT decoder_factory must return callable(samples, sample_rate)")
    return decoder


MultiTalkerASR = StreamingMultiTalkerASR
