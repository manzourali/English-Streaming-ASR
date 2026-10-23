"""Unified native SURT 2.0 streaming orchestration, without waveform re-decoding.

A configured backend must expose bounded dual-path mask and Zipformer states
and per-channel stateless RNN-T decoding. Offline icefall forward() is not a
streaming API and is deliberately rejected. See torch_surt.py for composition.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Mapping, Protocol

import numpy as np

from streaming_asr.audio.features import StreamingFbank
from streaming_asr.audio.stream import AudioChunk
from streaming_asr.exceptions import BackendNotAvailableError, StreamingStateError
from streaming_asr.streaming.engine import StreamingEngine


@dataclass(frozen=True)
class MultiTalkerStream:
    """An output channel, not a persistent speaker identity; text is a delta."""
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
    streaming_mode: str = "native_surt2"
    endpoints: tuple[int, ...] = ()
    available_sample: int = 0


@dataclass(frozen=True)
class BackendOutput:
    """New committed text and trailing blank counts for each output channel."""
    texts: tuple[str, ...]
    trailing_blank_frames: tuple[int, ...]


class SURTBackend(Protocol):
    num_channels: int
    chunk_frames: int
    right_context_frames: int
    frame_shift_samples: int
    max_state_bytes: int

    def reset(self) -> None: ...
    def step(self, features: np.ndarray, *, valid_frames: int, final: bool) -> BackendOutput:
        """Consume 32 new frames; extra right context must not advance caches."""
        ...
    def finish(self) -> BackendOutput:
        """Flush delayed model/token output, including exact chunk boundaries."""
        ...


class StreamingMultiTalkerASR(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> MultiTalkerTranscriptUpdate: ...
    @abstractmethod
    def finalize(self) -> MultiTalkerTranscriptUpdate: ...


class SURT2StreamingASR(StreamingMultiTalkerASR):
    """32-frame scheduling with bounded PCM/features and blank endpoint events.

    At most 31 intra-chunk future frames plus one explicit context frame are
    permitted (320 ms). Feature-window acquisition adds up to 25 ms separately.
    Endpointing emits channel events without resetting the acoustic state or
    dropping silence. Callers collect text deltas outside the acoustic loop.
    """
    streaming_mode = "native_surt2"

    def __init__(self, backend: SURTBackend, *, endpoint_silence_ms: int = 800):
        if (backend.chunk_frames != 32 or backend.right_context_frames not in (0, 1)
                or backend.frame_shift_samples != 160 or backend.num_channels != 2
                or backend.max_state_bytes <= 0):
            raise ValueError("SURT requires two channels, 32 frames, 10 ms shift and at most 320 ms modeling context")
        for method in ("reset", "step", "finish"):
            if not callable(getattr(backend, method, None)):
                raise TypeError(f"SURT backend requires {method}()")
        if endpoint_silence_ms <= 0:
            raise ValueError("endpoint_silence_ms must be positive")
        self.backend = backend
        self.endpoint_frames = (endpoint_silence_ms + 9) // 10
        if getattr(backend, "endpoint_frames", self.endpoint_frames) != self.endpoint_frames:
            raise ValueError("Backend and scheduler endpoint silence must agree")
        self.sample_rate = 16000
        self.features = StreamingFbank()
        self.engine = StreamingEngine()
        self._started = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SURT2StreamingASR":
        path = config.get("backend_factory")
        if not path:
            raise BackendNotAvailableError(
                "Configure model.backend_factory='module:factory' returning a native SURTBackend; "
                "supply matching trained DPRNN/Zipformer/stateless decoder weights and tokenizer. "
                "Waveform callables and offline state_dict files are not streaming backends.")
        try:
            module, name = str(path).split(":", 1)
            factory = getattr(import_module(module), name)
        except (ValueError, ImportError, AttributeError) as exc:
            raise BackendNotAvailableError(f"Cannot import SURT backend factory {path!r}") from exc
        backend = factory(dict(config))
        streaming = config.get("streaming", {})
        for key in ("chunk_frames", "right_context_frames"):
            if key in streaming and int(streaming[key]) != getattr(backend, key):
                raise ValueError(f"Configured {key} disagrees with the loaded model")
        if "chunk_duration_ms" in streaming and int(streaming["chunk_duration_ms"]) != 320:
            raise ValueError("SURT model chunks must be 320 ms")
        return cls(backend, endpoint_silence_ms=int(config.get("endpoint_silence_ms", 800)))

    def start(self) -> None:
        self.backend.reset()
        self.features.reset()
        self.engine.start()
        self.pending = np.empty((0, 80), np.float32)
        self.consumed_frames = 0
        self._active = [False] * self.backend.num_channels
        self._started = True
        self._final = None

    @property
    def buffered_samples(self) -> int:
        return len(self.features.buffer)

    @property
    def buffered_frames(self) -> int:
        return len(self.pending)

    def _collect(self, output: BackendOutput, start_sample: int, end_sample: int, final: bool):
        if len(output.texts) != 2 or len(output.trailing_blank_frames) != 2:
            raise ValueError("Backend must return both output channels")
        streams, endpoints = [], []
        for channel, (text, blanks) in enumerate(zip(output.texts, output.trailing_blank_frames)):
            if blanks < 0:
                raise ValueError("Trailing blank counts must be non-negative")
            self._active[channel] |= bool(text)
            endpoint = self._active[channel] and (final or blanks >= self.endpoint_frames)
            if endpoint:
                endpoints.append(channel)
                self._active[channel] = False
            streams.append(MultiTalkerStream(channel, text, text, start_sample / 16000,
                                            end_sample / 16000, final or endpoint))
        return streams, endpoints

    def _drain(self, *, final: bool):
        streams, endpoints = [], []
        width = 32 + self.backend.right_context_frames
        while len(self.pending) >= width or (final and len(self.pending)):
            valid = min(32, len(self.pending))
            block = np.zeros((width, 80), np.float32)
            copied = min(width, len(self.pending))
            block[:copied] = self.pending[:copied]
            last = final and len(self.pending) <= 32
            output = self.backend.step(block, valid_frames=valid, final=last)
            start = self.consumed_frames * 160
            self.consumed_frames += valid
            end = min(self.consumed_frames * 160, self.engine._next_sample)
            values, events = self._collect(output, start, end, False)
            streams.extend(values); endpoints.extend(events)
            self.pending = self.pending[valid:].copy()
        return streams, endpoints

    def process(self, audio_chunk: AudioChunk) -> MultiTalkerTranscriptUpdate:
        if not self._started or self._final is not None:
            raise StreamingStateError("SURT stream must be started and not finalized")
        if audio_chunk.sample_rate != 16000:
            raise ValueError("SURT requires 16 kHz PCM; resample explicitly before streaming")
        self.engine.process(audio_chunk)
        started = perf_counter()
        streams, endpoints = [], []
        first_frame = self.consumed_frames
        # Bound temporary features even when callers provide a whole recording.
        for offset in range(0, len(audio_chunk.samples), 5120):
            frames = self.features.accept(audio_chunk.samples[offset:offset + 5120])
            self.pending = np.concatenate((self.pending, frames))
            values, events = self._drain(final=False)
            streams.extend(values); endpoints.extend(events)
        return MultiTalkerTranscriptUpdate(audio_chunk.end_time, tuple(streams), audio_chunk.index,
            first_frame * .01, min(self.consumed_frames * 160, audio_chunk.end_sample) / 16000,
            processing_time=perf_counter() - started, endpoints=tuple(endpoints), available_sample=audio_chunk.end_sample)

    def finalize(self) -> MultiTalkerTranscriptUpdate:
        if not self._started:
            raise StreamingStateError("SURT stream must be started")
        if self._final is not None:
            return self._final
        started = perf_counter()
        first_frame = self.consumed_frames
        self.pending = np.concatenate((self.pending, self.features.accept(np.empty(0, np.float32), final=True)))
        streams, endpoints = self._drain(final=True)
        end = self.engine._next_sample
        values, events = self._collect(self.backend.finish(), end, end, True)
        streams.extend(values); endpoints.extend(events)
        self.engine.finalize()
        self._final = MultiTalkerTranscriptUpdate(end / 16000, tuple(streams), self.engine.state.chunk_index,
            first_frame * .01, end / 16000, True, perf_counter() - started,
            endpoints=tuple(endpoints), available_sample=end)
        return self._final


MultiTalkerASR = StreamingMultiTalkerASR
