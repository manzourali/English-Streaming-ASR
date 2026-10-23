"""Streaming VAD abstraction and the Phase 2 WebRTC backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.exceptions import BackendNotAvailableError, StreamingStateError


@dataclass(frozen=True)
class SpeechSegment:
    start_time: float
    end_time: float
    confidence: float | None = None


@dataclass(frozen=True)
class VADResult:
    speech: bool
    probability: float | None = None
    chunk_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    frame_count: int = 0
    segment: SpeechSegment | None = None
    completed_segments: tuple[SpeechSegment, ...] = ()

    @property
    def is_speech(self) -> bool:
        return self.speech


@dataclass
class VADState:
    started: bool = False
    finalized: bool = False
    previous_decision: bool = False
    speech_probability: float | None = None
    speech_duration: float = 0.0
    silence_duration: float = 0.0
    current_segment_start: float | None = None
    current_segment_end: float | None = None
    processed_chunks: int = 0


class StreamingVAD(ABC):
    def __init__(self, minimum_silence_ms: int = 300) -> None:
        if minimum_silence_ms < 0:
            raise ValueError("minimum_silence_ms must be non-negative")
        self.minimum_silence = minimum_silence_ms / 1000
        self._endpoint_silence = 0.0
        self.state = VADState()

    def start(self) -> None:
        self.state = VADState(started=True)
        self._endpoint_silence = 0.0

    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> VADResult: ...

    def finalize(self) -> SpeechSegment | None:
        if self.state.finalized:
            return None
        self.state.finalized = True
        if self.state.current_segment_start is None or self.state.current_segment_end is None:
            return None
        return SpeechSegment(self.state.current_segment_start, self.state.current_segment_end, self.state.speech_probability)

    def _ensure_started(self) -> None:
        if not self.state.started:
            self.start()
        if self.state.finalized:
            raise StreamingStateError("VAD stream has already been finalized")

    def _update_state(self, speech: bool, probability: float | None, start: float, end: float, chunk_index: int) -> VADResult:
        self._ensure_started()
        duration = max(0.0, end - start)
        completed = ()
        if speech:
            self._endpoint_silence = 0.0
            if self.state.current_segment_start is None:
                self.state.current_segment_start = start
            self.state.current_segment_end = end
            self.state.speech_duration += duration
        else:
            self.state.silence_duration += duration
            self._endpoint_silence += duration
            if self.state.current_segment_start is not None and self._endpoint_silence + 1e-9 >= self.minimum_silence:
                completed = (SpeechSegment(self.state.current_segment_start, self.state.current_segment_end, self.state.speech_probability),)
                self.state.current_segment_start = self.state.current_segment_end = None
        self.state.previous_decision = speech
        self.state.speech_probability = probability
        self.state.processed_chunks += 1
        segment = None
        if self.state.current_segment_start is not None and self.state.current_segment_end is not None:
            segment = SpeechSegment(self.state.current_segment_start, self.state.current_segment_end, probability)
        return VADResult(speech, probability, chunk_index, start, end, 1, segment, completed)


class DummyVAD(StreamingVAD):
    """Deterministic energy VAD for tests only; not an accuracy benchmark."""
    def __init__(self, threshold: float = 1e-4, minimum_silence_ms: int = 300):
        super().__init__(minimum_silence_ms)
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self.processed_chunks = 0

    def process(self, audio_chunk: AudioChunk) -> VADResult:
        energy = float((audio_chunk.samples ** 2).mean()) if len(audio_chunk.samples) else 0.0
        result = self._update_state(energy > self.threshold, min(1.0, energy / max(self.threshold, 1e-12)), audio_chunk.start_time, audio_chunk.end_time, audio_chunk.index)
        self.processed_chunks += 1
        return result


class WebRTCVADBackend(StreamingVAD):
    """Stateful WebRTC VAD with internal fixed-frame buffering.

    WebRTC returns a binary speech decision, not a calibrated probability.
    Therefore `probability` is 1.0/0.0 as a decision score, not a confidence
    estimate. Supported frame durations are 10, 20, and 30 ms.
    """
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, aggressiveness: int = 2, minimum_silence_ms: int = 300):
        super().__init__(minimum_silence_ms)
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("WebRTC VAD sample_rate must be 8000, 16000, 32000, or 48000 Hz")
        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")
        if aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("WebRTC VAD aggressiveness must be an integer from 0 to 3")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self.aggressiveness = aggressiveness
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0.0
        self._vad: Any = None

    def start(self) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise BackendNotAvailableError("WebRTC VAD requires the optional 'webrtcvad-wheels' package") from exc
        super().start()
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0.0

    def process(self, audio_chunk: AudioChunk) -> VADResult:
        self._ensure_started()
        if audio_chunk.sample_rate != self.sample_rate:
            raise ValueError(f"WebRTC VAD requires {self.sample_rate} Hz audio")
        if self._buffer.size == 0:
            self._buffer_start = audio_chunk.start_time
        self._buffer = np.concatenate((self._buffer, np.asarray(audio_chunk.samples, dtype=np.float32)))
        decisions, completed = [], []
        result = None
        while len(self._buffer) >= self.frame_samples:
            frame = np.clip(self._buffer[:self.frame_samples], -1.0, 1.0)
            decision = bool(self._vad.is_speech((frame * 32767.0).astype("<i2").tobytes(), self.sample_rate))
            decisions.append(decision)
            start_sample = round(self._buffer_start * self.sample_rate)
            end_sample = start_sample + self.frame_samples
            result = self._update_state(decision, float(decision), start_sample / self.sample_rate,
                                        end_sample / self.sample_rate, audio_chunk.index)
            completed.extend(result.completed_segments)
            self._buffer_start = end_sample / self.sample_rate
            self._buffer = self._buffer[self.frame_samples:].copy()
        if result is None:
            return VADResult(self.state.previous_decision, self.state.speech_probability,
                             audio_chunk.index, audio_chunk.start_time, audio_chunk.end_time, 0)
        return VADResult(any(decisions), sum(decisions) / len(decisions), audio_chunk.index,
                         result.start_time - (len(decisions) - 1) * self.frame_ms / 1000,
                         result.end_time, len(decisions), result.segment, tuple(completed))

    def finalize(self) -> SpeechSegment | None:
        if self.state.finalized:
            return None
        if self._buffer.size and self._vad is not None:
            frame = np.pad(self._buffer, (0, self.frame_samples - len(self._buffer)))
            pcm = np.clip(frame, -1.0, 1.0).astype(np.float32)
            decision = bool(self._vad.is_speech((pcm * 32767.0).astype(np.int16).tobytes(), self.sample_rate))
            tail = self._update_state(decision, float(decision), self._buffer_start, self._buffer_start + len(self._buffer) / self.sample_rate, self.state.processed_chunks)
            self._buffer = np.empty(0, dtype=np.float32)
            if tail.completed_segments:
                self.state.finalized = True
                return tail.completed_segments[-1]
        return super().finalize()


def get_vad_backend(name: str, config: dict[str, Any] | None = None) -> StreamingVAD:
    config = config or {}
    if name.lower() in {"webrtc", "webrtcvad", "webrtc_vad"}:
        return WebRTCVADBackend(sample_rate=int(config.get("sample_rate", 16000)), frame_ms=int(config.get("frame_ms", 30)), aggressiveness=int(config.get("aggressiveness", 2)), minimum_silence_ms=int(config.get("minimum_silence_ms", 300)))
    if name.lower() == "dummy":
        return DummyVAD(float(config.get("threshold", 1e-4)), minimum_silence_ms=int(config.get("minimum_silence_ms", 300)))
    raise ValueError(f"Unknown VAD backend: {name}")
