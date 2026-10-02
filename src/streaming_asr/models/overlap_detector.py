"""Streaming overlap-speech detection interfaces and lightweight baselines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.exceptions import StreamingStateError


class OverlapLabel(str, Enum):
    NO_SPEECH = "no_speech"
    SINGLE_SPEAKER = "single_speaker"
    OVERLAP = "overlap"


@dataclass(frozen=True)
class OverlapResult:
    label: OverlapLabel
    probability: float | None = None
    chunk_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    frame_count: int = 0
    spectral_complexity: float | None = None

    @property
    def overlap_probability(self) -> float | None:
        return self.probability


class StreamingOverlapDetector(ABC):
    def __init__(self) -> None:
        self.started = False
        self.finalized = False

    def start(self) -> None:
        self.started = True
        self.finalized = False

    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> OverlapResult: ...

    def finalize(self) -> OverlapResult | None:
        self.finalized = True
        return None

    def _ensure_started(self) -> None:
        if not self.started:
            self.start()
        if self.finalized:
            raise StreamingStateError("OSD stream has already been finalized")


class DummyOverlapDetector(StreamingOverlapDetector):
    """Energy-only integration stub; it cannot distinguish speakers."""
    def process(self, audio_chunk: AudioChunk) -> OverlapResult:
        self._ensure_started()
        energy = float(np.mean(audio_chunk.samples ** 2)) if len(audio_chunk.samples) else 0.0
        label = OverlapLabel.SINGLE_SPEAKER if energy > 1e-4 else OverlapLabel.NO_SPEECH
        return OverlapResult(label, 0.0, audio_chunk.index, audio_chunk.start_time, audio_chunk.end_time, 1)


class OracleOverlapDetector(StreamingOverlapDetector):
    """Ground-truth timing detector for integration tests only."""
    def __init__(self, intervals: list[tuple[float, float]], frame_ms: int = 30):
        super().__init__()
        self.intervals = intervals
        self.frame_ms = frame_ms

    def process(self, audio_chunk: AudioChunk) -> OverlapResult:
        self._ensure_started()
        midpoint = (audio_chunk.start_time + audio_chunk.end_time) / 2
        active = sum(start <= midpoint < end for start, end in self.intervals)
        label = OverlapLabel.OVERLAP if active >= 2 else OverlapLabel.SINGLE_SPEAKER if active == 1 else OverlapLabel.NO_SPEECH
        return OverlapResult(label, 1.0 if label is OverlapLabel.OVERLAP else 0.0, audio_chunk.index, audio_chunk.start_time, audio_chunk.end_time, 1)


class SpectralHeuristicOSD(StreamingOverlapDetector):
    """Causal spectral-peak OSD reference baseline.

    This is a transparent algorithmic baseline, not a pretrained neural OSD
    model. Each fixed frame is classified independently after buffering; no
    future frames or complete-utterance access are used. A frame is marked
    overlap when speech energy is present and at least two dominant spectral
    peaks are detected.
    """
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, speech_threshold: float = 1e-4, peak_threshold: float = 0.35, min_peak_distance_hz: float = 80.0):
        super().__init__()
        if frame_ms not in {10, 20, 30, 40, 80, 160, 320}:
            raise ValueError("frame_ms must be a supported positive streaming frame size")
        if sample_rate <= 0 or speech_threshold < 0 or not 0 < peak_threshold <= 1:
            raise ValueError("invalid OSD configuration")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self.speech_threshold = speech_threshold
        self.peak_threshold = peak_threshold
        self.min_peak_distance_bins = max(1, int(round(min_peak_distance_hz * self.frame_samples / sample_rate)))
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0.0
        self._last_index = 0

    def start(self) -> None:
        super().start()
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0.0
        self._last_index = 0

    def _classify(self, frame: np.ndarray) -> tuple[OverlapLabel, float, float]:
        energy = float(np.mean(frame * frame)) if len(frame) else 0.0
        if energy <= self.speech_threshold:
            return OverlapLabel.NO_SPEECH, 0.0, 0.0
        windowed = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(windowed))[1:]
        if not len(spectrum) or float(np.max(spectrum)) <= 0:
            return OverlapLabel.SINGLE_SPEAKER, 0.0, 0.0
        maximum = float(np.max(spectrum))
        peaks = [index for index in range(1, len(spectrum) - 1) if spectrum[index] >= spectrum[index - 1] and spectrum[index] >= spectrum[index + 1] and spectrum[index] >= self.peak_threshold * maximum]
        selected: list[int] = []
        for index in sorted(peaks, key=lambda item: spectrum[item], reverse=True):
            if all(abs(index - chosen) >= self.min_peak_distance_bins for chosen in selected):
                selected.append(index)
        probability = min(1.0, max(0.0, (len(selected) - 1) / 2.0))
        return (OverlapLabel.OVERLAP if len(selected) >= 2 else OverlapLabel.SINGLE_SPEAKER), probability, float(len(selected))

    def _process_frame(self, frame: np.ndarray, start: float, end: float, index: int) -> OverlapResult:
        label, probability, complexity = self._classify(frame)
        return OverlapResult(label, probability, index, start, end, 1, complexity)

    def process(self, audio_chunk: AudioChunk) -> OverlapResult:
        self._ensure_started()
        if audio_chunk.sample_rate != self.sample_rate:
            raise ValueError(f"OSD requires {self.sample_rate} Hz audio")
        if self._buffer.size == 0:
            self._buffer_start = audio_chunk.start_time
        self._buffer = np.concatenate((self._buffer, np.asarray(audio_chunk.samples, dtype=np.float32)))
        results: list[OverlapResult] = []
        while len(self._buffer) >= self.frame_samples:
            start = self._buffer_start
            end = start + self.frame_ms / 1000.0
            results.append(self._process_frame(self._buffer[:self.frame_samples], start, end, self._last_index))
            self._last_index += 1
            self._buffer = self._buffer[self.frame_samples:]
            self._buffer_start = end
        if not results:
            return OverlapResult(OverlapLabel.NO_SPEECH, 0.0, audio_chunk.index, audio_chunk.start_time, audio_chunk.end_time, 0, 0.0)
        return OverlapResult(max(results, key=lambda result: result.probability or 0.0).label, max(result.probability or 0.0 for result in results), audio_chunk.index, results[0].start_time, results[-1].end_time, len(results), max(result.spectral_complexity or 0.0 for result in results))

    def finalize(self) -> OverlapResult | None:
        if self.finalized:
            return None
        result = None
        if len(self._buffer):
            padded = np.pad(self._buffer, (0, self.frame_samples - len(self._buffer)))
            result = self._process_frame(padded, self._buffer_start, self._buffer_start + len(self._buffer) / self.sample_rate, self._last_index)
            self._buffer = np.empty(0, dtype=np.float32)
        super().finalize()
        return result


def get_overlap_detector(name: str, config: dict[str, Any] | None = None) -> StreamingOverlapDetector:
    config = config or {}
    normalized = name.lower()
    if normalized in {"heuristic", "spectral", "spectral_heuristic"}:
        return SpectralHeuristicOSD(sample_rate=int(config.get("sample_rate", 16000)), frame_ms=int(config.get("frame_ms", 30)), speech_threshold=float(config.get("speech_threshold", 1e-4)), peak_threshold=float(config.get("peak_threshold", 0.35)), min_peak_distance_hz=float(config.get("min_peak_distance_hz", 80.0)))
    if normalized == "dummy":
        return DummyOverlapDetector()
    raise ValueError(f"Unknown OSD backend: {name}")
