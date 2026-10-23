from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from streaming_asr.exceptions import InvalidAudioChunkError
from .preprocessing import to_mono


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    index: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.number):
            raise InvalidAudioChunkError("AudioChunk.samples must be a one-dimensional numeric array")
        if self.sample_rate <= 0 or self.index < 0 or self.end_time < self.start_time:
            raise InvalidAudioChunkError("Invalid sample rate, index, or timestamps")
        if not np.isfinite(samples).all() or not np.isfinite([self.start_time, self.end_time]).all():
            raise InvalidAudioChunkError("Audio samples and timestamps must be finite")
        start = round(self.start_time * self.sample_rate)
        end = round(self.end_time * self.sample_rate)
        if start < 0 or abs(start / self.sample_rate - self.start_time) > 1e-8 or abs(end / self.sample_rate - self.end_time) > 1e-8 or end - start != len(samples):
            raise InvalidAudioChunkError("Timestamps must match integer sample indices and sample count")
        if np.issubdtype(samples.dtype, np.integer):
            if samples.dtype != np.int16:
                raise InvalidAudioChunkError("Integer audio must be PCM16")
            samples = samples.astype(np.float32) / 32768.0
        object.__setattr__(self, "samples", samples.astype(np.float32, copy=False))

    @property
    def start_sample(self) -> int:
        return round(self.start_time * self.sample_rate)

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass
class AudioStream:
    chunks: Iterable[AudioChunk]
    _started: bool = field(default=False, init=False)

    def __iter__(self) -> Iterator[AudioChunk]:
        self._started = True
        yield from self.chunks

    @classmethod
    def from_array(cls, samples: np.ndarray, sample_rate: int, chunk_size: int, *, channel_axis: int = 1) -> "AudioStream":
        if sample_rate <= 0 or chunk_size <= 0:
            raise ValueError("sample_rate and chunk_size must be positive")
        samples = to_mono(samples, channel_axis=channel_axis)
        def chunks():
            for index, start in enumerate(range(0, len(samples), chunk_size)):
                part = samples[start:start + chunk_size]
                yield AudioChunk(part, sample_rate, start / sample_rate, (start + len(part)) / sample_rate, index)
        return cls(chunks())
