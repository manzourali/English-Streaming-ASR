from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from streaming_asr.exceptions import InvalidAudioChunkError


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
        object.__setattr__(self, "samples", samples.astype(np.float32, copy=False))

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
    def from_array(cls, samples: np.ndarray, sample_rate: int, chunk_size: int) -> "AudioStream":
        samples = np.asarray(samples, dtype=np.float32)
        chunks = []
        for index, start in enumerate(range(0, len(samples), chunk_size)):
            part = samples[start:start + chunk_size]
            chunks.append(AudioChunk(part, sample_rate, start / sample_rate, (start + len(part)) / sample_rate, index))
        return cls(chunks)

