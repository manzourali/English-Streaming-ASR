from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from streaming_asr.audio.stream import AudioChunk


@dataclass(frozen=True)
class VADResult:
    speech: bool
    probability: float | None = None
    chunk_index: int = 0


class StreamingVAD(ABC):
    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> VADResult: ...

    def finalize(self) -> None:
        pass


class DummyVAD(StreamingVAD):
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.processed_chunks = 0

    def process(self, audio_chunk: AudioChunk) -> VADResult:
        self.processed_chunks += 1
        energy = float((audio_chunk.samples ** 2).mean()) if len(audio_chunk.samples) else 0.0
        return VADResult(energy > self.threshold, min(1.0, energy / max(self.threshold, 1e-12)), audio_chunk.index)

