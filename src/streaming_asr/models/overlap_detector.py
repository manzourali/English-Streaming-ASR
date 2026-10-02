from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from streaming_asr.audio.stream import AudioChunk


class OverlapLabel(str, Enum):
    NO_SPEECH = "no_speech"
    SINGLE_SPEAKER = "single_speaker"
    OVERLAP = "overlap"


@dataclass(frozen=True)
class OverlapResult:
    label: OverlapLabel
    probability: float | None = None
    chunk_index: int = 0


class StreamingOverlapDetector(ABC):
    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> OverlapResult: ...

    def finalize(self) -> None:
        pass


class DummyOverlapDetector(StreamingOverlapDetector):
    def process(self, audio_chunk: AudioChunk) -> OverlapResult:
        import numpy as np
        energy = float(np.mean(audio_chunk.samples ** 2)) if len(audio_chunk.samples) else 0.0
        label = OverlapLabel.SINGLE_SPEAKER if energy > 1e-4 else OverlapLabel.NO_SPEECH
        return OverlapResult(label, None, audio_chunk.index)

