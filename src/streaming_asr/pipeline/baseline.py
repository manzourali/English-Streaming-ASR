from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.streaming.engine import StreamingEngine
from .streaming import StreamingASRPipeline

BaselinePipeline = StreamingASRPipeline


@dataclass(frozen=True)
class BaselineOutput:
    chunk: AudioChunk
    transcript: Any


class StreamingASRBaselinePipeline:
    """Phase 1 pipeline: audio chunks directly to WhisperRT, without VAD/OSD."""
    def __init__(self, asr: Any):
        self.asr = asr
        self.engine = StreamingEngine()

    def start(self) -> None:
        self.engine.start()
        self.asr.start()

    def process(self, chunk: AudioChunk) -> BaselineOutput:
        self.engine.process(chunk)
        return BaselineOutput(chunk, self.asr.process(chunk))

    def finalize(self):
        final = self.asr.finalize()
        self.engine.finalize()
        return final
