from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.models.overlap_detector import StreamingOverlapDetector
from streaming_asr.models.vad import StreamingVAD
from streaming_asr.streaming.engine import StreamingEngine


@dataclass(frozen=True)
class PipelineOutput:
    chunk_index: int
    vad: Any
    overlap: Any
    asr: Any = None


class StreamingASRPipeline:
    def __init__(self, vad: StreamingVAD | None = None, overlap_detector: StreamingOverlapDetector | None = None, asr: Any = None):
        self.engine = StreamingEngine()
        self.vad, self.overlap_detector, self.asr = vad, overlap_detector, asr

    def start(self) -> None:
        self.engine.start()
        if self.vad is not None and hasattr(self.vad, "start"):
            self.vad.start()
        if self.asr is not None and hasattr(self.asr, "start"):
            self.asr.start()

    def process(self, audio_chunk: AudioChunk) -> PipelineOutput:
        self.engine.process(audio_chunk)
        vad_result = self.vad.process(audio_chunk) if self.vad is not None else None
        overlap_result = self.overlap_detector.process(audio_chunk) if self.overlap_detector is not None else None
        asr_result = self.asr.process(audio_chunk) if self.asr is not None else None
        return PipelineOutput(audio_chunk.index, vad_result, overlap_result, asr_result)

    def finalize(self):
        if self.vad is not None:
            self.vad.finalize()
        if self.overlap_detector is not None:
            self.overlap_detector.finalize()
        if self.asr is not None and hasattr(self.asr, "finalize"):
            self.asr.finalize()
        return self.engine.finalize()
