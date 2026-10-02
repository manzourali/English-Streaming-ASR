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


class VADStreamingASRPipeline:
    """Phase 2 pipeline; VAD observes audio while ASR receives continuous audio.

    Keeping the audio continuous preserves WhisperRT context. VAD decisions and
    timings are exposed for routing/analysis in later phases rather than
    silently dropping non-speech frames.
    """
    def __init__(self, vad: Any, asr: Any):
        self.vad, self.asr = vad, asr
        self.engine = StreamingEngine()

    def start(self) -> None:
        self.engine.start()
        self.vad.start()
        self.asr.start()

    def process(self, chunk: AudioChunk) -> BaselineOutput:
        self.engine.process(chunk)
        vad_output = self.vad.process(chunk)
        asr_output = self.asr.process(chunk)
        return BaselineOutput(chunk, {"vad": vad_output, "asr": asr_output})

    def finalize(self) -> Any:
        vad_final = self.vad.finalize()
        asr_final = self.asr.finalize()
        self.engine.finalize()
        return {"vad": vad_final, "asr": asr_final}
