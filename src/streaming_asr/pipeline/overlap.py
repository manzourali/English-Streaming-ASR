"""Phase 7 adapter between structured multi-talker output and Phase 6 branches."""
from __future__ import annotations

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.models.multitalker import StreamingMultiTalkerASR
from .adaptive import ASRBranch


class MultiTalkerASRBranch(ASRBranch):
    """Use a structured multi-talker adapter as the adaptive overlap branch."""
    def __init__(self, model: StreamingMultiTalkerASR):
        self.model = model

    def start(self) -> None:
        self.model.start()

    def process(self, audio_chunk: AudioChunk):
        return self.model.process(audio_chunk)

    def finalize(self):
        return self.model.finalize()


OverlapPipeline = MultiTalkerASRBranch
