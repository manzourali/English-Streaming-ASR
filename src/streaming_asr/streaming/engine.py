from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streaming_asr.audio.stream import AudioChunk
from .state import StreamingState


@dataclass(frozen=True)
class EngineOutput:
    chunk: AudioChunk
    state: StreamingState
    payload: Any = None


class StreamingEngine:
    def __init__(self):
        self.state = StreamingState()

    def start(self) -> StreamingState:
        self.state.start()
        return self.state

    def process(self, chunk: AudioChunk, payload: Any = None) -> EngineOutput:
        self.state.update(chunk.index, chunk.end_time)
        return EngineOutput(chunk, self.state, payload)

    def finalize(self) -> StreamingState:
        if not self.state.finalized:
            self.state.finalize()
        return self.state

