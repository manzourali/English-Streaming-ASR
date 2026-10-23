from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streaming_asr.audio.stream import AudioChunk
from .state import StreamingState
from streaming_asr.exceptions import StreamingStateError


@dataclass(frozen=True)
class EngineOutput:
    chunk: AudioChunk
    state: StreamingState
    payload: Any = None


class StreamingEngine:
    def __init__(self):
        self.state = StreamingState()

    def start(self) -> StreamingState:
        self.state = StreamingState()
        self._next_sample = 0
        self._sample_rate = None
        self.state.start()
        return self.state

    def process(self, chunk: AudioChunk, payload: Any = None) -> EngineOutput:
        if not self.state.started or self.state.finalized:
            raise StreamingStateError("Stream must be started and not finalized")
        if chunk.start_sample != self._next_sample or self._sample_rate not in (None, chunk.sample_rate):
            raise StreamingStateError("Audio must be contiguous from sample zero at a fixed sample rate")
        self.state.update(chunk.index, chunk.end_sample / chunk.sample_rate)
        self._next_sample = chunk.end_sample
        self._sample_rate = chunk.sample_rate
        return EngineOutput(chunk, self.state, payload)

    def finalize(self) -> StreamingState:
        if not self.state.finalized:
            self.state.finalize()
        return self.state

