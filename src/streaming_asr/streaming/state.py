from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from streaming_asr.exceptions import StreamingStateError


@dataclass
class StreamingState:
    started: bool = False
    finalized: bool = False
    chunk_index: int = -1
    stream_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        if self.finalized:
            raise StreamingStateError("Cannot start a finalized stream")
        self.started = True

    def update(self, chunk_index: int, stream_time: float) -> None:
        if not self.started or self.finalized:
            raise StreamingStateError("Stream must be started and not finalized")
        if chunk_index <= self.chunk_index:
            raise StreamingStateError("Chunk indices must increase monotonically")
        self.chunk_index, self.stream_time = chunk_index, stream_time

    def finalize(self) -> None:
        if not self.started:
            raise StreamingStateError("Cannot finalize a stream that has not started")
        self.finalized = True


