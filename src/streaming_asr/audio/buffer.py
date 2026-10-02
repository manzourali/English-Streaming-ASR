from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from .stream import AudioChunk


class AudioBuffer:
    """Bounded FIFO buffer for incremental audio chunks."""
    def __init__(self, max_chunks: int = 32):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self._chunks: Deque[AudioChunk] = deque(maxlen=max_chunks)

    def append(self, chunk: AudioChunk) -> None:
        self._chunks.append(chunk)

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()

    def as_array(self) -> np.ndarray:
        return np.concatenate([chunk.samples for chunk in self._chunks]) if self._chunks else np.array([], dtype=np.float32)

