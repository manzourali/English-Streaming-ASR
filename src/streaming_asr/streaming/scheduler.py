from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkScheduler:
    chunk_ms: int = 320

    def __post_init__(self) -> None:
        if self.chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")

