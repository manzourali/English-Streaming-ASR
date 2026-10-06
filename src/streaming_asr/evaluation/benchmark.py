"""Compute-only summaries kept separate from ASR-quality aggregation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .records import EvaluationRecord


@dataclass(frozen=True)
class BenchmarkSummary:
    system: str
    samples: int
    rtf: float | None
    average_first_output_latency_seconds: float | None
    average_finalization_latency_seconds: float | None
    average_chunk_latency_seconds: float | None
    peak_gpu_memory_mib: float | None


class Benchmark:
    def run(self, records: Iterable[EvaluationRecord]) -> list[BenchmarkSummary]:
        grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            grouped[record.system].append(record)
        summaries = []
        for system, items in sorted(grouped.items()):
            processing = [item.processing_seconds for item in items if item.processing_seconds is not None]
            audio = [item.audio_seconds for item in items if item.audio_seconds is not None]
            rtf = sum(processing) / sum(audio) if processing and audio and sum(audio) else None
            average = lambda values: sum(values) / len(values) if values else None
            summaries.append(BenchmarkSummary(system, len(items), rtf, average([item.first_output_latency_seconds for item in items if item.first_output_latency_seconds is not None]), average([item.finalization_latency_seconds for item in items if item.finalization_latency_seconds is not None]), average([item.average_chunk_latency_seconds for item in items if item.average_chunk_latency_seconds is not None]), max((item.gpu_memory_mib for item in items if item.gpu_memory_mib is not None), default=None)))
        return summaries
