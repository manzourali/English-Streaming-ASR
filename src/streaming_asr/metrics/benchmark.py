from __future__ import annotations

from dataclasses import dataclass

from .asr import normalize_text, word_error_rate
from .streaming import average, real_time_factor


@dataclass(frozen=True)
class UtteranceMetrics:
    sample_id: str
    raw_reference: str
    normalized_reference: str
    raw_prediction: str
    normalized_prediction: str
    audio_duration: float
    processing_time: float
    rtf: float
    first_output_latency: float | None
    finalization_latency: float | None
    average_chunk_latency: float | None


def summarize(records: list[UtteranceMetrics]) -> dict[str, float | int | None]:
    if not records:
        return {"samples": 0, "wer": None, "rtf": None, "first_output_latency": None, "end_of_utterance_latency": None, "average_chunk_latency": None, "total_audio_seconds": 0.0}
    total_words = sum(len(r.normalized_reference.split()) for r in records)
    distance = sum(word_error_rate(r.normalized_reference, r.normalized_prediction) * len(r.normalized_reference.split()) for r in records)
    return {
        "samples": len(records),
        "wer": distance / total_words if total_words else 0.0,
        "rtf": sum(r.processing_time for r in records) / sum(r.audio_duration for r in records),
        "first_output_latency": average([r.first_output_latency for r in records if r.first_output_latency is not None]),
        "end_of_utterance_latency": average([r.finalization_latency for r in records if r.finalization_latency is not None]),
        "average_chunk_latency": average([r.average_chunk_latency for r in records if r.average_chunk_latency is not None]),
        "total_audio_seconds": sum(r.audio_duration for r in records),
    }
