from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .asr import corpus_word_error_rate, normalize_text


@dataclass(frozen=True)
class OverlapASRResult:
    mixture_id: str
    condition: str
    overlap_ratio: float
    overlap_duration: float
    relative_gain_db: float
    source_references: list[dict[str, Any]]
    concatenated_reference: str | None
    hypothesis: str
    wer: float | None
    wer_status: str
    audio_duration: float
    processing_time: float | None
    rtf: float | None
    first_output_latency: float | None
    end_of_utterance_latency: float | None
    average_chunk_latency: float | None
    output_updates: int
    incremental_hypotheses: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def control_reference(source_references: list[dict[str, Any]], overlap_exists: bool) -> str | None:
    """Construct a WER reference only for non-overlap time-ordered controls."""
    if overlap_exists:
        return None
    ordered = sorted(source_references, key=lambda source: source["start"])
    return normalize_text(" ".join(source["transcript"] for source in ordered))


def aggregate_overlap_results(results: list[OverlapASRResult]) -> dict[str, Any]:
    grouped: dict[str, list[OverlapASRResult]] = {}
    for result in results:
        grouped.setdefault(result.condition, []).append(result)
    summary: dict[str, Any] = {}
    for condition, records in grouped.items():
        valid = [record for record in records if record.wer is not None]
        references = [record.concatenated_reference or "" for record in valid]
        hypotheses = [record.hypothesis for record in valid]
        summary[condition] = {
            "num_examples": len(records),
            "num_valid_wer": len(valid),
            "num_not_measured_wer": len(records) - len(valid),
            "corpus_wer": corpus_word_error_rate(references, hypotheses) if valid else None,
            "wer_status": "MEASURED" if valid else "NOT_MEASURED",
            "total_audio_duration": sum(record.audio_duration for record in records),
            "mean_rtf": _mean([record.rtf for record in records]),
            "mean_first_output_latency": _mean([record.first_output_latency for record in records]),
            "mean_end_of_utterance_latency": _mean([record.end_of_utterance_latency for record in records]),
            "mean_chunk_latency": _mean([record.average_chunk_latency for record in records]),
            "mean_overlap_ratio": _mean([record.overlap_ratio for record in records]),
            "mean_overlap_duration": _mean([record.overlap_duration for record in records]),
        }
    return summary


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None
