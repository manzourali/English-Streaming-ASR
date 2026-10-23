"""Versioned, per-example records for Phase 9 evaluation and benchmarking."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationRecord:
    system: str
    sample_id: str
    split: str
    dataset: str
    metric: str
    value: float
    direction: str = "lower"
    error_count: float | None = None
    reference_units: int | None = None
    overlap_regime: str | None = None
    overlap_ratio: float | None = None
    relative_gain_db: float | None = None
    temporal_condition: str | None = None
    oracle: bool = False
    audio_seconds: float | None = None
    processing_seconds: float | None = None
    first_output_latency_seconds: float | None = None
    finalization_latency_seconds: float | None = None
    average_chunk_latency_seconds: float | None = None
    gpu_memory_mib: float | None = None
    error_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"lower", "higher"}:
            raise ValueError("direction must be lower or higher")
        if not self.system or not self.sample_id or not self.metric:
            raise ValueError("system, sample_id, and metric are required")
        if self.reference_units is not None and self.reference_units < 0:
            raise ValueError("reference_units must be non-negative")
        if self.error_count is not None and self.error_count < 0:
            raise ValueError("error_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_evaluation_records(records: Iterable[EvaluationRecord], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item.to_dict(), sort_keys=True) for item in records) + "\n", encoding="utf-8")


def read_evaluation_records(path: str | Path) -> list[EvaluationRecord]:
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        value["error_tags"] = tuple(value.get("error_tags", ()))
        values.append(EvaluationRecord(**value))
    return values


def surt_prediction_record(row: dict[str, Any], *, system: str, dataset: str,
                           split: str, checkpoint: str | None = None,
                           manifest_sha256: str | None = None) -> EvaluationRecord:
    """Convert only the native SURT runner schema with exact PI-WER counts."""
    return EvaluationRecord(system=system, sample_id=row["mixture_id"], split=split,
        dataset=dataset, metric="pi_wer", value=row["pi_wer"],
        error_count=row["errors"], reference_units=row["reference_words"],
        overlap_regime=row["regime"], overlap_ratio=row.get("overlap_ratio"),
        relative_gain_db=row.get("relative_gain_db"), audio_seconds=row["audio_seconds"],
        processing_seconds=row["processing_seconds"],
        finalization_latency_seconds=row["finalization_seconds"],
        metadata={"checkpoint": checkpoint, "manifest_sha256": manifest_sha256,
                  "streaming_mode": "native_surt2",
                  "timing_mode": "unpaced_file_processing",
                  "first_output_available_sample": row.get("first_output_available_sample"),
                  "first_output_processing_seconds": row.get("first_output_processing_seconds")})
