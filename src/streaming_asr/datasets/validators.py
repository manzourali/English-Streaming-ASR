from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from streaming_asr.audio.loader import load_audio
from .manifests import ManifestRecord, OverlapRecord
from .overlap_generator import calculate_overlap


def validate_manifest(records: Iterable[ManifestRecord]) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        if not record.audio_path:
            errors.append(f"record {index}: audio_path is empty")
        if record.sample_rate is not None and record.sample_rate <= 0:
            errors.append(f"record {index}: sample_rate must be positive")
    return errors


def validate_overlap_record(record: OverlapRecord, check_source_files: bool = True) -> list[str]:
    errors: list[str] = []
    mixture_path = Path(record.audio_path)
    if not mixture_path.is_file():
        errors.append(f"{record.mixture_id}: mixture file missing")
    else:
        try:
            samples, metadata = load_audio(mixture_path)
            if metadata.sample_rate != record.sample_rate:
                errors.append(f"{record.mixture_id}: sample rate mismatch")
            if not np.isfinite(samples).all() or not len(samples):
                errors.append(f"{record.mixture_id}: invalid mixture samples")
            if abs(len(samples) / record.sample_rate - record.duration) > 1 / record.sample_rate + 1e-6:
                errors.append(f"{record.mixture_id}: mixture duration mismatch")
        except Exception as exc:
            errors.append(f"{record.mixture_id}: cannot read mixture ({exc})")
    if record.num_speakers != len(record.sources) or record.num_speakers != 2:
        errors.append(f"{record.mixture_id}: expected exactly two sources")
    for source in record.sources:
        if source.start < 0 or source.end <= source.start or abs(source.end - source.start - source.duration) > 1e-5:
            errors.append(f"{record.mixture_id}: invalid source timing for {source.source_id}")
        if check_source_files and not Path(source.audio_path).is_file():
            errors.append(f"{record.mixture_id}: source file missing for {source.source_id}")
    a, b = record.sources
    start, end, duration = calculate_overlap(a.start, a.end, b.start, b.end)
    if abs(duration - record.overlap_duration) > 1e-5:
        errors.append(f"{record.mixture_id}: overlap duration mismatch")
    if (start, end) != (record.overlap_start, record.overlap_end):
        errors.append(f"{record.mixture_id}: overlap interval mismatch")
    expected_ratio = duration / min(a.duration, b.duration)
    if abs(expected_ratio - record.overlap_ratio) > 1e-5:
        errors.append(f"{record.mixture_id}: overlap ratio mismatch")
    return errors


def validate_overlap_records(records: Iterable[OverlapRecord], check_source_files: bool = True) -> list[str]:
    errors: list[str] = []
    for record in records:
        errors.extend(validate_overlap_record(record, check_source_files))
    return errors


def overlap_statistics(records: Iterable[OverlapRecord]) -> dict[str, object]:
    records = list(records)
    durations = [record.duration for record in records]
    overlaps = [record for record in records if record.overlap_exists]
    regimes: dict[str, int] = {}
    speakers: set[str] = set()
    for record in records:
        regimes[record.regime] = regimes.get(record.regime, 0) + 1
        speakers.update(source.speaker_id for source in record.sources if source.speaker_id is not None)
    return {
        "num_mixtures": len(records),
        "total_duration_seconds": float(sum(durations)),
        "mean_duration_seconds": float(np.mean(durations)) if durations else 0.0,
        "min_duration_seconds": float(min(durations)) if durations else 0.0,
        "max_duration_seconds": float(max(durations)) if durations else 0.0,
        "num_overlapping": len(overlaps),
        "num_non_overlapping": len(records) - len(overlaps),
        "mean_overlap_duration_seconds": float(np.mean([r.overlap_duration for r in overlaps])) if overlaps else 0.0,
        "mean_overlap_ratio": float(np.mean([r.overlap_ratio for r in overlaps])) if overlaps else 0.0,
        "regimes": regimes,
        "unique_speakers": len(speakers),
        "relative_gain_db": sorted({record.relative_gain_db for record in records}),
    }


def check_speaker_disjointness(split_records: dict[str, Iterable[OverlapRecord]]) -> list[str]:
    seen: dict[str, str] = {}
    errors: list[str] = []
    for split, records in split_records.items():
        for record in records:
            for source in record.sources:
                if source.speaker_id is None:
                    continue
                previous = seen.setdefault(source.speaker_id, split)
                if previous != split:
                    errors.append(f"speaker leakage: {source.speaker_id} in {previous} and {split}")
    return errors
