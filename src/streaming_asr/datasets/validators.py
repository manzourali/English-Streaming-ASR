from __future__ import annotations

from typing import Iterable

from .manifests import ManifestRecord


def validate_manifest(records: Iterable[ManifestRecord]) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        if not record.audio_path:
            errors.append(f"record {index}: audio_path is empty")
        if record.sample_rate is not None and record.sample_rate <= 0:
            errors.append(f"record {index}: sample_rate must be positive")
    return errors

