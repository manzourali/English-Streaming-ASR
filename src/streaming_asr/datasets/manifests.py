from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ManifestRecord:
    audio_path: str
    transcript: str = ""
    sample_rate: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OverlapSource:
    source_id: str
    speaker_id: str | None
    audio_path: str
    start: float
    end: float
    duration: float
    transcript: str
    gain_db: float = 0.0


@dataclass(frozen=True)
class OverlapRecord:
    mixture_id: str
    audio_path: str
    sample_rate: int
    duration: float
    num_speakers: int
    sources: list[OverlapSource]
    overlap_exists: bool
    overlap_start: float | None
    overlap_end: float | None
    overlap_duration: float
    overlap_ratio: float
    regime: str
    relative_gain_db: float
    split: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(record)) for record in records) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    return [ManifestRecord(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_overlap_manifest(records: Iterable[OverlapRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_overlap_manifest(path: str | Path) -> list[OverlapRecord]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        value["sources"] = [OverlapSource(**source) for source in value["sources"]]
        records.append(OverlapRecord(**value))
    return records
