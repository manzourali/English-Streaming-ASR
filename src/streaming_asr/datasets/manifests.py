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


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(record)) for record in records) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    return [ManifestRecord(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

