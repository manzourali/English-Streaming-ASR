"""Phase 8 training representations derived from Phase 3 overlap manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from streaming_asr.datasets.manifests import OverlapRecord


@dataclass(frozen=True)
class SurtTrainingExample:
    """Two unordered SURT target channels; IDs are never person identities."""
    sample_id: str
    audio_path: str
    sample_rate: int
    duration: float
    split: str
    num_speakers: int
    speaker_ids: tuple[str | None, ...]
    source_ids: tuple[str, ...]
    source_transcripts: tuple[str, ...]
    channel_targets: tuple[str, ...]
    overlap_start: float | None
    overlap_end: float | None
    overlap_ratio: float
    regime: str
    relative_gain_db: float
    mixing_parameters: dict[str, object]
    target_format: str = "heat_two_channel_transcripts"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def surt_training_example(record: OverlapRecord) -> SurtTrainingExample:
    """Create HEAT-style channels by source onset, as used by SURT recipes.

    This deterministic transformation is only valid for the project's current
    two-source Phase 3 data. The external recipe/tokenizer remains responsible
    for turning channel transcripts into its model-specific token IDs.
    """
    if record.num_speakers != 2 or len(record.sources) != 2:
        raise ValueError(f"{record.mixture_id}: Phase 8 currently requires exactly two sources")
    sources = sorted(record.sources, key=lambda source: (source.start, source.source_id))
    targets = tuple(source.transcript.strip() for source in sources)
    if not all(targets):
        raise ValueError(f"{record.mixture_id}: source transcript is empty")
    return SurtTrainingExample(
        sample_id=record.mixture_id,
        audio_path=record.audio_path,
        sample_rate=record.sample_rate,
        duration=record.duration,
        split=record.split,
        num_speakers=record.num_speakers,
        speaker_ids=tuple(source.speaker_id for source in sources),
        source_ids=tuple(source.source_id for source in sources),
        source_transcripts=tuple(source.transcript for source in sources),
        channel_targets=targets,
        overlap_start=record.overlap_start,
        overlap_end=record.overlap_end,
        overlap_ratio=record.overlap_ratio,
        regime=record.regime,
        relative_gain_db=record.relative_gain_db,
        mixing_parameters={"seed": record.seed, "regime": record.regime, "relative_gain_db": record.relative_gain_db, "overlap_duration": record.overlap_duration},
    )


def surt_training_examples(records: Iterable[OverlapRecord]) -> list[SurtTrainingExample]:
    return [surt_training_example(record) for record in records]


def write_surt_training_manifest(examples: Iterable[SurtTrainingExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(example.to_dict(), sort_keys=True) for example in examples) + "\n", encoding="utf-8")


def select_training_condition(examples: Iterable[SurtTrainingExample], condition: str) -> list[SurtTrainingExample]:
    """Select the small Phase 8 base/clean/overlap/mixed data conditions."""
    values = list(examples)
    if condition == "base":
        return []
    if condition == "clean":
        return [item for item in values if item.overlap_ratio == 0]
    if condition == "overlap":
        return [item for item in values if item.overlap_ratio > 0]
    if condition == "mixed":
        return values
    raise ValueError("training condition must be base, clean, overlap, or mixed")


def validate_training_examples(examples: Iterable[SurtTrainingExample]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for example in examples:
        if example.sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {example.sample_id}")
        seen_ids.add(example.sample_id)
        if example.sample_rate <= 0 or example.duration <= 0:
            errors.append(f"{example.sample_id}: invalid audio metadata")
        if len(example.channel_targets) != 2 or not all(example.channel_targets):
            errors.append(f"{example.sample_id}: expected two non-empty HEAT channel targets")
        if example.target_format != "heat_two_channel_transcripts":
            errors.append(f"{example.sample_id}: unsupported target format {example.target_format}")
    return errors


def leakage_errors(split_examples: dict[str, Iterable[SurtTrainingExample]]) -> list[str]:
    """Reject speaker/source/audio reuse across train, validation, and test."""
    errors: list[str] = []
    seen: dict[str, str] = {}
    for split, examples in split_examples.items():
        for example in examples:
            for prefix, values in (("speaker", example.speaker_ids), ("source", example.source_ids), ("audio", (example.audio_path,))):
                for value in values:
                    if value is None:
                        continue
                    key = f"{prefix}:{value}"
                    prior = seen.setdefault(key, split)
                    if prior != split:
                        errors.append(f"{prefix} leakage: {value} in {prior} and {split}")
    return errors
