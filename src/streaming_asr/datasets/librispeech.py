"""Prepare speaker-disjoint local LibriSpeech manifests without HF audio APIs."""
from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import random

from .manifests import ManifestRecord, write_manifest


def _scan(directory: Path) -> list[ManifestRecord]:
    records = []
    for transcript in sorted(directory.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            identifier, text = line.split(maxsplit=1)
            audio = transcript.parent / f"{identifier}.flac"
            if not audio.is_file():
                audio = transcript.parent / f"{identifier}.wav"
            if not audio.is_file():
                raise FileNotFoundError(f"Missing audio for {identifier}")
            records.append(ManifestRecord(str(audio.resolve()), text, 16000,
                {"source_id": identifier, "speaker_id": identifier.split("-")[0]}))
    if not records:
        raise ValueError(f"No LibriSpeech transcripts found in {directory}")
    return records


def _balanced_limit(records, limit):
    by_speaker = defaultdict(list)
    for record in records:
        by_speaker[record.metadata["speaker_id"]].append(record)
    balanced = [record for group in zip_longest(*by_speaker.values()) for record in group if record is not None]
    return balanced[:limit] if limit else balanced


def prepare_librispeech(source_root, output_root, *, layout="pilot", max_per_split=100, seed=42):
    """Pilot partitions dev-clean speakers; standard preserves official splits.

    Pilot output is for development only. It is not an official test-clean
    benchmark. Source paths are provenance; generated mixtures are portable.
    """
    if max_per_split is not None and max_per_split < 2:
        raise ValueError("max_per_split must be at least two")
    source_root, output_root = Path(source_root), Path(output_root)
    if layout == "pilot":
        records = _scan(source_root / "dev-clean")
        speakers = sorted({r.metadata["speaker_id"] for r in records})
        if len(speakers) < 6:
            raise ValueError("Pilot preparation requires at least six speakers")
        random.Random(seed).shuffle(speakers)
        first = min(len(speakers) - 4, max(2, int(.6 * len(speakers))))
        second = min(len(speakers) - first - 2, max(2, int(.2 * len(speakers))))
        groups = {"train": speakers[:first], "validation": speakers[first:first + second],
                  "test": speakers[first + second:]}
        splits = {split: [r for r in records if r.metadata["speaker_id"] in set(group)]
                  for split, group in groups.items()}
    elif layout == "standard":
        splits = {split: _scan(source_root / name) for split, name in
                  {"train": "train-clean-100", "validation": "dev-clean", "test": "test-clean"}.items()}
    else:
        raise ValueError("layout must be pilot or standard")
    seen = set()
    summary = {}
    for split, records in splits.items():
        records = _balanced_limit(records, max_per_split)
        speakers = {r.metadata["speaker_id"] for r in records}
        if len(speakers) < 2 or speakers & seen:
            raise ValueError(f"{split} needs at least two speakers and no split leakage")
        seen.update(speakers)
        path = output_root / f"{split}.jsonl"
        write_manifest(records, path)
        summary[split] = {"manifest": str(path), "utterances": len(records), "speakers": len(speakers)}
    return {"layout": layout, "purpose": "development_only" if layout == "pilot" else "standard_splits",
            "splits": summary}
