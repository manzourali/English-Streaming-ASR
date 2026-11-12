"""Convert this project's overlap JSONL manifests to Lhotse manifests.

Input is the schema emitted by ``write_overlap_manifest``; arbitrary FastMSS
metadata must first be mapped to that schema. Output contains mono recordings,
one timed supervision per speaker, and one cut per mixture, without features.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import warnings
import wave

from .manifests import read_overlap_manifest


def convert(manifest: str | Path, output_dir: str | Path, split: str) -> dict[str, Path]:
    if not split or Path(split).name != split or split in {".", ".."}:
        raise ValueError("split must be a single nonempty path component")
    records = read_overlap_manifest(manifest)
    if not records:
        raise ValueError("Input manifest is empty")
    recordings, supervisions, cuts = [], [], []
    ids: set[str] = set()
    for record in records:
        identifier = record.mixture_id
        if not identifier or identifier in ids:
            raise ValueError(f"Empty or duplicate mixture ID: {identifier!r}")
        ids.add(identifier)
        if record.split != split:
            raise ValueError(f"{identifier}: split does not match {split!r}")
        with wave.open(record.audio_path, "rb") as audio:
            rate, frames, channels = audio.getframerate(), audio.getnframes(), audio.getnchannels()
        if channels != 1 or rate != 16000 or frames <= 0:
            raise ValueError(f"{identifier}: expected nonempty mono 16 kHz WAV")
        duration = frames / rate
        if record.sample_rate != rate or not math.isfinite(record.duration) or abs(record.duration - duration) > 1 / rate + 1e-8:
            raise ValueError(f"{identifier}: audio length or sample rate mismatch")
        if record.num_speakers != 2 or len(record.sources) != 2:
            raise ValueError(f"{identifier}: exactly two sources are required")
        speakers = [source.speaker_id for source in record.sources]
        if any(not speaker or not speaker.strip() for speaker in speakers) or speakers[0] == speakers[1]:
            raise ValueError(f"{identifier}: two distinct speaker IDs are required")
        recording = {"id": identifier, "sources": [{"type": "file", "channels": [0], "source": str(Path(record.audio_path).resolve())}], "sampling_rate": rate, "num_samples": frames, "duration": duration, "channel_ids": [0]}
        local_supervisions = []
        for index, source in enumerate(record.sources, 1):
            if not source.transcript or not source.transcript.strip():
                raise ValueError(f"{identifier}: speaker {index} transcript is empty")
            if not all(math.isfinite(value) for value in (source.start, source.end, source.duration)) or source.start < 0 or source.start >= duration or source.duration <= 0 or abs(source.end - source.start - source.duration) > 1 / rate + 1e-8 or source.end > duration + 1 / rate + 1e-8:
                raise ValueError(f"{identifier}: invalid speaker {index} segment")
            supervision = {"id": f"{identifier}-speaker{index}", "recording_id": identifier, "start": source.start, "duration": min(source.duration, duration - source.start), "channel": 0, "text": source.transcript, "speaker": source.speaker_id, "custom": {"source_index": index}}
            local_supervisions.append(supervision)
        overlap = max(0.0, min(source.end for source in record.sources) - max(source.start for source in record.sources))
        ratio = overlap / min(source.duration for source in record.sources)
        if not math.isfinite(record.overlap_ratio) or not 0 <= record.overlap_ratio <= 1 or abs(ratio - record.overlap_ratio) > 0.01:
            warnings.warn(f"{identifier}: declared overlap ratio differs from source timing", stacklevel=2)
        recordings.append(recording)
        supervisions.extend(local_supervisions)
        cuts.append({"id": identifier, "start": 0.0, "duration": duration, "channel": 0, "supervisions": local_supervisions, "recording": recording, "type": "MonoCut"})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for kind, rows in (("recordings", recordings), ("supervisions", supervisions), ("cuts", cuts)):
        path = output_dir / f"persian_{kind}_{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs[kind] = path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Overlap JSONL produced by this project's generator")
    parser.add_argument("output_dir", help="Destination for Lhotse JSONL manifests")
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    for path in convert(args.manifest, args.output_dir, args.split).values():
        print(path)


if __name__ == "__main__":
    main()
