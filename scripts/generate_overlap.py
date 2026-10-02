#!/usr/bin/env python3
"""Generate and validate a small or configured synthetic overlap dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.audio.loader import load_audio, write_audio
from streaming_asr.datasets.loaders import iter_librispeech
from streaming_asr.datasets.overlap_generator import OverlapGenerator, SourceUtterance
from streaming_asr.datasets.validators import check_speaker_disjointness, overlap_statistics, validate_overlap_records
from streaming_asr.utils.config import Config
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def _demo_sources(root: Path, sample_rate: int) -> dict[str, list[SourceUtterance]]:
    sources: dict[str, list[SourceUtterance]] = {}
    for split, speaker_names in {"train": ["train_a", "train_b", "train_c"], "validation": ["dev_a", "dev_b", "dev_c"], "test": ["test_a", "test_b", "test_c"]}.items():
        items = []
        for index, speaker in enumerate(speaker_names):
            duration = 0.6 + 0.1 * index
            samples = (0.25 * np.sin(np.linspace(0, 2 * np.pi * (180 + index * 50) * duration, int(sample_rate * duration), endpoint=False))).astype(np.float32)
            path = root / "sources" / split / f"{speaker}.wav"
            write_audio(path, samples, sample_rate)
            items.append(SourceUtterance(f"{split}_{index}", speaker, samples, sample_rate, f"synthetic source {speaker}", str(path)))
        sources[split] = items
    return sources


def _source_from_example(example, root: Path, sample_rate: int) -> SourceUtterance:
    source_path = str(example.audio) if isinstance(example.audio, (str, Path)) else str(example.sample_id)
    if isinstance(example.audio, (str, Path)):
        samples, metadata = load_audio(example.audio)
    else:
        samples = np.asarray(example.audio["array"], dtype=np.float32)
        metadata = type("Metadata", (), {"sample_rate": int(example.audio["sampling_rate"])})()
    return SourceUtterance(example.sample_id, example.speaker_id, samples, metadata.sample_rate, example.reference, source_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overlap_dataset.yaml")
    parser.add_argument("--demo", action="store_true", help="Use tiny deterministic fixture sources; no download")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    paths = ProjectPaths.from_config(config, ROOT)
    paths.ensure_output_dirs()
    seed = int(config.get("experiment.seed", 42))
    set_seed(seed)
    sample_rate = int(config.get("audio.sample_rate", 16000))
    output_root = Path(config.get("output.root", paths.outputs / "synthetic_overlap"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if args.demo:
        sources = _demo_sources(output_root, sample_rate)
    else:
        source_splits = config.get("data.source_splits", {"train": "train.100", "validation": "validation", "test": "test"})
        sources = {}
        for split, source_split in source_splits.items():
            data_config = dict(config.get("data", {}))
            data_config["split"] = source_split
            max_source = int(config.get("generation.max_source_samples", 20))
            data_config["max_samples"] = max_source
            sources[split] = [_source_from_example(example, output_root, sample_rate) for example in iter_librispeech(data_config)]
    regimes = {name: float(value.get("target_ratio", value)) for name, value in config.get("overlap.regimes", {"low": {"target_ratio": 0.25}, "medium": {"target_ratio": 0.5}, "high": {"target_ratio": 0.8}, "control": {"target_ratio": 0.0}}).items()}
    generator = OverlapGenerator(output_root, sample_rate, seed, bool(config.get("generation.normalize_rms", True)))
    results = generator.generate_dataset(sources, num_samples=int(config.get("generation.num_samples", 3)), regimes=regimes, relative_gains_db=[float(value) for value in config.get("mix.relative_gain_db", [0.0, -3.0])])
    errors = []
    for records in results.values():
        errors.extend(validate_overlap_records(records, check_source_files=args.demo))
    errors.extend(check_speaker_disjointness(results))
    stats = {split: overlap_statistics(records) for split, records in results.items()}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    summary = {"status": "success" if not errors else "failed", "output_root": str(output_root), "splits": stats, "validation_errors": errors}
    print(json.dumps(summary, indent=2))
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
