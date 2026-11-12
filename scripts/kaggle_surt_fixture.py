#!/usr/bin/env python3
"""Build and validate a small English-tone fixture for a pinned SURT recipe.

This is plumbing data, not speech. Run only in the approved Kaggle environment.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

LABELS = {"0L": 0.0, "0S": 0.0, "OV10": 0.1, "OV20": 0.2, "OV30": 0.3, "OV40": 0.4}
MANIFEST_NAMES = (
    "cuts_train_comb_full_sources",
    "cuts_train_comb_ov40_sources",
    "cuts_dev_libricss-sdm",
    "cuts_dev_libricss-ihm-mix",
    "cuts_test_libricss-ihm-mix",
)
RATE = 16000
SOURCE_SECONDS = 1.0
ICEFALL_COMMIT = "f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1"


def probe_empty_decode_group(decoder) -> str:
    try:
        decoder.decode_dataset([], SimpleNamespace(decoding_method="greedy_search"), None, None)
    except UnboundLocalError as exc:
        return f"UnboundLocalError: {exc}"
    return "returned without exception"


def build(root: Path) -> None:
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, Recording, SupervisionSegment
    from lhotse.features.io import NumpyFilesWriter
    from lhotse.features.kaldifeat import (
        KaldifeatFbank, KaldifeatFbankConfig, KaldifeatFrameOptions, KaldifeatMelOptions,
    )

    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output directory must be empty: {root}")
    audio_dir = root / "audio"
    manifest_dir = root / "manifests"
    feature_dir = root / "features"
    for directory in (audio_dir, manifest_dir, feature_dir):
        directory.mkdir(parents=True, exist_ok=True)

    extractor = KaldifeatFbank(KaldifeatFbankConfig(
        frame_opts=KaldifeatFrameOptions(sampling_rate=RATE, dither=0.0),
        mel_opts=KaldifeatMelOptions(num_bins=80), device="cpu",
    ))
    sample_index = np.arange(RATE, dtype=np.float64)
    window = np.sin(np.pi * sample_index / RATE) ** 2
    source_arrays = [
        (0.1 * window * np.sin(2 * np.pi * frequency * sample_index / RATE)).astype(np.float32)
        for frequency in (220, 330)
    ]
    source_paths = []
    for index, samples in enumerate(source_arrays):
        path = audio_dir / f"source-{index}.wav"
        sf.write(path, samples, RATE, subtype="PCM_16")
        source_paths.append(path)

    cuts = []
    with NumpyFilesWriter(feature_dir) as storage:
        sources = []
        for index, path in enumerate(source_paths):
            recording = Recording.from_file(path, recording_id=f"source-{index}")
            sources.append(recording.to_cut().compute_and_store_features(extractor, storage))
        source_features = [cut.load_features() for cut in sources]

        for label, ratio in LABELS.items():
            offset = 1.2 if label == "0L" else 1.05 if label == "0S" else 1.0 - ratio
            offset_samples = round(offset * RATE)
            signal = np.zeros(offset_samples + RATE, dtype=np.float32)
            signal[:RATE] += sources[0].load_audio()[0]
            signal[offset_samples:] += sources[1].load_audio()[0]
            path = audio_dir / f"fixture-{label}.wav"
            sf.write(path, signal, RATE, subtype="PCM_16")
            recording = Recording.from_file(path, recording_id=f"fixture-{label}")
            cut = recording.to_cut()
            cut.supervisions = [
                SupervisionSegment(id=f"{label}-a", recording_id=recording.id, start=0.0,
                                   duration=SOURCE_SECONDS, channel=0, speaker="speaker-a", text="HELLO"),
                SupervisionSegment(id=f"{label}-b", recording_id=recording.id, start=offset,
                                   duration=SOURCE_SECONDS, channel=0, speaker="speaker-b", text="WORLD"),
            ]
            cut = cut.compute_and_store_features(extractor, storage)
            cut.source_feats = storage.store_array(
                f"{label}-source-features", np.concatenate(source_features, axis=0)
            )
            cut.source_feat_offsets = [0, len(source_features[0])]
            cuts.append(cut)

    training = [cut for cut in cuts if cut.id == "fixture-OV40"]
    for name in MANIFEST_NAMES:
        selected = training if name.startswith("cuts_train_") else cuts
        CutSet.from_cuts(selected).to_file(manifest_dir / f"{name}.jsonl.gz")
    print(json.dumps({"root": str(root), "cuts": len(cuts), "training_cuts": len(training)}))


def validate(root: Path, recipe_dir: Path) -> None:
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet
    from lhotse.dataset import K2SurtDataset, PrecomputedFeatures
    from lhotse.features.kaldifeat import (
        KaldifeatFbank, KaldifeatFbankConfig, KaldifeatFrameOptions, KaldifeatMelOptions,
    )
    from lhotse.utils import compute_num_frames

    root = root.resolve()
    recipe_dir = recipe_dir.resolve()
    decoder = recipe_dir / "dprnn_zipformer" / "decode.py"
    if not decoder.is_file():
        raise ValueError(f"Pinned recipe decode.py is missing: {decoder}")
    icefall_root = recipe_dir.parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(icefall_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != ICEFALL_COMMIT:
        raise ValueError(f"Expected Icefall {ICEFALL_COMMIT}, found {commit}")
    sys.path.insert(0, str(icefall_root))
    sys.path.insert(0, str(decoder.parent))
    import decode

    if tuple(decode.OVERLAP_RATIOS) != tuple(LABELS):
        raise ValueError("Recipe decode group labels differ from fixture labels")
    empty_group_result = probe_empty_decode_group(decode)

    dataset = K2SurtDataset(return_cuts=True, return_sources=True,
                            input_strategy=PrecomputedFeatures(), num_channels=2, strict=True)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(
        frame_opts=KaldifeatFrameOptions(sampling_rate=RATE, dither=0.0),
        mel_opts=KaldifeatMelOptions(num_bins=80), device="cpu",
    ))
    expected_sources = []
    for index in range(2):
        waveform, rate = sf.read(root / "audio" / f"source-{index}.wav", dtype="float32")
        if rate != RATE or waveform.ndim != 1 or len(waveform) != RATE:
            raise ValueError("Invalid source audio")
        expected_sources.append(extractor.extract(waveform, RATE))
    summary = {}
    for name in MANIFEST_NAMES:
        path = root / "manifests" / f"{name}.jsonl.gz"
        cuts = CutSet.from_file(path).to_eager()
        is_training = name.startswith("cuts_train_")
        expected = {"fixture-OV40"} if is_training else {f"fixture-{label}" for label in LABELS}
        if {cut.id for cut in cuts} != expected or len(cuts) != len(expected):
            raise ValueError(f"Unexpected or missing cuts in {path}")
        counts = {label: sum(label in cut.id for cut in cuts) for label in LABELS}
        if not is_training and any(count != 1 for count in counts.values()):
            raise ValueError(f"Decode groups are empty or ambiguous: {counts}")
        for cut in cuts:
            if cut.sampling_rate != RATE or cut.duration < 1 or len(cut.supervisions) != 2:
                raise ValueError(f"Invalid recording or supervisions: {cut.id}")
            ordered = sorted(cut.supervisions, key=lambda item: (item.start, item.speaker))
            if [item.text for item in ordered] != ["HELLO", "WORLD"]:
                raise ValueError(f"Invalid supervision text/order: {cut.id}")
            if len({item.speaker for item in ordered}) != 2:
                raise ValueError(f"Invalid speaker identities: {cut.id}")
            for supervision in ordered:
                if (supervision.recording_id != cut.recording_id or supervision.channel != 0
                        or supervision.start < 0 or supervision.duration <= 0
                        or supervision.end > cut.duration + 1 / RATE):
                    raise ValueError(f"Invalid supervision bounds: {cut.id}")
            features = cut.load_features()
            waveform = cut.load_audio()
            source_features = cut.load_source_feats()
            offsets = cut.source_feat_offsets
            if (features.ndim != 2 or features.shape[1] != 80 or features.shape[0] != cut.num_frames
                    or source_features.ndim != 2 or source_features.shape[1] != 80
                    or not np.isfinite(features).all() or not np.isfinite(source_features).all()):
                raise ValueError(f"Invalid feature matrices: {cut.id}")
            if not np.isfinite(waveform).all() or not np.allclose(
                features, extractor.extract(waveform, RATE), rtol=1e-4, atol=1e-4
            ):
                raise ValueError(f"Mixture features do not match audio: {cut.id}")
            if len(offsets) != 2 or offsets[0] != 0 or not 0 < offsets[1] < len(source_features):
                raise ValueError(f"Invalid source offsets: {cut.id}")
            for supervision, source, expected_source in zip(
                ordered, np.split(source_features, offsets[1:]), expected_sources
            ):
                frames = (compute_num_frames(supervision.end, cut.frame_shift, RATE)
                          - compute_num_frames(supervision.start, cut.frame_shift, RATE))
                if abs(len(source) - frames) > 2:
                    raise ValueError(f"Source frames do not match supervision: {cut.id}")
                if source.shape != expected_source.shape or not np.allclose(
                    source, expected_source, rtol=1e-4, atol=1e-4
                ):
                    raise ValueError(f"Source features do not match source audio: {cut.id}")
            label = cut.id.removeprefix("fixture-")
            overlap = max(0.0, min(item.end for item in ordered) - max(item.start for item in ordered))
            if not math.isclose(overlap / SOURCE_SECONDS, LABELS[label], abs_tol=1 / RATE):
                raise ValueError(f"Overlap does not match label: {cut.id}")
            batch = dataset[CutSet.from_cuts([cut])]
            if (batch["inputs"].shape[0] != 1 or batch["inputs"].shape[-1] != 80
                    or len(batch["source_feats"][0]) != 2
                    or len(batch["source_boundaries"][0]) != 2):
                raise ValueError(f"K2SurtDataset rejected source contract: {cut.id}")
            if is_training and batch["text"][0] != ["HELLO", "WORLD"]:
                raise ValueError(f"Training targets are not split between branches: {cut.id}")
        summary[name] = {"cuts": len(cuts), "groups": counts}
    print(json.dumps({"status": "validated", "empty_decode_group_probe": empty_group_result,
                      "manifests": summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("output", type=Path)
    validation_parser = commands.add_parser("validate")
    validation_parser.add_argument("output", type=Path)
    validation_parser.add_argument("--recipe-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output)
    else:
        validate(args.output, args.recipe_dir)


if __name__ == "__main__":
    main()
