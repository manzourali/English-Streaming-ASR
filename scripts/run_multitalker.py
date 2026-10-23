#!/usr/bin/env python3
"""Run continuous SURT 2.0 inference and optional corpus PI-WER scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.audio.loader import load_audio
from streaming_asr.audio.stream import AudioStream
from streaming_asr.datasets.manifests import read_overlap_manifest
from streaming_asr.exceptions import BackendNotAvailableError
from streaming_asr.evaluation.records import surt_prediction_record, write_evaluation_records
from streaming_asr.metrics.asr import permutation_invariant_word_error_rate
from streaming_asr.models.multitalker import SURT2StreamingASR
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def recognize(model, audio_path):
    samples, metadata = load_audio(audio_path)
    if metadata.sample_rate != 16000:
        raise ValueError(f"{audio_path}: expected 16000 Hz, got {metadata.sample_rate}")
    model.start()
    texts = [[], []]
    endpoints = []
    first_output_sample = first_output_processing = None
    started = time.perf_counter()

    def collect(update):
        nonlocal first_output_sample, first_output_processing
        for stream in update.streams:
            if stream.new_text:
                texts[stream.stream_id].append(stream.new_text)
                if first_output_sample is None:
                    first_output_sample = update.available_sample
                    first_output_processing = time.perf_counter() - started
        endpoints.extend({"stream_id": channel, "available_sample": update.available_sample}
                         for channel in update.endpoints)

    for chunk in AudioStream.from_array(samples, 16000, 5120):
        collect(model.process(chunk))
    finalize_started = time.perf_counter()
    collect(model.finalize())
    finalize_seconds = time.perf_counter() - finalize_started
    return {
        "audio_path": str(audio_path),
        "hypothesis_streams": [" ".join(parts) for parts in texts],
        "audio_seconds": len(samples) / 16000,
        "processing_seconds": time.perf_counter() - started,
        "first_output_available_sample": first_output_sample,
        "first_output_processing_seconds": first_output_processing,
        "finalization_seconds": finalize_seconds,
        "endpoints": endpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multitalker.yaml")
    parser.add_argument("--override")
    parser.add_argument("--audio", help="Transcribe one 16 kHz file without a manifest or WER scoring")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validate-only", action="store_true", help="Validate input paths, not model compatibility")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    config = Config.from_yaml(args.config, args.override)
    values = config.to_dict()
    records = []
    if args.audio:
        if not Path(args.audio).is_file():
            parser.error(f"Audio file not found: {args.audio}")
    else:
        manifest = Path(values["data"]["manifest"])
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        records = read_overlap_manifest(manifest)
        limit = args.max_samples or values["data"].get("max_samples")
        if limit is not None:
            records = records[:int(limit)]
        if not records:
            parser.error("Manifest has no selected records")
    if args.validate_only:
        print(json.dumps({"status": "inputs_validated", "records": len(records),
                          "audio": args.audio, "model_executed": False}, indent=2))
        return 0

    set_seed(int(values["experiment"].get("seed", 42)))
    model_config = dict(values["model"])
    model_config.setdefault("device", values.get("runtime", {}).get("device", "cpu"))
    model = SURT2StreamingASR.from_config(model_config)
    if args.audio:
        print(json.dumps(recognize(model, args.audio), indent=2))
        return 0

    paths = ProjectPaths.from_config(config, ROOT)
    paths.ensure_output_dirs()
    run_dir, logger = create_run(values, paths.outputs)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    total_errors = total_words = 0
    rows = []
    for record in records:
        row = recognize(model, record.audio_path)
        references = [source.transcript for source in record.sources]
        metric = permutation_invariant_word_error_rate(references, row["hypothesis_streams"])
        total_errors += metric.errors
        total_words += metric.reference_words
        row.update(mixture_id=record.mixture_id, regime=record.regime,
                   overlap_ratio=record.overlap_ratio, relative_gain_db=record.relative_gain_db,
                   reference_streams=references, pi_wer=metric.wer,
                   errors=metric.errors, reference_words=metric.reference_words,
                   assignment=metric.assignment)
        rows.append(row)
    audio_seconds = sum(row["audio_seconds"] for row in rows)
    processing_seconds = sum(row["processing_seconds"] for row in rows)
    metrics = {
        "samples": len(rows), "errors": total_errors, "reference_words": total_words,
        "corpus_permutation_invariant_wer": total_errors / total_words if total_words else None,
        "audio_seconds": audio_seconds, "processing_seconds": processing_seconds,
        "rtf": processing_seconds / audio_seconds if audio_seconds else None,
        "streaming_mode": model.streaming_mode, "chunk_frames": 32,
        "right_context_frames": model.backend.right_context_frames,
        "max_acoustic_state_bytes": model.backend.max_state_bytes,
        "timing_note": "Unpaced file inference; processing time excludes model load and file I/O. "
                       "Available-sample timestamps measure acquisition; endpoint delay is additional.",
    }
    prediction_path = paths.outputs / "predictions" / f"{run_dir.name}_multitalker.jsonl"
    evaluation_path = paths.outputs / "predictions" / f"{run_dir.name}_evaluation_records.jsonl"
    write_evaluation_records([surt_prediction_record(row,
        system=values["experiment"].get("system", "surt_base"),
        dataset=values["data"].get("dataset_id", "synthetic_overlap"),
        split=records[index].split, checkpoint=model_config.get("checkpoint"),
        manifest_sha256=manifest_sha256)
        for index, row in enumerate(rows)], evaluation_path)
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (paths.outputs / "metrics" / f"{run_dir.name}_multitalker.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Completed %d continuous multi-talker recordings", len(rows))
    print(json.dumps({"metrics": metrics, "predictions": str(prediction_path),
                      "evaluation_records": str(evaluation_path)}, indent=2))
    return 0


def cli() -> int:
    try:
        return main()
    except BackendNotAvailableError as exc:
        raise SystemExit(f"SURT cannot start: {exc}")


if __name__ == "__main__":
    raise SystemExit(cli())
