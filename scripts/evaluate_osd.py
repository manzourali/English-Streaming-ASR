#!/usr/bin/env python3
"""Evaluate streaming OSD against independent Phase 3 timing labels."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.audio.loader import load_audio
from streaming_asr.audio.stream import AudioStream
from streaming_asr.datasets.manifests import read_overlap_manifest
from streaming_asr.metrics.osd import align_frames, event_detection_delays, ground_truth_frames, overlap_metrics, predictions_to_frames
from streaming_asr.models.overlap_detector import get_overlap_detector
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overlap_detection.yaml")
    parser.add_argument("--override", help="Optional YAML override, useful for Kaggle paths")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override)
    paths = ProjectPaths.from_config(config, ROOT)
    paths.ensure_output_dirs()
    set_seed(int(config.get("experiment.seed", 42)))
    run_dir, logger = create_run(config.to_dict(), paths.outputs)
    manifest_root = Path(config.get("data.manifest_root", "outputs/synthetic_overlap"))
    if not manifest_root.is_absolute():
        manifest_root = ROOT / manifest_root
    splits = config.get("data.splits", ["test"])
    records = []
    for split in splits:
        records.extend(read_overlap_manifest(manifest_root / "manifests" / f"{split}.jsonl"))
    if args.max_samples is not None:
        records = records[:args.max_samples]
    frame_ms = int(config.get("osd.frame_ms", 30))
    backend_config = dict(config.get("osd", {}))
    backend_config["sample_rate"] = int(config.get("audio.sample_rate", 16000))
    detector_name = config.get("osd.backend", "heuristic")
    results = []
    all_pairs = []
    all_delays = []
    started = time.perf_counter()
    for record in records:
        samples, metadata = load_audio(record.audio_path)
        detector = get_overlap_detector(detector_name, backend_config)
        detector.start()
        predictions = []
        chunk_size = int(metadata.sample_rate * frame_ms / 1000)
        for chunk in AudioStream.from_array(samples, metadata.sample_rate, chunk_size):
            predictions.append(detector.process(chunk))
        final = detector.finalize()
        if final is not None:
            predictions.append(final)
        reference_frames = ground_truth_frames(record, frame_ms)
        predicted_frames = predictions_to_frames(predictions)
        pairs = align_frames(reference_frames, predicted_frames)
        all_pairs.extend(pairs)
        delays = event_detection_delays(reference_frames, predicted_frames)
        all_delays.extend(delays)
        results.append({"mixture_id": record.mixture_id, "regime": record.regime, "overlap_ratio": record.overlap_ratio, "relative_gain_db": record.relative_gain_db, "reference_frames": [frame.__dict__ | {"label": frame.label.value} for frame in reference_frames], "predictions": [{"start_time": result.start_time, "end_time": result.end_time, "label": result.label.value, "overlap_probability": result.probability, "spectral_complexity": result.spectral_complexity} for result in predictions], "metrics": overlap_metrics(pairs), "detection_delays": delays, "audio_duration": record.duration})
    elapsed = time.perf_counter() - started
    aggregate = overlap_metrics(all_pairs)
    aggregate.update({"backend": detector_name, "frame_ms": frame_ms, "num_examples": len(records), "mean_detection_latency": sum(all_delays) / len(all_delays) if all_delays else None, "processing_time": elapsed, "total_audio_duration": sum(record.duration for record in records), "osd_rtf": elapsed / sum(record.duration for record in records) if records else None, "latency_status": "MEASURED" if all_delays else "NOT_MEASURED"})
    output_dir = paths.outputs / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{run_dir.name}.jsonl"
    results_path.write_text("\n".join(json.dumps(result) for result in results) + "\n", encoding="utf-8")
    metrics_path = output_dir / f"{run_dir.name}.json"
    metrics_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    report_path = paths.outputs / "reports" / f"phase5_{run_dir.name}.md"
    report_path.write_text("# Phase 5 — Streaming OSD\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in aggregate.items()) + "\n\nThis is a heuristic OSD baseline on synthetic mixtures; it is not diarization or source separation.\n", encoding="utf-8")
    logger.info("Completed OSD evaluation: %s", aggregate)
    print(json.dumps({"metrics": str(metrics_path), "results": str(results_path), "report": str(report_path), "summary": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
