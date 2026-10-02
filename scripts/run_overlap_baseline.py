#!/usr/bin/env python3
"""Evaluate the existing single-stream WhisperRT directly on Phase 3 mixtures."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.audio.loader import load_audio
from streaming_asr.audio.stream import AudioStream
from streaming_asr.datasets.manifests import read_overlap_manifest
from streaming_asr.exceptions import BackendNotAvailableError
from streaming_asr.metrics.overlap_baseline import OverlapASRResult, aggregate_overlap_results, control_reference
from streaming_asr.models.whisperrt import TranscriptAccumulator, WhisperRTStreamingASR, resolve_device
from streaming_asr.pipeline.baseline import StreamingASRBaselinePipeline
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def _manifest_paths(config: Config) -> list[Path]:
    root = Path(config.get("data.manifest_root", "outputs/synthetic_overlap"))
    if not root.is_absolute():
        root = ROOT / root
    conditions = config.get("data.conditions", ["train", "validation", "test"])
    return [root / "manifests" / f"{condition}.jsonl" for condition in conditions]


def _load_records(config: Config, max_samples: int | None) -> list:
    records = []
    for path in _manifest_paths(config):
        if not path.is_file():
            raise FileNotFoundError(f"Overlap manifest not found: {path}. Generate Phase 3 data first.")
        records.extend(read_overlap_manifest(path))
    return records[:max_samples] if max_samples is not None else records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overlap_baseline.yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validate-only", action="store_true", help="Validate manifests and write NOT_MEASURED summary without loading WhisperRT")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    paths = ProjectPaths.from_config(config, ROOT)
    paths.ensure_output_dirs()
    set_seed(int(config.get("experiment.seed", 42)))
    run_dir, logger = create_run(config.to_dict(), paths.outputs)
    records = _load_records(config, args.max_samples)
    output_dir = paths.outputs / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{run_dir.name}.jsonl"
    if args.validate_only:
        summary = {"status": "validated_only", "num_examples": len(records), "wer": "NOT_MEASURED", "model_run": "NOT_RUN"}
        (output_dir / f"{run_dir.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    model_config = dict(config.get("model", {}))
    model_config.update({"device": resolve_device(config.get("runtime.device", "auto")), "chunk_ms": config.get("audio.chunk_ms", 300)})
    load_started = time.perf_counter()
    asr = WhisperRTStreamingASR.from_config(model_config)
    model_load_time = time.perf_counter() - load_started
    results: list[OverlapASRResult] = []
    with result_path.open("w", encoding="utf-8") as handle:
        for record in records:
            samples, metadata = load_audio(record.audio_path)
            if metadata.sample_rate != int(config.get("audio.sample_rate", 16000)):
                raise ValueError(f"Sample-rate mismatch for {record.mixture_id}")
            pipeline = StreamingASRBaselinePipeline(asr)
            accumulator = TranscriptAccumulator()
            pipeline.start()
            chunk_size = int(metadata.sample_rate * int(config.get("audio.chunk_ms", 300)) / 1000)
            stream_started = time.perf_counter()
            first_output_latency = None
            incremental = []
            for chunk in AudioStream.from_array(samples, metadata.sample_rate, chunk_size):
                update = pipeline.process(chunk).transcript
                accumulator.update(update)
                incremental.append({"chunk_index": update.chunk_index, "timestamp": update.end_time, "text": update.text, "is_final": update.is_final})
                if first_output_latency is None and update.text:
                    first_output_latency = time.perf_counter() - stream_started
            final_started = time.perf_counter()
            final = pipeline.finalize()
            end_latency = time.perf_counter() - final_started
            hypothesis = accumulator.finalize().text if accumulator.latest else (final.text if final else "")
            processing_time = time.perf_counter() - stream_started
            source_refs = [source.__dict__ for source in record.sources]
            reference = control_reference(source_refs, record.overlap_exists)
            wer = None
            if reference is not None:
                from streaming_asr.metrics.asr import word_error_rate
                wer = word_error_rate(reference, hypothesis)
            result = OverlapASRResult(record.mixture_id, record.regime, record.overlap_ratio, record.overlap_duration, record.relative_gain_db, source_refs, reference, hypothesis, wer, "MEASURED" if wer is not None else "NOT_MEASURED_OVERLAP_SINGLE_STREAM", record.duration, processing_time, processing_time / record.duration if record.duration else None, first_output_latency, end_latency, float(np.mean(asr.processing_times)) if asr.processing_times else None, len(incremental), incremental)
            results.append(result)
            handle.write(json.dumps(result.to_dict()) + "\n")
    summary = aggregate_overlap_results(results)
    payload = {"status": "success", "model_load_time": model_load_time, "model": config.get("model.name"), "device": asr.device, "dtype": asr.actual_dtype, "chunk_ms": config.get("audio.chunk_ms"), "vad_enabled": bool(config.get("vad.enabled", False)), "num_examples": len(results), "conditions": summary}
    (output_dir / f"{run_dir.name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = paths.outputs / "reports" / f"phase4_{run_dir.name}.md"
    report.write_text("# Phase 4 — WhisperRT Under Synthetic Overlap\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in payload.items() if key != "conditions") + "\n\n## Conditions\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in summary.items()) + "\n\nWER is not measured for overlapping mixtures because a single-stream hypothesis has no justified speaker assignment.\n", encoding="utf-8")
    logger.info("Completed Phase 4 with %d examples", len(results))
    print(json.dumps({"results": str(result_path), "metrics": str(output_dir / f"{run_dir.name}.json"), "report": str(report), "conditions": summary}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackendNotAvailableError as exc:
        raise SystemExit(f"Phase 4 cannot start: {exc}")
