#!/usr/bin/env python3
"""Run the Phase 1 WhisperRT streaming baseline."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.audio.loader import load_audio
from streaming_asr.audio.preprocessing import to_mono
from streaming_asr.audio.stream import AudioStream
from streaming_asr.datasets.loaders import iter_librispeech
from streaming_asr.exceptions import BackendNotAvailableError
from streaming_asr.metrics.asr import normalize_text
from streaming_asr.metrics.benchmark import UtteranceMetrics, summarize
from streaming_asr.models.whisperrt import WhisperRTStreamingASR, TranscriptAccumulator, resolve_device
from streaming_asr.pipeline.baseline import StreamingASRBaselinePipeline
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def _audio_array(audio: object) -> tuple[np.ndarray, int]:
    if isinstance(audio, (str, Path)):
        samples, metadata = load_audio(audio)
        return samples, metadata.sample_rate
    if isinstance(audio, dict):
        return to_mono(np.asarray(audio["array"], dtype=np.float32)), int(audio["sampling_rate"])
    raise ValueError("Dataset audio must be a path or an audio mapping with array and sampling_rate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whisperrt_baseline.yaml")
    parser.add_argument("--override")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override)
    values = config.to_dict()
    if args.max_samples is not None:
        values.setdefault("data", {})["max_samples"] = args.max_samples
    paths = ProjectPaths.from_config(Config(values), ROOT)
    paths.ensure_output_dirs()
    set_seed(int(config.get("experiment.seed", 42)))
    run_dir, logger = create_run(values, paths.outputs)
    run_id = run_dir.name
    model_values = dict(config.get("model", {}))
    model_values.update({"device": resolve_device(config.get("runtime.device", "auto")), "chunk_ms": config.get("audio.chunk_ms", 300)})
    logger.info("Loading WhisperRT model %s", model_values.get("name"))
    load_started = time.perf_counter()
    asr = WhisperRTStreamingASR.from_config(model_values)
    logger.info("Model loaded in %.3fs", time.perf_counter() - load_started)
    records: list[UtteranceMetrics] = []
    prediction_path = paths.outputs / "predictions" / f"{run_id}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as prediction_file:
        for example in iter_librispeech(dict(config.get("data", {}))):
            samples, sample_rate = _audio_array(example.audio)
            target_rate = int(config.get("audio.sample_rate", 16000))
            if sample_rate != target_rate:
                raise ValueError(f"Expected {target_rate} Hz audio, received {sample_rate} Hz for {example.sample_id}")
            chunk_size = int(target_rate * int(config.get("audio.chunk_ms", 300)) / 1000)
            pipeline = StreamingASRBaselinePipeline(asr)
            accumulator = TranscriptAccumulator()
            pipeline.start()
            process_started = time.perf_counter()
            first_output_latency = None
            for chunk in AudioStream.from_array(samples, target_rate, chunk_size):
                output = pipeline.process(chunk)
                accumulator.update(output.transcript)
                if first_output_latency is None and output.transcript.text:
                    first_output_latency = time.perf_counter() - process_started
            finalization_started = time.perf_counter()
            final = pipeline.finalize()
            finalization_latency = time.perf_counter() - finalization_started
            prediction = accumulator.finalize().text if accumulator.latest else (final.text if final else "")
            processing_time = time.perf_counter() - process_started
            duration = len(samples) / target_rate
            record = UtteranceMetrics(example.sample_id, example.reference, normalize_text(example.reference), prediction, normalize_text(prediction), duration, processing_time, processing_time / duration if duration else 0.0, first_output_latency, finalization_latency, float(np.mean(asr.processing_times)) if asr.processing_times else None)
            records.append(record)
            prediction_file.write(json.dumps(record.__dict__) + "\n")
    metrics = summarize(records)
    metrics.update({"model": config.get("model.name"), "device": asr.device, "dtype": asr.actual_dtype, "chunk_ms": config.get("audio.chunk_ms"), "sample_rate": config.get("audio.sample_rate"), "dataset": config.get("data.dataset"), "split": config.get("data.split")})
    metrics_path = paths.outputs / "metrics" / f"{run_id}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = paths.outputs / "reports" / f"{run_id}.md"
    report_path.write_text("# Phase 1 — WhisperRT Streaming Baseline\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in metrics.items()) + "\n\n## Limitations\n\nThis run evaluates clean single-speaker speech only; VAD, OSD, overlap, and multi-talker recognition are not included.\n", encoding="utf-8")
    logger.info("Completed baseline with %d samples", len(records))
    print(json.dumps({"run_id": run_id, "metrics": metrics, "predictions": str(prediction_path), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackendNotAvailableError as exc:
        raise SystemExit(f"Phase 1 cannot start: {exc}")
