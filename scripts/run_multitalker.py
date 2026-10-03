#!/usr/bin/env python3
"""Run the configured Phase 7 external SURT-compatible decoder directly."""
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
from streaming_asr.metrics.asr import permutation_invariant_word_error_rate
from streaming_asr.metrics.streaming import real_time_factor
from streaming_asr.models.multitalker import SURT2WindowedASR
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitalker.yaml")
    parser.add_argument("--override")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override); values = config.to_dict()
    manifest = Path(values["data"]["manifest"])
    if not manifest.is_absolute(): manifest = ROOT / manifest
    records = read_overlap_manifest(manifest)
    limit = args.max_samples if args.max_samples is not None else values["data"].get("max_samples")
    if limit: records = records[:int(limit)]
    if args.validate_only:
        print(json.dumps({"status": "validated", "manifest": str(manifest), "records": len(records), "model": values["model"]["name"], "streaming_mode": "low_latency_windowed"}, indent=2)); return 0
    paths = ProjectPaths.from_config(config, ROOT); paths.ensure_output_dirs()
    set_seed(int(values["experiment"].get("seed", 42)))
    run_dir, logger = create_run(values, paths.outputs)
    model = SURT2WindowedASR.from_config(values["model"])
    rows, wers = [], []
    processing_started = time.perf_counter()
    audio_seconds = 0.0
    for record in records:
        samples, metadata = load_audio(record.audio_path)
        if metadata.sample_rate != int(values["audio"]["sample_rate"]):
            raise ValueError(f"{record.mixture_id}: unsupported sample rate {metadata.sample_rate}")
        model.start(); first_output = None; record_started = time.perf_counter()
        chunk_samples = metadata.sample_rate * int(values["model"]["streaming"]["chunk_duration_ms"]) // 1000
        for chunk in AudioStream.from_array(samples, metadata.sample_rate, chunk_samples):
            output = model.process(chunk)
            if output.streams and first_output is None: first_output = time.perf_counter() - record_started
        final = model.finalize()
        hypothesis = [stream.text for stream in (final.streams if final else ())]
        metric = permutation_invariant_word_error_rate([source.transcript for source in record.sources], hypothesis)
        wers.append(metric.wer); audio_seconds += record.duration
        rows.append({"mixture_id": record.mixture_id, "regime": record.regime, "reference_streams": [source.transcript for source in record.sources], "hypothesis_streams": hypothesis, "pi_wer": metric.wer, "assignment": metric.assignment, "first_output_latency": first_output})
    processing_seconds = time.perf_counter() - processing_started
    metrics = {"samples": len(rows), "mean_permutation_invariant_wer": sum(wers) / len(wers) if wers else None, "audio_seconds": audio_seconds, "processing_seconds": processing_seconds, "rtf": real_time_factor(processing_seconds, audio_seconds) if audio_seconds else None, "streaming_mode": "low_latency_windowed", "note": "Measured only when a verified external SURT decoder_factory is configured."}
    prediction_path = paths.outputs / "predictions" / f"{run_dir.name}_multitalker.jsonl"; prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (paths.outputs / "metrics" / f"{run_dir.name}_multitalker.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = paths.outputs / "reports" / f"{run_dir.name}_multitalker.md"; report_path.write_text("# Phase 7 — SURT 2.0\n\n```json\n" + json.dumps(metrics, indent=2) + "\n```\n\nThis adapter is LOW-LATENCY WINDOWED unless its configured external decoder exposes verified native streaming state.\n", encoding="utf-8")
    logger.info("Completed %d multi-talker samples", len(rows)); print(json.dumps({"metrics": metrics, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
