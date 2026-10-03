#!/usr/bin/env python3
"""Run Phase 6 always-normal, oracle, and predicted adaptive controls."""
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
from streaming_asr.datasets.manifests import OverlapRecord, read_overlap_manifest
from streaming_asr.metrics.adaptive import route_durations, routing_metrics, transition_metrics
from streaming_asr.metrics.streaming import real_time_factor
from streaming_asr.models.overlap_detector import OracleOverlapDetector, get_overlap_detector, OverlapLabel
from streaming_asr.models.vad import get_vad_backend
from streaming_asr.models.whisperrt import WhisperRTStreamingASR
from streaming_asr.models.multitalker import SURT2WindowedASR
from streaming_asr.pipeline.adaptive import AdaptiveStreamingASRPipeline, AlwaysNormalRoutingPolicy, RoutingPolicy
from streaming_asr.pipeline.overlap import MultiTalkerASRBranch
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def _label(record: OverlapRecord, timestamp: float) -> OverlapLabel:
    active = sum(source.start <= timestamp < source.end for source in record.sources)
    return OverlapLabel.OVERLAP if active >= 2 else OverlapLabel.SINGLE_SPEAKER if active else OverlapLabel.NO_SPEECH


def _policy(values: dict, mode: str):
    routing = values["routing"]
    cls = AlwaysNormalRoutingPolicy if mode == "always_normal" else RoutingPolicy
    return cls(
        overlap_threshold=float(routing["overlap_threshold"]),
        enter_overlap_frames=int(routing["enter_overlap_frames"]),
        exit_overlap_frames=int(routing["exit_overlap_frames"]),
        min_overlap_duration_ms=float(routing.get("min_overlap_duration_ms", 0)),
        use_vad=bool(routing.get("use_vad", True)),
        idle_behavior=str(routing.get("idle_behavior", "forward_normal")),
    )


def _run_mode(values: dict, records: list[OverlapRecord], mode: str) -> dict[str, object]:
    sample_rate = int(values["audio"]["sample_rate"])
    chunk_samples = sample_rate * int(values["audio"]["chunk_ms"]) // 1000
    all_pairs, all_states, all_events, all_delays, traces = [], [], [], [], []
    model_config = dict(values["model"])
    model_config["chunk_ms"] = values["audio"]["chunk_ms"]
    # Loading is intentionally excluded from audio-processing RTF, matching
    # the Phase 1 benchmark definition. ``start`` resets it per recording.
    asr = WhisperRTStreamingASR.from_config(model_config)
    overlap_branch = None
    if mode != "always_normal" and values.get("branches", {}).get("overlap") == "surt2":
        overlap_branch = MultiTalkerASRBranch(SURT2WindowedASR.from_config(values["multitalker"]))
    wall_started = time.perf_counter()
    for record in records:
        samples, metadata = load_audio(record.audio_path)
        if metadata.sample_rate != sample_rate:
            raise ValueError(f"{record.mixture_id}: expected {sample_rate} Hz, got {metadata.sample_rate}")
        vad = get_vad_backend(values["vad"]["backend"], {**values["vad"], "sample_rate": sample_rate}) if values["vad"].get("enabled", True) else None
        predicted = get_overlap_detector(values["osd"]["backend"], {**values["osd"], "sample_rate": sample_rate}) if values["osd"].get("enabled", True) else None
        oracle = OracleOverlapDetector([(source.start, source.end) for source in record.sources])
        history_ms = values["routing"].get("overlap_pre_roll_ms", values["routing"].get("history_ms", 0)) if overlap_branch is not None else values["routing"].get("history_ms", 0)
        pipeline = AdaptiveStreamingASRPipeline(asr, overlap_branch, vad=vad, overlap_detector=predicted, oracle_detector=oracle, oracle=mode == "oracle", policy=_policy(values, mode), history_ms=float(history_ms), fallback_route=values["routing"].get("fallback_route", "normal"))
        pipeline.start()
        for chunk in AudioStream.from_array(samples, sample_rate, chunk_samples):
            output = pipeline.process(chunk)
            truth = _label(record, (chunk.start_time + chunk.end_time) / 2)
            all_pairs.append((truth, output.route))
            all_states.append((output.state, chunk.duration))
            traces.append({"mixture_id": record.mixture_id, "chunk_index": chunk.index, "timestamp": output.timestamp, "ground_truth": truth.value, "route": output.route.value, "state": output.state.value, "confidence": output.confidence, "controller_error": output.controller_error})
        pipeline.finalize()
        all_events.extend(pipeline.events)
        if record.overlap_start is not None:
            all_delays.extend(transition_metrics(pipeline.events, [record.overlap_start])["detection_to_routing_latency_seconds"])
    processing_seconds = time.perf_counter() - wall_started
    audio_seconds = sum(record.duration for record in records)
    metrics = routing_metrics(all_pairs)
    metrics.update(route_durations(all_states))
    metrics.update(transition_metrics(all_events))
    metrics["detection_to_routing_latency_seconds"] = all_delays
    metrics["mean_detection_to_routing_latency_seconds"] = sum(all_delays) / len(all_delays) if all_delays else None
    metrics.update({"mode": mode, "oracle_routing": mode == "oracle", "audio_seconds": audio_seconds, "processing_seconds": processing_seconds, "rtf": real_time_factor(processing_seconds, audio_seconds) if audio_seconds else None, "note": "Both logical routes use the same WhisperRT control branch; these results measure controller behavior and overhead, not overlap-ASR improvement."})
    return {"metrics": metrics, "traces": traces, "events": [event.to_dict() for event in all_events]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adaptive.yaml")
    parser.add_argument("--override")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--modes", nargs="+", choices=["always_normal", "oracle", "predicted"], default=["always_normal", "oracle", "predicted"])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override)
    values = config.to_dict()
    manifest = Path(values["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    records = read_overlap_manifest(manifest)
    limit = args.max_samples if args.max_samples is not None else values["data"].get("max_samples")
    if limit:
        records = records[:int(limit)]
    if args.validate_only:
        print(json.dumps({"manifest": str(manifest), "records": len(records), "modes": args.modes, "status": "validated"}, indent=2)); return 0
    paths = ProjectPaths.from_config(config, ROOT); paths.ensure_output_dirs()
    set_seed(int(values["experiment"].get("seed", 42)))
    run_dir, logger = create_run(values, paths.outputs)
    reports = {mode: _run_mode(values, records, mode) for mode in args.modes}
    for mode, result in reports.items():
        (paths.outputs / "metrics" / f"{run_dir.name}_{mode}.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        (paths.outputs / "predictions" / f"{run_dir.name}_{mode}_routing.jsonl").write_text("\n".join(json.dumps(row) for row in result["traces"]) + "\n", encoding="utf-8")
        (paths.outputs / "predictions" / f"{run_dir.name}_{mode}_events.json").write_text(json.dumps(result["events"], indent=2), encoding="utf-8")
    report = "# Phase 6 — Adaptive Streaming ASR\n\n" + "\n".join(f"## {mode}\n\n```json\n{json.dumps(value['metrics'], indent=2)}\n```" for mode, value in reports.items()) + "\n\nOracle routing is **not a deployable system**. Both routes use the same Phase 6 control branch; no recognition improvement is claimed.\n"
    report_path = paths.outputs / "reports" / f"{run_dir.name}_adaptive.md"; report_path.write_text(report, encoding="utf-8")
    logger.info("Completed Phase 6 modes: %s", ", ".join(args.modes))
    print(json.dumps({"run_id": run_dir.name, "report": str(report_path), "metrics": {mode: result["metrics"] for mode, result in reports.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
