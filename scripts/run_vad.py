#!/usr/bin/env python3
"""Run the Phase 2 VAD benchmark on a supplied WAV or deterministic demo signal."""
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
from streaming_asr.exceptions import BackendNotAvailableError
from streaming_asr.models.vad import get_vad_backend
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/streaming_vad.yaml")
    parser.add_argument("--input", help="Optional mono WAV file; without it, runs an unscored synthetic demo")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    paths = ProjectPaths.from_config(config, ROOT)
    paths.ensure_output_dirs()
    set_seed(int(config.get("experiment.seed", 42)))
    run_dir, logger = create_run(config.to_dict(), paths.outputs)
    sample_rate = int(config.get("audio.sample_rate", 16000))
    if args.input:
        samples, metadata = load_audio(args.input)
        if metadata.sample_rate != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz audio, received {metadata.sample_rate} Hz")
        scored = False
    else:
        frame = np.zeros(sample_rate // 10, dtype=np.float32)
        tone = np.sin(np.linspace(0, 30, sample_rate // 5)).astype(np.float32)
        samples = np.concatenate([frame, tone, frame])
        scored = False
    vad = get_vad_backend(config.get("vad.backend", "webrtc"), {"sample_rate": sample_rate, **config.get("vad", {})})
    chunk_size = int(sample_rate * int(config.get("audio.chunk_ms", 320)) / 1000)
    vad.start()
    decisions = []
    started = time.perf_counter()
    for chunk in AudioStream.from_array(samples, sample_rate, chunk_size):
        result = vad.process(chunk)
        decisions.append(result.__dict__)
    final_segment = vad.finalize()
    processing_time = time.perf_counter() - started
    output = {"phase": "phase2", "backend": config.get("vad.backend"), "sample_rate": sample_rate, "frame_ms": config.get("vad.frame_ms"), "audio_duration": len(samples) / sample_rate, "processing_time": processing_time, "vad_rtf": processing_time / (len(samples) / sample_rate), "scored": scored, "decisions": decisions, "final_segment": final_segment.__dict__ if final_segment else None}
    (paths.outputs / "metrics" / f"{run_dir.name}.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info("Completed VAD run: %s", output)
    print(json.dumps({key: value for key, value in output.items() if key != "decisions"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackendNotAvailableError as exc:
        raise SystemExit(f"Phase 2 cannot start: {exc}")
