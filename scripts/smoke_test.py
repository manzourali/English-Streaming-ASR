#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.audio.stream import AudioStream
from streaming_asr.models.overlap_detector import DummyOverlapDetector
from streaming_asr.models.vad import DummyVAD, get_vad_backend
from streaming_asr.datasets.overlap_generator import OverlapGenerator, SourceUtterance
from streaming_asr.datasets.validators import validate_overlap_records
from streaming_asr.pipeline.streaming import StreamingASRPipeline
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--real-vad", action="store_true")
    parser.add_argument("--demo-overlap", action="store_true")
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override)
    paths = ProjectPaths.from_config(config, ROOT)
    try:
        paths.ensure_output_dirs()
    except OSError:
        # Allows validating kaggle.yaml on a non-Kaggle machine without
        # pretending that /kaggle/working is available locally.
        if config.get("runtime.environment") != "kaggle":
            raise
        paths = ProjectPaths.from_config(Config({"runtime": {"environment": "local"}}), ROOT)
        paths.ensure_output_dirs()
    set_seed(int(config.get("experiment.seed", 42)))
    run_dir, logger = create_run(config.to_dict(), paths.outputs)
    sample_rate = int(config.get("audio.sample_rate", 16000))
    chunk_size = int(sample_rate * int(config.get("audio.chunk_ms", 320)) / 1000)
    samples = np.concatenate([np.zeros(chunk_size), np.sin(np.linspace(0, 20, chunk_size)), np.zeros(chunk_size)])
    if args.phase == 2 and args.real_vad:
        vad = get_vad_backend(config.get("vad.backend", "webrtc"), {"sample_rate": sample_rate, **config.get("vad", {})})
    else:
        vad = DummyVAD()
    pipeline = StreamingASRPipeline(vad, None if args.phase == 2 else DummyOverlapDetector())
    pipeline.start()
    count = 0
    for chunk in AudioStream.from_array(samples, sample_rate, chunk_size):
        pipeline.process(chunk)
        count += 1
    state = pipeline.finalize()
    overlap_ok = None
    if args.phase == 3 and args.demo_overlap:
        root = paths.outputs / "smoke_overlap"
        source_a = SourceUtterance("a", "speaker_a", np.ones(sample_rate // 5, dtype=np.float32) * 0.1, sample_rate, "alpha", str(root / "a.wav"))
        source_b = SourceUtterance("b", "speaker_b", np.ones(sample_rate // 5, dtype=np.float32) * 0.1, sample_rate, "bravo", str(root / "b.wav"))
        generator = OverlapGenerator(root, sample_rate, 42)
        record = generator.generate_pair(source_a, source_b, split="test", regime="medium", target_ratio=0.5)
        overlap_ok = not validate_overlap_records([record], check_source_files=False)
    result = {"phase": f"phase{args.phase}", "experiment": config.get("experiment.name"), "status": "success", "environment": config.get("runtime.environment"), "seed": config.get("experiment.seed"), "chunks": count, "finalized": state.finalized, "real_vad": bool(args.real_vad and args.phase == 2), "demo_overlap": overlap_ok}
    (run_dir / "smoke_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Smoke test completed: %s", result)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
