#!/usr/bin/env python3
"""Report native export readiness separately from downloaded official weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED = ("metadata.json", "mask.pt", "encoder.pt", "decoder.pt", "joiner.pt", "tokenizer.model")


def inspect_bundle(root):
    if root is None:
        return {"status": "native_export_missing", "ready": False,
                "reason": "Official checkpoint downloads do not implement the project's streaming export interface."}
    root = Path(root)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        return {"status": "incomplete_bundle", "ready": False, "missing": missing}
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("branch_tying") and not (root / "branch_tying.pt").is_file():
        return {"status": "incomplete_bundle", "ready": False, "missing": ["branch_tying.pt"]}
    valid = (metadata.get("architecture") == "surt2-dprnn-zipformer-stateless"
             and metadata.get("feature_profile") == "kaldi80-snip-pcm16-v1"
             and metadata.get("right_context_frames") in (0, 1)
             and all(isinstance(metadata.get(key), int) and metadata[key] > 0
                     for key in ("context_size", "encoder_subsampling", "max_state_bytes"))
             and isinstance(metadata.get("blank_id"), int) and metadata["blank_id"] >= 0)
    return {"status": "bundle_present_not_executed" if valid else "incompatible_metadata",
            "ready": False, "structure_valid": valid, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir")
    parser.add_argument("--probe", action="store_true", help="Load exports and decode synthetic silence; not an accuracy benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()
    result = inspect_bundle(args.export_dir)
    if args.probe and result.get("structure_valid"):
        try:
            import numpy as np
            from streaming_asr.audio.stream import AudioStream
            from streaming_asr.models.multitalker import SURT2StreamingASR
            model = SURT2StreamingASR.from_config({
                "backend_factory": "streaming_asr.models.torch_surt:load_torchscript_backend",
                "checkpoint": args.export_dir, "device": args.device})
            model.start()
            for chunk in AudioStream.from_array(np.zeros(16000, np.float32), 16000, 5120):
                model.process(chunk)
            model.finalize()
            result.update(status="native_smoke_passed", ready=True,
                          accuracy_verified=False, feature_checkpoint_parity_verified=False)
        except Exception as exc:
            result.update(status="native_probe_failed", ready=False, error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2))
    return 1 if args.require_native and not result["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
