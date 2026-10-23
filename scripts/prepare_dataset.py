#!/usr/bin/env python3
"""Build real local LibriSpeech manifests for Kaggle data preparation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streaming_asr.datasets.librispeech import prepare_librispeech


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Directory containing dev-clean or official split directories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--layout", choices=["pilot", "standard"], default="pilot")
    parser.add_argument("--max-per-split", type=int, default=100)
    args = parser.parse_args()
    result = prepare_librispeech(args.source_root, args.output_root, layout=args.layout,
                                max_per_split=args.max_per_split)
    (Path(args.output_root) / "preparation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
