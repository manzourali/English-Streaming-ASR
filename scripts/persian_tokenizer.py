#!/usr/bin/env python3
"""Normalize transcripts, train a BPE model, or measure OOV coverage."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streaming_asr.datasets.persian_text import normalize_file, oov_coverage, train_bpe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize")
    normalize.add_argument("input")
    normalize.add_argument("output")
    train = commands.add_parser("train")
    train.add_argument("input")
    train.add_argument("model_prefix")
    train.add_argument("--vocab-size", type=int, default=500)
    coverage = commands.add_parser("coverage")
    coverage.add_argument("model")
    coverage.add_argument("input")
    args = parser.parse_args()
    if args.command == "normalize":
        print(json.dumps({"lines": normalize_file(args.input, args.output)}))
    elif args.command == "train":
        train_bpe(args.input, args.model_prefix, args.vocab_size)
    else:
        print(json.dumps(oov_coverage(args.model, args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
