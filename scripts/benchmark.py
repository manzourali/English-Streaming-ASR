#!/usr/bin/env python3
"""Compute-only Phase 9 benchmark summary from per-example records."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.evaluation import Benchmark, read_evaluation_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    args = parser.parse_args()
    records = [record for path in args.records for record in read_evaluation_records(path)]
    result = [asdict(item) for item in Benchmark().run(records)]
    print(json.dumps({"records": len(records), "benchmarks": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
