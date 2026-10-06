#!/usr/bin/env python3
"""One entry point for Phase 10 code, smoke, and full-result reproduction."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> int:
    print("+", " ".join(command)); return subprocess.run(command, cwd=ROOT).returncode


def _not_ready() -> list[str]:
    needed = []
    for label, path in (("WhisperRT checkpoint", None), ("SURT checkpoint", None), ("fine-tuned SURT checkpoint", None), ("Phase 3 final test manifest", ROOT / "data/manifests/test.jsonl")):
        if path is None or not path.is_file(): needed.append(label)
    return needed


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--level", choices=("code", "small", "full"), default="small")
    args = parser.parse_args()
    commands = [[sys.executable, "scripts/check_environment.py"], [sys.executable, "-m", "pytest", "-q"], [sys.executable, "scripts/validate_notebooks.py"]]
    if args.level in {"small", "full"}:
        commands += [[sys.executable, "scripts/smoke_test.py", "--phase", "8", "--config", "configs/training.yaml"], [sys.executable, "scripts/smoke_test.py", "--phase", "9", "--config", "configs/evaluation.yaml"], [sys.executable, "scripts/evaluate.py", "--config", "configs/evaluation.yaml", "--validate-only"], [sys.executable, "scripts/audit_artifact.py"]]
    for command in commands:
        if _run(command): return 1
    if args.level == "full":
        needed = _not_ready()
        if needed:
            print("NOT REPRODUCED: full results require " + ", ".join(needed)); return 2
        return _run([sys.executable, "scripts/evaluate.py", "--config", "configs/evaluation.yaml", "--records", "outputs/predictions/final_records.jsonl"])
    print(f"REPRODUCTION LEVEL {args.level.upper()}: VERIFIED")
    return 0


if __name__ == "__main__": raise SystemExit(main())
