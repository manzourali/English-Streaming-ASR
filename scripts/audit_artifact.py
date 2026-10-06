#!/usr/bin/env python3
"""Non-destructive Phase 10 audit for provenance, secrets, files, and licenses."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.utils.provenance import reproducibility_manifest

SENSITIVE = {"aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"), "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"), "generic_secret_assignment": re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"][^'\"]+")}


def _audit_files() -> tuple[list[Path], list[Path]]:
    try:
        names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        untracked = [line[3:] for line in subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).splitlines() if line.startswith("?? ")]
        tracked_paths, untracked_paths = [ROOT / name for name in names], [ROOT / name for name in untracked]
        return tracked_paths + untracked_paths, untracked_paths
    except (OSError, subprocess.CalledProcessError):
        paths = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
        return paths, []


def audit(max_file_bytes: int = 5 * 1024 * 1024) -> dict[str, object]:
    files, untracked = _audit_files(); suspicious = []
    for path in files:
        try:
            if path.stat().st_size > max_file_bytes: suspicious.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
            if path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".ipynb"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for name, pattern in SENSITIVE.items():
                    if pattern.search(text): suspicious.append({"path": str(path.relative_to(ROOT)), "finding": name})
        except OSError:
            continue
    required = ["LICENSE", "README.md", "CITATION.cff", "pyproject.toml", "requirements.txt", "environment.yml", ".gitignore"]
    return {"audited_files": len(files), "untracked_files": [str(path.relative_to(ROOT)) for path in untracked], "large_or_sensitive_findings": suspicious, "required_artifacts": {name: (ROOT / name).is_file() for name in required}, "provenance": reproducibility_manifest(ROOT, ["pyproject.toml", "requirements.txt", "environment.yml", "configs/evaluation.yaml", "master_prompt.md"])}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true")
    args = parser.parse_args(); report = audit()
    if args.write:
        target = ROOT / "outputs" / "reproducibility_manifest.json"; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["output"] = str(target)
    print(json.dumps(report, indent=2))
    return 0 if not report["large_or_sensitive_findings"] and all(report["required_artifacts"].values()) else 1


if __name__ == "__main__": raise SystemExit(main())
