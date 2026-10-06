"""Lightweight provenance and hashing for reproducible research artifacts."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: str | Path) -> dict[str, object]:
    root = Path(root)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
        return {"revision": revision, "worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "NOT AVAILABLE", "worktree_dirty": None}


def reproducibility_manifest(root: str | Path, files: Iterable[str | Path]) -> dict[str, object]:
    root = Path(root).resolve()
    entries = {}
    for item in files:
        path = root / item if not Path(item).is_absolute() else Path(item)
        entries[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = sha256_file(path) if path.is_file() else "MISSING"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(root),
        "python": sys.version,
        "platform": platform.platform(),
        "sha256": entries,
    }
