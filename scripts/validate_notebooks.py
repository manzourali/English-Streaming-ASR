#!/usr/bin/env python3
"""Validate committed notebook JSON without executing model/data-dependent cells."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    failures = []
    for path in notebooks:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
                raise ValueError("expected nbformat 4 with a cells list")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{path.name}: {exc}")
    for failure in failures: print(failure)
    print(f"Notebook JSON validation: {len(notebooks) - len(failures)}/{len(notebooks)} valid")
    return 0 if not failures else 1


if __name__ == "__main__": raise SystemExit(main())
