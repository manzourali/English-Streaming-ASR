#!/usr/bin/env python3
"""Validate all active notebook JSON and Python cells without executing them."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    notebooks = sorted(ROOT.glob("*.ipynb")) + sorted((ROOT / "notebooks").glob("*.ipynb"))
    failures = []
    for path in notebooks:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
                raise ValueError("expected nbformat 4 with a cells list")
            for index, cell in enumerate(payload["cells"]):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    compile(source, f"{path.name}:cell-{index}", "exec")
                    if cell.get("outputs") or cell.get("execution_count") is not None:
                        raise ValueError("Committed workflow notebooks must have cleared outputs")
        except (OSError, ValueError, SyntaxError) as exc:
            failures.append(f"{path.name}: {exc}")
    for failure in failures:
        print(failure)
    print(f"Notebook JSON/Python validation: {len(notebooks) - len(failures)}/{len(notebooks)} valid; cells not executed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
