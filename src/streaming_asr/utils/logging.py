"""Minimal JSON metadata and text logging for experiment runs."""
from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


def create_run(config: Mapping[str, Any], output_root: str | Path) -> tuple[Path, logging.Logger]:
    experiment = str(config.get("experiment", {}).get("name", "experiment"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / "logs" / f"{stamp}_{experiment}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    metadata = {
        "experiment": experiment,
        "timestamp_utc": stamp,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_environment": config.get("runtime", {}).get("environment", "local"),
        "seed": config.get("experiment", {}).get("seed"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger = logging.getLogger(f"streaming_asr.{experiment}.{stamp}")
    logger.setLevel(getattr(logging, str(config.get("logging", {}).get("level", "INFO")).upper()))
    logger.handlers.clear()
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return run_dir, logger

