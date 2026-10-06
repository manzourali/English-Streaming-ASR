#!/usr/bin/env python3
"""Aggregate frozen Phase 9 per-example records into reproducible reports."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.evaluation import Benchmark, Evaluator, phase9_report, read_evaluation_records
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOT AVAILABLE"


def _worktree_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _freeze(values: dict) -> dict[str, object]:
    frozen = dict(values["frozen_system"])
    if frozen.get("git_commit") == "auto":
        frozen["git_commit"] = _git_revision()
    frozen["worktree_dirty"] = _worktree_dirty()
    frozen["seed"] = values["experiment"]["seed"]
    frozen["test_set_protected"] = values["evaluation"]["test_set_protected"]
    return frozen


def _validate(values: dict) -> dict[str, object]:
    required = {"git_commit", "vad_config", "osd_config", "routing_config", "training_config", "train_manifest", "validation_manifest", "test_manifest"}
    missing = sorted(required - set(values.get("frozen_system", {})))
    if missing:
        raise ValueError("evaluation freeze is missing: " + ", ".join(missing))
    if not values["evaluation"].get("test_set_protected"):
        raise ValueError("Phase 9 requires test_set_protected: true")
    return {"status": "validated", "freeze": _freeze(values), "primary_systems": values["systems"]["primary"], "oracle_systems": values["systems"].get("oracle", []), "ablation_rows": len(values.get("ablations", []))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--override")
    parser.add_argument("--records", nargs="*", help="JSONL per-example EvaluationRecord files")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="Generate a planning report with NOT MEASURED cells")
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override); values = config.to_dict()
    validation = _validate(values)
    if args.validate_only:
        print(json.dumps(validation, indent=2)); return 0
    record_paths = args.records if args.records is not None else values["evaluation"].get("input_records", [])
    if not record_paths and not args.allow_empty:
        raise ValueError("No per-example records supplied; use --validate-only or --allow-empty for a planning report")
    records = [record for path in record_paths for record in read_evaluation_records(path)]
    paths = ProjectPaths.from_config(config, ROOT); paths.ensure_output_dirs()
    set_seed(int(values["experiment"]["seed"])); run_dir, logger = create_run(values, paths.outputs)
    evaluator = Evaluator(int(values["evaluation"]["bootstrap_iterations"]), int(values["experiment"]["seed"]))
    summaries = evaluator.summarize(records); benchmarks = Benchmark().run(records)
    strata = {name: {group: [asdict(item) for item in items] for group, items in evaluator.stratify(records, name).items()} for name in values["evaluation"].get("strata", [])}
    taxonomy = evaluator.error_taxonomy(records)
    report = phase9_report(validation["freeze"], summaries, benchmarks, values["systems"]["primary"], strata, taxonomy, values.get("ablations", []))
    report_path = paths.outputs / "reports" / f"{run_dir.name}_phase9.md"; report_path.write_text(report, encoding="utf-8")
    result = {"freeze": validation["freeze"], "records": len(records), "summaries": [asdict(item) for item in summaries], "benchmarks": [asdict(item) for item in benchmarks], "strata": strata, "error_taxonomy": taxonomy, "report": str(report_path)}
    (run_dir / "evaluation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Phase 9 evaluation processed %d records", len(records)); print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
