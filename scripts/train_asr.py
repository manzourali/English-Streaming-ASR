#!/usr/bin/env python3
"""Phase 8 external-SURT training feasibility and execution entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_asr.datasets.manifests import read_overlap_manifest
from streaming_asr.exceptions import BackendNotAvailableError
from streaming_asr.training import SurtTrainingCollator, Trainer, assess_surt_peft, leakage_errors, load_training_backend, select_training_condition, surt_training_examples, validate_training_examples
from streaming_asr.utils.config import Config
from streaming_asr.utils.logging import create_run
from streaming_asr.utils.paths import ProjectPaths
from streaming_asr.utils.reproducibility import set_seed


def _manifest(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _load_splits(values: dict):
    dataset = values["dataset"]
    splits = {name: surt_training_examples(read_overlap_manifest(_manifest(dataset[f"{name}_manifest"]))) for name in ("train", "validation", "test")}
    errors = [error for examples in splits.values() for error in validate_training_examples(examples)] + leakage_errors(splits)
    if errors:
        raise ValueError("Training manifest validation failed:\n- " + "\n- ".join(errors))
    max_duration = float(dataset.get("max_duration", 0) or 0)
    if max_duration > 0:
        splits = {name: [example for example in examples if example.duration <= max_duration] for name, examples in splits.items()}
    selected = select_training_condition(splits["train"], dataset["condition"])
    if dataset["condition"] != "base" and not selected:
        raise ValueError(f"No training examples satisfy condition={dataset['condition']!r}")
    return splits, selected


def _batches(examples, batch_size: int, collator: SurtTrainingCollator):
    return [collator(examples[index:index + batch_size]) for index in range(0, len(examples), batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--override")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Validate manifests/collator contract without loading a model")
    args = parser.parse_args()
    config = Config.from_yaml(args.config, args.override); values = config.to_dict()
    splits, selected = _load_splits(values)
    peft = assess_surt_peft(False)
    validation = {"status": "validated", "strategy": values["model"]["strategy"], "condition": values["dataset"]["condition"], "examples": {key: len(value) for key, value in splits.items()}, "selected_train_examples": len(selected), "target_format": "heat_two_channel_transcripts", "peft": peft.__dict__}
    if args.validate_only or args.smoke:
        print(json.dumps(validation, indent=2)); return 0
    if values["dataset"]["condition"] == "base":
        raise BackendNotAvailableError("Base is an evaluation-only condition; no fine-tuning batches should be created.")
    paths = ProjectPaths.from_config(config, ROOT); paths.ensure_output_dirs()
    set_seed(int(values["experiment"]["seed"]))
    run_dir, logger = create_run(values, paths.outputs)
    backend = load_training_backend(values["model"].get("backend_factory"), values)
    counts = backend.parameter_counts()
    collator = SurtTrainingCollator(int(values["dataset"]["sample_rate"]))
    train_batches = _batches(selected, int(values["training"]["batch_size"]), collator)
    validation_batches = _batches(splits["validation"], int(values["training"]["batch_size"]), collator)
    trainer = Trainer(backend, paths.checkpoints / run_dir.name, early_stopping_patience=int(values["training"]["early_stopping_patience"]))
    if values["training"].get("resume_from"):
        trainer.resume(values["training"]["resume_from"])
    history = trainer.fit(train_batches, validation_batches, epochs=int(values["training"]["epochs"]), eval_steps=int(values["training"]["eval_steps"]), save_steps=int(values["training"]["save_steps"]))
    result = {**validation, "parameter_counts": {"total": counts.total, "trainable": counts.trainable, "frozen": counts.frozen, "trainable_percent": counts.trainable_percent}, "history": history, "checkpoint_root": str(paths.checkpoints / run_dir.name)}
    (run_dir / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Training completed with %d steps", len(history)); print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackendNotAvailableError as exc:
        raise SystemExit(str(exc))
