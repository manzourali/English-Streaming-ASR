from pathlib import Path

import numpy as np
import pytest

from streaming_asr.datasets.manifests import OverlapRecord, OverlapSource
from streaming_asr.training import ExternalSurtTrainingBackend, ParameterCounts, SurtTrainingCollator, Trainer, assess_surt_peft, leakage_errors, select_training_condition, surt_training_example, validate_training_examples


def _record(mixture_id="mix", split="train", speaker_a="a", speaker_b="b", overlap_ratio=0.5):
    sources = [
        OverlapSource("later", speaker_b, "b.wav", 0.2, 0.8, 0.6, "second source"),
        OverlapSource("first", speaker_a, "a.wav", 0.0, 0.6, 0.6, "first source"),
    ]
    return OverlapRecord(mixture_id, "mix.wav", 10, 1.0, 2, sources, overlap_ratio > 0, 0.2 if overlap_ratio > 0 else None, 0.6 if overlap_ratio > 0 else None, 0.4 if overlap_ratio > 0 else 0.0, overlap_ratio, "medium", 0.0, split, 1)


def test_surt_training_targets_are_deterministic_channel_targets():
    example = surt_training_example(_record())
    assert example.channel_targets == ("first source", "second source")
    assert validate_training_examples([example]) == []
    assert select_training_condition([example], "overlap") == [example]
    assert select_training_condition([example], "clean") == []


def test_training_leakage_rejects_speaker_source_and_audio_reuse():
    train = surt_training_example(_record("train", "train", "same", "b"))
    validation = surt_training_example(_record("valid", "validation", "same", "c"))
    errors = leakage_errors({"train": [train], "validation": [validation]})
    assert any("speaker leakage" in error for error in errors)
    assert any("audio leakage" in error for error in errors)


def test_collator_pads_audio_and_retains_two_channel_text_targets():
    examples = [surt_training_example(_record("one")), surt_training_example(_record("two", speaker_a="c", speaker_b="d"))]
    batch = SurtTrainingCollator(10)(examples, [np.ones(2), np.ones(3)])
    assert batch.input_values.shape == (2, 3)
    assert batch.attention_mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch.channel_targets[0] == ("first source", "second source")


class _Backend(ExternalSurtTrainingBackend):
    def __init__(self): self.loaded = False
    def parameter_counts(self): return ParameterCounts(100, 10)
    def train_step(self, batch): return {"loss": 1.0}
    def validation_step(self, batch): return {"loss": 0.5}
    def save_checkpoint(self, directory): (Path(directory) / "backend.txt").write_text("ok", encoding="utf-8")
    def load_checkpoint(self, directory): self.loaded = (Path(directory) / "backend.txt").is_file()


def test_trainer_saves_and_reloads_checkpoint_state(tmp_path):
    backend = _Backend(); trainer = Trainer(backend, tmp_path, early_stopping_patience=1)
    history = trainer.fit(["train"], ["validation"], epochs=1, eval_steps=1, save_steps=1)
    assert history[-1]["validation_loss"] == 0.5
    assert (tmp_path / "best" / "training_state.json").is_file()
    resumed = Trainer(backend, tmp_path); state = resumed.resume(tmp_path / "last")
    assert backend.loaded and state.step == 1


def test_training_fails_on_invalid_loss_and_peft_is_explicitly_unsupported(tmp_path):
    class BadBackend(_Backend):
        def train_step(self, batch): return {"loss": float("nan")}
    with pytest.raises(FloatingPointError):
        Trainer(BadBackend(), tmp_path).fit(["x"], ["y"], epochs=1, eval_steps=1, save_steps=1)
    assert assess_surt_peft().status == "NOT TECHNICALLY SUPPORTED"
