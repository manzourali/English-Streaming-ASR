"""Numpy-first batch collation before external SURT recipe tokenization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from streaming_asr.audio.loader import load_audio
from .data import SurtTrainingExample


@dataclass(frozen=True)
class CollatedSurtBatch:
    input_values: np.ndarray
    attention_mask: np.ndarray
    input_lengths: np.ndarray
    channel_targets: tuple[tuple[str, str], ...]
    sample_ids: tuple[str, ...]
    sample_rate: int


class SurtTrainingCollator:
    """Pad mono audio but leave exact tokenizer/loss work to the SURT recipe."""
    def __init__(self, sample_rate: int = 16000, pad_value: float = 0.0):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate, self.pad_value = sample_rate, pad_value

    def __call__(self, examples: Iterable[SurtTrainingExample], audio_arrays: Iterable[np.ndarray] | None = None) -> CollatedSurtBatch:
        examples = list(examples)
        if not examples:
            raise ValueError("cannot collate an empty batch")
        arrays = list(audio_arrays) if audio_arrays is not None else [self._load(example) for example in examples]
        if len(arrays) != len(examples):
            raise ValueError("audio_arrays must match example count")
        normalized = [self._validate_array(item, example.sample_id) for item, example in zip(arrays, examples)]
        length = max(len(item) for item in normalized)
        values = np.full((len(normalized), length), self.pad_value, dtype=np.float32)
        mask = np.zeros((len(normalized), length), dtype=np.int64)
        for index, item in enumerate(normalized):
            values[index, :len(item)] = item
            mask[index, :len(item)] = 1
        return CollatedSurtBatch(values, mask, np.asarray([len(item) for item in normalized], dtype=np.int64), tuple(tuple(example.channel_targets) for example in examples), tuple(example.sample_id for example in examples), self.sample_rate)

    def _load(self, example: SurtTrainingExample) -> np.ndarray:
        audio, metadata = load_audio(example.audio_path)
        if metadata.sample_rate != self.sample_rate:
            raise ValueError(f"{example.sample_id}: expected {self.sample_rate} Hz, got {metadata.sample_rate}")
        return audio

    @staticmethod
    def _validate_array(value: np.ndarray, sample_id: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError(f"{sample_id}: audio must be finite non-empty mono samples")
        return array
