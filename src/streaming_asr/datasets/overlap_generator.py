"""Deterministic two-speaker synthetic overlap generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from streaming_asr.audio.loader import write_audio
from .manifests import OverlapRecord, OverlapSource, write_overlap_manifest


@dataclass(frozen=True)
class SourceUtterance:
    source_id: str
    speaker_id: str | None
    samples: np.ndarray
    sample_rate: int
    transcript: str
    audio_path: str


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    if len(samples) < 2:
        return np.asarray(samples, dtype=np.float32)
    duration = len(samples) / source_rate
    old_times = np.linspace(0.0, duration, len(samples), endpoint=False)
    new_times = np.linspace(0.0, duration, int(round(duration * target_rate)), endpoint=False)
    return np.interp(new_times, old_times, samples).astype(np.float32)


def calculate_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> tuple[float | None, float | None, float]:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return (start, end, end - start) if end > start else (None, None, 0.0)


class OverlapGenerator:
    """Generate reproducible two-source mixtures.

    `overlap_ratio` is overlap duration divided by the duration of the shorter
    source. Thus 0 means no overlap and 1 means the shorter source is fully
    contained in the overlap interval.
    """
    def __init__(self, output_root: str | Path, sample_rate: int = 16000, seed: int = 42, normalize_rms: bool = True):
        self.output_root = Path(output_root)
        self.sample_rate = sample_rate
        self.seed = seed
        self.normalize_rms = normalize_rms

    def generate_pair(self, source_a: SourceUtterance, source_b: SourceUtterance, *, split: str, regime: str, target_ratio: float, relative_gain_db: float = 0.0, mixture_id: str | None = None, gap_ms: int = 0) -> OverlapRecord:
        if not 0.0 <= target_ratio <= 1.0:
            raise ValueError("target_ratio must be between 0 and 1")
        if source_a.speaker_id is not None and source_b.speaker_id is not None and source_a.speaker_id == source_b.speaker_id:
            raise ValueError("Two-speaker mixtures must use different speakers")
        a = _resample(source_a.samples, source_a.sample_rate, self.sample_rate)
        b = _resample(source_b.samples, source_b.sample_rate, self.sample_rate)
        if not len(a) or not len(b):
            raise ValueError("Source utterances must be non-empty")
        duration_a, duration_b = len(a) / self.sample_rate, len(b) / self.sample_rate
        shorter = min(duration_a, duration_b)
        start_b = duration_a + gap_ms / 1000.0 if target_ratio == 0 else max(0.0, duration_a - target_ratio * shorter)
        end_b = start_b + duration_b
        overlap_start, overlap_end, overlap_duration = calculate_overlap(0.0, duration_a, start_b, end_b)
        mixture_duration = max(duration_a, end_b)
        gain_b = 10.0 ** (relative_gain_db / 20.0)
        if self.normalize_rms:
            rms_a, rms_b = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
            if rms_a > 0:
                a = a / rms_a
            if rms_b > 0:
                b = b / rms_b
        mixture = np.zeros(int(np.ceil(mixture_duration * self.sample_rate)), dtype=np.float32)
        mixture[:len(a)] += a
        offset_b = int(round(start_b * self.sample_rate))
        mixture[offset_b:offset_b + len(b)] += b * gain_b
        peak = float(np.max(np.abs(mixture))) if len(mixture) else 0.0
        if peak > 0.999:
            mixture *= 0.999 / peak
        mixture_id = mixture_id or f"{split}_{source_a.source_id}_{source_b.source_id}_{regime}"
        audio_path = self.output_root / "audio" / split / f"{mixture_id}.wav"
        write_audio(audio_path, mixture, self.sample_rate)
        sources = [
            OverlapSource(source_a.source_id, source_a.speaker_id, source_a.audio_path, 0.0, duration_a, duration_a, source_a.transcript, 0.0),
            OverlapSource(source_b.source_id, source_b.speaker_id, source_b.audio_path, start_b, end_b, duration_b, source_b.transcript, relative_gain_db),
        ]
        return OverlapRecord(mixture_id, str(audio_path), self.sample_rate, mixture_duration, 2, sources, overlap_duration > 0, overlap_start, overlap_end, overlap_duration, overlap_duration / shorter if shorter else 0.0, regime, relative_gain_db, split, self.seed)

    def generate_dataset(self, sources_by_split: dict[str, list[SourceUtterance]], *, num_samples: int, regimes: dict[str, float], relative_gains_db: list[float] | None = None) -> dict[str, list[OverlapRecord]]:
        if not regimes:
            raise ValueError("At least one overlap regime is required")
        rng = np.random.default_rng(self.seed)
        gains = relative_gains_db or [0.0]
        outputs: dict[str, list[OverlapRecord]] = {}
        for split, sources in sources_by_split.items():
            if len(sources) < 2:
                raise ValueError(f"Split {split} needs at least two source utterances")
            records: list[OverlapRecord] = []
            regime_names = list(regimes)
            for index in range(num_samples):
                order = rng.permutation(len(sources))
                first, second = sources[int(order[0])], sources[int(order[1])]
                if first.speaker_id is not None and second.speaker_id is not None and first.speaker_id == second.speaker_id:
                    alternatives = [item for item in sources if item.speaker_id != first.speaker_id]
                    if alternatives:
                        second = alternatives[int(rng.integers(len(alternatives)))]
                    else:
                        raise ValueError(f"Split {split} does not contain two distinct speakers")
                regime = regime_names[index % len(regime_names)]
                records.append(self.generate_pair(first, second, split=split, regime=regime, target_ratio=float(regimes[regime]), relative_gain_db=float(gains[index % len(gains)]), mixture_id=f"{split}_{index:06d}_{regime}"))
            outputs[split] = records
            write_overlap_manifest(records, self.output_root / "manifests" / f"{split}.jsonl")
        return outputs
