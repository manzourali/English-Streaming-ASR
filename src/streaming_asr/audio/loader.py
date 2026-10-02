from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import to_mono


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    num_samples: int
    channels: int = 1


def load_audio(path: str | Path) -> tuple[np.ndarray, AudioMetadata]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Loading files requires the optional 'soundfile' package") from exc
    samples, sample_rate = sf.read(str(path), always_2d=False)
    channels = 1 if np.asarray(samples).ndim == 1 else int(np.asarray(samples).shape[1])
    mono = to_mono(samples)
    return mono, AudioMetadata(int(sample_rate), len(mono), channels)

