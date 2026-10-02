from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

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
        samples, sample_rate = sf.read(str(path), always_2d=False)
    except ImportError:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels)
    channels = 1 if np.asarray(samples).ndim == 1 else int(np.asarray(samples).shape[1])
    mono = to_mono(samples)
    return mono, AudioMetadata(int(sample_rate), len(mono), channels)


def write_audio(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write deterministic mono PCM16 WAV without requiring SoundFile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Audio to write must be a finite one-dimensional array")
    pcm = np.clip(array, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
