"""Explicit frame-major audio conversion; integer input is signed PCM16."""
from __future__ import annotations
import numpy as np


def to_mono(samples: np.ndarray, *, channel_axis: int = 1) -> np.ndarray:
    array = np.asarray(samples)
    if np.issubdtype(array.dtype, np.integer):
        if array.dtype != np.int16:
            raise ValueError("Integer audio must be signed PCM16")
        array = array.astype(np.float32) / 32768.0
    else:
        array = array.astype(np.float32)
    if array.ndim == 2:
        if channel_axis not in (0, 1, -1, -2) or array.shape[channel_axis] == 0:
            raise ValueError("channel_axis must identify a nonempty channel dimension")
        array = array.mean(axis=channel_axis)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Audio must be finite mono or a two-dimensional channel array")
    return array
