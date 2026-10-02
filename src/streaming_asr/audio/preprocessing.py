from __future__ import annotations

import numpy as np


def to_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=1 if array.shape[1] <= 8 else 0).astype(np.float32)
    raise ValueError("Audio must be one- or two-dimensional")

