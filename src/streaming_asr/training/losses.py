"""Training-loss validation independent of a specific external SURT runtime."""
from __future__ import annotations

import math


def validate_loss(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("training loss is NaN or infinite")
    return value
