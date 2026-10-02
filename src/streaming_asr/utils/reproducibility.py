from __future__ import annotations

import random
from typing import Any


def set_seed(seed: int) -> dict[str, Any]:
    """Seed lightweight dependencies; seed PyTorch when it is installed."""
    random.seed(seed)
    details: dict[str, Any] = {"python": seed, "numpy": False, "torch": False}
    try:
        import numpy as np
        np.random.seed(seed)
        details["numpy"] = True
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        details["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return details

