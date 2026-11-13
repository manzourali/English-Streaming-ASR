"""Device selection shared by native SURT backends; no model imports."""
from __future__ import annotations

from streaming_asr.exceptions import BackendNotAvailableError


def resolve_device(requested: str) -> str:
    requested = str(requested)
    indexed_cuda = requested.startswith("cuda:") and requested[5:].isdigit()
    if requested not in {"auto", "cpu", "cuda"} and not indexed_cuda:
        raise ValueError("runtime.device must be auto, cpu, cuda, or cuda:N")
    try:
        import torch
    except ImportError:
        if requested.startswith("cuda"):
            raise BackendNotAvailableError("CUDA was requested but PyTorch is not installed")
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise BackendNotAvailableError("CUDA was requested but is unavailable")
    if indexed_cuda and int(requested[5:]) >= torch.cuda.device_count():
        raise BackendNotAvailableError(f"CUDA device index is unavailable: {requested}")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
