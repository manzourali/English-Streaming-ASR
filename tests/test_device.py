"""SURT device selection without downloading or importing an ASR model."""
import sys
from types import SimpleNamespace

import pytest

from streaming_asr.exceptions import BackendNotAvailableError
from streaming_asr.utils.device import resolve_device


def test_missing_torch_allows_cpu_and_rejects_requested_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto") == resolve_device("cpu") == "cpu"
    with pytest.raises(BackendNotAvailableError, match="not installed"):
        resolve_device("cuda")


def test_auto_and_explicit_gpu_selection(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(BackendNotAvailableError, match="index"):
        resolve_device("cuda:2")
    with pytest.raises(ValueError):
        resolve_device("cuda:bad")


def test_cuda_unavailable_is_not_silently_accepted(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert resolve_device("auto") == "cpu"
    with pytest.raises(BackendNotAvailableError, match="unavailable"):
        resolve_device("cuda:0")
