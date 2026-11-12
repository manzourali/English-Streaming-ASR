from streaming_asr.models.whisperrt import TranscriptAccumulator, TranscriptUpdate, resolve_device, validate_dtype
import sys
from types import SimpleNamespace

import pytest


def test_transcript_accumulator_keeps_latest_partial_only():
    accumulator = TranscriptAccumulator()
    accumulator.update(TranscriptUpdate("the", 0, 0.0, 0.3))
    accumulator.update(TranscriptUpdate("the quick", 1, 0.3, 0.6))
    assert accumulator.text == "the quick"
    assert accumulator.finalize().is_final


def test_cpu_dtype_validation():
    validate_dtype("auto", "cpu")
    assert resolve_device("cpu") == "cpu"


def test_auto_device_is_kept_by_loaded_adapter(monkeypatch):
    from streaming_asr.models import whisperrt
    monkeypatch.setattr(whisperrt, "resolve_device", lambda _: "cuda")
    seen = []
    backend = SimpleNamespace(load=lambda config: seen.append(config) or object())
    monkeypatch.setattr(whisperrt, "get_model_backend", lambda _: backend)
    model = whisperrt.WhisperRTStreamingASR.from_config({"device": "auto"})
    assert model.device == "cuda" and model.config["device"] == "cuda"
    assert seen[0]["device"] == "cuda"


def test_t4_rejects_native_bfloat16(monkeypatch):
    cuda = SimpleNamespace(get_device_capability=lambda _: (7, 5))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(ValueError, match="T4"):
        validate_dtype("bfloat16", "cuda")
    validate_dtype("float16", "cuda")
    cuda.get_device_capability = lambda _: (8, 0)
    validate_dtype("bfloat16", "cuda")


def test_explicit_second_gpu_is_validated(monkeypatch):
    from streaming_asr.exceptions import BackendNotAvailableError
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert resolve_device("cuda:1") == "cuda:1"
    with pytest.raises(BackendNotAvailableError, match="index"):
        resolve_device("cuda:2")
    with pytest.raises(ValueError):
        resolve_device("cuda:bad")
