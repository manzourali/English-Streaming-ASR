import numpy as np
import pytest

from streaming_asr.audio.stream import AudioStream
from streaming_asr.exceptions import BackendNotAvailableError, StreamingStateError
from streaming_asr.models.multitalker import SURT2WindowedASR
from streaming_asr.pipeline.overlap import MultiTalkerASRBranch


def test_structured_streaming_output_and_duplicate_suppression():
    outputs = iter([["hello", "world"], ["hello there", "world again"]])
    model = SURT2WindowedASR(lambda samples, rate: next(outputs), sample_rate=10, chunk_duration_ms=100, hop_duration_ms=100, context_duration_ms=200)
    model.start()
    first, second = [model.process(chunk) for chunk in AudioStream.from_array(np.ones(2), 10, 1)]
    assert [stream.text for stream in first.streams] == ["hello", "world"]
    assert [stream.new_text for stream in second.streams] == ["there", "again"]
    assert second.streaming_mode == "low_latency_windowed"
    final = model.finalize()
    assert final is not None and final.is_final


def test_hop_keeps_state_and_schema_handles_mapping():
    model = SURT2WindowedASR(lambda samples, rate: {"streams": [{"stream_id": 2, "text": "alpha", "confidence": 0.8}]}, sample_rate=10, chunk_duration_ms=100, hop_duration_ms=200)
    model.start()
    first, second = [model.process(chunk) for chunk in AudioStream.from_array(np.ones(2), 10, 1)]
    assert first.streams == ()
    assert second.streams[0].stream_id == 2 and second.streams[0].confidence == 0.8


def test_revised_partial_replaces_rolling_text_without_reemitting_it():
    outputs = iter([["hello world"], ["hello there"]])
    model = SURT2WindowedASR(lambda samples, rate: next(outputs), sample_rate=10, chunk_duration_ms=100)
    model.start()
    first, second = [model.process(chunk) for chunk in AudioStream.from_array(np.ones(2), 10, 1)]
    assert first.streams[0].new_text == "hello world"
    assert second.streams[0].text == "hello there" and second.streams[0].new_text == ""


def test_invalid_output_and_lifecycle_fail_clearly():
    model = SURT2WindowedASR(lambda samples, rate: 123, sample_rate=10, chunk_duration_ms=100)
    with pytest.raises(StreamingStateError):
        model.process(next(iter(AudioStream.from_array(np.ones(1), 10, 1))))
    model.start()
    with pytest.raises(ValueError):
        model.process(next(iter(AudioStream.from_array(np.ones(1), 10, 1))))
    with pytest.raises(BackendNotAvailableError):
        SURT2WindowedASR.from_config({})


def test_overlap_branch_preserves_structured_output():
    model = SURT2WindowedASR(lambda samples, rate: ["one", "two"], sample_rate=10, chunk_duration_ms=100)
    branch = MultiTalkerASRBranch(model); branch.start()
    output = branch.process(next(iter(AudioStream.from_array(np.ones(1), 10, 1))))
    assert len(output.streams) == 2
    assert branch.finalize().is_final
