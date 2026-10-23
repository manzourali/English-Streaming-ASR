"""Streaming scheduler tests with a deterministic, explicitly non-ASR backend."""
import numpy as np
import pytest

from streaming_asr.audio.features import StreamingFbank
from streaming_asr.audio.stream import AudioChunk, AudioStream
from streaming_asr.models.multitalker import BackendOutput, SURT2StreamingASR
from streaming_asr.exceptions import BackendNotAvailableError, StreamingStateError


class RecordingBackend:
    num_channels = 2
    chunk_frames = 32
    right_context_frames = 0
    frame_shift_samples = 160
    max_state_bytes = 1024

    def reset(self):
        self.calls = []
        self.finishes = 0

    def step(self, features, *, valid_frames, final):
        self.calls.append((features.copy(), valid_frames, final))
        return BackendOutput(("alpha", "beta"), (0, 0))

    def finish(self):
        self.finishes += 1
        return BackendOutput(("tail", ""), (0, 0))


def run(samples, size, right=0):
    backend = RecordingBackend()
    backend.right_context_frames = right
    model = SURT2StreamingASR(backend)
    model.start()
    for chunk in AudioStream.from_array(samples, 16000, size):
        model.process(chunk)
        assert model.buffered_samples < 400
        assert model.buffered_frames < 32 + right
    final = model.finalize()
    assert model.finalize() is final
    assert backend.finishes == 1
    return model, backend, final


@pytest.mark.parametrize("length", [0, 1, 159, 160, 399, 400, 5119, 5120, 5360, 10241])
def test_eof_conserves_all_frame_starts_and_original_timestamps(length):
    model, backend, final = run(np.ones(length, np.float32), 137)
    assert sum(call[1] for call in backend.calls) == (length + 159) // 160
    assert final.timestamp == length / 16000
    assert final.is_final and final.streams[-2].text == "tail"
    assert model.buffered_samples == model.buffered_frames == 0
    assert all(call[0].shape == (32, 80) for call in backend.calls)


def test_feature_and_model_results_are_independent_of_input_partition():
    samples = np.random.default_rng(1).normal(0, .03, 20007).astype(np.float32)
    _, first, _ = run(samples, 1)
    _, second, _ = run(samples, len(samples))
    assert len(first.calls) == len(second.calls)
    for left, right in zip(first.calls, second.calls):
        np.testing.assert_allclose(left[0], right[0], rtol=1e-6)
        assert left[1:] == right[1:]


def test_right_context_waits_for_actual_future_samples():
    backend = RecordingBackend(); backend.right_context_frames = 1
    model = SURT2StreamingASR(backend); model.start()
    # 32 complete windows exist at 5360 samples; frame 33 requires 5520.
    chunks = iter(AudioStream.from_array(np.ones(5520), 16000, 5360))
    model.process(next(chunks))
    assert backend.calls == []
    model.process(next(chunks))
    assert len(backend.calls) == 1
    assert backend.calls[0][0].shape == (33, 80)
    model.finalize()
    assert sum(call[1] for call in backend.calls) == 35


def test_acoustic_buffers_remain_bounded_over_long_recording():
    class ConstantBackend(RecordingBackend):
        def step(self, features, *, valid_frames, final):
            return BackendOutput(("", ""), (100, 100))
    model = SURT2StreamingASR(ConstantBackend()); model.start()
    for index in range(2000):
        start = index * 5120
        model.process(AudioChunk(np.zeros(5120), 16000, start / 16000, (start + 5120) / 16000, index))
        assert model.buffered_samples < 400 and model.buffered_frames < 32
    model.finalize()


def test_endpoint_occurs_once_per_channel_without_resetting_model():
    class EndpointBackend(RecordingBackend):
        def step(self, features, *, valid_frames, final):
            self.calls.append(None)
            return BackendOutput(("word" if len(self.calls) == 1 else "", ""),
                                 (0 if len(self.calls) == 1 else 100, 100))
    model = SURT2StreamingASR(EndpointBackend()); model.start()
    events = []
    for chunk in AudioStream.from_array(np.zeros(20000), 16000, 5120):
        events.extend(model.process(chunk).endpoints)
    assert events == [0]


def test_configuration_and_lifecycle_reject_invalid_backends():
    with pytest.raises(BackendNotAvailableError):
        SURT2StreamingASR.from_config({})
    backend = RecordingBackend(); backend.right_context_frames = 32
    with pytest.raises(ValueError):
        SURT2StreamingASR(backend)
    model = SURT2StreamingASR(RecordingBackend())
    with pytest.raises(StreamingStateError):
        model.finalize()
    model.start(); model.finalize()
    with pytest.raises(StreamingStateError):
        model.process(AudioChunk(np.zeros(0), 16000, 0, 0, 0))
