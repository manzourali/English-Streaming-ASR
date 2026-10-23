import numpy as np
import pytest

from streaming_asr.audio.stream import AudioChunk, AudioStream
from streaming_asr.models.vad import DummyVAD, WebRTCVADBackend, get_vad_backend


def test_dummy_vad_schema_and_state():
    vad = DummyVAD()
    result = vad.process(AudioChunk(np.ones(4), 4, 0, 1, 0))
    assert result.speech is True
    assert result.is_speech is True
    assert vad.processed_chunks == 1
    assert result.start_time == 0 and result.end_time == 1


def test_dummy_vad_state_reset():
    vad = DummyVAD()
    vad.process(AudioChunk(np.ones(4), 4, 0, 1, 0))
    vad.finalize()
    vad.start()
    assert vad.state.processed_chunks == 0


def test_webrtc_configuration_and_frame_buffer():
    vad = WebRTCVADBackend(sample_rate=16000, frame_ms=30)
    assert vad.frame_samples == 480
    with pytest.raises(ValueError): WebRTCVADBackend(frame_ms=25)
    with pytest.raises(ValueError): WebRTCVADBackend(sample_rate=22050)


def test_backend_selection():
    assert isinstance(get_vad_backend("dummy"), DummyVAD)


def test_completed_segments_hangover_and_eof_are_exposed():
    vad = DummyVAD(minimum_silence_ms=200)
    samples = np.concatenate((np.ones(1600), np.zeros(1600), np.ones(1600), np.zeros(3200), np.ones(10)))
    results = [vad.process(chunk) for chunk in AudioStream.from_array(samples, 16000, 1600)]
    events = [segment for result in results for segment in result.completed_segments]
    assert len(events) == 1
    assert events[0].start_time == 0 and events[0].end_time == .3
    final = vad.finalize()
    assert final.start_time == .5 and final.end_time == len(samples) / 16000
    assert vad.finalize() is None
