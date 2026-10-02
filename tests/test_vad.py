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
