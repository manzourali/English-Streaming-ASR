import numpy as np
import pytest

from streaming_asr.audio.buffer import AudioBuffer
from streaming_asr.audio.stream import AudioChunk, AudioStream
from streaming_asr.exceptions import InvalidAudioChunkError

def test_chunk_metadata_and_duration():
    chunk = AudioChunk(np.zeros(160), 16000, 0.0, 0.01, 0)
    assert chunk.duration == pytest.approx(0.01)
    assert chunk.sample_rate == 16000

def test_buffer_concatenates_chunks():
    buffer = AudioBuffer(2)
    for chunk in AudioStream.from_array(np.ones(5), 5, 2): buffer.append(chunk)
    assert len(buffer) == 2
    assert len(buffer.as_array()) == 3

def test_invalid_chunk_rejected():
    with pytest.raises(InvalidAudioChunkError): AudioChunk(np.zeros((2, 2)), 16000, 0, 1, 0)


def test_timestamp_duration_and_sample_grid_are_validated():
    for start, end in [(0, .02), (.00001, .01001), (float("nan"), .01)]:
        with pytest.raises(InvalidAudioChunkError):
            AudioChunk(np.zeros(160), 16000, start, end, 0)


def test_pcm16_and_explicit_channel_conversion():
    from streaming_asr.audio.preprocessing import to_mono
    samples = np.full((10, 16), 16384, dtype=np.int16)
    assert to_mono(samples).shape == (10,)
    np.testing.assert_allclose(to_mono(samples), .5)
    np.testing.assert_allclose(to_mono(samples.T, channel_axis=0), .5)
    chunks = list(AudioStream.from_array(samples, 16000, 3))
    np.testing.assert_allclose(np.concatenate([c.samples for c in chunks]), .5)
    assert chunks[-1].end_sample == 10
    assert AudioChunk(np.array([-32768], np.int16), 16000, 0, 1 / 16000, 0).samples[0] == -1


def test_invalid_stream_shape_and_chunk_size():
    with pytest.raises(ValueError):
        AudioStream.from_array(np.zeros(100), 16000, 0)
