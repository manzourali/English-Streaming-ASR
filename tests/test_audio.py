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
