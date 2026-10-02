import numpy as np
from streaming_asr.audio.stream import AudioChunk
from streaming_asr.models.vad import DummyVAD

def test_dummy_vad_schema_and_state():
    vad = DummyVAD()
    result = vad.process(AudioChunk(np.ones(4), 4, 0, 1, 0))
    assert result.speech is True
    assert vad.processed_chunks == 1

