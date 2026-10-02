import numpy as np
import pytest
from streaming_asr.audio.stream import AudioStream
from streaming_asr.streaming.engine import StreamingEngine
from streaming_asr.exceptions import StreamingStateError

def test_engine_lifecycle_and_indices():
    engine = StreamingEngine(); engine.start()
    outputs = [engine.process(c) for c in AudioStream.from_array(np.ones(5), 5, 2)]
    assert [o.chunk.index for o in outputs] == [0, 1, 2]
    assert engine.finalize().finalized
    assert engine.finalize().finalized

def test_engine_requires_start():
    with pytest.raises(StreamingStateError): StreamingEngine().process(next(iter(AudioStream.from_array(np.ones(1), 1, 1))))

