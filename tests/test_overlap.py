import numpy as np
from streaming_asr.audio.stream import AudioChunk
from streaming_asr.models.overlap_detector import DummyOverlapDetector, OverlapLabel

def test_dummy_overlap_labels_silence():
    result = DummyOverlapDetector().process(AudioChunk(np.zeros(4), 4, 0, 1, 0))
    assert result.label is OverlapLabel.NO_SPEECH

