import numpy as np
from streaming_asr.audio.stream import AudioStream
from streaming_asr.models.overlap_detector import DummyOverlapDetector
from streaming_asr.models.vad import DummyVAD
from streaming_asr.pipeline.streaming import StreamingASRPipeline

def test_pipeline_propagates_state():
    pipeline = StreamingASRPipeline(DummyVAD(), DummyOverlapDetector()); pipeline.start()
    output = pipeline.process(next(iter(AudioStream.from_array(np.ones(2), 2, 2))))
    state = pipeline.finalize()
    assert output.chunk_index == 0 and state.finalized

