"""The old routing entry point now runs one continuous recognizer."""
import numpy as np
import streaming_asr.pipeline.adaptive as adaptive
from streaming_asr.audio.stream import AudioStream
from streaming_asr.pipeline.streaming import StreamingASRPipeline


def test_dynamic_router_is_removed_and_alias_is_unified():
    assert not hasattr(adaptive, "RoutingPolicy")
    assert adaptive.AdaptivePipeline is StreamingASRPipeline
    class Recognizer:
        def start(self): self.count = 0
        def process(self, chunk): self.count += len(chunk.samples)
        def finalize(self): return self.count
    asr = Recognizer()
    pipeline = adaptive.AdaptivePipeline(asr=asr); pipeline.start()
    for chunk in AudioStream.from_array(np.zeros(1001), 16000, 300):
        pipeline.process(chunk)
    assert pipeline.finalize().asr == 1001
