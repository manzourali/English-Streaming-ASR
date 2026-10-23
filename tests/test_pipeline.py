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


def test_pipeline_can_run_vad_without_overlap_detector():
    pipeline = StreamingASRPipeline(DummyVAD(), None)
    pipeline.start()
    output = pipeline.process(next(iter(AudioStream.from_array(np.ones(2), 2, 2))))
    pipeline.finalize()
    assert output.vad.speech and output.overlap is None


def test_finalization_exposes_all_tails_once():
    class Component:
        def start(self): self.finals = 0
        def finalize(self): self.finals += 1; return "tail"
    vad, osd, asr = Component(), Component(), Component()
    pipeline = StreamingASRPipeline(vad, osd, asr); pipeline.start()
    final = pipeline.finalize()
    assert (final.vad, final.overlap, final.asr) == ("tail", "tail", "tail")
    assert pipeline.finalize() is final
    assert vad.finals == osd.finals == asr.finals == 1
