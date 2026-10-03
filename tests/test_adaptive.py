import numpy as np

from streaming_asr.audio.stream import AudioStream
from streaming_asr.models.overlap_detector import OracleOverlapDetector
from streaming_asr.models.vad import DummyVAD
from streaming_asr.pipeline.adaptive import ASRBranch, AdaptiveStreamingASRPipeline, Route, RoutingPolicy, RoutingState


class RecordingBranch(ASRBranch):
    def __init__(self):
        self.starts = 0
        self.indices = []
        self.finalized = 0

    def start(self):
        self.starts += 1

    def process(self, chunk):
        self.indices.append(chunk.index)
        return f"chunk-{chunk.index}"

    def finalize(self):
        self.finalized += 1
        return "final"


def chunks(values):
    return list(AudioStream.from_array(np.asarray(values, dtype=np.float32), 10, 1))


def test_state_machine_and_oracle_routing_are_incremental():
    # Active source counts produce: single, overlap, single, no speech.
    normal, overlap = RecordingBranch(), RecordingBranch()
    pipeline = AdaptiveStreamingASRPipeline(
        normal,
        overlap,
        vad=DummyVAD(threshold=0.01),
        oracle_detector=OracleOverlapDetector([(0.0, 0.3), (0.1, 0.2)]),
        oracle=True,
        policy=RoutingPolicy(enter_overlap_frames=1, exit_overlap_frames=1),
    )
    outputs = [pipeline.process(chunk) for chunk in chunks([1, 1, 1, 0])]
    result = pipeline.finalize()
    assert [item.state for item in outputs] == [RoutingState.SINGLE_SPEAKER, RoutingState.OVERLAP, RoutingState.SINGLE_SPEAKER, RoutingState.NO_SPEECH]
    assert [item.route for item in outputs] == [Route.NORMAL, Route.OVERLAP, Route.NORMAL, Route.NORMAL]
    assert normal.indices == [0, 2, 3] and overlap.indices == [1]
    assert result["routing_mode"] == "oracle"


def test_hysteresis_requires_confirmed_enter_and_exit():
    policy = RoutingPolicy(overlap_threshold=0.5, enter_overlap_frames=2, exit_overlap_frames=2, use_vad=False)
    detector = OracleOverlapDetector([(0.0, 10.0), (0.0, 10.0)])
    detector.start()
    chunks_ = chunks([1, 1, 1, 1])
    first = policy.decide(None, detector.process(chunks_[0]), 0.1)
    second = policy.decide(None, detector.process(chunks_[1]), 0.2)
    assert first.route is Route.NORMAL and second.route is Route.OVERLAP
    no_overlap = OracleOverlapDetector([(0.0, 10.0)])
    no_overlap.start()
    assert policy.decide(None, no_overlap.process(chunks_[2]), 0.3).route is Route.OVERLAP
    assert policy.decide(None, no_overlap.process(chunks_[3]), 0.4).route is Route.NORMAL


def test_shared_branch_prevents_duplicate_audio_and_finalization():
    branch = RecordingBranch()
    pipeline = AdaptiveStreamingASRPipeline(
        branch,
        vad=DummyVAD(threshold=0.01),
        oracle_detector=OracleOverlapDetector([(0.0, 0.3), (0.1, 0.2)]),
        oracle=True,
        policy=RoutingPolicy(),
        history_ms=200,
    )
    for chunk in chunks([1, 1, 1]):
        pipeline.process(chunk)
    pipeline.finalize()
    assert branch.indices == [0, 1, 2]
    assert branch.finalized == 1


def test_idle_skip_does_not_drop_future_route_audio():
    branch = RecordingBranch()
    pipeline = AdaptiveStreamingASRPipeline(
        branch,
        vad=DummyVAD(threshold=0.01),
        oracle_detector=OracleOverlapDetector([(0.1, 0.2)]),
        oracle=True,
        policy=RoutingPolicy(idle_behavior="skip"),
    )
    outputs = [pipeline.process(chunk) for chunk in chunks([0, 1])]
    assert outputs[0].route is Route.IDLE and outputs[0].transcript_update is None
    assert branch.indices == [1]


def test_osd_failure_has_deterministic_normal_fallback():
    class BrokenOSD:
        def start(self): pass
        def finalize(self): pass
        def process(self, chunk): raise RuntimeError("unavailable")

    branch = RecordingBranch()
    pipeline = AdaptiveStreamingASRPipeline(branch, vad=DummyVAD(threshold=0.01), overlap_detector=BrokenOSD(), policy=RoutingPolicy())
    output = pipeline.process(chunks([1])[0])
    assert output.route is Route.NORMAL and output.controller_error == "osd:RuntimeError"
