import numpy as np

from streaming_asr.audio.stream import AudioStream
from streaming_asr.datasets.overlap_generator import SourceUtterance, OverlapGenerator
from streaming_asr.metrics.osd import OSDFrame, align_frames, event_detection_delays, ground_truth_frames, overlap_metrics
from streaming_asr.models.overlap_detector import OverlapLabel, SpectralHeuristicOSD, get_overlap_detector


def test_osd_metrics_perfect_and_error_cases():
    reference = [
        OSDFrame(0.0, 0.1, OverlapLabel.NO_SPEECH),
        OSDFrame(0.1, 0.2, OverlapLabel.OVERLAP),
        OSDFrame(0.2, 0.3, OverlapLabel.SINGLE_SPEAKER),
    ]
    assert overlap_metrics([(frame.label, frame.label) for frame in reference])["overlap_f1"] == 1.0
    predicted = [
        OSDFrame(0.0, 0.1, OverlapLabel.SINGLE_SPEAKER),
        OSDFrame(0.1, 0.2, OverlapLabel.NO_SPEECH),
        OSDFrame(0.2, 0.3, OverlapLabel.OVERLAP),
    ]
    metrics = overlap_metrics(align_frames(reference, predicted))
    assert metrics["overlap_precision"] == 0.0
    assert metrics["overlap_recall"] == 0.0
    assert metrics["confusion_matrix"]["overlap"]["no_speech"] == 1
    assert overlap_metrics([(OverlapLabel.NO_SPEECH, OverlapLabel.NO_SPEECH)])["overlap_f1"] is None
    all_positive = overlap_metrics([(OverlapLabel.OVERLAP, OverlapLabel.OVERLAP)])
    assert all_positive["overlap_precision"] == 1.0 and all_positive["overlap_recall"] == 1.0


def test_event_delay_and_alignment_are_explicit():
    reference = [OSDFrame(0.0, 0.1, OverlapLabel.NO_SPEECH), OSDFrame(0.1, 0.2, OverlapLabel.OVERLAP)]
    predicted = [OSDFrame(0.0, 0.15, OverlapLabel.SINGLE_SPEAKER), OSDFrame(0.15, 0.25, OverlapLabel.OVERLAP)]
    pairs = align_frames(reference, predicted)
    assert pairs[0][1] is OverlapLabel.SINGLE_SPEAKER
    assert np.isclose(event_detection_delays(reference, predicted)[0], 0.05)


def test_ground_truth_uses_source_intervals_independently(tmp_path):
    sample_rate = 1000
    source_a = SourceUtterance("a", "A", np.ones(200, dtype=np.float32), sample_rate, "a", str(tmp_path / "a.wav"))
    source_b = SourceUtterance("b", "B", np.ones(100, dtype=np.float32), sample_rate, "b", str(tmp_path / "b.wav"))
    record = OverlapGenerator(tmp_path, sample_rate, seed=1).generate_pair(source_a, source_b, split="test", regime="medium", target_ratio=0.5)
    frames = ground_truth_frames(record, 50)
    assert frames
    assert any(frame.label is OverlapLabel.OVERLAP for frame in frames)


def test_spectral_detector_is_streaming_and_restarts():
    detector = SpectralHeuristicOSD(sample_rate=1000, frame_ms=20, speech_threshold=1e-8)
    audio = np.sin(2 * np.pi * 100 * np.arange(40) / 1000).astype(np.float32)
    detector.start()
    outputs = [detector.process(chunk) for chunk in AudioStream.from_array(audio, 1000, 10)]
    detector.finalize()
    assert len(outputs) == 4
    assert [output.frame_count for output in outputs] == [0, 1, 0, 1]
    detector.start()
    assert detector.process(list(AudioStream.from_array(audio[:10], 1000, 10))[0]).frame_count == 0
    assert get_overlap_detector("heuristic", {"sample_rate": 1000, "frame_ms": 20}).frame_ms == 20
