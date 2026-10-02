from streaming_asr.metrics.overlap_baseline import aggregate_overlap_results, control_reference


def test_control_reference_and_overlap_exclusion():
    refs = [{"start": 1.0, "transcript": "bravo"}, {"start": 0.0, "transcript": "alpha"}]
    assert control_reference(refs, False) == "alpha bravo"
    assert control_reference(refs, True) is None


def test_aggregate_handles_missing_wer():
    summary = aggregate_overlap_results([])
    assert summary == {}
