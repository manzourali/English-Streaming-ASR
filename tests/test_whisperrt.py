from streaming_asr.models.whisperrt import TranscriptAccumulator, TranscriptUpdate, resolve_device, validate_dtype


def test_transcript_accumulator_keeps_latest_partial_only():
    accumulator = TranscriptAccumulator()
    accumulator.update(TranscriptUpdate("the", 0, 0.0, 0.3))
    accumulator.update(TranscriptUpdate("the quick", 1, 0.3, 0.6))
    assert accumulator.text == "the quick"
    assert accumulator.finalize().is_final


def test_cpu_dtype_validation():
    validate_dtype("auto", "cpu")
    assert resolve_device("cpu") == "cpu"
