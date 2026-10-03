import pytest
from streaming_asr.metrics.asr import character_error_rate, permutation_invariant_word_error_rate, word_error_rate
from streaming_asr.metrics.overlap import binary_f1
from streaming_asr.metrics.streaming import real_time_factor
from streaming_asr.metrics.vad import binary_vad_metrics

def test_deterministic_metrics():
    assert word_error_rate("one two", "one two") == 0
    assert character_error_rate("abc", "adc") == pytest.approx(1 / 3)
    assert binary_f1([True, False], [True, True]) == pytest.approx(2 / 3)
    assert real_time_factor(2, 4) == pytest.approx(0.5)
    assert binary_vad_metrics([True, False], [True, True]).precision == pytest.approx(0.5)


def test_permutation_invariant_wer_is_not_speaker_attributed():
    result = permutation_invariant_word_error_rate(["hello there", "general kenobi"], ["general kenobi", "hello there"])
    assert result.wer == 0 and result.assignment == (1, 0)
    assert permutation_invariant_word_error_rate(["one"], ["one", "extra"]).wer == pytest.approx(1.0)
