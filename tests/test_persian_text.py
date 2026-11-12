"""Tests for Persian normalization and BPE preparation."""
import importlib.util
import sys
import types

import pytest

from streaming_asr.datasets.persian_text import normalize_persian, oov_coverage, train_bpe


def test_normalization_transforms_digits_and_arabic_letters(monkeypatch):
    normalizer = type("Normalizer", (), {"normalize": lambda self, text: text})
    monkeypatch.setitem(sys.modules, "hazm", types.SimpleNamespace(Normalizer=normalizer))
    monkeypatch.setitem(sys.modules, "num2words", types.SimpleNamespace(
        num2words=lambda number, lang: f"number-{number}"))
    assert normalize_persian("كِتاب يک ۱۲ می‌رود") == "کتاب یک number-12 می‌رود"
    with pytest.raises(ValueError):
        normalize_persian("۱۲٫۵")


@pytest.mark.skipif(not all(importlib.util.find_spec(name) for name in ("hazm", "num2words")),
                    reason="Persian normalization dependencies are optional")
def test_persian_normalization():
    result = normalize_persian("كِتاب يک ۱۲ می‌رود")
    assert "کتاب" in result
    assert "یک" in result
    assert "دوازده" in result
    assert "می‌رود" in result
    with pytest.raises(ValueError):
        normalize_persian("۱۲٫۵")


@pytest.mark.skipif(not importlib.util.find_spec("sentencepiece"),
                    reason="SentencePiece is optional")
def test_bpe_round_trip_and_oov(tmp_path):
    import sentencepiece as spm

    text = tmp_path / "text"
    text.write_text("سلام دنیا\nکتاب خوب\nمی‌رود خانه\n", encoding="utf-8")
    prefix = tmp_path / "bpe"
    train_bpe(text, prefix, 64)
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    assert processor.decode(processor.encode("می‌رود خانه")) == "می‌رود خانه"
    assert processor.piece_to_id("<blk>") == 0
    assert oov_coverage(str(prefix) + ".model", text)["word_oov_rate"] == 0
    text.write_text("سلام 🦊\n", encoding="utf-8")
    assert oov_coverage(str(prefix) + ".model", text)["unknown_words"] == 1
