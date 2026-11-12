"""Persian text preparation and SentencePiece vocabulary utilities."""
import re
import unicodedata
from pathlib import Path


_ARABIC_LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUMBER = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"[0-9]+[٫./][0-9]+")


def normalize_persian(text: str) -> str:
    """Normalize Persian text, converting integer digit sequences into words.

    Decimal numbers and numeric dates must be expanded upstream because their
    spoken form depends on context. ZWNJ remains a distinct character.
    """
    try:
        from hazm import Normalizer
        from num2words import num2words
    except ImportError as exc:
        raise RuntimeError("Install the persian extra: pip install '.[persian]'") from exc

    text = unicodedata.normalize("NFC", text).translate(_ARABIC_LETTERS).translate(_DIGITS)
    text = "".join(char for char in text if not unicodedata.combining(char))
    if _DECIMAL.search(text):
        raise ValueError("Expand decimal numbers and numeric dates before normalization")
    text = Normalizer().normalize(text)
    text = text.translate(_ARABIC_LETTERS).translate(_DIGITS)
    text = _NUMBER.sub(lambda match: num2words(int(match.group()), lang="fa"), text)
    return " ".join(text.split())


def normalize_file(source: str | Path, destination: str | Path) -> int:
    """Normalize one transcript per input line, preserving line boundaries."""
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    normalized = [normalize_persian(line) for line in lines]
    if any(not line for line in normalized):
        raise ValueError("Empty transcript after normalization")
    Path(destination).write_text("".join(line + "\n" for line in normalized), encoding="utf-8")
    return len(normalized)


def train_bpe(text_path: str | Path, model_prefix: str | Path, vocab_size: int = 500) -> None:
    """Train an Icefall-style BPE model with blank ID 0 and unknown ID 1."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install the persian extra: pip install '.[persian]'") from exc
    if vocab_size < 8:
        raise ValueError("vocab_size must be at least 8")
    spm.SentencePieceTrainer.train(
        input=str(text_path), model_prefix=str(model_prefix), vocab_size=vocab_size,
        model_type="bpe", pad_id=0, pad_piece="<blk>", unk_id=1, bos_id=-1, eos_id=-1,
        normalization_rule_name="identity", hard_vocab_limit=False,
        shuffle_input_sentence=False,
    )


def oov_coverage(model_path: str | Path, text_path: str | Path) -> dict:
    """Measure words and emitted pieces containing the unknown token."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install the persian extra: pip install '.[persian]'") from exc
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    words = Path(text_path).read_text(encoding="utf-8").split()
    word_ids = [processor.encode(word, out_type=int) for word in words]
    unknown_words = sum(processor.unk_id() in ids for ids in word_ids)
    pieces = sum(len(ids) for ids in word_ids)
    unknown_pieces = sum(ids.count(processor.unk_id()) for ids in word_ids)
    return {"words": len(words), "unknown_words": unknown_words,
            "word_oov_rate": unknown_words / len(words) if words else None,
            "pieces": pieces, "unknown_pieces": unknown_pieces,
            "piece_oov_rate": unknown_pieces / pieces if pieces else None}
