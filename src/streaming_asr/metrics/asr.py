from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return " ".join(text.split())

def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    return _edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)


def corpus_word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses lengths must match")
    ref = normalize_text(" ".join(references)).split()
    hyp = normalize_text(" ".join(hypotheses)).split()
    return _edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = list(normalize_text(reference)), list(normalize_text(hypothesis))
    return _edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)
