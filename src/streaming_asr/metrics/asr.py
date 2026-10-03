from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import permutations


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


@dataclass(frozen=True)
class PermutationWER:
    wer: float
    assignment: tuple[int, ...]
    reference_words: int


def permutation_invariant_word_error_rate(references: list[str], hypotheses: list[str]) -> PermutationWER:
    """Best stream permutation WER for unordered multi-talker outputs.

    Streams are output channels rather than enrolled/identified speakers. Empty
    padding makes unmatched reference or hypothesis streams count as deletions
    or insertions instead of being silently discarded.
    """
    if not references and not hypotheses:
        return PermutationWER(0.0, (), 0)
    count = max(len(references), len(hypotheses))
    refs, hyps = list(references) + [""] * (count - len(references)), list(hypotheses) + [""] * (count - len(hypotheses))
    ref_words = sum(len(normalize_text(item).split()) for item in refs)
    best_errors, best_assignment = None, tuple(range(count))
    for assignment in permutations(range(count)):
        errors = sum(_edit_distance(normalize_text(refs[index]).split(), normalize_text(hyps[target]).split()) for index, target in enumerate(assignment))
        if best_errors is None or errors < best_errors:
            best_errors, best_assignment = errors, assignment
    assert best_errors is not None
    return PermutationWER(best_errors / ref_words if ref_words else (0.0 if best_errors == 0 else 1.0), best_assignment, ref_words)
