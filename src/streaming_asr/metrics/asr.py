from __future__ import annotations

def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    return _edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = list(reference), list(hypothesis)
    return _edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)

