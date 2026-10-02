def binary_f1(reference: list[bool], hypothesis: list[bool]) -> float:
    if len(reference) != len(hypothesis):
        raise ValueError("reference and hypothesis lengths must match")
    tp = sum(r and h for r, h in zip(reference, hypothesis))
    fp = sum((not r) and h for r, h in zip(reference, hypothesis))
    fn = sum(r and (not h) for r, h in zip(reference, hypothesis))
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)

