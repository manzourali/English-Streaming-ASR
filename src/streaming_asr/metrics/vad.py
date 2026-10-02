from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryVADMetrics:
    precision: float | None
    recall: float | None
    f1: float | None
    false_alarm_rate: float | None
    miss_rate: float | None


def binary_vad_metrics(reference: list[bool], hypothesis: list[bool]) -> BinaryVADMetrics:
    if len(reference) != len(hypothesis):
        raise ValueError("reference and hypothesis lengths must match")
    tp = sum(r and h for r, h in zip(reference, hypothesis))
    fp = sum((not r) and h for r, h in zip(reference, hypothesis))
    fn = sum(r and (not h) for r, h in zip(reference, hypothesis))
    tn = sum((not r) and (not h) for r, h in zip(reference, hypothesis))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return BinaryVADMetrics(precision, recall, f1, fp / (fp + tn) if fp + tn else None, fn / (fn + tp) if fn + tp else None)
