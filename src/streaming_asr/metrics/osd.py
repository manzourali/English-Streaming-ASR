from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from streaming_asr.datasets.manifests import OverlapRecord
from streaming_asr.models.overlap_detector import OverlapLabel, OverlapResult


@dataclass(frozen=True)
class OSDFrame:
    start_time: float
    end_time: float
    label: OverlapLabel


def ground_truth_frames(record: OverlapRecord, frame_ms: int) -> list[OSDFrame]:
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    frame_duration = frame_ms / 1000.0
    frames: list[OSDFrame] = []
    index = 0
    while index * frame_duration < record.duration - 1e-9:
        start = index * frame_duration
        end = min(record.duration, start + frame_duration)
        active = sum(max(start, source.start) < min(end, source.end) for source in record.sources)
        label = OverlapLabel.OVERLAP if active >= 2 else OverlapLabel.SINGLE_SPEAKER if active == 1 else OverlapLabel.NO_SPEECH
        frames.append(OSDFrame(start, end, label))
        index += 1
    return frames


def predictions_to_frames(results: Iterable[OverlapResult]) -> list[OSDFrame]:
    frames: list[OSDFrame] = []
    for result in results:
        if result.frame_count == 0:
            continue
        frames.append(OSDFrame(result.start_time, result.end_time, result.label))
    return frames


def align_frames(reference: list[OSDFrame], predictions: list[OSDFrame]) -> list[tuple[OverlapLabel, OverlapLabel]]:
    """Align by frame midpoint; frame duration differences are explicit, not hidden."""
    aligned = []
    for frame in reference:
        midpoint = (frame.start_time + frame.end_time) / 2
        candidates = [prediction for prediction in predictions if prediction.start_time <= midpoint < prediction.end_time]
        if candidates:
            aligned.append((frame.label, candidates[0].label))
    return aligned


def confusion_matrix(pairs: list[tuple[OverlapLabel, OverlapLabel]]) -> dict[str, dict[str, int]]:
    matrix = {actual.value: {predicted.value: 0 for predicted in OverlapLabel} for actual in OverlapLabel}
    for actual, predicted in pairs:
        matrix[actual.value][predicted.value] += 1
    return matrix


def overlap_metrics(pairs: list[tuple[OverlapLabel, OverlapLabel]]) -> dict[str, object]:
    tp = sum(actual is OverlapLabel.OVERLAP and predicted is OverlapLabel.OVERLAP for actual, predicted in pairs)
    fp = sum(actual is not OverlapLabel.OVERLAP and predicted is OverlapLabel.OVERLAP for actual, predicted in pairs)
    fn = sum(actual is OverlapLabel.OVERLAP and predicted is not OverlapLabel.OVERLAP for actual, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"num_frames": len(pairs), "overlap_precision": precision, "overlap_recall": recall, "overlap_f1": f1, "confusion_matrix": confusion_matrix(pairs)}


def event_detection_delays(reference: list[OSDFrame], predictions: list[OSDFrame]) -> list[float]:
    true_events = _events(reference, OverlapLabel.OVERLAP)
    predicted_events = _events(predictions, OverlapLabel.OVERLAP)
    delays = []
    for true_start, _ in true_events:
        starts = [predicted_start for predicted_start, predicted_end in predicted_events if predicted_end > true_start]
        if starts:
            delays.append(starts[0] - true_start)
    return delays


def _events(frames: list[OSDFrame], label: OverlapLabel) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []
    start = end = None
    for frame in frames:
        if frame.label is label:
            if start is None:
                start = frame.start_time
            end = frame.end_time
        elif start is not None:
            events.append((start, end))
            start = end = None
    if start is not None:
        events.append((start, end))
    return events
