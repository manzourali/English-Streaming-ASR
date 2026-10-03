"""Metrics for Phase 6 routing traces; independent of ASR model quality."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from streaming_asr.models.overlap_detector import OverlapLabel
from streaming_asr.pipeline.adaptive import Route, RoutingEvent, RoutingState


def routing_metrics(pairs: Iterable[tuple[OverlapLabel, Route]]) -> dict[str, object]:
    values = list(pairs)
    tp = sum(actual is OverlapLabel.OVERLAP and route is Route.OVERLAP for actual, route in values)
    fp = sum(actual is not OverlapLabel.OVERLAP and route is Route.OVERLAP for actual, route in values)
    fn = sum(actual is OverlapLabel.OVERLAP and route is not Route.OVERLAP for actual, route in values)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    matrix = {label.value: {route.value: 0 for route in Route} for label in OverlapLabel}
    for actual, route in values:
        matrix[actual.value][route.value] += 1
    return {"num_frames": len(values), "routing_precision": precision, "routing_recall": recall, "routing_f1": f1, "routing_confusion": matrix}


def route_durations(states: Iterable[tuple[RoutingState, float]]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for state, duration in states:
        totals[state.value] += max(0.0, duration)
    return {state.value: totals[state.value] for state in RoutingState}


def transition_metrics(events: Iterable[RoutingEvent], true_overlap_starts: Iterable[float] = ()) -> dict[str, object]:
    values = list(events)
    entries = [event for event in values if event.new_state is RoutingState.OVERLAP and event.previous_state is not RoutingState.OVERLAP]
    delays = []
    for start in true_overlap_starts:
        match = next((event for event in entries if event.timestamp >= start), None)
        if match is not None:
            delays.append(match.timestamp - start)
    return {"transition_count": len(values), "overlap_transition_count": len(entries), "detection_to_routing_latency_seconds": delays, "mean_detection_to_routing_latency_seconds": sum(delays) / len(delays) if delays else None}
