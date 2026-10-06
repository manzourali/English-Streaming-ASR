"""Aggregation, comparisons, strata, and uncertainty for frozen Phase 9 runs."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .records import EvaluationRecord


@dataclass(frozen=True)
class MetricSummary:
    system: str
    metric: str
    direction: str
    samples: int
    value: float | None
    confidence_interval_95: tuple[float, float] | None
    dataset: str
    split: str
    oracle: bool


def _aggregate(records: list[EvaluationRecord]) -> float:
    weighted = [record for record in records if record.error_count is not None and record.reference_units]
    if weighted and len(weighted) == len(records):
        units = sum(record.reference_units or 0 for record in records)
        return sum(record.error_count or 0 for record in records) / units if units else 0.0
    return sum(record.value for record in records) / len(records)


def bootstrap_interval(records: list[EvaluationRecord], *, iterations: int = 1000, seed: int = 42) -> tuple[float, float] | None:
    if len(records) < 2:
        return None
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    values = sorted(_aggregate([records[rng.randrange(len(records))] for _ in records]) for _ in range(iterations))
    lower, upper = values[int(0.025 * (iterations - 1))], values[int(0.975 * (iterations - 1))]
    return lower, upper


class Evaluator:
    def __init__(self, bootstrap_iterations: int = 1000, seed: int = 42):
        self.bootstrap_iterations, self.seed = bootstrap_iterations, seed

    def summarize(self, records: Iterable[EvaluationRecord]) -> list[MetricSummary]:
        grouped: dict[tuple[str, str, str, str, bool, str], list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            grouped[(record.system, record.metric, record.dataset, record.split, record.oracle, record.direction)].append(record)
        return [MetricSummary(system, metric, direction, len(items), _aggregate(items), bootstrap_interval(items, iterations=self.bootstrap_iterations, seed=self.seed), dataset, split, oracle) for (system, metric, dataset, split, oracle, direction), items in sorted(grouped.items())]

    def stratify(self, records: Iterable[EvaluationRecord], key: str) -> dict[str, list[MetricSummary]]:
        if key not in {"overlap_regime", "temporal_condition"}:
            raise ValueError("supported strata are overlap_regime and temporal_condition")
        groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            value = getattr(record, key)
            if value is not None:
                groups[str(value)].append(record)
        return {name: self.summarize(items) for name, items in sorted(groups.items())}

    def error_taxonomy(self, records: Iterable[EvaluationRecord]) -> dict[str, int]:
        return dict(sorted(Counter(tag for record in records for tag in record.error_tags).items()))

    def paired_improvement(self, baseline: Iterable[EvaluationRecord], candidate: Iterable[EvaluationRecord]) -> dict[str, float | int | str | None]:
        base, new = {record.sample_id: record for record in baseline}, {record.sample_id: record for record in candidate}
        if set(base) != set(new) or not base:
            raise ValueError("improvement requires identical non-empty sample sets")
        first = next(iter(base.values()))
        if any(record.metric != first.metric or record.direction != first.direction for record in [*base.values(), *new.values()]):
            raise ValueError("improvement requires comparable metric and direction")
        base_value, candidate_value = _aggregate(list(base.values())), _aggregate(list(new.values()))
        difference = base_value - candidate_value if first.direction == "lower" else candidate_value - base_value
        relative = difference / abs(base_value) if base_value else None
        return {"metric": first.metric, "direction": first.direction, "samples": len(base), "baseline": base_value, "candidate": candidate_value, "absolute_improvement": difference, "relative_improvement": relative}
