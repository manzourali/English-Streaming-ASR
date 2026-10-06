"""Optional record-driven Phase 9 figures; no values are hard-coded."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from streaming_asr.evaluation.benchmark import BenchmarkSummary
from streaming_asr.evaluation.evaluator import MetricSummary


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise RuntimeError("Evaluation figures require optional matplotlib") from exc
    return plt


def plot_metric_by_overlap(strata: dict[str, list[MetricSummary]], metric: str):
    """Compare one metric across existing overlap strata; skips absent values."""
    plt = _plt()
    figure, axis = plt.subplots(figsize=(8, 4))
    systems = sorted({item.system for values in strata.values() for item in values if item.metric == metric and item.value is not None})
    labels = list(strata)
    for system in systems:
        values = [next((item.value for item in strata[label] if item.system == system and item.metric == metric), None) for label in labels]
        axis.plot(labels, values, marker="o", label=system)
    axis.set(xlabel="overlap condition", ylabel=metric, title=f"{metric} by overlap condition")
    if systems: axis.legend()
    figure.tight_layout()
    return figure


def plot_latency_accuracy(summaries: Iterable[MetricSummary], benchmarks: Iterable[BenchmarkSummary], metric: str):
    """Plot measured system metric against first-output latency only."""
    plt = _plt()
    metric_lookup = {item.system: item.value for item in summaries if item.metric == metric and item.value is not None}
    figure, axis = plt.subplots(figsize=(6, 4))
    for item in benchmarks:
        if item.system in metric_lookup and item.average_first_output_latency_seconds is not None:
            axis.scatter(item.average_first_output_latency_seconds, metric_lookup[item.system])
            axis.annotate(item.system, (item.average_first_output_latency_seconds, metric_lookup[item.system]))
    axis.set(xlabel="average first-output latency (s)", ylabel=metric, title=f"Latency–{metric} trade-off")
    figure.tight_layout()
    return figure


def plot_computational_cost(benchmarks: Iterable[BenchmarkSummary]):
    """Compare RTF only for systems where processing and audio time were recorded."""
    plt = _plt()
    values = [item for item in benchmarks if item.rtf is not None]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([item.system for item in values], [item.rtf for item in values])
    axis.set(xlabel="system", ylabel="RTF", title="Computational cost")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure
