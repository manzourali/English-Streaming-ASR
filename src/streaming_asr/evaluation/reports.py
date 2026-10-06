"""Programmatic Phase 9 report tables; never hand-enter measured values."""
from __future__ import annotations

from typing import Iterable

from .benchmark import BenchmarkSummary
from .evaluator import MetricSummary


NOT_MEASURED = "NOT MEASURED"


def _number(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{value:.4g}"


def main_results_table(summaries: Iterable[MetricSummary], systems: Iterable[str]) -> str:
    lookup = {(item.system, item.metric): item for item in summaries if not item.oracle}
    metrics = sorted({item.metric for item in lookup.values()}) or ["ASR metric"]
    lines = ["| System | " + " | ".join(metrics) + " |", "| --- | " + " | ".join("---:" for _ in metrics) + " |"]
    for system in systems:
        lines.append("| " + system + " | " + " | ".join(_number(lookup.get((system, metric)).value if (system, metric) in lookup else None) for metric in metrics) + " |")
    return "\n".join(lines)


def computational_table(summaries: Iterable[BenchmarkSummary], systems: Iterable[str]) -> str:
    lookup = {item.system: item for item in summaries}
    lines = ["| System | RTF | Avg first-output latency (s) | Avg finalization latency (s) | Peak GPU memory (MiB) |", "| --- | ---: | ---: | ---: | ---: |"]
    for system in systems:
        item = lookup.get(system)
        values = [item.rtf, item.average_first_output_latency_seconds, item.average_finalization_latency_seconds, item.peak_gpu_memory_mib] if item else [None] * 4
        lines.append("| " + system + " | " + " | ".join(_number(value) for value in values) + " |")
    return "\n".join(lines)


def ablation_table(rows: Iterable[dict[str, object]]) -> str:
    lines = ["| System configuration | VAD | OSD | Routing | Multi-talker | Fine-tuning | ASR metric | RTF |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        flags = ["Yes" if row.get(key) else "No" for key in ("vad", "osd", "routing", "multitalker", "finetuned")]
        lines.append("| " + str(row["name"]) + " | " + " | ".join(flags + [NOT_MEASURED, NOT_MEASURED]) + " |")
    return "\n".join(lines)


def osd_table(summaries: Iterable[MetricSummary]) -> str:
    lookup = {item.metric: item.value for item in summaries if item.metric.startswith("osd_") and not item.oracle}
    return "| Condition | Precision | Recall | F1 | Detection delay |\n| --- | ---: | ---: | ---: | ---: |\n| OSD | " + " | ".join(_number(lookup.get(metric)) for metric in ("osd_precision", "osd_recall", "osd_f1", "osd_detection_delay")) + " |"


def phase9_report(freeze: dict[str, object], summaries: Iterable[MetricSummary], benchmarks: Iterable[BenchmarkSummary], systems: Iterable[str], strata: dict[str, object], taxonomy: dict[str, int], ablations: Iterable[dict[str, object]] = ()) -> str:
    systems = list(systems)
    return "# Phase 9 — Comprehensive Evaluation\n\n## Frozen configuration\n\n```json\n" + __import__("json").dumps(freeze, indent=2, sort_keys=True) + "\n```\n\n## Main results\n\n" + main_results_table(summaries, systems) + "\n\n## OSD results\n\n" + osd_table(summaries) + "\n\n## Ablation plan/results\n\n" + ablation_table(ablations) + "\n\n## Computational results\n\n" + computational_table(benchmarks, systems) + "\n\n## Stratification\n\n```json\n" + __import__("json").dumps(strata, indent=2, default=str, sort_keys=True) + "\n```\n\n## Error taxonomy\n\n```json\n" + __import__("json").dumps(taxonomy, indent=2, sort_keys=True) + "\n```\n\nOracle rows, if present in input records, are not included in primary deployable-system tables. Missing values remain `NOT MEASURED`.\n"
