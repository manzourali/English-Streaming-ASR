"""Optional visualizations for saved Phase 6 routing traces."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def plot_routing_timeline(traces: Iterable[Mapping[str, object]]):
    """Plot ground truth, routed state, and confidence against chunk time.

    Matplotlib is optional so the core streaming package stays lightweight.
    ``traces`` accepts JSONL rows written by ``scripts/run_adaptive.py``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is optional
        raise RuntimeError("Routing plots require the optional matplotlib package") from exc
    rows = list(traces)
    times = [float(row["timestamp"]) for row in rows]
    encode = {"no_speech": 0, "single_speaker": 1, "overlap": 2, "idle": 0, "normal": 1}
    figure, axis = plt.subplots(figsize=(12, 3.5))
    axis.step(times, [encode.get(str(row.get("ground_truth")), -1) for row in rows], where="post", label="ground truth")
    axis.step(times, [encode.get(str(row.get("state")), -1) for row in rows], where="post", label="routing state")
    axis.step(times, [encode.get(str(row.get("route")), -1) for row in rows], where="post", label="selected route", alpha=0.7)
    axis.set(yticks=[0, 1, 2], yticklabels=["no speech / idle", "single / normal", "overlap"], xlabel="time (s)", title="Phase 6 routing timeline")
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_route_distribution(traces: Iterable[Mapping[str, object]]):
    """Plot chunk-count route distribution for a routing trace."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is optional
        raise RuntimeError("Routing plots require the optional matplotlib package") from exc
    counts = Counter(str(row.get("route", "unknown")) for row in traces)
    figure, axis = plt.subplots(figsize=(5, 3.5))
    axis.bar(list(counts), list(counts.values()))
    axis.set(xlabel="route", ylabel="chunks", title="Phase 6 route distribution")
    figure.tight_layout()
    return figure
