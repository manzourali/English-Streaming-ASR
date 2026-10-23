"""Historical trace types for reading prior routing experiments only."""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

class RoutingState(str, Enum):
    NO_SPEECH = "no_speech"
    SINGLE_SPEAKER = "single_speaker"
    OVERLAP = "overlap"


class Route(str, Enum):
    IDLE = "idle"
    NORMAL = "normal"
    OVERLAP = "overlap"


@dataclass(frozen=True)
class RoutingDecision:
    state: RoutingState
    route: Route
    confidence: float | None
    reason: str


@dataclass(frozen=True)
class RoutingEvent:
    timestamp: float
    previous_state: RoutingState
    new_state: RoutingState
    previous_route: Route
    route: Route
    overlap_probability: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("previous_state", "new_state", "previous_route", "route"):
            value[key] = value[key].value
        return value


