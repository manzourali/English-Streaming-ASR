"""Phase 6 adaptive, incremental ASR routing.

The default shared-branch mode sends every live chunk to one ASR adapter while
changing only its logical route. This preserves causal decoder context through
short overlap and silence events. Separate branches are supported for Phase 7;
bounded history is replayed for context but never emitted as transcript output.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from time import perf_counter
from typing import Any

from streaming_asr.audio.stream import AudioChunk
from streaming_asr.models.overlap_detector import OverlapLabel, OverlapResult, StreamingOverlapDetector
from streaming_asr.models.vad import StreamingVAD, VADResult
from streaming_asr.streaming.engine import StreamingEngine


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


@dataclass(frozen=True)
class AdaptivePipelineOutput:
    chunk_index: int
    timestamp: float
    vad: VADResult | None
    overlap: OverlapResult | None
    state: RoutingState
    route: Route
    confidence: float | None
    transcript_update: Any = None
    branch_output: Any = None
    transition: RoutingEvent | None = None
    processing_time: float | None = None
    controller_error: str | None = None


class ASRBranch(ABC):
    """Implementation-independent incremental ASR branch contract."""
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def process(self, audio_chunk: AudioChunk) -> Any: ...

    @abstractmethod
    def finalize(self) -> Any: ...


class ASRAdapterBranch(ASRBranch):
    """Adapt the existing WhisperRT-style ``start/process/finalize`` API."""
    def __init__(self, asr: Any):
        self.asr = asr

    def start(self) -> None:
        self.asr.start()

    def process(self, audio_chunk: AudioChunk) -> Any:
        return self.asr.process(audio_chunk)

    def finalize(self) -> Any:
        return self.asr.finalize()


class RoutingPolicy:
    """Thresholded three-state policy with configurable temporal persistence."""
    def __init__(self, overlap_threshold: float = 0.5, enter_overlap_frames: int = 1, exit_overlap_frames: int = 1, min_overlap_duration_ms: float = 0.0, use_vad: bool = True, idle_behavior: str = "forward_normal") -> None:
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between 0 and 1")
        if enter_overlap_frames < 1 or exit_overlap_frames < 1 or min_overlap_duration_ms < 0:
            raise ValueError("routing persistence values must be positive")
        if idle_behavior not in {"forward_normal", "skip"}:
            raise ValueError("idle_behavior must be 'forward_normal' or 'skip'")
        self.overlap_threshold = overlap_threshold
        self.enter_overlap_frames = enter_overlap_frames
        self.exit_overlap_frames = exit_overlap_frames
        self.min_overlap_duration = min_overlap_duration_ms / 1000.0
        self.use_vad = use_vad
        self.idle_behavior = idle_behavior
        self.start()

    def start(self) -> None:
        self.state = RoutingState.NO_SPEECH
        self._enter_count = self._exit_count = 0
        self._overlap_candidate_start: float | None = None

    def decide(self, vad_result: VADResult | None, osd_result: OverlapResult | None, timestamp: float) -> RoutingDecision:
        if self.use_vad and (vad_result is None or not vad_result.speech):
            self._enter_count = self._exit_count = 0
            self._overlap_candidate_start = None
            self.state = RoutingState.NO_SPEECH
            return self._decision(None, "vad_no_speech")
        probability = self._probability(osd_result)
        is_overlap = probability >= self.overlap_threshold
        if self.state is RoutingState.OVERLAP:
            self._exit_count = 0 if is_overlap else self._exit_count + 1
            if self._exit_count >= self.exit_overlap_frames:
                self.state = RoutingState.SINGLE_SPEAKER
                self._enter_count = 0
                self._overlap_candidate_start = None
                return self._decision(probability, "overlap_exit_confirmed")
            return self._decision(probability, "overlap_persisted" if is_overlap else "overlap_exit_pending")
        if is_overlap:
            self._enter_count += 1
            self._overlap_candidate_start = self._overlap_candidate_start if self._overlap_candidate_start is not None else timestamp
            duration_ready = timestamp - self._overlap_candidate_start >= self.min_overlap_duration
            if self._enter_count >= self.enter_overlap_frames and duration_ready:
                self.state = RoutingState.OVERLAP
                self._exit_count = 0
                return self._decision(probability, "overlap_enter_confirmed")
            self.state = RoutingState.SINGLE_SPEAKER
            return self._decision(probability, "overlap_enter_pending")
        self._enter_count = self._exit_count = 0
        self._overlap_candidate_start = None
        self.state = RoutingState.SINGLE_SPEAKER
        return self._decision(probability, "single_speaker")

    @staticmethod
    def _probability(result: OverlapResult | None) -> float:
        if result is None:
            return 0.0
        if result.overlap_probability is not None:
            return float(result.overlap_probability)
        return 1.0 if result.label is OverlapLabel.OVERLAP else 0.0

    def _decision(self, probability: float | None, reason: str) -> RoutingDecision:
        if self.state is RoutingState.OVERLAP:
            route = Route.OVERLAP
        elif self.state is RoutingState.SINGLE_SPEAKER:
            route = Route.NORMAL
        else:
            route = Route.NORMAL if self.idle_behavior == "forward_normal" else Route.IDLE
        return RoutingDecision(self.state, route, probability, reason)


class AlwaysNormalRoutingPolicy(RoutingPolicy):
    """Phase 6 mandatory control: observe state but never switch ASR route."""
    def decide(self, vad_result: VADResult | None, osd_result: OverlapResult | None, timestamp: float) -> RoutingDecision:
        decision = super().decide(vad_result, osd_result, timestamp)
        return RoutingDecision(decision.state, Route.NORMAL, decision.confidence, "always_normal_control")


class AdaptiveStreamingASRPipeline:
    """Incrementally coordinate VAD, OSD, routing and generic ASR branches.

    Oracle mode uses its supplied timing detector only as a clearly labelled,
    non-deployable experimental routing signal.
    """
    def __init__(self, normal_branch: ASRBranch | Any, overlap_branch: ASRBranch | Any | None = None, *, vad: StreamingVAD | None = None, overlap_detector: StreamingOverlapDetector | None = None, oracle_detector: StreamingOverlapDetector | None = None, oracle: bool = False, policy: RoutingPolicy | None = None, history_ms: float = 0.0, fallback_route: Route | str = Route.NORMAL) -> None:
        if history_ms < 0:
            raise ValueError("history_ms must be non-negative")
        self.normal_branch = self._as_branch(normal_branch)
        self.overlap_branch = self._as_branch(overlap_branch) if overlap_branch is not None else self.normal_branch
        self.vad, self.overlap_detector = vad, overlap_detector
        self.oracle_detector, self.oracle = oracle_detector, oracle
        if oracle and oracle_detector is None:
            raise ValueError("oracle routing requires oracle_detector")
        self.policy = policy or RoutingPolicy()
        self.history_seconds = history_ms / 1000.0
        self.fallback_route = Route(fallback_route)
        self.engine = StreamingEngine()
        self.events: list[RoutingEvent] = []
        self._history: deque[AudioChunk] = deque()
        self._started = False
        self._last_state = RoutingState.NO_SPEECH
        self._last_route = Route.IDLE
        self._branch_started: set[int] = set()
        self._branch_processed: dict[int, set[int]] = {}

    @staticmethod
    def _as_branch(value: ASRBranch | Any) -> ASRBranch:
        return value if isinstance(value, ASRBranch) else ASRAdapterBranch(value)

    @property
    def routing_mode(self) -> str:
        return "oracle" if self.oracle else "predicted"

    def start(self) -> None:
        self.engine.start(); self.policy.start(); self.events.clear(); self._history.clear()
        self._branch_started.clear(); self._branch_processed.clear()
        self._last_state, self._last_route = RoutingState.NO_SPEECH, Route.IDLE
        for component in (self.vad, self.overlap_detector, self.oracle_detector if self.oracle else None):
            if component is not None:
                component.start()
        self._start_branch(self.normal_branch)
        self._started = True

    def process(self, audio_chunk: AudioChunk) -> AdaptivePipelineOutput:
        if not self._started:
            self.start()
        started = perf_counter()
        self.engine.process(audio_chunk)
        controller_errors: list[str] = []
        try:
            vad_result = self.vad.process(audio_chunk) if self.vad is not None else None
        except Exception as exc:  # A controller failure must not corrupt branch state.
            vad_result = None
            controller_errors.append(f"vad:{type(exc).__name__}")
        try:
            predicted = self.overlap_detector.process(audio_chunk) if self.overlap_detector is not None else None
        except Exception as exc:
            predicted = None
            controller_errors.append(f"osd:{type(exc).__name__}")
        try:
            selected = self.oracle_detector.process(audio_chunk) if self.oracle and self.oracle_detector is not None else predicted
        except Exception as exc:
            selected = predicted
            controller_errors.append(f"oracle_osd:{type(exc).__name__}")
        decision = self.policy.decide(vad_result, selected, audio_chunk.end_time)
        transition = self._transition(audio_chunk, decision)
        route, branch_output = decision.route, None
        try:
            branch = self._branch_for(route)
            if branch is not None:
                self._start_branch(branch)
                if transition is not None and branch is not self._branch_for(self._last_route):
                    self._replay_history(branch, audio_chunk.index)
                branch_output = self._process_once(branch, audio_chunk)
        except Exception:
            if self.fallback_route is route:
                raise
            route = self.fallback_route
            branch = self._branch_for(route)
            if branch is None:
                raise
            self._start_branch(branch)
            branch_output = self._process_once(branch, audio_chunk)
        self._remember(audio_chunk)
        self._last_state, self._last_route = decision.state, route
        return AdaptivePipelineOutput(audio_chunk.index, audio_chunk.end_time, vad_result, selected, decision.state, route, decision.confidence, branch_output, branch_output, transition, perf_counter() - started, ";".join(controller_errors) or None)

    def finalize(self) -> dict[str, Any]:
        finals: dict[str, Any] = {}
        finalized: set[int] = set()
        for name, branch in (("normal", self.normal_branch), ("overlap", self.overlap_branch)):
            if id(branch) in self._branch_started and id(branch) not in finalized:
                finals[name] = branch.finalize()
                finalized.add(id(branch))
        for component in (self.vad, self.overlap_detector, self.oracle_detector if self.oracle else None):
            if component is not None:
                component.finalize()
        self._started = False
        return {"engine": self.engine.finalize(), "branches": finals, "events": [event.to_dict() for event in self.events], "routing_mode": self.routing_mode}

    def _transition(self, chunk: AudioChunk, decision: RoutingDecision) -> RoutingEvent | None:
        if decision.state is self._last_state and decision.route is self._last_route:
            return None
        event = RoutingEvent(chunk.end_time, self._last_state, decision.state, self._last_route, decision.route, decision.confidence, decision.reason)
        self.events.append(event)
        return event

    def _branch_for(self, route: Route) -> ASRBranch | None:
        return None if route is Route.IDLE else self.overlap_branch if route is Route.OVERLAP else self.normal_branch

    def _start_branch(self, branch: ASRBranch) -> None:
        key = id(branch)
        if key not in self._branch_started:
            branch.start(); self._branch_started.add(key); self._branch_processed[key] = set()

    def _process_once(self, branch: ASRBranch, chunk: AudioChunk) -> Any:
        seen = self._branch_processed[id(branch)]
        if chunk.index in seen:
            return None
        output = branch.process(chunk)
        seen.add(chunk.index)
        return output

    def _replay_history(self, branch: ASRBranch, current_index: int) -> None:
        if self.history_seconds > 0:
            for chunk in self._history:
                if chunk.index < current_index:
                    self._process_once(branch, chunk)

    def _remember(self, chunk: AudioChunk) -> None:
        if self.history_seconds <= 0:
            return
        self._history.append(chunk)
        cutoff = chunk.end_time - self.history_seconds
        while self._history and self._history[0].end_time <= cutoff:
            self._history.popleft()


# Backwards-compatible public name for callers that imported the Phase 0 stub.
AdaptivePipeline = AdaptiveStreamingASRPipeline
