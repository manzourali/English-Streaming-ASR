"""External-recipe training lifecycle with reproducible metadata/checkpoints."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping

from streaming_asr.exceptions import BackendNotAvailableError
from .losses import validate_loss


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable

    @property
    def trainable_percent(self) -> float:
        return 100 * self.trainable / self.total if self.total else 0.0


@dataclass(frozen=True)
class TrainingState:
    epoch: int = 0
    step: int = 0
    best_validation_loss: float | None = None
    bad_validation_checks: int = 0


class ExternalSurtTrainingBackend(ABC):
    """The only supported route to real SURT adaptation in this repository."""
    @abstractmethod
    def parameter_counts(self) -> ParameterCounts: ...

    @abstractmethod
    def train_step(self, batch: Any) -> Mapping[str, float]: ...

    @abstractmethod
    def validation_step(self, batch: Any) -> Mapping[str, float]: ...

    @abstractmethod
    def save_checkpoint(self, directory: Path) -> None: ...

    @abstractmethod
    def load_checkpoint(self, directory: Path) -> None: ...


class Trainer:
    def __init__(self, backend: ExternalSurtTrainingBackend, checkpoint_dir: str | Path, *, early_stopping_patience: int = 3):
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = early_stopping_patience
        self.state = TrainingState()

    def resume(self, directory: str | Path) -> TrainingState:
        directory = Path(directory)
        self.backend.load_checkpoint(directory)
        self.state = TrainingState(**json.loads((directory / "training_state.json").read_text(encoding="utf-8")))
        return self.state

    def fit(self, train_batches: Iterable[Any], validation_batches: Iterable[Any], *, epochs: int, eval_steps: int, save_steps: int) -> list[dict[str, float]]:
        if epochs < 1 or eval_steps < 1 or save_steps < 1:
            raise ValueError("epochs, eval_steps, and save_steps must be positive")
        history: list[dict[str, float]] = []
        train_batches, validation_batches = list(train_batches), list(validation_batches)
        if not train_batches or not validation_batches:
            raise ValueError("training and validation batches must be non-empty")
        for epoch in range(self.state.epoch + 1, epochs + 1):
            for batch in train_batches:
                metrics = {key: float(value) for key, value in self.backend.train_step(batch).items()}
                metrics["loss"] = validate_loss(metrics["loss"])
                self.state = TrainingState(epoch, self.state.step + 1, self.state.best_validation_loss, self.state.bad_validation_checks)
                metrics.update({"epoch": float(epoch), "step": float(self.state.step)})
                history.append(metrics)
                if self.state.step % eval_steps == 0:
                    validation = self._validate(validation_batches)
                    history.append({**validation, "epoch": float(epoch), "step": float(self.state.step)})
                    if self.state.bad_validation_checks > self.patience:
                        self._save("last")
                        return history
                if self.state.step % save_steps == 0:
                    self._save("last")
        self._save("last")
        return history

    def _validate(self, batches: list[Any]) -> dict[str, float]:
        losses = [validate_loss(self.backend.validation_step(batch)["loss"]) for batch in batches]
        loss = sum(losses) / len(losses)
        best = self.state.best_validation_loss
        improved = best is None or loss < best
        self.state = TrainingState(self.state.epoch, self.state.step, loss if improved else best, 0 if improved else self.state.bad_validation_checks + 1)
        if improved:
            self._save("best")
        return {"validation_loss": loss}

    def _save(self, name: str) -> Path:
        directory = self.checkpoint_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        self.backend.save_checkpoint(directory)
        (directory / "training_state.json").write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        return directory


def load_training_backend(factory_path: str | None, config: Mapping[str, Any]) -> ExternalSurtTrainingBackend:
    if not factory_path:
        raise BackendNotAvailableError("TRAINING NOT TECHNICALLY SUPPORTED: configure training.backend_factory for an exact, verified Icefall/SURT training recipe. The Phase 7 decoder_factory is inference-only.")
    try:
        module_name, attribute = factory_path.split(":", 1)
        backend = getattr(import_module(module_name), attribute)(dict(config))
    except (ValueError, ImportError, AttributeError) as exc:
        raise BackendNotAvailableError(f"Unable to load configured training.backend_factory {factory_path!r}") from exc
    if not isinstance(backend, ExternalSurtTrainingBackend):
        raise BackendNotAvailableError("training.backend_factory must return ExternalSurtTrainingBackend")
    return backend
