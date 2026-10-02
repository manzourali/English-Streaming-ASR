"""Environment-aware, safe project path resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from streaming_asr.exceptions import ConfigurationError


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    inputs: Path
    outputs: Path
    checkpoints: Path

    @classmethod
    def from_config(cls, config: object, root: str | Path | None = None) -> "ProjectPaths":
        environment = config.get("runtime.environment", "local")
        if environment not in {"local", "kaggle"}:
            raise ConfigurationError(f"Unsupported runtime environment: {environment}")
        project_root = Path(root or config.get("paths.root", Path.cwd())).resolve()
        if environment == "kaggle":
            data = Path(config.get("paths.data", "/kaggle/input"))
            inputs = Path("/kaggle/input")
            outputs = Path(config.get("paths.outputs", "/kaggle/working/outputs"))
            checkpoints = Path(config.get("paths.checkpoints", "/kaggle/working/checkpoints"))
        else:
            data = project_root / config.get("paths.data", "data")
            inputs = data
            outputs = project_root / config.get("paths.outputs", "outputs")
            checkpoints = project_root / config.get("paths.checkpoints", "checkpoints")
        return cls(project_root, data, inputs, outputs, checkpoints)

    def ensure_output_dirs(self) -> None:
        for path in (self.outputs, self.checkpoints):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "metrics", "predictions", "figures", "reports"):
            (self.outputs / name).mkdir(parents=True, exist_ok=True)

