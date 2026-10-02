"""Small YAML configuration loader with recursive overrides."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from streaming_asr.exceptions import ConfigurationError


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class Config:
    def __init__(self, values: Mapping[str, Any], source: str | None = None):
        self._values = deepcopy(dict(values))
        self.source = source

    @classmethod
    def from_yaml(cls, path: str | Path, override: str | Path | None = None) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}")
        values = _read_yaml(path)
        if override:
            override_path = Path(override)
            if not override_path.is_file():
                raise ConfigurationError(f"Override file not found: {override_path}")
            values = deep_merge(values, _read_yaml(override_path))
        return cls(values, str(path))

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self._values
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, *keys: str) -> None:
        missing = [key for key in keys if self.get(key) is None]
        if missing:
            raise ConfigurationError(f"Missing required configuration fields: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ConfigurationError(f"Top-level YAML value must be a mapping: {path}")
    return values

