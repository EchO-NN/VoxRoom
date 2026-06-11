from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

try:
    import yaml
except Exception:  # pragma: no cover - only used in minimal envs
    yaml = None


class AttrDict(dict):
    """Small dict wrapper with recursive attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _to_attr(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({k: _to_attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr(v) for v in value]
    return value


def deep_update(base: dict, override: Mapping[str, Any]) -> dict:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_nested(data: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    cur: Any = data
    for key in dotted_path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError("expected a boolean value, got %r" % value)


def load_yaml(path: Union[str, os.PathLike]) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load benchmark configuration")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: Optional[Union[str, os.PathLike]] = None, overrides: Optional[Mapping[str, Any]] = None) -> AttrDict:
    cfg_path = Path(path or "voxroom_online/isaac_runtime/configs/voxroom_online.isaac_runtime.yaml")
    data = load_yaml(cfg_path)
    if "voxroom_policy" in data and "sgnav" not in data:
        # Compatibility bridge for the copied runtime, whose internal policy
        # namespace predates the VoxRoom-Online project name.
        data["sgnav"] = data["voxroom_policy"]
    if overrides:
        deep_update(data, overrides)
    return _to_attr(data)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
