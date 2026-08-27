from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass
class BaselineResult:
    label_map: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_arrays: dict[str, np.ndarray] = field(default_factory=dict)


class OfflineBaselineRunner(Protocol):
    baseline_name: str

    def start_scene(self, scene_id: str) -> None:
        ...

    def segment_snapshot(self, snapshot_path: Path, arrays: Mapping[str, Any]) -> BaselineResult:
        ...

    def end_scene(self) -> None:
        ...


class MissingOriginalImplementationError(RuntimeError):
    pass

