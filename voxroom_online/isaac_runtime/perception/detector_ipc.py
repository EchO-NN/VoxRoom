from __future__ import annotations

from typing import Any, List

from voxroom_online.isaac_runtime.perception.detection_types import Detection2D
from voxroom_online.isaac_runtime.perception.detector_base import DetectorBase


class SubprocessDetector(DetectorBase):
    """Disabled detector IPC placeholder for the geometry-only VoxRoom pipeline."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "VoxRoom-Online does not launch detector subprocesses. "
            "Run with --detector none for the room-segmentation benchmark."
        )

    def set_vocabulary(self, categories: List[str]) -> None:
        self.vocab = list(categories)

    def detect(self, rgb: Any) -> List[Detection2D]:
        return []
