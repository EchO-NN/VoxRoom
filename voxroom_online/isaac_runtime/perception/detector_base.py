from __future__ import annotations

from typing import Any, List

from voxroom_online.isaac_runtime.perception.detection_types import Detection2D


class DetectorBase:
    def set_vocabulary(self, categories: List[str]) -> None:
        raise NotImplementedError

    def detect(self, rgb: Any) -> List[Detection2D]:
        raise NotImplementedError


class DryRunDetector(DetectorBase):
    """Detector used for smoke tests; returns no detections."""

    def __init__(self) -> None:
        self.vocab: List[str] = []

    def set_vocabulary(self, categories: List[str]) -> None:
        self.vocab = list(categories)

    def detect(self, rgb: Any) -> List[Detection2D]:
        return []
