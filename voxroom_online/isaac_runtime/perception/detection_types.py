from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Literal, Optional, Tuple

import numpy as np


MIN_VALID_DETECTION_CONFIDENCE = 0.45


def detection_confidence_is_valid(confidence: float, min_confidence: float = MIN_VALID_DETECTION_CONFIDENCE) -> bool:
    return float(confidence) > float(min_confidence)


def bbox_touches_image_edge(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_px: float = 2,
    margin_ratio: float = 0.0,
) -> bool:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    margin = max(float(margin_px), float(margin_ratio) * float(max(int(image_width), int(image_height))))
    return bool(
        x1 <= margin
        or y1 <= margin
        or x2 >= float(int(image_width) - 1) - margin
        or y2 >= float(int(image_height) - 1) - margin
    )


@dataclass
class Detection2D:
    category: str
    raw_label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    class_id: Optional[int] = None
    mask: Optional[np.ndarray] = None
    bbox_touches_edge: bool = False
    used_for_object_track: bool = True
    reject_reason: Optional[str] = None


@dataclass
class Detection3D:
    category: str
    raw_label: str
    confidence: float
    center_world: Tuple[float, float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    point_cloud_world: Optional[np.ndarray] = None
    bbox_world: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


@dataclass
class RawObjectDetection:
    frame_id: int
    category: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    mask_area_px: int = 0
    bbox_touches_image_edge: bool = False
    mask_touches_image_edge: bool = False
    depth_support_ratio: float = 0.0
    projected_footprint: Optional[np.ndarray] = None
    visibility_status: str = "unknown"
    associated_track_id: Optional[str] = None
    used_for_policy_graph: bool = False
    reject_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "frame_id": int(self.frame_id),
            "category": self.category,
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "mask_area_px": int(self.mask_area_px),
            "bbox_touches_image_edge": bool(self.bbox_touches_image_edge),
            "mask_touches_image_edge": bool(self.mask_touches_image_edge),
            "depth_support_ratio": float(self.depth_support_ratio),
            "visibility_status": self.visibility_status,
            "associated_track_id": self.associated_track_id,
            "used_for_policy_graph": bool(self.used_for_policy_graph),
            "reject_reason": self.reject_reason,
        }


@dataclass
class FusedInstance:
    instance_id: str
    category: str
    node_type: Literal["object", "room"]
    confidence: float
    point_cloud_world: np.ndarray
    bbox_world: np.ndarray
    center_world: np.ndarray
    last_mask: Optional[np.ndarray]
    last_seen_step: int
    observed_count: int
    source: str = "rgbd_projected_detection"
    class_conf_sums: Dict[str, float] = field(default_factory=dict)
    class_hits: Dict[str, int] = field(default_factory=dict)
    valid_detection_count: int = 0
    total_conf_sum: float = 0.0
    edge_rejected_count: int = 0
    visibility_status_counts: Dict[str, int] = field(default_factory=dict)
    geometry_confidence: float = 1.0
    center_world_stable: Optional[np.ndarray] = None
    center_world_visible: Optional[np.ndarray] = None
    full_mask_reference: Optional[np.ndarray] = None
    last_visible_mask: Optional[np.ndarray] = None
    center_estimation_mode: str = "full_mask"
    is_stable: bool = True
    used_for_policy_graph: bool = True
    parent_track_id: Optional[str] = None
    child_track_ids: Tuple[str, ...] = ()

    @property
    def stable_category(self) -> str:
        if not self.class_conf_sums:
            return self.category
        return max(sorted(self.class_conf_sums), key=lambda key: float(self.class_conf_sums[key]))

    @property
    def winner_detection_count(self) -> int:
        return int(self.class_hits.get(self.stable_category, self.observed_count))

    @property
    def mean_confidence(self) -> float:
        hits = max(1, self.winner_detection_count)
        return float(self.class_conf_sums.get(self.stable_category, float(self.confidence) * hits)) / float(hits)

    @property
    def label_entropy(self) -> float:
        total = float(sum(max(0.0, float(v)) for v in self.class_conf_sums.values()))
        if total <= 1e-9:
            return 0.0
        entropy = 0.0
        for value in self.class_conf_sums.values():
            p = max(0.0, float(value)) / total
            if p > 1e-12:
                entropy -= p * math.log(p)
        return float(entropy)
