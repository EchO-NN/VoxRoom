from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.perception.detection_types import (
    MIN_VALID_DETECTION_CONFIDENCE,
    Detection2D,
    Detection3D,
    FusedInstance,
    bbox_touches_image_edge,
    detection_confidence_is_valid,
)
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.isaac_runtime.sensors.depth_backproject import detection_to_world_points, world_points_bbox


DEFAULT_ROOM_CATEGORIES = {
    "bedroom",
    "bathroom",
    "kitchen",
    "living_room",
    "living room",
    "dining_room",
    "dining room",
    "office",
    "hallway",
    "unknown_room",
    "unknown room",
}


class FusedInstanceRegistry:
    """Cross-frame registry for projected RGB-D projected instances."""

    def __init__(
        self,
        merge_distance_m: float = 0.75,
        merge_iou_3d: float = 0.15,
        max_points_per_instance: int = 4096,
        room_categories: Optional[Iterable[str]] = None,
        min_valid_confidence: float = MIN_VALID_DETECTION_CONFIDENCE,
        reject_edge_touching_bboxes: bool = False,
        bbox_edge_margin_px: float = 0,
        bbox_edge_margin_ratio: float = 0.0,
        partial_class_weight: float = 0.25,
        min_geometry_confidence: float = 0.50,
        partial_stability_min_observations: int = 3,
        mask_iou_association_threshold: float = 0.20,
        mask_containment_track_match_threshold: float = 0.60,
        footprint_iou_association_threshold: float = 0.15,
        child_containment_threshold: float = 0.60,
        child_area_ratio_threshold: float = 0.35,
    ):
        self.merge_distance_m = float(merge_distance_m)
        self.merge_iou_3d = float(merge_iou_3d)
        self.max_points_per_instance = max(1, int(max_points_per_instance))
        self.room_categories = {normalize_category(name) for name in (room_categories or DEFAULT_ROOM_CATEGORIES)}
        self.min_valid_confidence = float(min_valid_confidence)
        self.reject_edge_touching_bboxes = bool(reject_edge_touching_bboxes)
        self.bbox_edge_margin_px = float(bbox_edge_margin_px)
        self.bbox_edge_margin_ratio = float(bbox_edge_margin_ratio)
        self.partial_class_weight = float(partial_class_weight)
        self.min_geometry_confidence = float(min_geometry_confidence)
        self.partial_stability_min_observations = int(partial_stability_min_observations)
        self.mask_iou_association_threshold = float(mask_iou_association_threshold)
        self.mask_containment_track_match_threshold = float(mask_containment_track_match_threshold)
        self.footprint_iou_association_threshold = float(footprint_iou_association_threshold)
        self.child_containment_threshold = float(child_containment_threshold)
        self.child_area_ratio_threshold = float(child_area_ratio_threshold)
        self.instances: List[FusedInstance] = []
        self._next_id = 0
        self.raw_detection_log: List[dict] = []
        self.raw_rejected_detections_count = 0

    def reset(self) -> None:
        self.instances = []
        self._next_id = 0
        self.raw_detection_log = []
        self.raw_rejected_detections_count = 0

    def update(
        self,
        detections: Iterable[Detection2D],
        depth: np.ndarray,
        intr: CameraIntrinsics,
        camera_pose_world: Tuple[float, float, float, float],
        step_id: int,
        depth_max_m: float = 5.0,
        min_points: int = 20,
        stride: int = 4,
    ) -> List[FusedInstance]:
        for det in detections:
            raw_record = self._raw_detection_record(det, step_id, intr.width, intr.height)
            self.raw_detection_log.append(raw_record)
            if not detection_confidence_is_valid(float(det.confidence), self.min_valid_confidence):
                raw_record["used_for_object_track"] = False
                raw_record["reject_reason"] = "low_confidence"
                continue
            points = detection_to_world_points(
                det,
                depth,
                intr,
                camera_pose_world,
                depth_max_m=depth_max_m,
                min_points=min_points,
                stride=stride,
            )
            if points is None or len(points) == 0:
                raw_record["used_for_object_track"] = False
                raw_record["reject_reason"] = "insufficient_depth_points"
                raw_record["visibility_status"] = "partial_occluded"
                self.raw_rejected_detections_count += 1
                continue
            depth_support_ratio = _depth_support_ratio(det.mask, points)
            raw_record["depth_support_ratio"] = float(depth_support_ratio)
            raw_record["projected_footprint"] = _footprint_from_bbox_world(world_points_bbox(points))
            category = normalize_category(det.category)
            node_type = self._node_type_for_category(category)
            bbox_world = world_points_bbox(points)
            center_world = np.median(np.asarray(points, dtype=np.float32), axis=0).astype(np.float32)
            visibility_status = self._visibility_status(det, raw_record, depth_support_ratio)
            raw_record["visibility_status"] = visibility_status
            parent = self._contained_parent(det, category, node_type)
            match = None if parent is not None else self._find_match(category, node_type, center_world, bbox_world, det)
            if match is None:
                class_conf_sums = {category: float(det.confidence)}
                class_hits = {category: 1}
                is_full = visibility_status == "full"
                instance_id = "%s_%d" % (node_type, self._next_id)
                instance = FusedInstance(
                    instance_id=instance_id,
                    category=category,
                    node_type=node_type,
                    confidence=float(det.confidence),
                    point_cloud_world=self._downsample_points(points),
                    bbox_world=bbox_world,
                    center_world=center_world,
                    last_mask=self._copy_mask(det.mask),
                    last_seen_step=int(step_id),
                    observed_count=1,
                    class_conf_sums=class_conf_sums,
                    class_hits=class_hits,
                    valid_detection_count=1,
                    total_conf_sum=float(det.confidence),
                    visibility_status_counts={visibility_status: 1},
                    geometry_confidence=1.0 if is_full else 0.25,
                    center_world_stable=center_world.copy() if is_full else None,
                    center_world_visible=center_world.copy(),
                    full_mask_reference=self._copy_mask(det.mask) if is_full else None,
                    last_visible_mask=self._copy_mask(det.mask),
                    center_estimation_mode="full_mask" if is_full else "visible_extent_low_conf",
                    is_stable=bool(is_full),
                    used_for_policy_graph=bool(is_full),
                    parent_track_id=getattr(parent, "instance_id", None) if parent is not None else None,
                )
                self.instances.append(instance)
                if parent is not None:
                    parent.child_track_ids = tuple(sorted(set(parent.child_track_ids + (instance_id,))))
                self._next_id += 1
                raw_record["associated_track_id"] = instance_id
                raw_record["used_for_object_track"] = True
                raw_record["used_for_policy_graph"] = bool(instance.used_for_policy_graph)
                continue
            self._merge(match, points, bbox_world, center_world, det, step_id, visibility_status=visibility_status)
            raw_record["associated_track_id"] = match.instance_id
            raw_record["used_for_object_track"] = True
            raw_record["used_for_policy_graph"] = bool(match.used_for_policy_graph)
        return list(self.instances)

    def to_detections_3d(self, node_type: str = "object") -> List[Detection3D]:
        out: List[Detection3D] = []
        for instance in self.instances:
            if instance.node_type != node_type or not bool(getattr(instance, "used_for_policy_graph", True)):
                continue
            out.append(
                Detection3D(
                    category=instance.category,
                    raw_label=instance.category,
                    confidence=float(instance.mean_confidence),
                    center_world=tuple(float(v) for v in instance.center_world),
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    point_cloud_world=instance.point_cloud_world,
                    bbox_world=instance.bbox_world,
                    mask=instance.last_mask,
                )
            )
        return out

    def _find_match(
        self,
        category: str,
        node_type: str,
        center_world: Sequence[float],
        bbox_world: np.ndarray,
        det: Optional[Detection2D] = None,
    ) -> Optional[FusedInstance]:
        best: Optional[FusedInstance] = None
        best_dist = float("inf")
        for instance in self.instances:
            _ = category
            if instance.node_type != node_type:
                continue
            dist = float(np.linalg.norm(np.asarray(instance.center_world[:2], dtype=np.float32) - np.asarray(center_world[:2], dtype=np.float32)))
            reference_mask = instance.full_mask_reference if instance.full_mask_reference is not None else instance.last_mask
            mask_iou = _mask_iou(det.mask, reference_mask) if det is not None else 0.0
            mask_containment = _mask_containment(det.mask, reference_mask) if det is not None else 0.0
            det_is_partial = bool(det is not None and (det.bbox_touches_edge or mask_touches_image_edge(det.mask)))
            if (
                det_is_partial
                and bool(instance.is_stable)
                and mask_containment >= self.mask_containment_track_match_threshold
            ):
                return instance
            if mask_iou >= self.mask_iou_association_threshold and (dist <= self.merge_distance_m * 2.0 or det_is_partial):
                return instance
            if dist > self.merge_distance_m or dist >= best_dist:
                continue
            iou = bbox_iou_3d(instance.bbox_world, bbox_world)
            if iou < max(self.merge_iou_3d, self.footprint_iou_association_threshold):
                continue
            best = instance
            best_dist = dist
        return best

    def _merge(
        self,
        instance: FusedInstance,
        points: np.ndarray,
        bbox_world: np.ndarray,
        center_world: np.ndarray,
        det: Detection2D,
        step_id: int,
        visibility_status: str = "full",
    ) -> None:
        old_count = max(1, int(instance.observed_count))
        merged_points = np.concatenate([np.asarray(instance.point_cloud_world, dtype=np.float32), np.asarray(points, dtype=np.float32)], axis=0)
        is_full = visibility_status == "full"
        if is_full:
            instance.point_cloud_world = self._downsample_points(merged_points)
            instance.bbox_world = world_points_bbox(instance.point_cloud_world)
            weighted_center = (np.asarray(instance.center_world, dtype=np.float32) * float(old_count) + np.asarray(center_world, dtype=np.float32)) / float(old_count + 1)
            bbox_center = np.mean(np.asarray(bbox_world, dtype=np.float32), axis=0)
            instance.center_world = ((weighted_center + bbox_center) * 0.5).astype(np.float32)
            instance.center_world_stable = instance.center_world.copy()
            instance.full_mask_reference = self._copy_mask(det.mask)
            instance.geometry_confidence = min(1.0, max(float(instance.geometry_confidence), 0.85))
            instance.center_estimation_mode = "full_mask"
            instance.is_stable = True
        else:
            instance.center_world_visible = np.asarray(center_world, dtype=np.float32).copy()
            if instance.center_world_stable is not None:
                instance.center_world = np.asarray(instance.center_world_stable, dtype=np.float32).copy()
                instance.center_estimation_mode = "inherited_full_mask"
            else:
                instance.center_world = np.asarray(center_world, dtype=np.float32).copy()
                instance.center_estimation_mode = "visible_extent_low_conf"
            if int(instance.observed_count) + 1 >= self.partial_stability_min_observations:
                instance.is_stable = True
                instance.geometry_confidence = max(float(instance.geometry_confidence), self.min_geometry_confidence)
        instance.confidence = max(float(instance.confidence), float(det.confidence))
        instance.last_mask = self._copy_mask(det.mask)
        instance.last_visible_mask = self._copy_mask(det.mask)
        instance.last_seen_step = int(step_id)
        instance.observed_count = old_count + 1
        instance.visibility_status_counts[visibility_status] = int(instance.visibility_status_counts.get(visibility_status, 0)) + 1
        category = normalize_category(det.category)
        weight = 1.0 if is_full else float(self.partial_class_weight)
        instance.class_conf_sums[category] = float(instance.class_conf_sums.get(category, 0.0)) + float(det.confidence) * weight
        instance.class_hits[category] = int(instance.class_hits.get(category, 0)) + 1
        instance.valid_detection_count = int(instance.valid_detection_count) + 1
        instance.total_conf_sum = float(instance.total_conf_sum) + float(det.confidence) * weight
        instance.category = instance.stable_category
        has_full_or_prior_full_geometry = instance.full_mask_reference is not None and instance.center_world_stable is not None
        instance.used_for_policy_graph = bool(
            instance.is_stable
            and has_full_or_prior_full_geometry
            and float(instance.geometry_confidence) >= float(self.min_geometry_confidence)
        )

    def _node_type_for_category(self, category: str) -> str:
        return "room" if normalize_category(category) in self.room_categories else "object"

    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float32)
        if len(arr) <= self.max_points_per_instance:
            return arr.copy()
        indices = np.linspace(0, len(arr) - 1, self.max_points_per_instance).astype(np.int64)
        return arr[indices].copy()

    @staticmethod
    def _copy_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return np.asarray(mask).astype(bool).copy()

    def _raw_detection_record(self, det: Detection2D, step_id: int, width: int, height: int) -> dict:
        touches = bbox_touches_image_edge(
            det.bbox_xyxy,
            width,
            height,
            margin_px=self.bbox_edge_margin_px,
            margin_ratio=self.bbox_edge_margin_ratio,
        )
        mask_area = int(np.count_nonzero(det.mask)) if det.mask is not None else 0
        mask_edge = mask_touches_image_edge(det.mask) if det.mask is not None else False
        det.bbox_touches_edge = bool(touches)
        det.used_for_object_track = True
        det.reject_reason = None
        return {
            "raw_detection_id": "frame_%04d_det_%04d" % (int(step_id), len(self.raw_detection_log)),
            "frame_id": int(step_id),
            "step": int(step_id),
            "category": normalize_category(det.category),
            "raw_category": normalize_category(det.category),
            "raw_label": str(det.raw_label),
            "confidence": float(det.confidence),
            "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
            "bbox_touches_edge": bool(touches),
            "bbox_touches_image_edge": bool(touches),
            "mask_touches_image_edge": bool(mask_edge),
            "mask_area_px": int(mask_area),
            "depth_support_ratio": 0.0,
            "projected_footprint": None,
            "visibility_status": "partial_edge" if touches or mask_edge else "unknown",
            "associated_track_id": None,
            "used_for_object_track": True,
            "used_for_policy_graph": False,
            "reject_reason": None,
        }

    def _visibility_status(self, det: Detection2D, raw_record: dict, depth_support_ratio: float) -> str:
        if bool(raw_record.get("bbox_touches_edge")) or bool(raw_record.get("mask_touches_image_edge")):
            return "partial_edge"
        if float(depth_support_ratio) < 0.10 or int(raw_record.get("mask_area_px", 0)) < 8:
            return "partial_occluded"
        return "full"

    def _contained_parent(self, det: Detection2D, category: str, node_type: str) -> Optional[FusedInstance]:
        if det.mask is None or node_type != "object":
            return None
        mask = np.asarray(det.mask, dtype=bool)
        area = max(1, int(np.count_nonzero(mask)))
        for instance in self.instances:
            if instance.node_type != "object" or normalize_category(instance.category) == category:
                continue
            parent_mask = instance.full_mask_reference if instance.full_mask_reference is not None else instance.last_mask
            if parent_mask is None or np.asarray(parent_mask).shape != mask.shape:
                continue
            parent_arr = np.asarray(parent_mask, dtype=bool)
            parent_area = max(1, int(np.count_nonzero(parent_arr)))
            containment = float(np.count_nonzero(mask & parent_arr)) / float(area)
            area_ratio = float(area) / float(parent_area)
            if containment >= self.child_containment_threshold and area_ratio < self.child_area_ratio_threshold:
                return instance
        return None

    def gnn_snapshot(self) -> dict:
        return {
            "raw_detection_count": int(len(self.raw_detection_log)),
            "stable_track_count": int(len([item for item in self.instances if bool(item.used_for_policy_graph)])),
            "tentative_track_count": int(len([item for item in self.instances if not bool(item.used_for_policy_graph)])),
            "mask_association_count": int(len([item for item in self.raw_detection_log if item.get("associated_track_id")])),
            "partial_edge_count": int(len([item for item in self.raw_detection_log if item.get("visibility_status") == "partial_edge"])),
            "contained_child_count": int(len([item for item in self.instances if item.parent_track_id])),
            "raw_detections": list(self.raw_detection_log),
            "object_tracks": [fused_instance_to_track_dict(item) for item in self.instances],
        }


def bbox_iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != (2, 3) or bb.shape != (2, 3):
        return 0.0
    aa = _inflate_degenerate_bbox(aa)
    bb = _inflate_degenerate_bbox(bb)
    inter_min = np.maximum(aa[0], bb[0])
    inter_max = np.minimum(aa[1], bb[1])
    inter_extent = np.maximum(inter_max - inter_min, 0.0)
    extent_a = np.maximum(aa[1] - aa[0], 0.0)
    extent_b = np.maximum(bb[1] - bb[0], 0.0)
    if float(np.prod(extent_a)) <= 1e-9 or float(np.prod(extent_b)) <= 1e-9:
        inter_area = float(np.prod(inter_extent[:2]))
        area_a = float(np.prod(extent_a[:2]))
        area_b = float(np.prod(extent_b[:2]))
        denom_area = area_a + area_b - inter_area
        return 0.0 if denom_area <= 1e-9 else float(inter_area / denom_area)
    inter_vol = float(np.prod(inter_extent))
    vol_a = float(np.prod(extent_a))
    vol_b = float(np.prod(extent_b))
    denom = vol_a + vol_b - inter_vol
    if denom <= 1e-9:
        return 0.0
    return float(inter_vol / denom)


def _inflate_degenerate_bbox(bbox: np.ndarray, min_extent: float = 0.20) -> np.ndarray:
    out = np.asarray(bbox, dtype=np.float32).copy()
    center = np.mean(out, axis=0)
    extent = out[1] - out[0]
    half = np.maximum(extent, float(min_extent)) * 0.5
    out[0] = center - half
    out[1] = center + half
    return out


def mask_touches_image_edge(mask: Optional[np.ndarray], margin_px: int = 0) -> bool:
    if mask is None:
        return False
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not np.any(arr):
        return False
    margin = max(0, int(margin_px))
    if margin <= 0:
        return bool(np.any(arr[0, :]) or np.any(arr[-1, :]) or np.any(arr[:, 0]) or np.any(arr[:, -1]))
    return bool(
        np.any(arr[: margin + 1, :])
        or np.any(arr[-(margin + 1) :, :])
        or np.any(arr[:, : margin + 1])
        or np.any(arr[:, -(margin + 1) :])
    )


def _mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape or aa.ndim != 2:
        return 0.0
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return 0.0 if union <= 0 else float(inter) / float(union)


def _depth_support_ratio(mask: Optional[np.ndarray], points: Optional[np.ndarray]) -> float:
    point_count = 0 if points is None else int(len(points))
    if mask is None:
        return 1.0 if point_count > 0 else 0.0
    mask_area = max(1, int(np.count_nonzero(np.asarray(mask, dtype=bool))))
    # Points are sampled with depth stride, so comparing them directly with the
    # full mask area would mark every normal detection as occluded. Use a small
    # support budget that still catches near-empty/occluded masks.
    required = max(6, min(30, int(round(math.sqrt(float(mask_area)) * 0.5))))
    return float(min(1.0, float(point_count) / float(required)))


def _footprint_from_bbox_world(bbox_world: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if bbox_world is None:
        return None
    arr = np.asarray(bbox_world, dtype=np.float32)
    if arr.shape != (2, 3):
        return None
    return [
        [float(arr[0, 0]), float(arr[0, 1])],
        [float(arr[1, 0]), float(arr[0, 1])],
        [float(arr[1, 0]), float(arr[1, 1])],
        [float(arr[0, 0]), float(arr[1, 1])],
    ]


def fused_instance_to_track_dict(instance: FusedInstance) -> dict:
    return {
        "track_id": str(instance.instance_id),
        "category": normalize_category(instance.category),
        "stable_category": normalize_category(instance.stable_category),
        "confidence": float(instance.confidence),
        "mean_confidence": float(instance.mean_confidence),
        "valid_detection_count": int(instance.valid_detection_count),
        "winner_detection_count": int(instance.winner_detection_count),
        "observed_count": int(instance.observed_count),
        "class_conf_sums": {str(k): float(v) for k, v in sorted(instance.class_conf_sums.items())},
        "class_hits": {str(k): int(v) for k, v in sorted(instance.class_hits.items())},
        "label_entropy": float(instance.label_entropy),
        "visibility_status_counts": {str(k): int(v) for k, v in sorted(instance.visibility_status_counts.items())},
        "geometry_confidence": float(instance.geometry_confidence),
        "center_estimation_mode": str(instance.center_estimation_mode),
        "center_world": [float(v) for v in np.asarray(instance.center_world, dtype=np.float32).tolist()],
        "center_world_stable": None
        if instance.center_world_stable is None
        else [float(v) for v in np.asarray(instance.center_world_stable, dtype=np.float32).tolist()],
        "center_world_visible": None
        if instance.center_world_visible is None
        else [float(v) for v in np.asarray(instance.center_world_visible, dtype=np.float32).tolist()],
        "is_stable": bool(instance.is_stable),
        "used_for_policy_graph": bool(instance.used_for_policy_graph),
        "used_for_room_label": bool(instance.used_for_policy_graph),
        "used_for_goal_candidate": bool(instance.used_for_policy_graph),
        "used_for_stop": bool(instance.used_for_policy_graph),
        "parent_track_id": instance.parent_track_id,
        "child_track_ids": list(instance.child_track_ids),
    }


def _mask_containment(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape or aa.ndim != 2:
        return 0.0
    denom = int(np.count_nonzero(aa))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero(aa & bb)) / float(denom)
