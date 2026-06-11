from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, world_xy_to_grid
from voxroom_online.isaac_runtime.perception.detection_types import (
    MIN_VALID_DETECTION_CONFIDENCE,
    Detection3D,
    FusedInstance,
    detection_confidence_is_valid,
)


@dataclass
class ObjectNode:
    node_id: int
    category: str
    center_world: Tuple[float, float, float]
    center_grid: Tuple[int, int]
    confidence: float
    observed_count: int
    last_seen_step: int
    raw_label: str = ""
    point_cloud_world: Optional[np.ndarray] = None
    bbox_world: Optional[np.ndarray] = None
    last_mask: Optional[np.ndarray] = None
    source_instance_id: str = ""
    source: str = ""
    class_conf_sums: Dict[str, float] = field(default_factory=dict)
    class_hits: Dict[str, int] = field(default_factory=dict)
    valid_detection_count: int = 0
    total_conf_sum: float = 0.0
    edge_rejected_count: int = 0
    raw_rejected_detections_count: int = 0
    visibility_status_counts: Dict[str, int] = field(default_factory=dict)
    geometry_confidence: float = 1.0
    center_estimation_mode: str = "full_mask"
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
        hits = max(1, int(self.winner_detection_count))
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

    def to_dict(self) -> dict:
        return {
            "node_id": int(self.node_id),
            "track_id": "obj_%04d" % int(self.node_id),
            "category": self.category,
            "stable_category": self.stable_category,
            "center_world": tuple(float(v) for v in self.center_world),
            "center_grid": tuple(int(v) for v in self.center_grid),
            "confidence": float(self.confidence),
            "mean_confidence": float(self.mean_confidence),
            "detection_count": int(self.valid_detection_count or self.observed_count),
            "valid_detection_count": int(self.valid_detection_count or self.observed_count),
            "winner_detection_count": int(self.winner_detection_count),
            "observed_count": int(self.observed_count),
            "last_seen_step": int(self.last_seen_step),
            "raw_label": self.raw_label,
            "point_cloud_world": _array_to_list(self.point_cloud_world),
            "bbox_world": _array_to_list(self.bbox_world),
            "source_instance_id": self.source_instance_id,
            "source": self.source,
            "class_conf_sums": {str(k): float(v) for k, v in sorted(self.class_conf_sums.items())},
            "class_hits": {str(k): int(v) for k, v in sorted(self.class_hits.items())},
            "label_entropy": float(self.label_entropy),
            "visibility_status_counts": {str(k): int(v) for k, v in sorted(self.visibility_status_counts.items())},
            "geometry_confidence": float(self.geometry_confidence),
            "center_estimation_mode": str(self.center_estimation_mode),
            "used_for_policy_graph": bool(self.used_for_policy_graph),
            "parent_track_id": self.parent_track_id,
            "child_track_ids": list(self.child_track_ids),
            "edge_rejected_count": int(self.edge_rejected_count),
            "raw_rejected_detections_count": int(self.raw_rejected_detections_count),
        }


class ObjectMemory:
    def __init__(self, merge_radius_m: float = 0.5, min_valid_confidence: float = MIN_VALID_DETECTION_CONFIDENCE):
        self.merge_radius_m = float(merge_radius_m)
        self.min_valid_confidence = float(min_valid_confidence)
        self.nodes: List[ObjectNode] = []
        self._next_id = 0

    def reset(self) -> None:
        self.nodes = []
        self._next_id = 0

    @staticmethod
    def _category_key(category: str) -> str:
        return normalize_category(category)

    @staticmethod
    def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
        aa = np.asarray(a[:2], dtype=np.float32)
        bb = np.asarray(b[:2], dtype=np.float32)
        return float(np.linalg.norm(aa - bb))

    def find_match(self, category: str, center_world: Sequence[float]) -> Optional[ObjectNode]:
        _ = category
        best = None
        best_dist = float("inf")
        for node in self.nodes:
            dist = self._xy_distance(center_world, node.center_world)
            if dist < self.merge_radius_m and dist < best_dist:
                best = node
                best_dist = dist
        return best

    def update(self, detections: List[Detection3D], step_id: int, map_info: Optional[MapInfo] = None) -> List[ObjectNode]:
        changed: List[ObjectNode] = []
        for det in detections:
            if not detection_confidence_is_valid(float(det.confidence), self.min_valid_confidence):
                continue
            center = tuple(float(v) for v in det.center_world)
            category = self._category_key(det.category)
            matched = self.find_match(category, center)
            if map_info is not None:
                center_grid = world_xy_to_grid(center[0], center[1], map_info)
            else:
                center_grid = (0, 0)
            if matched is None:
                class_conf_sums = {category: float(det.confidence)}
                class_hits = {category: 1}
                node = ObjectNode(
                    node_id=self._next_id,
                    category=category,
                    center_world=center,
                    center_grid=center_grid,
                    confidence=float(det.confidence),
                    observed_count=1,
                    last_seen_step=int(step_id),
                    raw_label=det.raw_label,
                    point_cloud_world=_copy_array(det.point_cloud_world),
                    bbox_world=_copy_array(det.bbox_world),
                    last_mask=_copy_array(det.mask),
                    source=getattr(det, "source", ""),
                    class_conf_sums=class_conf_sums,
                    class_hits=class_hits,
                    valid_detection_count=1,
                    total_conf_sum=float(det.confidence),
                )
                self._next_id += 1
                self.nodes.append(node)
                changed.append(node)
                continue
            old_count = matched.observed_count
            old_center = np.asarray(matched.center_world, dtype=np.float32)
            new_center = np.asarray(center, dtype=np.float32)
            merged = (old_center * old_count + new_center) / float(old_count + 1)
            matched.center_world = tuple(float(v) for v in merged)
            if map_info is not None:
                matched.center_grid = world_xy_to_grid(float(merged[0]), float(merged[1]), map_info)
            else:
                matched.center_grid = center_grid
            matched.confidence = max(float(matched.confidence), float(det.confidence))
            _accumulate_category_observation(matched, category, float(det.confidence))
            matched.category = matched.stable_category
            matched.point_cloud_world = _merge_point_clouds(matched.point_cloud_world, det.point_cloud_world)
            matched.bbox_world = _copy_array(det.bbox_world) if det.bbox_world is not None else _bbox_from_points(matched.point_cloud_world)
            matched.last_mask = _copy_array(det.mask)
            matched.observed_count += 1
            matched.last_seen_step = int(step_id)
            changed.append(matched)
        self.dedupe(map_info=map_info)
        return changed

    def update_fused_instances(
        self,
        instances: Sequence[FusedInstance],
        step_id: int,
        map_info: Optional[MapInfo] = None,
    ) -> List[ObjectNode]:
        changed: List[ObjectNode] = []
        for instance in instances:
            if instance.node_type != "object":
                continue
            if not bool(getattr(instance, "used_for_policy_graph", True)):
                continue
            if not detection_confidence_is_valid(float(instance.confidence), self.min_valid_confidence):
                continue
            category = self._category_key(instance.category)
            center = tuple(float(v) for v in instance.center_world)
            matched = self._find_match_for_fused_instance(instance, center)
            center_grid = world_xy_to_grid(center[0], center[1], map_info) if map_info is not None else (0, 0)
            class_conf_sums, class_hits, valid_detection_count, total_conf_sum = _category_accumulators_from_instance(instance)
            if matched is None:
                node = ObjectNode(
                    node_id=self._next_id,
                    category=category,
                    center_world=center,
                    center_grid=center_grid,
                    confidence=float(instance.confidence),
                    observed_count=int(instance.observed_count),
                    last_seen_step=int(step_id),
                    raw_label=instance.category,
                    point_cloud_world=_copy_array(instance.point_cloud_world),
                    bbox_world=_copy_array(instance.bbox_world),
                    last_mask=_copy_array(instance.last_mask),
                    source_instance_id=instance.instance_id,
                    source=instance.source,
                    class_conf_sums=class_conf_sums,
                    class_hits=class_hits,
                    valid_detection_count=int(valid_detection_count),
                    total_conf_sum=float(total_conf_sum),
                    edge_rejected_count=int(getattr(instance, "edge_rejected_count", 0)),
                    raw_rejected_detections_count=int(getattr(instance, "edge_rejected_count", 0)),
                    visibility_status_counts={
                        str(k): int(v)
                        for k, v in dict(getattr(instance, "visibility_status_counts", {}) or {}).items()
                    },
                    geometry_confidence=float(getattr(instance, "geometry_confidence", 1.0)),
                    center_estimation_mode=str(getattr(instance, "center_estimation_mode", "full_mask")),
                    used_for_policy_graph=bool(getattr(instance, "used_for_policy_graph", True)),
                    parent_track_id=getattr(instance, "parent_track_id", None),
                    child_track_ids=tuple(str(v) for v in getattr(instance, "child_track_ids", ()) or ()),
                )
                node.category = node.stable_category
                self._next_id += 1
                self.nodes.append(node)
                changed.append(node)
                continue
            matched.center_world = center
            matched.center_grid = center_grid
            matched.confidence = max(float(matched.confidence), float(instance.confidence))
            matched.observed_count = max(int(matched.observed_count), int(instance.observed_count))
            matched.last_seen_step = int(step_id)
            matched.raw_label = instance.category
            matched.point_cloud_world = _copy_array(instance.point_cloud_world)
            matched.bbox_world = _copy_array(instance.bbox_world)
            matched.last_mask = _copy_array(instance.last_mask)
            matched.source_instance_id = instance.instance_id
            matched.source = instance.source
            matched.class_conf_sums = class_conf_sums
            matched.class_hits = class_hits
            matched.valid_detection_count = int(valid_detection_count)
            matched.total_conf_sum = float(total_conf_sum)
            matched.edge_rejected_count = int(getattr(instance, "edge_rejected_count", matched.edge_rejected_count))
            matched.raw_rejected_detections_count = int(getattr(instance, "edge_rejected_count", matched.raw_rejected_detections_count))
            matched.visibility_status_counts = {
                str(k): int(v)
                for k, v in dict(getattr(instance, "visibility_status_counts", {}) or {}).items()
            }
            matched.geometry_confidence = float(getattr(instance, "geometry_confidence", matched.geometry_confidence))
            matched.center_estimation_mode = str(getattr(instance, "center_estimation_mode", matched.center_estimation_mode))
            matched.used_for_policy_graph = bool(getattr(instance, "used_for_policy_graph", True))
            matched.parent_track_id = getattr(instance, "parent_track_id", matched.parent_track_id)
            matched.child_track_ids = tuple(str(v) for v in getattr(instance, "child_track_ids", ()) or ())
            matched.category = matched.stable_category
            changed.append(matched)
        self.dedupe(map_info=map_info)
        return changed

    def _find_match_for_fused_instance(self, instance: FusedInstance, center_world: Sequence[float]) -> Optional[ObjectNode]:
        instance_id = str(getattr(instance, "instance_id", "") or "")
        if instance_id:
            for node in self.nodes:
                if str(node.source_instance_id or "") == instance_id:
                    return node
        return self.find_match(getattr(instance, "category", ""), center_world)

    def dedupe(self, map_info: Optional[MapInfo] = None) -> None:
        if len(self.nodes) < 2:
            return
        merged: List[ObjectNode] = []
        for node in sorted(self.nodes, key=lambda item: int(item.node_id)):
            node.category = self._category_key(node.category)
            match = None
            best_dist = float("inf")
            for existing in merged:
                dist = self._xy_distance(existing.center_world, node.center_world)
                if dist < self.merge_radius_m and dist < best_dist:
                    match = existing
                    best_dist = dist
            if match is None:
                _ensure_category_accumulator(node)
                merged.append(node)
                continue
            self._merge_node_into(match, node, map_info=map_info)
        self.nodes = merged

    def dedupe_goal_candidates(
        self,
        goal_category: str,
        merge_radius_m: float,
        map_info: Optional[MapInfo] = None,
    ) -> None:
        goal_key = self._category_key(goal_category)
        old_radius = float(self.merge_radius_m)
        self.merge_radius_m = max(old_radius, float(merge_radius_m))
        goal_nodes = [node for node in self.nodes if self._category_key(node.category) == goal_key]
        other_nodes = [node for node in self.nodes if self._category_key(node.category) != goal_key]
        if len(goal_nodes) < 2:
            self.merge_radius_m = old_radius
            return
        merged_goals: List[ObjectNode] = []
        for node in sorted(goal_nodes, key=lambda item: int(item.node_id)):
            match = None
            best_dist = float("inf")
            for existing in merged_goals:
                dist = self._xy_distance(existing.center_world, node.center_world)
                if dist < self.merge_radius_m and dist < best_dist:
                    match = existing
                    best_dist = dist
            if match is None:
                merged_goals.append(node)
            else:
                self._merge_node_into(match, node, map_info=map_info)
        self.nodes = sorted(other_nodes + merged_goals, key=lambda item: int(item.node_id))
        self.merge_radius_m = old_radius

    def _merge_node_into(self, target: ObjectNode, duplicate: ObjectNode, map_info: Optional[MapInfo] = None) -> None:
        target_count = max(1, int(target.observed_count))
        dup_count = max(1, int(duplicate.observed_count))
        total = float(target_count + dup_count)
        target_center = np.asarray(target.center_world, dtype=np.float32)
        duplicate_center = np.asarray(duplicate.center_world, dtype=np.float32)
        merged_center = (target_center * float(target_count) + duplicate_center * float(dup_count)) / total
        target.center_world = tuple(float(v) for v in merged_center)
        if map_info is not None:
            target.center_grid = world_xy_to_grid(float(merged_center[0]), float(merged_center[1]), map_info)
        target.confidence = max(float(target.confidence), float(duplicate.confidence))
        _merge_category_accumulators(target, duplicate)
        target.category = target.stable_category
        target.point_cloud_world = _merge_point_clouds(target.point_cloud_world, duplicate.point_cloud_world)
        target.bbox_world = _bbox_from_points(target.point_cloud_world) if target.point_cloud_world is not None else target.bbox_world
        if duplicate.last_mask is not None:
            target.last_mask = _copy_array(duplicate.last_mask)
        target.observed_count = int(target_count + dup_count)
        target.last_seen_step = max(int(target.last_seen_step), int(duplicate.last_seen_step))
        if not target.raw_label and duplicate.raw_label:
            target.raw_label = duplicate.raw_label
        if not target.source_instance_id and duplicate.source_instance_id:
            target.source_instance_id = duplicate.source_instance_id
        if not target.source and duplicate.source:
            target.source = duplicate.source

    def to_dicts(self) -> List[dict]:
        return [node.to_dict() for node in self.nodes]


def _copy_array(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value).copy()


def _ensure_category_accumulator(node: ObjectNode) -> None:
    if not node.class_conf_sums:
        category = normalize_category(node.category)
        count = max(1, int(node.observed_count))
        node.class_conf_sums = {category: float(node.confidence) * float(count)}
        node.class_hits = {category: count}
        node.valid_detection_count = count
        node.total_conf_sum = float(node.confidence) * float(count)
        node.category = category


def _accumulate_category_observation(node: ObjectNode, category: str, confidence: float) -> None:
    _ensure_category_accumulator(node)
    label = normalize_category(category)
    node.class_conf_sums[label] = float(node.class_conf_sums.get(label, 0.0)) + float(confidence)
    node.class_hits[label] = int(node.class_hits.get(label, 0)) + 1
    node.valid_detection_count = int(node.valid_detection_count) + 1
    node.total_conf_sum = float(node.total_conf_sum) + float(confidence)


def _merge_category_accumulators(target: ObjectNode, duplicate: ObjectNode) -> None:
    _ensure_category_accumulator(target)
    _ensure_category_accumulator(duplicate)
    for category, value in duplicate.class_conf_sums.items():
        key = normalize_category(category)
        target.class_conf_sums[key] = float(target.class_conf_sums.get(key, 0.0)) + float(value)
    for category, value in duplicate.class_hits.items():
        key = normalize_category(category)
        target.class_hits[key] = int(target.class_hits.get(key, 0)) + int(value)
    target.valid_detection_count = int(target.valid_detection_count) + int(duplicate.valid_detection_count)
    target.total_conf_sum = float(target.total_conf_sum) + float(duplicate.total_conf_sum)
    target.edge_rejected_count = int(target.edge_rejected_count) + int(duplicate.edge_rejected_count)
    target.raw_rejected_detections_count = int(target.raw_rejected_detections_count) + int(duplicate.raw_rejected_detections_count)
    for status, count in duplicate.visibility_status_counts.items():
        key = str(status)
        target.visibility_status_counts[key] = int(target.visibility_status_counts.get(key, 0)) + int(count)
    target.geometry_confidence = max(float(target.geometry_confidence), float(duplicate.geometry_confidence))
    if target.center_estimation_mode != "full_mask" and duplicate.center_estimation_mode == "full_mask":
        target.center_estimation_mode = duplicate.center_estimation_mode
    target.used_for_policy_graph = bool(target.used_for_policy_graph and duplicate.used_for_policy_graph)
    if not target.parent_track_id and duplicate.parent_track_id:
        target.parent_track_id = duplicate.parent_track_id
    target.child_track_ids = tuple(sorted(set(tuple(target.child_track_ids) + tuple(duplicate.child_track_ids))))


def _category_accumulators_from_instance(instance: FusedInstance) -> Tuple[Dict[str, float], Dict[str, int], int, float]:
    sums = {
        normalize_category(category): float(value)
        for category, value in dict(getattr(instance, "class_conf_sums", {}) or {}).items()
    }
    hits = {
        normalize_category(category): int(value)
        for category, value in dict(getattr(instance, "class_hits", {}) or {}).items()
    }
    category = normalize_category(getattr(instance, "category", "object"))
    if not sums:
        count = max(1, int(getattr(instance, "observed_count", 1)))
        sums = {category: float(getattr(instance, "confidence", 0.0)) * float(count)}
        hits = {category: count}
    valid_detection_count = int(getattr(instance, "valid_detection_count", 0) or sum(max(0, int(v)) for v in hits.values()))
    total_conf_sum = float(getattr(instance, "total_conf_sum", 0.0) or sum(float(v) for v in sums.values()))
    return sums, hits, valid_detection_count, total_conf_sum


def _array_to_list(value: Optional[np.ndarray]):
    if value is None:
        return None
    return np.asarray(value).tolist()


def _merge_point_clouds(a: Optional[np.ndarray], b: Optional[np.ndarray], max_points: int = 4096) -> Optional[np.ndarray]:
    if a is None and b is None:
        return None
    parts = [np.asarray(item, dtype=np.float32) for item in (a, b) if item is not None and len(item) > 0]
    if not parts:
        return None
    merged = np.concatenate(parts, axis=0)
    if len(merged) <= int(max_points):
        return merged.copy()
    indices = np.linspace(0, len(merged) - 1, int(max_points)).astype(np.int64)
    return merged[indices].copy()


def _bbox_from_points(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or len(points) == 0:
        return None
    arr = np.asarray(points, dtype=np.float32)
    return np.stack([np.min(arr, axis=0), np.max(arr, axis=0)], axis=0)
