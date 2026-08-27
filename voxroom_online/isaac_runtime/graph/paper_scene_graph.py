from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.mapping.coordinate_transform import grid_to_world_xy
from voxroom_online.isaac_runtime.mapping.room_segmentation import RoomMask, assign_objects_to_room_masks
from voxroom_online.isaac_runtime.perception.detection_types import FusedInstance
from voxroom_online.isaac_runtime.perception.object_memory import ObjectMemory

PAPER_RELATED_CATEGORY_PAIRS = {
    tuple(sorted((normalize_category(a), normalize_category(b))))
    for a, b in [
        ("bed", "nightstand"),
        ("wardrobe", "dresser"),
        ("bookshelf", "chair"),
        ("counter", "stove"),
        ("table", "chair"),
        ("bathroom sink", "mirror"),
        ("sink", "mirror"),
        ("shower", "bathtub"),
        ("refrigerator", "freezer"),
        ("oven", "microwave"),
        ("washing machine", "dryer"),
        ("sofa", "table"),
        ("desk", "office chair"),
        ("desk", "chair"),
        ("computer", "monitor"),
        ("computer", "tv"),
        ("piano", "bench"),
        ("fireplace", "mantel"),
        ("table", "mirror"),
        ("window", "curtains"),
        ("closet", "hangers"),
        ("bathroom cabinet", "toiletries"),
        ("living room rug", "coffee table"),
        ("kitchen cabinet", "dishes"),
        ("dining room chandelier", "dining table"),
        ("clock", "wall"),
        ("floor lamp", "reading chair"),
        ("couch", "throw pillows"),
        ("bookcase", "books"),
        ("tv", "sofa"),
        ("tv", "table"),
        ("vase", "table"),
        ("lamp", "table"),
    ]
}


@dataclass
class ObjectNode:
    id: str
    category: str
    confidence: float
    point_cloud_world: np.ndarray
    bbox_world: np.ndarray
    center_world: np.ndarray
    observed_count: int
    last_seen_step: int
    mean_confidence: float = 0.0
    detection_count: int = 0
    winner_detection_count: int = 0
    room_id: Optional[str] = None
    center_grid: Optional[Tuple[int, int]] = None
    is_new_node: bool = False
    is_goal_node: bool = False
    room_assignment_metadata: Dict[str, object] = field(default_factory=dict)
    source_instance_id: str = ""
    parent_track_id: Optional[str] = None
    child_track_ids: List[str] = field(default_factory=list)


@dataclass
class RoomNode:
    id: str
    room_type: str
    confidence: float
    region_polygon_world: Optional[np.ndarray]
    point_cloud_world: Optional[np.ndarray]
    contained_object_ids: List[str] = field(default_factory=list)
    mask_id: Optional[str] = None
    category_source: str = "unknown"
    mask_source: str = "unknown"
    area_m2: float = 0.0
    centroid_xy: Optional[Tuple[float, float]] = None
    boundary_unknown_fraction: float = 0.0
    is_partial: bool = False
    doorway_edges: List[dict] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class GroupNode:
    id: str
    category_summary: str
    object_ids: List[str]
    center_world: np.ndarray
    room_id: Optional[str]


@dataclass
class ObjectEdge:
    src_id: str
    dst_id: str
    relation: str
    confidence: float = 1.0
    is_short_edge: bool = False
    source: str = "paper_scene_graph"
    pruning_debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class AffiliationEdge:
    src_id: str
    dst_id: str
    relation: str
    confidence: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


class PaperSceneGraph:
    def __init__(self, related_category_pairs: Optional[Iterable[Tuple[str, str]]] = None):
        self.related_category_pairs = {
            tuple(sorted((normalize_category(a), normalize_category(b)))) for a, b in (related_category_pairs or PAPER_RELATED_CATEGORY_PAIRS)
        }
        self.object_nodes: Dict[str, ObjectNode] = {}
        self.room_nodes: Dict[str, RoomNode] = {}
        self.group_nodes: Dict[str, GroupNode] = {}
        self.object_edges: List[ObjectEdge] = []
        self.affiliation_edges: List[AffiliationEdge] = []
        self.new_object_ids: List[str] = []
        self.room_masks: Dict[str, RoomMask] = {}
        self.object_room_assignments: Dict[str, object] = {}
        self.version = 0

    def reset(self) -> None:
        self.object_nodes = {}
        self.room_nodes = {}
        self.group_nodes = {}
        self.object_edges = []
        self.affiliation_edges = []
        self.new_object_ids = []
        self.room_masks = {}
        self.object_room_assignments = {}
        self.version = 0

    def update_object_and_room_nodes(self, fused_instances: Sequence[FusedInstance]) -> List[ObjectNode]:
        new_objects: List[ObjectNode] = []
        self.new_object_ids = []
        for instance in fused_instances:
            if instance.node_type == "room":
                self._upsert_room(instance)
            else:
                node = self._upsert_object(instance)
                if node.is_new_node:
                    new_objects.append(node)
                    self.new_object_ids.append(node.id)
        if not self.room_nodes:
            self._ensure_unknown_room()
        self.update_affiliation_edges()
        self.version += 1
        return new_objects

    def update_from_object_memory(self, object_memory: ObjectMemory, map_info=None) -> None:
        previous_ids = set(self.object_nodes)
        self.new_object_ids = []
        for mem_node in object_memory.nodes:
            node_id = "object:%s" % int(mem_node.node_id)
            is_new = node_id not in previous_ids
            points = _points_or_center(mem_node.point_cloud_world, mem_node.center_world)
            bbox = _bbox_or_points(mem_node.bbox_world, points)
            self.object_nodes[node_id] = ObjectNode(
                id=node_id,
                category=normalize_category(mem_node.category),
                confidence=float(mem_node.confidence),
                point_cloud_world=points,
                bbox_world=bbox,
                center_world=np.asarray(mem_node.center_world, dtype=np.float32),
                observed_count=int(mem_node.observed_count),
                last_seen_step=int(mem_node.last_seen_step),
                mean_confidence=float(getattr(mem_node, "mean_confidence", mem_node.confidence)),
                detection_count=int(getattr(mem_node, "valid_detection_count", mem_node.observed_count) or mem_node.observed_count),
                winner_detection_count=int(getattr(mem_node, "winner_detection_count", mem_node.observed_count)),
                center_grid=tuple(int(v) for v in mem_node.center_grid),
                is_new_node=is_new,
                source_instance_id=str(getattr(mem_node, "source_instance_id", "") or ""),
                parent_track_id=getattr(mem_node, "parent_track_id", None),
                child_track_ids=list(getattr(mem_node, "child_track_ids", ()) or ()),
            )
            if is_new:
                self.new_object_ids.append(node_id)
        self._append_containment_edges_from_tracks()
        if not self.room_nodes:
            self._ensure_unknown_room()
        self.update_affiliation_edges(map_info=map_info)
        self.version += 1

    def _append_containment_edges_from_tracks(self) -> None:
        track_to_node = {
            str(node.source_instance_id): node_id
            for node_id, node in self.object_nodes.items()
            if str(node.source_instance_id or "")
        }
        existing = {(edge.src_id, edge.dst_id, edge.relation) for edge in self.object_edges}
        for node_id, node in self.object_nodes.items():
            parent_track_id = str(node.parent_track_id or "")
            parent_node_id = track_to_node.get(parent_track_id)
            if not parent_node_id or parent_node_id == node_id:
                continue
            for relation in ("supported_by", "inside_or_on"):
                key = (node_id, parent_node_id, relation)
                if key not in existing:
                    self.object_edges.append(ObjectEdge(src_id=node_id, dst_id=parent_node_id, relation=relation, confidence=1.0, is_short_edge=True, source="mask_containment"))
                    existing.add(key)

    def update_room_nodes_from_room_map(self, room_map: np.ndarray, map_info, room_names: Sequence[str]) -> None:
        arr = np.asarray(room_map)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[0] == 0:
            return
        next_rooms: Dict[str, RoomNode] = {}
        for idx, name in enumerate(room_names):
            if idx >= arr.shape[0]:
                break
            mask = np.asarray(arr[idx]) > 0.0
            if not np.any(mask):
                continue
            points = []
            for r, c in np.argwhere(mask):
                wx, wy = grid_to_world_xy(int(r), int(c), map_info)
                points.append([float(wx), float(wy), 0.0])
            room_type = normalize_category(str(name)).replace("_", " ")
            room_id = "room:%s" % room_type.replace(" ", "_")
            next_rooms[room_id] = RoomNode(
                id=room_id,
                room_type=room_type,
                confidence=1.0,
                region_polygon_world=None,
                point_cloud_world=np.asarray(points, dtype=np.float32),
                contained_object_ids=[],
                category_source="oracle_rooms_json",
                mask_source="preprocessed_rooms_json",
            )
        if next_rooms:
            unknown = self.room_nodes.get("room:unknown_room")
            self.room_nodes = next_rooms
            if unknown is not None:
                self.room_nodes.setdefault("room:unknown_room", unknown)
            self.update_affiliation_edges()
            self.version += 1

    def update_room_nodes_from_room_masks(self, room_masks: Sequence[RoomMask], labels: Optional[Dict[str, object]] = None, map_info=None) -> None:
        labels = dict(labels or {})
        next_rooms: Dict[str, RoomNode] = {}
        self.room_masks = {room.room_id: room for room in room_masks if not room.stale}
        for room in room_masks:
            if room.stale:
                continue
            label = labels.get(room.room_id)
            category = normalize_category(getattr(label, "category", "unknown")).replace("_", " ")
            if category == "unknown room":
                category = "unknown"
            confidence = float(getattr(label, "confidence", room.confidence))
            source = str(getattr(label, "backend", "unknown"))
            rr, cc = np.nonzero(room.mask)
            points = []
            if map_info is not None:
                for r, c in zip(rr, cc):
                    wx, wy = grid_to_world_xy(int(r), int(c), map_info)
                    points.append([float(wx), float(wy), 0.0])
            elif rr.size:
                points = [[float(c), float(r), 0.0] for r, c in zip(rr, cc)]
            room_id = "room:%s" % room.room_id
            next_rooms[room_id] = RoomNode(
                id=room_id,
                room_type=category,
                confidence=float(confidence),
                region_polygon_world=None,
                point_cloud_world=np.asarray(points, dtype=np.float32) if points else None,
                contained_object_ids=[],
                mask_id=room.room_id,
                category_source="vlm_object_evidence" if source == "vlm" else source,
                mask_source=room.source,
                area_m2=float(room.area_m2),
                centroid_xy=tuple(float(v) for v in room.centroid_xy),
                boundary_unknown_fraction=float(room.boundary_unknown_fraction),
                is_partial=bool(room.is_partial),
                doorway_edges=list(room.doorway_edges),
                metadata={
                    "unknown_reason": getattr(label, "unknown_reason", None),
                    "supporting_objects": list(getattr(label, "supporting_objects", []) or []),
                    "conflicting_evidence": list(getattr(label, "conflicting_evidence", []) or []),
                    "rationale": str(getattr(label, "rationale", "")),
                    "mask_confidence": float(room.mask_confidence),
                    "observed_free_cells": int(room.observed_free_cells),
                },
            )
        if next_rooms:
            self.room_nodes = next_rooms
            self.update_affiliation_edges(map_info=map_info)
            self.version += 1

    def update_affiliation_edges(self, map_info=None) -> None:
        self.affiliation_edges = []
        for room in self.room_nodes.values():
            room.contained_object_ids = []
        if self.room_masks and map_info is not None:
            self.object_room_assignments = assign_objects_to_room_masks(self.object_nodes.values(), list(self.room_masks.values()), map_info)
        else:
            self.object_room_assignments = {}
        for obj in self.object_nodes.values():
            room = self._find_room_for_object(obj)
            obj.room_id = room.id if room is not None else None
            obj.room_assignment_metadata = {}
            if room is None:
                continue
            assignment = self.object_room_assignments.get(obj.id)
            metadata = assignment.to_edge_metadata() if assignment is not None else {}
            obj.room_assignment_metadata = dict(metadata)
            room.contained_object_ids.append(obj.id)
            self.affiliation_edges.append(
                AffiliationEdge(
                    src_id=obj.id,
                    dst_id=room.id,
                    relation="belongs_to",
                    confidence=float(metadata.get("assignment_confidence", 1.0)),
                    metadata=metadata,
                )
            )
        for group in self.group_nodes.values():
            room_ids = {self.object_nodes[obj_id].room_id for obj_id in group.object_ids if obj_id in self.object_nodes}
            room_ids.discard(None)
            if len(room_ids) == 1:
                group.room_id = next(iter(room_ids))
                self.affiliation_edges.append(
                    AffiliationEdge(
                        src_id=group.id,
                        dst_id=group.room_id,
                        relation="belongs_to",
                        metadata={"source": "member_object_room_majority"},
                    )
                )

    def update_group_nodes(self, eps_m: float = 0.5) -> None:
        self.group_nodes = {}
        objects_by_room: Dict[str, List[str]] = {}
        for obj_id, obj in self.object_nodes.items():
            room_id = obj.room_id or "room:unknown_room"
            objects_by_room.setdefault(room_id, []).append(obj_id)
        for room_id, object_ids in objects_by_room.items():
            for members in _cluster_object_ids_by_distance(self.object_nodes, object_ids, eps_m=float(eps_m)):
                if not members:
                    continue
                centers = [self.object_nodes[mid].center_world for mid in members]
                member_key = "-".join(mid.split(":", 1)[-1] for mid in sorted(members))
                group_id = "group:%s:%s" % (room_id.split(":", 1)[-1], member_key)
                self.group_nodes[group_id] = GroupNode(
                    id=group_id,
                    category_summary=_group_summary(self.object_nodes, self.object_edges, members),
                    object_ids=sorted(members),
                    center_world=np.mean(np.asarray(centers, dtype=np.float32), axis=0),
                    room_id=room_id,
                )
        self.update_affiliation_edges()
        self.version += 1

    def find_goal_candidates(self, goal_category: str) -> List[ObjectNode]:
        goal = normalize_category(goal_category)
        return [node for node in self.object_nodes.values() if normalize_category(node.category) == goal]

    def _upsert_object(self, instance: FusedInstance) -> ObjectNode:
        node_id = "object:%s" % instance.instance_id
        node = ObjectNode(
            id=node_id,
            category=normalize_category(instance.category),
            confidence=float(instance.confidence),
            point_cloud_world=np.asarray(instance.point_cloud_world, dtype=np.float32).copy(),
            bbox_world=np.asarray(instance.bbox_world, dtype=np.float32).copy(),
            center_world=np.asarray(instance.center_world, dtype=np.float32).copy(),
            observed_count=int(instance.observed_count),
            last_seen_step=int(instance.last_seen_step),
            mean_confidence=float(getattr(instance, "mean_confidence", instance.confidence)),
            detection_count=int(getattr(instance, "valid_detection_count", instance.observed_count) or instance.observed_count),
            winner_detection_count=int(getattr(instance, "winner_detection_count", instance.observed_count)),
            is_new_node=node_id not in self.object_nodes,
            source_instance_id=str(getattr(instance, "instance_id", "") or ""),
            parent_track_id=getattr(instance, "parent_track_id", None),
            child_track_ids=list(getattr(instance, "child_track_ids", ()) or ()),
        )
        self.object_nodes[node_id] = node
        return node

    def _upsert_room(self, instance: FusedInstance) -> RoomNode:
        room_type = normalize_category(instance.category)
        room_id = "room:%s" % instance.instance_id
        room = RoomNode(
            id=room_id,
            room_type=room_type,
            confidence=float(instance.confidence),
            region_polygon_world=None,
            point_cloud_world=np.asarray(instance.point_cloud_world, dtype=np.float32).copy(),
            contained_object_ids=[],
        )
        self.room_nodes[room_id] = room
        return room

    def _ensure_unknown_room(self) -> None:
        self.room_nodes.setdefault(
            "room:unknown_room",
            RoomNode(
                id="room:unknown_room",
                room_type="unknown_room",
                confidence=0.0,
                region_polygon_world=None,
                point_cloud_world=None,
                contained_object_ids=[],
            ),
        )

    def _find_room_for_object(self, obj: ObjectNode) -> Optional[RoomNode]:
        assignment = self.object_room_assignments.get(obj.id)
        if assignment is not None and getattr(assignment, "room_id", None):
            room = self.room_nodes.get("room:%s" % assignment.room_id)
            if room is not None:
                return room
        concrete_rooms = [room for room in self.room_nodes.values() if room.point_cloud_world is not None and len(room.point_cloud_world) > 0]
        for room in concrete_rooms:
            room_bbox = _bbox_from_points(room.point_cloud_world)
            if _object_inside_room(obj, room_bbox):
                return room
        return self.room_nodes.get("room:unknown_room")


def _points_or_center(points: Optional[np.ndarray], center_world: Sequence[float]) -> np.ndarray:
    if points is not None and len(points) > 0:
        return np.asarray(points, dtype=np.float32).copy()
    return np.asarray([center_world], dtype=np.float32)


def _bbox_or_points(bbox: Optional[np.ndarray], points: np.ndarray) -> np.ndarray:
    if bbox is not None:
        arr = np.asarray(bbox, dtype=np.float32)
        if arr.shape == (2, 3):
            return arr.copy()
    return _bbox_from_points(points)


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    return np.stack([np.min(arr, axis=0), np.max(arr, axis=0)], axis=0)


def _cluster_object_ids_by_distance(
    object_nodes: Dict[str, ObjectNode],
    object_ids: Sequence[str],
    eps_m: float,
) -> List[List[str]]:
    ids = sorted(object_ids)
    visited = set()
    clusters: List[List[str]] = []
    for obj_id in ids:
        if obj_id in visited:
            continue
        visited.add(obj_id)
        stack = [obj_id]
        cluster = []
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            cur_center = np.asarray(object_nodes[cur].center_world[:2], dtype=np.float32)
            for other_id in ids:
                if other_id in visited:
                    continue
                other_center = np.asarray(object_nodes[other_id].center_world[:2], dtype=np.float32)
                if float(np.linalg.norm(cur_center - other_center)) <= float(eps_m):
                    visited.add(other_id)
                    stack.append(other_id)
        clusters.append(sorted(cluster))
    return clusters


def _group_summary(object_nodes: Dict[str, ObjectNode], object_edges: Sequence[ObjectEdge], members: Sequence[str]) -> str:
    member_set = set(members)
    categories = sorted({object_nodes[obj_id].category for obj_id in members if obj_id in object_nodes})
    relations = []
    for edge in object_edges:
        if edge.src_id in member_set and edge.dst_id in member_set:
            src = object_nodes[edge.src_id].category
            dst = object_nodes[edge.dst_id].category
            relations.append("%s %s %s" % (src, edge.relation, dst))
    if relations:
        return "; ".join(relations[:4])
    return ", ".join(categories)


def _object_inside_room(obj: ObjectNode, room_bbox: np.ndarray) -> bool:
    center = np.asarray(obj.center_world, dtype=np.float32)
    if abs(float(room_bbox[1, 2] - room_bbox[0, 2])) <= 1e-6:
        if np.all(center[:2] >= room_bbox[0, :2]) and np.all(center[:2] <= room_bbox[1, :2]):
            return True
        points = np.asarray(obj.point_cloud_world, dtype=np.float32)
        if len(points) == 0:
            return False
        inside_xy = np.all((points[:, :2] >= room_bbox[0, :2]) & (points[:, :2] <= room_bbox[1, :2]), axis=1)
        return float(np.count_nonzero(inside_xy)) / float(len(points)) >= 0.5
    if np.all(center >= room_bbox[0]) and np.all(center <= room_bbox[1]):
        return True
    points = np.asarray(obj.point_cloud_world, dtype=np.float32)
    if len(points) == 0:
        return False
    inside = np.all((points >= room_bbox[0]) & (points <= room_bbox[1]), axis=1)
    return float(np.count_nonzero(inside)) / float(len(points)) >= 0.5
