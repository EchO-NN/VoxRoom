from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from voxroom_online.isaac_runtime.mapping.vertical_profile import VerticalProfileMap, ensure_vertical_profile


GridCell = Tuple[int, int]


@dataclass
class RoomSegmentationConfig:
    enabled: bool = True
    algorithm: str = "voxel_occupancy_door_wall_v33"
    legacy_watershed_allowed: str = "debug_only"
    source_grid: str = "online_depth_observed"
    update_every_steps: int = 5
    proposal_mode: str = "distance_watershed"
    finalization_mode: str = "doorway_constrained_merge"
    use_structural_obstacle_mask: bool = True
    suppress_furniture_obstacles: bool = True
    min_observed_free_cells: int = 50
    min_room_area_m2: float = 1.5
    max_clutter_component_area_m2: float = 4.0
    min_wall_line_length_m: float = 1.5
    wall_like_aspect_ratio_min: float = 3.0
    morphology_close_radius_m: float = 0.20
    morphology_open_radius_m: float = 0.10
    distance_smooth_sigma_cells: float = 1.0
    seed_min_clearance_m: float = 0.45
    seed_min_distance_m: float = 1.2
    doorway_width_min_m: float = 0.55
    doorway_width_max_m: float = 1.45
    doorway_clearance_max_m: float = 0.85
    doorway_wall_support_min_ratio: float = 0.35
    doorway_unknown_support_max_ratio: float = 0.20
    doorway_endpoint_wall_distance_m: float = 0.35
    doorway_neck_ratio_max: float = 0.55
    open_boundary_merge: bool = True
    merge_same_semantic_category: bool = True
    merge_unknown_into_open_labeled_region: bool = True
    use_premerge_labels_for_open_plan_merge: bool = True
    min_label_reliability_for_functional_split: float = 0.65
    unknown_allows_functional_split: bool = False
    final_label_after_merge: bool = True
    merge_wide_openings: bool = True
    small_segment_merge_area_m2: float = 1.2
    id_iou_threshold: float = 0.35
    unknown_boundary_confidence_penalty: bool = True
    stale_ttl_steps: int = 2
    debug_dump: bool = False
    debug_dir: str = "debug/voxroom_roomseg"
    vertical_free_z_min_m: float = 0.20
    vertical_free_z_max_m: float = 2.00
    vertical_free_min_free_rays: int = 1
    vertical_free_min_observed_rays: int = 1
    legacy_options: Dict[str, object] = field(default_factory=dict)
    resolution_m: float = 0.05
    map_info: Optional[MapInfo] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]] = None, **overrides) -> "RoomSegmentationConfig":
        raw = dict(data or {})
        vertical_or_free = dict(raw.get("vertical_or_free", {}) or {})
        if vertical_or_free:
            raw.setdefault("vertical_free_z_min_m", vertical_or_free.get("z_min_m", 0.20))
            raw.setdefault("vertical_free_z_max_m", vertical_or_free.get("z_max_m", 2.00))
            raw.setdefault("vertical_free_min_free_rays", vertical_or_free.get("min_free_rays", 1))
            raw.setdefault("vertical_free_min_observed_rays", vertical_or_free.get("min_observed_rays", 1))
        raw.update({key: value for key, value in overrides.items() if value is not None})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class RoomMask:
    room_id: str
    mask: np.ndarray
    centroid_xy: Tuple[float, float]
    area_m2: float
    boundary_unknown_fraction: float
    doorway_edges: List[dict]
    confidence: float
    source: str = "voxel_occupancy_door_wall_v33"
    observed_free_cells: int = 0
    mask_confidence: float = 0.0
    is_partial: bool = False
    step: int = 0
    stale: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ObjectRoomAssignment:
    object_id: str
    room_id: Optional[str]
    edge_type: str = "contains"
    source: str = "online_geometry_mask_overlap"
    centroid_inside: bool = False
    mask_overlap_ratio: float = 0.0
    assignment_confidence: float = 0.0
    ambiguous: bool = False

    def to_edge_metadata(self) -> dict:
        return {
            "edge_type": self.edge_type,
            "source": self.source,
            "centroid_inside": bool(self.centroid_inside),
            "mask_overlap_ratio": float(self.mask_overlap_ratio),
            "assignment_confidence": float(self.assignment_confidence),
            "ambiguous": bool(self.ambiguous),
        }


@dataclass
class RoomAdjacencyEvidence:
    room_a_label: int
    room_b_label: int
    boundary_cells: int
    boundary_length_m: float
    neck_width_m: float
    min_clearance_m: float
    mean_clearance_m: float
    wall_support_left: float
    wall_support_right: float
    obstacle_support_ratio: float
    unknown_support_ratio: float
    endpoints_touch_structural_wall: bool
    separates_large_regions: bool
    verified_doorway: bool
    merge_reason: str
    boundary_cells_sample: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "room_a_label": int(self.room_a_label),
            "room_b_label": int(self.room_b_label),
            "boundary_cells": int(self.boundary_cells),
            "boundary_length_m": float(self.boundary_length_m),
            "neck_width_m": float(self.neck_width_m),
            "min_clearance_m": float(self.min_clearance_m),
            "mean_clearance_m": float(self.mean_clearance_m),
            "wall_support_left": float(self.wall_support_left),
            "wall_support_right": float(self.wall_support_right),
            "obstacle_support_ratio": float(self.obstacle_support_ratio),
            "unknown_support_ratio": float(self.unknown_support_ratio),
            "endpoints_touch_structural_wall": bool(self.endpoints_touch_structural_wall),
            "separates_large_regions": bool(self.separates_large_regions),
            "verified_doorway": bool(self.verified_doorway),
            "merge_reason": self.merge_reason,
            "boundary_cells_sample": [list(map(int, cell)) for cell in self.boundary_cells_sample],
        }


@dataclass
class RoomProposalState:
    proposal_labels: np.ndarray
    structural_free_mask: np.ndarray
    structural_obstacle_mask: np.ndarray
    unknown_mask: np.ndarray
    distance_m: np.ndarray
    step: int = 0
    debug: Dict[str, object] = field(default_factory=dict)


class OnlineRoomSegmenter:
    def __init__(self, config: Optional[RoomSegmentationConfig | Mapping[str, object]] = None):
        if isinstance(config, RoomSegmentationConfig):
            self.config = config
        else:
            self.config = RoomSegmentationConfig.from_mapping(config or {})
        self._previous: Dict[str, RoomMask] = {}
        self._last_live_ids: set[str] = set()
        self._next_room_index = 1
        self.last_debug: Dict[str, object] = {}

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Optional[Iterable[object]] = None,
        vertical_profile: Optional[VerticalProfileMap] = None,
    ) -> List[RoomMask]:
        if not self.config.enabled:
            self.last_debug = {"enabled": False, "room_count": 0}
            return []
        free = np.asarray(observed_free_mask, dtype=bool)
        occ = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool)
        unknown = np.asarray(unknown_mask, dtype=bool)
        if free.shape != occ.shape or free.shape != unknown.shape:
            raise ValueError("room segmentation masks must have the same HxW shape")

        proposal_rooms, proposal_state = self.build_proposals(
            occupancy_map,
            observed_free_mask,
            obstacle_mask,
            unknown_mask,
            step=step,
            object_memory=object_memory,
            vertical_profile=vertical_profile,
        )
        _ = proposal_rooms
        return self.finalize_proposals(proposal_state, proposal_semantic_labels=None)

    def build_proposals(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Optional[Iterable[object]] = None,
        vertical_profile: Optional[VerticalProfileMap] = None,
    ) -> Tuple[List[RoomMask], RoomProposalState]:
        free = np.asarray(observed_free_mask, dtype=bool)
        occ = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool)
        unknown = np.asarray(unknown_mask, dtype=bool)
        if free.shape != occ.shape or free.shape != unknown.shape:
            raise ValueError("room proposal masks must have the same HxW shape")
        vertical_debug: dict = {}
        vertical_source_active = False
        if _uses_vertical_free_source(self.config):
            vertical_free, vertical_observed, vertical_debug = _vertical_free_watershed_masks(
                vertical_profile=vertical_profile,
                shape=free.shape,
                config=self.config,
            )
            if np.any(vertical_observed):
                vertical_source_active = True
                free = vertical_free
                occ = vertical_observed & ~vertical_free
                unknown = ~vertical_observed
        structural_obstacles, obstacle_debug = _build_structural_obstacle_mask_with_debug(
            occ,
            free,
            unknown,
            object_memory,
            self.config,
        )
        free_for_rooms = free.copy()
        if bool(self.config.use_structural_obstacle_mask):
            free_for_rooms |= occ & ~structural_obstacles & ~unknown
            obstacle_for_rooms = structural_obstacles
        else:
            obstacle_for_rooms = occ
        if vertical_source_active:
            structural = free & ~unknown
            obstacle_for_rooms = occ
            structural_obstacles = occ
        else:
            structural = build_structural_free_mask(free_for_rooms, obstacle_for_rooms, unknown, self.config)
        proposal_rooms, proposal_state = build_room_proposals(
            structural,
            unknown,
            self.config,
            step=int(step),
            structural_obstacle_mask=structural_obstacles,
        )
        proposal_state.debug["structural_obstacle_mask"] = obstacle_debug
        if vertical_debug:
            proposal_state.debug.update(vertical_debug)
        return proposal_rooms, proposal_state

    def finalize_proposals(
        self,
        proposal_state: RoomProposalState,
        proposal_semantic_labels: Optional[Mapping[int, object]] = None,
    ) -> List[RoomMask]:
        room_masks, seg_debug = finalize_room_proposals(
            proposal_state,
            self.config,
            proposal_semantic_labels=proposal_semantic_labels,
        )
        room_masks = self._assign_stable_ids(room_masks, int(proposal_state.step))
        self._previous = {room.room_id: room for room in room_masks}
        self._last_live_ids = {room.room_id for room in room_masks if not room.stale}
        self.last_debug = room_segmentation_debug(
            room_masks,
            proposal_state.structural_free_mask,
            segmentation_debug=seg_debug,
            structural_obstacle_debug=proposal_state.debug.get("structural_obstacle_mask"),
        )
        return room_masks

    def _assign_stable_ids(self, rooms: Sequence[RoomMask], step: int) -> List[RoomMask]:
        previous_items = [(room_id, room) for room_id, room in self._previous.items() if not room.stale]
        matches: Dict[int, str] = {}
        used_prev: set[str] = set()
        candidates = []
        for idx, room in enumerate(rooms):
            for room_id, prev in previous_items:
                iou = _mask_iou(room.mask, prev.mask)
                dist = _centroid_distance(room.centroid_xy, prev.centroid_xy)
                if iou >= self.config.id_iou_threshold or (iou > 0.0 and dist <= max(2.0, 2.0 * self.config.seed_min_distance_m)):
                    candidates.append((iou, -dist, idx, room_id))
        candidates.sort(reverse=True)
        for _iou, _neg_dist, idx, room_id in candidates:
            if idx in matches or room_id in used_prev:
                continue
            matches[idx] = room_id
            used_prev.add(room_id)
        out: List[RoomMask] = []
        for idx, room in enumerate(rooms):
            room_id = matches.get(idx)
            if room_id is None:
                room_id = "room_%04d" % self._next_room_index
                self._next_room_index += 1
            room.room_id = room_id
            room.step = int(step)
            out.append(room)
        label_to_room = {
            int(room.metadata["label_id"]): room.room_id
            for room in out
            if isinstance(room.metadata, dict) and room.metadata.get("label_id") is not None and not room.stale
        }
        for room in out:
            enriched_edges = []
            for edge in room.doorway_edges:
                item = dict(edge)
                if item.get("room_a_label") is not None:
                    item["room_a"] = label_to_room.get(int(item["room_a_label"]))
                if item.get("room_b_label") is not None:
                    item["room_b"] = label_to_room.get(int(item["room_b_label"]))
                if item.get("room_a") and item.get("room_b") and item.get("room_a") != item.get("room_b"):
                    enriched_edges.append(item)
            room.doorway_edges = enriched_edges
        stale_ttl = max(0, int(self.config.stale_ttl_steps))
        if stale_ttl:
            for room_id, prev in self._previous.items():
                if room_id in used_prev:
                    continue
                if int(step) - int(prev.step) <= stale_ttl:
                    stale = RoomMask(
                        room_id=room_id,
                        mask=np.asarray(prev.mask, dtype=bool).copy(),
                        centroid_xy=tuple(float(v) for v in prev.centroid_xy),
                        area_m2=float(prev.area_m2),
                        boundary_unknown_fraction=float(prev.boundary_unknown_fraction),
                        doorway_edges=list(prev.doorway_edges),
                        confidence=float(prev.confidence),
                        source=prev.source,
                        observed_free_cells=int(prev.observed_free_cells),
                        mask_confidence=float(prev.mask_confidence),
                        is_partial=bool(prev.is_partial),
                        step=int(prev.step),
                        stale=True,
                        metadata={**dict(prev.metadata), "stale_until_step": int(prev.step) + stale_ttl},
                    )
                    out.append(stale)
        return out


MOVABLE_ROOM_CLUTTER_CATEGORIES = {
    "armchair",
    "bed",
    "bench",
    "cabinet",
    "chair",
    "couch",
    "desk",
    "dining_table",
    "dresser",
    "lamp",
    "nightstand",
    "ottoman",
    "picture",
    "plant",
    "shelf",
    "sofa",
    "stool",
    "table",
    "television",
    "tv",
    "wardrobe",
}


def build_structural_obstacle_mask(
    obstacle_mask: np.ndarray,
    observed_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
    object_memory: Optional[Iterable[object]],
    config: RoomSegmentationConfig,
) -> np.ndarray:
    structural, _debug = _build_structural_obstacle_mask_with_debug(
        obstacle_mask,
        observed_free_mask,
        unknown_mask,
        object_memory,
        config,
    )
    return structural


def _build_structural_obstacle_mask_with_debug(
    obstacle_mask: np.ndarray,
    observed_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
    object_memory: Optional[Iterable[object]],
    config: RoomSegmentationConfig,
) -> Tuple[np.ndarray, dict]:
    obstacles = np.asarray(obstacle_mask, dtype=bool)
    free = np.asarray(observed_free_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if obstacles.shape != free.shape or obstacles.shape != unknown.shape:
        raise ValueError("structural obstacle inputs must have the same HxW shape")
    candidate_obstacles = obstacles & ~unknown
    object_mask, object_categories = _object_clutter_mask(candidate_obstacles.shape, object_memory)
    structural = np.zeros_like(candidate_obstacles, dtype=bool)
    cell_area = float(config.resolution_m) ** 2
    max_clutter_cells = max(1, int(round(float(config.max_clutter_component_area_m2) / max(cell_area, 1e-9))))
    min_wall_line_cells = max(1, int(round(float(config.min_wall_line_length_m) / max(float(config.resolution_m), 1e-9))))
    components_debug: List[dict] = []
    for idx, component in enumerate(_connected_components(candidate_obstacles), start=1):
        rows = np.asarray([cell[0] for cell in component], dtype=np.int32)
        cols = np.asarray([cell[1] for cell in component], dtype=np.int32)
        area_cells = int(len(component))
        area_m2 = float(area_cells) * cell_area
        height_cells = int(np.max(rows) - np.min(rows) + 1) if rows.size else 0
        width_cells = int(np.max(cols) - np.min(cols) + 1) if cols.size else 0
        long_axis = max(height_cells, width_cells)
        short_axis = max(1, min(height_cells, width_cells))
        aspect_ratio = float(long_axis) / float(short_axis)
        comp_mask = np.zeros_like(candidate_obstacles, dtype=bool)
        comp_mask[rows, cols] = True
        overlap_categories = sorted({object_categories[cell] for cell in zip(rows.tolist(), cols.tolist()) if cell in object_categories})
        overlap_object = bool(np.any(comp_mask & object_mask))
        wall_like = bool(long_axis >= min_wall_line_cells and aspect_ratio >= float(config.wall_like_aspect_ratio_min))
        compact_clutter = bool(area_cells <= max_clutter_cells and not wall_like)
        kept_as_wall = bool(wall_like and not (bool(config.suppress_furniture_obstacles) and overlap_object))
        if not kept_as_wall and not compact_clutter and area_cells > max_clutter_cells:
            kept_as_wall = bool(aspect_ratio >= max(1.6, float(config.wall_like_aspect_ratio_min) * 0.65) and long_axis >= max(4, min_wall_line_cells // 2))
        if kept_as_wall:
            structural |= comp_mask
        components_debug.append(
            {
                "component_index": int(idx),
                "area_cells": int(area_cells),
                "area_m2": float(area_m2),
                "height_cells": int(height_cells),
                "width_cells": int(width_cells),
                "aspect_ratio": float(aspect_ratio),
                "line_support": float(1.0 if wall_like else 0.0),
                "kept_as_wall": bool(kept_as_wall),
                "removed_as_clutter": bool(not kept_as_wall),
                "overlap_object_categories": overlap_categories,
            }
        )
    debug = {
        "enabled": bool(config.use_structural_obstacle_mask),
        "source": "structural_obstacle_filter",
        "input_obstacle_cells": int(np.count_nonzero(obstacles)),
        "unknown_obstacle_cells_ignored": int(np.count_nonzero(obstacles & unknown)),
        "structural_obstacle_cells": int(np.count_nonzero(structural)),
        "suppressed_obstacle_cells": int(np.count_nonzero(candidate_obstacles & ~structural)),
        "object_clutter_cells": int(np.count_nonzero(object_mask)),
        "components": components_debug,
    }
    return structural.astype(bool), debug


def build_structural_free_mask(
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: RoomSegmentationConfig,
) -> np.ndarray:
    free = np.asarray(observed_free_mask, dtype=bool).copy()
    obstacles = np.asarray(obstacle_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    cell_area = float(config.resolution_m) ** 2
    max_clutter_cells = max(0, int(round(float(config.max_clutter_component_area_m2) / max(cell_area, 1e-9))))
    if max_clutter_cells > 0 and np.any(obstacles) and not bool(config.use_structural_obstacle_mask):
        for component in _connected_components(obstacles):
            if len(component) <= max_clutter_cells:
                for row, col in component:
                    if not unknown[row, col]:
                        free[row, col] = True
    if bool(config.use_structural_obstacle_mask):
        free &= ~obstacles
    free &= ~unknown
    close_radius = _radius_cells(config.morphology_close_radius_m, config.resolution_m)
    open_radius = _radius_cells(config.morphology_open_radius_m, config.resolution_m)
    if close_radius:
        free = _binary_close(free, close_radius)
    if open_radius:
        free = _binary_open(free, open_radius)
    free &= ~unknown
    return free.astype(bool)


def _uses_vertical_free_source(config: RoomSegmentationConfig) -> bool:
    source = str(getattr(config, "source_grid", "") or "").strip().lower()
    algorithm = str(getattr(config, "algorithm", "") or "").strip().lower()
    return source in {
        "vertical_profile_free_0p2_2p0",
        "vertical_free_0p2_2p0",
        "watershed_vertical_free",
    } or "vertical_free" in algorithm


def _vertical_free_watershed_masks(
    *,
    vertical_profile: Optional[VerticalProfileMap],
    shape: tuple[int, int],
    config: RoomSegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    vp = ensure_vertical_profile(vertical_profile, shape)
    indices = _vertical_band_indices(
        vp,
        z_min_m=float(config.vertical_free_z_min_m),
        z_max_m=float(config.vertical_free_z_max_m),
    )
    if not indices:
        free = np.zeros(shape, dtype=bool)
        observed = np.zeros(shape, dtype=bool)
    else:
        band_names = tuple(str(vp.band_names[idx]) for idx in indices)
        free = vp.reliable_free_mask(
            min_free_rays=int(config.vertical_free_min_free_rays),
            min_observed_rays=int(config.vertical_free_min_observed_rays),
            band_names=band_names,
        )
        observed_count = np.sum(np.asarray(vp.observed_count[indices], dtype=np.uint32), axis=0)
        observed = observed_count >= int(config.vertical_free_min_observed_rays)
    debug = {
        "roomseg_input_source": "vertical_profile_free_0p2_2p0_watershed",
        "vertical_free_source": "vertical_profile_0p2_2p0",
        "vertical_or_free_z_min_m": float(config.vertical_free_z_min_m),
        "vertical_or_free_z_max_m": float(config.vertical_free_z_max_m),
        "vertical_or_free_map": free.astype(bool),
        "vertical_or_free_cells": int(np.count_nonzero(free)),
        "vertical_free_room_domain": free.astype(bool),
        "vertical_free_added_to_roomseg_cells": int(np.count_nonzero(free)),
        "vertical_observed_map": observed.astype(bool),
        "vertical_observed_cells": int(np.count_nonzero(observed)),
        "observed_not_vertical_free": observed & ~free,
        "observed_not_vertical_free_cells": int(np.count_nonzero(observed & ~free)),
        "initial_roomseg_free": free.astype(bool),
        "initial_roomseg_occupied": (observed & ~free).astype(bool),
        "repaired_roomseg_free": free.astype(bool),
        "repaired_roomseg_occupied": (observed & ~free).astype(bool),
        "repaired_roomseg_unknown": (~observed).astype(bool),
        "navigation_free_added_to_roomseg_cells": 0,
        "navigation_free_added_to_strict_roomseg_cells": 0,
    }
    return free.astype(bool), observed.astype(bool), debug


def _vertical_band_indices(
    vertical_profile: VerticalProfileMap,
    *,
    z_min_m: float,
    z_max_m: float,
) -> list[int]:
    lo = float(z_min_m)
    hi = float(z_max_m)
    return [
        int(idx)
        for idx, (_name, (band_lo, band_hi)) in enumerate(zip(vertical_profile.band_names, vertical_profile.band_ranges_m))
        if float(band_hi) > lo and float(band_lo) < hi
    ]


def build_room_proposals(
    structural_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: RoomSegmentationConfig,
    step: int = 0,
    structural_obstacle_mask: Optional[np.ndarray] = None,
) -> Tuple[List[RoomMask], RoomProposalState]:
    free = np.asarray(structural_free_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    structural_obstacles = np.asarray(structural_obstacle_mask, dtype=bool) if structural_obstacle_mask is not None else ~free & ~unknown
    proposal_labels = np.zeros_like(free, dtype=np.int32)
    distance_global = np.zeros_like(free, dtype=np.float32)
    proposal_rooms: List[RoomMask] = []
    if not np.any(free):
        state = RoomProposalState(proposal_labels, free, structural_obstacles, unknown, distance_global, step=int(step), debug=_empty_room_segmentation_debug())
        return [], state
    min_cells = max(1, int(config.min_observed_free_cells))
    min_area_cells = max(1, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    components = _connected_components(free)
    if len(components) == 1:
        component_list = components
    else:
        component_list = [comp for comp in components if len(comp) >= min_cells and len(comp) >= min_area_cells]
        if not component_list and components:
            component_list = [max(components, key=len)]
    next_label = 1
    for component in component_list:
        comp_mask = np.zeros_like(free, dtype=bool)
        for row, col in component:
            comp_mask[row, col] = True
        local_labels, distance_m = _watershed_component(comp_mask, config)
        distance_global[comp_mask] = distance_m[comp_mask]
        for local_label in sorted(int(v) for v in np.unique(local_labels) if int(v) > 0):
            mask = local_labels == local_label
            if int(np.count_nonzero(mask)) < min_cells and proposal_rooms:
                continue
            proposal_labels[mask] = int(next_label)
            room = _room_from_mask("proposal_%04d" % int(next_label), mask, unknown, [], config, step)
            room.metadata["label_id"] = int(next_label)
            room.metadata["proposal_labels"] = [int(next_label)]
            room.metadata["is_premerge_proposal"] = True
            proposal_rooms.append(room)
            next_label += 1
    if not proposal_rooms and np.any(free):
        proposal_labels[free] = 1
        room = _room_from_mask("proposal_0001", free, unknown, [], config, step)
        room.metadata["label_id"] = 1
        room.metadata["proposal_labels"] = [1]
        room.metadata["is_premerge_proposal"] = True
        proposal_rooms.append(room)
    debug = {
        "proposal_mode": str(config.proposal_mode),
        "finalization_mode": "premerge_proposals",
        "proposal_room_count": int(len(proposal_rooms)),
        "final_room_count": int(len(proposal_rooms)),
        "proposal_room_masks": _proposal_masks_debug(proposal_labels),
        "merge_operations": [],
        "doorway_edges": [],
        "adjacency_evidence": [],
    }
    state = RoomProposalState(
        proposal_labels=proposal_labels,
        structural_free_mask=free,
        structural_obstacle_mask=structural_obstacles,
        unknown_mask=unknown,
        distance_m=distance_global,
        step=int(step),
        debug=debug,
    )
    return proposal_rooms, state


def finalize_room_proposals(
    proposal_state: RoomProposalState,
    config: RoomSegmentationConfig,
    proposal_semantic_labels: Optional[Mapping[int, object]] = None,
) -> Tuple[List[RoomMask], dict]:
    labels = np.asarray(proposal_state.proposal_labels, dtype=np.int32)
    if not np.any(labels > 0):
        return [], _empty_room_segmentation_debug()
    final_labels, finalization_debug, doorway_edges = merge_open_plan_proposals(
        proposal_labels=labels,
        structural_free_mask=proposal_state.structural_free_mask,
        structural_obstacle_mask=proposal_state.structural_obstacle_mask,
        unknown_mask=proposal_state.unknown_mask,
        distance_m=proposal_state.distance_m,
        config=config,
        proposal_semantic_labels=proposal_semantic_labels,
    )
    min_cells = max(1, int(config.min_observed_free_cells))
    room_masks: List[RoomMask] = []
    original_labels = np.asarray(proposal_state.proposal_labels, dtype=np.int32)
    for label_id in sorted(int(v) for v in np.unique(final_labels) if int(v) > 0):
        mask = final_labels == label_id
        if int(np.count_nonzero(mask)) < min_cells and room_masks:
            continue
        room_edges = [
            dict(edge)
            for edge in doorway_edges
            if int(edge.get("room_a_label", -1)) == int(label_id) or int(edge.get("room_b_label", -1)) == int(label_id)
        ]
        room = _room_from_mask("pending", mask, proposal_state.unknown_mask, room_edges, config, int(proposal_state.step))
        room.metadata["label_id"] = int(label_id)
        room.metadata["source_finalization_mode"] = str(config.finalization_mode)
        room.metadata["proposal_labels"] = sorted(int(v) for v in np.unique(original_labels[mask]) if int(v) > 0)
        room_masks.append(room)
    finalization_debug["proposal_room_count"] = int(len([v for v in np.unique(labels) if int(v) > 0]))
    finalization_debug["final_room_count"] = int(len(room_masks))
    return room_masks, finalization_debug


def segment_room_masks(
    structural_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: RoomSegmentationConfig,
    step: int = 0,
    structural_obstacle_mask: Optional[np.ndarray] = None,
    proposal_semantic_labels: Optional[Mapping[int, str]] = None,
    return_debug: bool = False,
) -> List[RoomMask] | Tuple[List[RoomMask], dict]:
    free = np.asarray(structural_free_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    structural_obstacles = np.asarray(structural_obstacle_mask, dtype=bool) if structural_obstacle_mask is not None else ~free & ~unknown
    if not np.any(free):
        debug = _empty_room_segmentation_debug()
        return ([], debug) if return_debug else []
    min_cells = max(1, int(config.min_observed_free_cells))
    min_area_cells = max(1, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    components = _connected_components(free)
    if not components:
        debug = _empty_room_segmentation_debug()
        return ([], debug) if return_debug else []
    if len(components) == 1:
        component_list = components
    else:
        component_list = [comp for comp in components if len(comp) >= min_cells and len(comp) >= min_area_cells]
        if not component_list:
            component_list = [max(components, key=len)]

    all_rooms: List[RoomMask] = []
    merged_debug: dict = {
        "proposal_room_count": 0,
        "final_room_count": 0,
        "proposal_room_masks": [],
        "merge_operations": [],
        "doorway_edges": [],
        "functional_split_edges": [],
        "adjacency_evidence": [],
        "adjacency_decisions": [],
    }
    label_offset = 0
    for component in component_list:
        comp_mask = np.zeros_like(free, dtype=bool)
        for row, col in component:
            comp_mask[row, col] = True
        labels, distance_m = _watershed_component(comp_mask, config)
        proposal_count = len([v for v in np.unique(labels) if int(v) > 0])
        if str(config.finalization_mode).strip().lower() == "doorway_constrained_merge":
            labels, finalization_debug, doorway_edges = merge_open_plan_proposals(
                proposal_labels=labels,
                structural_free_mask=free,
                structural_obstacle_mask=structural_obstacles,
                unknown_mask=unknown,
                distance_m=distance_m,
                config=config,
                proposal_semantic_labels=proposal_semantic_labels,
            )
        else:
            labels, doorway_edges = _refine_labels_by_doorways(labels, distance_m, config)
            finalization_debug = {
                "proposal_room_count": proposal_count,
                "final_room_count": len([v for v in np.unique(labels) if int(v) > 0]),
                "proposal_room_masks": _proposal_masks_debug(labels),
                "merge_operations": [],
                "doorway_edges": list(doorway_edges),
                "adjacency_evidence": [],
            }
        merged_debug["proposal_room_count"] += int(finalization_debug.get("proposal_room_count", proposal_count) or 0)
        merged_debug["merge_operations"].extend(list(finalization_debug.get("merge_operations") or []))
        merged_debug["doorway_edges"].extend(list(finalization_debug.get("doorway_edges") or []))
        merged_debug["functional_split_edges"].extend(list(finalization_debug.get("functional_split_edges") or []))
        merged_debug["adjacency_evidence"].extend(list(finalization_debug.get("adjacency_evidence") or []))
        for item in list(finalization_debug.get("proposal_room_masks") or []):
            proposal_item = dict(item)
            proposal_item["component_index"] = int(len(merged_debug["proposal_room_masks"]) + 1)
            merged_debug["proposal_room_masks"].append(proposal_item)
        for label_id in sorted(v for v in np.unique(labels) if int(v) > 0):
            mask = labels == label_id
            if not np.any(mask):
                continue
            if int(np.count_nonzero(mask)) < min_cells and len(all_rooms) > 0:
                continue
            room = _room_from_mask("pending", mask, unknown, doorway_edges, config, step)
            room.metadata["label_id"] = int(label_id + label_offset)
            room.metadata["source_finalization_mode"] = str(config.finalization_mode)
            room.metadata["proposal_labels"] = sorted(
                int(v) for v in np.unique(labels[mask]) if int(v) > 0
            )
            all_rooms.append(room)
        label_offset += max([int(v) for v in np.unique(labels) if int(v) > 0] or [0])
    if not all_rooms and np.any(free):
        all_rooms.append(_room_from_mask("pending", free, unknown, [], config, step))
    merged_debug["final_room_count"] = int(len(all_rooms))
    if not merged_debug["proposal_room_masks"]:
        merged_debug["proposal_room_masks"] = _proposal_masks_debug(free.astype(np.int32))
        merged_debug["proposal_room_count"] = int(len(merged_debug["proposal_room_masks"]))
    return (all_rooms, merged_debug) if return_debug else all_rooms


def merge_open_plan_proposals(
    proposal_labels: np.ndarray,
    structural_free_mask: np.ndarray,
    structural_obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    distance_m: np.ndarray,
    config: RoomSegmentationConfig,
    proposal_semantic_labels: Optional[Mapping[int, str]] = None,
) -> Tuple[np.ndarray, dict, List[dict]]:
    labels = np.asarray(proposal_labels, dtype=np.int32).copy()
    positive = [int(v) for v in np.unique(labels) if int(v) > 0]
    proposal_debug = _proposal_masks_debug(labels)
    if len(positive) <= 1:
        debug = {
            "proposal_room_count": int(len(positive)),
            "final_room_count": int(len(positive)),
            "proposal_room_masks": proposal_debug,
            "merge_operations": [],
            "doorway_edges": [],
            "adjacency_evidence": [],
        }
        return labels, debug, []

    adjacency = _proposal_adjacency(labels)
    parent = {label: label for label in positive}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        root = min(ra, rb)
        other = max(ra, rb)
        parent[other] = root
        return root

    merge_operations: List[dict] = []
    doorway_edges: List[dict] = []
    functional_split_edges: List[dict] = []
    evidence_rows: List[dict] = []
    semantic_labels = {int(k): _semantic_label_info(v, config) for k, v in dict(proposal_semantic_labels or {}).items()}
    for (a, b), boundary_cells in sorted(adjacency.items()):
        evidence = compute_room_adjacency_evidence(
            labels,
            int(a),
            int(b),
            boundary_cells,
            structural_free_mask,
            structural_obstacle_mask,
            unknown_mask,
            distance_m,
            config,
        )
        reason = evidence.merge_reason
        left_info = semantic_labels.get(int(a), _semantic_label_info("", config))
        right_info = semantic_labels.get(int(b), _semantic_label_info("", config))
        verified_structural = bool(evidence.verified_doorway)
        functional_split = False if verified_structural else should_keep_open_plan_functional_split(left_info, right_info, config)
        decision = "keep_split" if verified_structural or functional_split else "merge"
        if functional_split:
            reason = "reliable_different_room_types"
            evidence.merge_reason = reason
        elif not verified_structural:
            sem_a = str(left_info["category"])
            sem_b = str(right_info["category"])
            if (
                bool(config.merge_same_semantic_category)
                and sem_a
                and sem_b
                and sem_a == sem_b
                and sem_a not in {"unknown", "unknown_room"}
            ):
                reason = "open_plan_no_verified_doorway_same_semantic_%s" % sem_a
                evidence.merge_reason = reason
            elif (
                bool(config.merge_unknown_into_open_labeled_region)
                and {sem_a, sem_b} & {"unknown", "unknown_room", ""}
                and (sem_a or sem_b)
            ):
                reason = "open_unknown_into_open_labeled_region"
                evidence.merge_reason = reason
        evidence_payload = evidence.to_dict()
        evidence_payload.update(
            {
                "left": "proposal_%s" % int(a),
                "right": "proposal_%s" % int(b),
                "verified_structural_boundary": verified_structural,
                "left_premerge_category": left_info["category"],
                "right_premerge_category": right_info["category"],
                "left_reliable": bool(left_info["is_reliable"]),
                "right_reliable": bool(right_info["is_reliable"]),
                "left_label_reliability": float(left_info["label_reliability"]),
                "right_label_reliability": float(right_info["label_reliability"]),
                "decision": decision,
                "reason": reason,
            }
        )
        evidence_rows.append(evidence_payload)
        if evidence.verified_doorway:
            doorway_edges.append(
                {
                    "room_a_label": int(a),
                    "room_b_label": int(b),
                    "edge_type": "adjacent_via_doorway",
                    "doorway_width_m": float(evidence.neck_width_m),
                    "boundary_mean_clearance_m": float(evidence.mean_clearance_m),
                    "source": "doorway_constrained_merge",
                    "boundary_cells_sample": evidence.boundary_cells_sample,
                    "confidence": float(
                        np.clip(
                            min(evidence.wall_support_left, evidence.wall_support_right)
                            * (1.0 - evidence.unknown_support_ratio),
                            0.0,
                            1.0,
                        )
                    ),
                }
            )
            continue
        if functional_split:
            functional_split_edges.append(
                {
                    "room_a_label": int(a),
                    "room_b_label": int(b),
                    "edge_type": "adjacent_open_plan_functional_split",
                    "source": "premerge_room_recognition",
                    "confidence": float(min(left_info["label_reliability"], right_info["label_reliability"])),
                    "left_premerge_category": left_info["category"],
                    "right_premerge_category": right_info["category"],
                    "reason": reason,
                }
            )
            continue
        if bool(config.open_boundary_merge):
            root = union(int(a), int(b))
            merge_operations.append(
                {
                    "from_labels": [int(a), int(b)],
                    "to_label": int(root),
                    "to_room_id": None,
                    "reason": reason,
                    "boundary_cells": int(evidence.boundary_cells),
                    "boundary_cells_sample": evidence.boundary_cells_sample,
                }
            )

    root_to_final: Dict[int, int] = {}
    next_label = 1
    out = np.zeros_like(labels, dtype=np.int32)
    final_members: Dict[int, List[int]] = {}
    for label in positive:
        root = find(label)
        if root not in root_to_final:
            root_to_final[root] = next_label
            next_label += 1
        final = root_to_final[root]
        out[labels == label] = final
        final_members.setdefault(final, []).append(label)
    for op in merge_operations:
        root = find(int(op["to_label"]))
        op["to_label"] = int(root_to_final.get(root, root))
        op["from_labels"] = sorted({int(v) for v in final_members.get(int(op["to_label"]), op["from_labels"])})
    remapped_edges = []
    for edge in doorway_edges:
        a = root_to_final.get(find(int(edge["room_a_label"])))
        b = root_to_final.get(find(int(edge["room_b_label"])))
        if a is None or b is None or a == b:
            continue
        remapped_edges.append({**edge, "room_a_label": int(a), "room_b_label": int(b)})
    debug = {
        "proposal_mode": str(config.proposal_mode),
        "finalization_mode": str(config.finalization_mode),
        "proposal_room_count": int(len(positive)),
        "final_room_count": int(len([v for v in np.unique(out) if int(v) > 0])),
        "proposal_room_masks": proposal_debug,
        "merge_operations": merge_operations,
        "doorway_edges": remapped_edges,
        "functional_split_edges": functional_split_edges,
        "adjacency_evidence": evidence_rows,
        "adjacency_decisions": evidence_rows,
    }
    return out, debug, remapped_edges


def compute_room_adjacency_evidence(
    labels: np.ndarray,
    a: int,
    b: int,
    boundary_cells: Sequence[GridCell],
    structural_free_mask: np.ndarray,
    structural_obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    distance_m: np.ndarray,
    config: RoomSegmentationConfig,
) -> RoomAdjacencyEvidence:
    unique_cells = sorted({(int(r), int(c)) for r, c in boundary_cells})
    boundary_count = int(len(unique_cells))
    resolution = float(config.resolution_m)
    neck_width_m = max(resolution, math.sqrt(float(max(1, boundary_count))) * resolution)
    clearances = [float(distance_m[cell]) for cell in unique_cells if np.isfinite(float(distance_m[cell]))]
    min_clearance = float(np.min(clearances)) if clearances else 0.0
    mean_clearance = float(np.mean(clearances)) if clearances else 0.0
    boundary_mask = np.zeros_like(labels, dtype=bool)
    for row, col in unique_cells:
        if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
            boundary_mask[row, col] = True
    support_radius = max(1, _radius_cells(config.doorway_endpoint_wall_distance_m, resolution))
    support_region = _dilate(boundary_mask, support_radius)
    support_area = max(1, int(np.count_nonzero(support_region)))
    obstacle_support_ratio = float(np.count_nonzero(support_region & structural_obstacle_mask)) / float(support_area)
    unknown_support_ratio = float(np.count_nonzero(support_region & unknown_mask)) / float(support_area)
    endpoint_a, endpoint_b = _boundary_endpoints(unique_cells)
    wall_support_left = _endpoint_wall_support(endpoint_a, structural_obstacle_mask, support_radius)
    wall_support_right = _endpoint_wall_support(endpoint_b, structural_obstacle_mask, support_radius)
    endpoints_touch_structural_wall = bool(wall_support_left > 0.0 and wall_support_right > 0.0)
    separates_large_regions = _closing_boundary_separates(labels, int(a), int(b), unique_cells, structural_free_mask, config)
    width_ok = float(config.doorway_width_min_m) <= neck_width_m <= float(config.doorway_width_max_m)
    clearance_ok = mean_clearance <= float(config.doorway_clearance_max_m)
    unknown_ok = unknown_support_ratio <= float(config.doorway_unknown_support_max_ratio)
    wall_ok = endpoints_touch_structural_wall and max(wall_support_left, wall_support_right) >= float(config.doorway_wall_support_min_ratio)
    verified = bool(width_ok and clearance_ok and unknown_ok and wall_ok and separates_large_regions)
    if verified:
        reason = "verified_structural_doorway"
    elif not unknown_ok:
        reason = "unknown_supported_boundary_not_wall"
    elif not wall_ok:
        reason = "open_plan_no_verified_doorway_poor_wall_support"
    elif not width_ok and neck_width_m > float(config.doorway_width_max_m):
        reason = "wide_open_boundary"
    elif not separates_large_regions:
        reason = "boundary_does_not_separate_rooms"
    else:
        reason = "open_plan_no_verified_doorway"
    return RoomAdjacencyEvidence(
        room_a_label=int(a),
        room_b_label=int(b),
        boundary_cells=boundary_count,
        boundary_length_m=float(boundary_count) * resolution,
        neck_width_m=float(neck_width_m),
        min_clearance_m=float(min_clearance),
        mean_clearance_m=float(mean_clearance),
        wall_support_left=float(wall_support_left),
        wall_support_right=float(wall_support_right),
        obstacle_support_ratio=float(obstacle_support_ratio),
        unknown_support_ratio=float(unknown_support_ratio),
        endpoints_touch_structural_wall=bool(endpoints_touch_structural_wall),
        separates_large_regions=bool(separates_large_regions),
        verified_doorway=bool(verified),
        merge_reason=reason,
        boundary_cells_sample=[list(map(int, cell)) for cell in _sample_cells(unique_cells, 96)],
    )


def should_keep_open_plan_functional_split(left_label: Mapping[str, object], right_label: Mapping[str, object], config: RoomSegmentationConfig) -> bool:
    left_category = str(left_label.get("category", "")).strip().lower()
    right_category = str(right_label.get("category", "")).strip().lower()
    unknowns = {"", "unknown", "unknown_room"}
    if not bool(config.use_premerge_labels_for_open_plan_merge):
        return False
    if not bool(config.unknown_allows_functional_split) and (left_category in unknowns or right_category in unknowns):
        return False
    if not bool(left_label.get("is_reliable", False)) or not bool(right_label.get("is_reliable", False)):
        return False
    if left_category == right_category:
        return False
    return True


def _semantic_label_info(value: object, config: RoomSegmentationConfig) -> dict:
    category = getattr(value, "category", None)
    if category is None and isinstance(value, Mapping):
        category = value.get("category")
    if category is None:
        category = str(value or "")
    category = str(category).strip().lower().replace(" ", "_")
    if category in {"unknown_room", ""}:
        category = "unknown" if category else ""
    reliability = getattr(value, "label_reliability", None)
    if reliability is None and isinstance(value, Mapping):
        reliability = value.get("label_reliability", value.get("confidence"))
    if reliability is None:
        reliability = 1.0 if category and category != "unknown" else 0.0
    is_reliable = getattr(value, "is_reliable", None)
    if is_reliable is None and isinstance(value, Mapping):
        is_reliable = value.get("is_reliable")
    if is_reliable is None:
        is_reliable = float(reliability) >= float(config.min_label_reliability_for_functional_split)
    if category in {"", "unknown"}:
        is_reliable = False
    return {
        "category": category,
        "label_reliability": float(np.clip(float(reliability), 0.0, 1.0)),
        "is_reliable": bool(is_reliable),
    }


def assign_objects_to_room_masks(
    object_nodes: Iterable[object],
    room_masks: Sequence[RoomMask],
    map_info: MapInfo,
    overlap_ambiguity_margin: float = 0.10,
) -> Dict[str, ObjectRoomAssignment]:
    rooms = [room for room in room_masks if not room.stale]
    assignments: Dict[str, ObjectRoomAssignment] = {}
    if not rooms:
        return assignments
    for obj in object_nodes:
        object_id = str(getattr(obj, "id", getattr(obj, "node_id", "")))
        center = np.asarray(getattr(obj, "center_world", [math.nan, math.nan, 0.0]), dtype=np.float32)
        centroid_inside = False
        centroid_room: Optional[str] = None
        if len(center) >= 2 and np.all(np.isfinite(center[:2])):
            row, col = world_xy_to_grid(float(center[0]), float(center[1]), map_info)
            if is_inside_grid(row, col, map_info):
                for room in rooms:
                    if bool(room.mask[row, col]):
                        centroid_inside = True
                        centroid_room = room.room_id
                        break
        object_cells = _object_cells(obj, map_info)
        overlap_scores: List[Tuple[float, str]] = []
        if object_cells:
            denom = max(1, len(object_cells))
            for room in rooms:
                hits = sum(1 for row, col in object_cells if 0 <= row < room.mask.shape[0] and 0 <= col < room.mask.shape[1] and room.mask[row, col])
                overlap_scores.append((float(hits) / float(denom), room.room_id))
        elif centroid_room is not None:
            overlap_scores.append((1.0, centroid_room))
        overlap_scores.sort(reverse=True)
        if centroid_room is not None:
            best_score = next((score for score, rid in overlap_scores if rid == centroid_room), 1.0)
            best_room = centroid_room
        elif overlap_scores:
            best_score, best_room = overlap_scores[0]
        else:
            assignments[object_id] = ObjectRoomAssignment(object_id=object_id, room_id=None)
            continue
        if best_score <= 0.0 and not centroid_inside:
            assignments[object_id] = ObjectRoomAssignment(object_id=object_id, room_id=None)
            continue
        second = overlap_scores[1][0] if len(overlap_scores) > 1 else 0.0
        ambiguous = bool((best_score - second) < float(overlap_ambiguity_margin) and second > 0.0)
        confidence = 1.0 if centroid_inside else float(best_score)
        if ambiguous:
            confidence = min(confidence, max(0.0, best_score - second + 0.5))
        assignments[object_id] = ObjectRoomAssignment(
            object_id=object_id,
            room_id=best_room,
            centroid_inside=centroid_inside,
            mask_overlap_ratio=float(best_score),
            assignment_confidence=float(np.clip(confidence, 0.0, 1.0)),
            ambiguous=ambiguous,
        )
    return assignments


def room_segmentation_debug(
    room_masks: Sequence[RoomMask],
    structural_free_mask: Optional[np.ndarray] = None,
    segmentation_debug: Optional[Mapping[str, object]] = None,
    structural_obstacle_debug: Optional[Mapping[str, object]] = None,
) -> dict:
    live = [room for room in room_masks if not room.stale]
    base = {
        "source": "online_geometry_watershed",
        "proposal_mode": "distance_watershed",
        "finalization_mode": "doorway_constrained_merge",
        "room_count": int(len(live)),
        "final_room_count": int(len(live)),
        "proposal_room_count": int(len(live)),
        "merge_operations": [],
        "doorway_edges": [],
        "adjacency_evidence": [],
        "structural_free_cells": int(np.count_nonzero(structural_free_mask)) if structural_free_mask is not None else None,
        "rooms": [room_mask_to_dict(room, include_mask=False) for room in room_masks],
    }
    if segmentation_debug:
        base.update({key: value for key, value in dict(segmentation_debug).items() if key != "rooms"})
        base["final_room_count"] = int(len(live))
        base["room_count"] = int(len(live))
    if structural_obstacle_debug:
        base["structural_obstacle_mask"] = dict(structural_obstacle_debug)
    _attach_room_ids_to_merge_debug(base, live)
    return base


def room_mask_to_dict(room: RoomMask, include_mask: bool = False) -> dict:
    payload = {
        "room_id": room.room_id,
        "area_m2": float(room.area_m2),
        "centroid_xy": [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
        "boundary_unknown_fraction": float(room.boundary_unknown_fraction),
        "observed_free_cells": int(room.observed_free_cells),
        "mask_confidence": float(room.mask_confidence),
        "confidence": float(room.confidence),
        "is_partial": bool(room.is_partial),
        "source": room.source,
        "stale": bool(room.stale),
        "doorway_edges": list(room.doorway_edges),
        "metadata": dict(room.metadata),
    }
    if include_mask:
        payload["mask"] = np.asarray(room.mask, dtype=bool).astype(np.uint8).tolist()
    return payload


def _attach_room_ids_to_merge_debug(debug: dict, rooms: Sequence[RoomMask]) -> None:
    label_to_room = {
        int(room.metadata["label_id"]): room.room_id
        for room in rooms
        if isinstance(room.metadata, Mapping) and room.metadata.get("label_id") is not None
    }
    for op in list(debug.get("merge_operations") or []):
        if not isinstance(op, dict):
            continue
        to_label = op.get("to_label")
        if to_label is not None:
            op["to_room_id"] = label_to_room.get(int(to_label), op.get("to_room_id"))
    for edge in list(debug.get("doorway_edges") or []):
        if not isinstance(edge, dict):
            continue
        if edge.get("room_a_label") is not None:
            edge["room_a"] = label_to_room.get(int(edge["room_a_label"]))
        if edge.get("room_b_label") is not None:
            edge["room_b"] = label_to_room.get(int(edge["room_b_label"]))


def _watershed_component(component_mask: np.ndarray, config: RoomSegmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy import ndimage as ndi

        distance_cells = ndi.distance_transform_edt(component_mask)
        if float(config.distance_smooth_sigma_cells) > 0:
            distance_cells = ndi.gaussian_filter(distance_cells, sigma=float(config.distance_smooth_sigma_cells))
    except Exception:
        distance_cells = _distance_transform_fallback(component_mask)
    distance_m = np.asarray(distance_cells, dtype=np.float32) * float(config.resolution_m)
    seeds = _seed_points(component_mask, distance_m, config)
    markers = np.zeros_like(component_mask, dtype=np.int32)
    for idx, (row, col) in enumerate(seeds, start=1):
        markers[row, col] = idx
    if not seeds:
        return component_mask.astype(np.int32), distance_m
    try:
        from skimage.segmentation import watershed

        labels = watershed(-distance_m, markers=markers, mask=component_mask).astype(np.int32)
    except Exception:
        labels = _seeded_region_grow(component_mask, distance_m, seeds)
    if len([v for v in np.unique(labels) if int(v) > 0]) <= 1:
        bottleneck = _split_by_structural_bottleneck(component_mask, config)
        if bottleneck is not None:
            labels = bottleneck
    return labels, distance_m


def _split_by_structural_bottleneck(mask: np.ndarray, config: RoomSegmentationConfig) -> Optional[np.ndarray]:
    """Fallback split for classic wall-with-doorway geometry.

    Watershed can collapse two rooms connected by a narrow doorway into a single
    basin when the distance field has one broad maximum. This deterministic
    refinement detects a low-free-width column/row inside a large component and
    splits across that bottleneck while keeping doorway cells adjacent.
    """
    src = np.asarray(mask, dtype=bool)
    rr, cc = np.nonzero(src)
    if rr.size == 0:
        return None
    r0, r1 = int(np.min(rr)), int(np.max(rr)) + 1
    c0, c1 = int(np.min(cc)), int(np.max(cc)) + 1
    sub = src[r0:r1, c0:c1]
    min_side_cells = max(4, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9) * 0.25)))

    def split_vertical() -> Optional[np.ndarray]:
        sums = np.sum(sub, axis=0)
        if sums.size < 8 or float(np.max(sums)) <= 0.0:
            return None
        candidates = np.where((sums > 0) & (sums <= 0.45 * float(np.max(sums))))[0]
        best = None
        for col in candidates:
            left = int(np.count_nonzero(sub[:, : col + 1]))
            right = int(np.count_nonzero(sub[:, col + 1 :]))
            if left >= min_side_cells and right >= min_side_cells:
                score = float(sums[col]) + abs(left - right) * 1e-4
                if best is None or score < best[0]:
                    best = (score, int(col))
        if best is None:
            return None
        col = best[1]
        out = np.zeros_like(src, dtype=np.int32)
        left_mask = np.zeros_like(sub, dtype=bool)
        right_mask = np.zeros_like(sub, dtype=bool)
        left_mask[:, : col + 1] = sub[:, : col + 1]
        right_mask[:, col + 1 :] = sub[:, col + 1 :]
        if not np.any(right_mask):
            return None
        view = out[r0:r1, c0:c1]
        view[left_mask] = 1
        view[right_mask] = 2
        return out

    def split_horizontal() -> Optional[np.ndarray]:
        sums = np.sum(sub, axis=1)
        if sums.size < 8 or float(np.max(sums)) <= 0.0:
            return None
        candidates = np.where((sums > 0) & (sums <= 0.45 * float(np.max(sums))))[0]
        best = None
        for row in candidates:
            top = int(np.count_nonzero(sub[: row + 1, :]))
            bottom = int(np.count_nonzero(sub[row + 1 :, :]))
            if top >= min_side_cells and bottom >= min_side_cells:
                score = float(sums[row]) + abs(top - bottom) * 1e-4
                if best is None or score < best[0]:
                    best = (score, int(row))
        if best is None:
            return None
        row = best[1]
        out = np.zeros_like(src, dtype=np.int32)
        top_mask = np.zeros_like(sub, dtype=bool)
        bottom_mask = np.zeros_like(sub, dtype=bool)
        top_mask[: row + 1, :] = sub[: row + 1, :]
        bottom_mask[row + 1 :, :] = sub[row + 1 :, :]
        if not np.any(bottom_mask):
            return None
        view = out[r0:r1, c0:c1]
        view[top_mask] = 1
        view[bottom_mask] = 2
        return out

    vertical = split_vertical()
    horizontal = split_horizontal()
    if vertical is None:
        return horizontal
    if horizontal is None:
        return vertical
    # Prefer the split whose two sides are more balanced.
    v_counts = [np.count_nonzero(vertical == 1), np.count_nonzero(vertical == 2)]
    h_counts = [np.count_nonzero(horizontal == 1), np.count_nonzero(horizontal == 2)]
    return vertical if abs(v_counts[0] - v_counts[1]) <= abs(h_counts[0] - h_counts[1]) else horizontal


def _seed_points(mask: np.ndarray, distance_m: np.ndarray, config: RoomSegmentationConfig) -> List[GridCell]:
    candidates: List[GridCell] = []
    try:
        from skimage.feature import peak_local_max

        min_distance_cells = max(1, _radius_cells(config.seed_min_distance_m, config.resolution_m))
        coords = peak_local_max(
            distance_m,
            min_distance=min_distance_cells,
            threshold_abs=float(config.seed_min_clearance_m),
            labels=mask.astype(np.uint8),
            exclude_border=False,
        )
        candidates = [(int(row), int(col)) for row, col in coords]
    except Exception:
        candidates = _local_maxima(mask, distance_m, float(config.seed_min_clearance_m))
    if not candidates:
        rr, cc = np.nonzero(mask)
        if rr.size:
            center_idx = int(np.argmin((rr - float(np.mean(rr))) ** 2 + (cc - float(np.mean(cc))) ** 2))
            candidates = [(int(rr[center_idx]), int(cc[center_idx]))]
    candidates.sort(key=lambda cell: float(distance_m[cell]), reverse=True)
    seeds: List[GridCell] = []
    min_cells = max(1.0, float(config.seed_min_distance_m) / max(float(config.resolution_m), 1e-9))
    for cell in candidates:
        if float(distance_m[cell]) < float(config.seed_min_clearance_m) and seeds:
            continue
        if all(math.hypot(cell[0] - other[0], cell[1] - other[1]) >= min_cells for other in seeds):
            seeds.append(cell)
    return seeds


def _refine_labels_by_doorways(labels: np.ndarray, distance_m: np.ndarray, config: RoomSegmentationConfig) -> Tuple[np.ndarray, List[dict]]:
    labels = np.asarray(labels, dtype=np.int32).copy()
    positive = [int(v) for v in np.unique(labels) if int(v) > 0]
    if len(positive) <= 1:
        return labels, []
    adjacency: Dict[Tuple[int, int], List[GridCell]] = {}
    h, w = labels.shape
    for row in range(h):
        for col in range(w):
            a = int(labels[row, col])
            if a <= 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = row + dr, col + dc
                if rr >= h or cc >= w:
                    continue
                b = int(labels[rr, cc])
                if b <= 0 or b == a:
                    continue
                key = tuple(sorted((a, b)))
                adjacency.setdefault(key, []).append((row, col))
                adjacency.setdefault(key, []).append((rr, cc))
    parent = {label: label for label in positive}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    areas = {label: int(np.count_nonzero(labels == label)) * float(config.resolution_m) ** 2 for label in positive}
    doorway_edges: List[dict] = []
    for (a, b), cells in sorted(adjacency.items()):
        unique_cells = sorted(set(cells))
        width_m = max(float(config.resolution_m), math.sqrt(float(len(unique_cells))) * float(config.resolution_m))
        clearances = [float(distance_m[cell]) for cell in unique_cells if np.isfinite(float(distance_m[cell]))]
        mean_clearance = float(np.mean(clearances)) if clearances else 0.0
        likely_doorway = (
            float(config.doorway_width_min_m) <= width_m <= float(config.doorway_width_max_m)
            and mean_clearance <= float(config.doorway_clearance_max_m)
        )
        tiny = min(float(areas.get(a, 0.0)), float(areas.get(b, 0.0))) < float(config.small_segment_merge_area_m2)
        wide_open = bool(config.merge_wide_openings and width_m > float(config.doorway_width_max_m))
        if likely_doorway and not tiny:
            doorway_edges.append(
                {
                    "room_a_label": int(a),
                    "room_b_label": int(b),
                    "edge_type": "adjacent_via_doorway",
                    "doorway_width_m": float(width_m),
                    "boundary_mean_clearance_m": float(mean_clearance),
                    "source": "online_geometry_watershed_boundary",
                }
            )
        else:
            if tiny or wide_open:
                union(a, b)
    remap: Dict[int, int] = {}
    next_label = 1
    out = np.zeros_like(labels, dtype=np.int32)
    for label in positive:
        root = find(label)
        if root not in remap:
            remap[root] = next_label
            next_label += 1
        out[labels == label] = remap[root]
    remapped_edges = []
    for edge in doorway_edges:
        a = remap.get(find(int(edge["room_a_label"])))
        b = remap.get(find(int(edge["room_b_label"])))
        if a is None or b is None or a == b:
            continue
        remapped_edges.append({**edge, "room_a_label": int(a), "room_b_label": int(b)})
    return out, remapped_edges


def _room_from_mask(room_id: str, mask: np.ndarray, unknown: np.ndarray, doorway_edges: Sequence[dict], config: RoomSegmentationConfig, step: int) -> RoomMask:
    mask_bool = np.asarray(mask, dtype=bool)
    rr, cc = np.nonzero(mask_bool)
    centroid_grid = (float(np.mean(rr)), float(np.mean(cc))) if rr.size else (0.0, 0.0)
    if config.map_info is not None and rr.size:
        cx, cy = grid_to_world_xy(int(round(centroid_grid[0])), int(round(centroid_grid[1])), config.map_info)
        centroid_xy = (float(cx), float(cy))
    else:
        centroid_xy = (float(centroid_grid[1]) * float(config.resolution_m), float(centroid_grid[0]) * float(config.resolution_m))
    cells = int(np.count_nonzero(mask_bool))
    boundary = _boundary(mask_bool)
    boundary_count = int(np.count_nonzero(boundary))
    unknown_fraction = float(np.count_nonzero(boundary & unknown)) / float(max(1, boundary_count))
    area_m2 = float(cells) * float(config.resolution_m) ** 2
    partial = bool(unknown_fraction >= 0.35 or cells < int(config.min_observed_free_cells))
    confidence = 1.0
    if bool(config.unknown_boundary_confidence_penalty):
        confidence -= 0.5 * unknown_fraction
    if partial:
        confidence -= 0.15
    confidence = float(np.clip(confidence, 0.05, 1.0))
    mask_confidence = confidence
    return RoomMask(
        room_id=room_id,
        mask=mask_bool,
        centroid_xy=centroid_xy,
        area_m2=area_m2,
        boundary_unknown_fraction=unknown_fraction,
        doorway_edges=[dict(edge) for edge in doorway_edges],
        confidence=confidence,
        source="online_geometry_watershed"
        if "watershed" in str(getattr(config, "algorithm", "")).lower()
        else "voxel_occupancy_door_wall_v33",
        observed_free_cells=cells,
        mask_confidence=mask_confidence,
        is_partial=partial,
        step=int(step),
        metadata={"centroid_grid": [float(centroid_grid[0]), float(centroid_grid[1])]},
    )


def _object_cells(obj: object, map_info: MapInfo) -> List[GridCell]:
    cells = set()
    center_grid = getattr(obj, "center_grid", None)
    if center_grid is not None:
        cells.add((int(center_grid[0]), int(center_grid[1])))
    points = getattr(obj, "point_cloud_world", None)
    if points is not None:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            for x, y in arr[:, :2]:
                row, col = world_xy_to_grid(float(x), float(y), map_info)
                if is_inside_grid(row, col, map_info):
                    cells.add((int(row), int(col)))
    bbox = getattr(obj, "bbox_world", None)
    if bbox is not None:
        arr = np.asarray(bbox, dtype=np.float32)
        if arr.shape == (2, 3):
            xs = np.linspace(float(arr[0, 0]), float(arr[1, 0]), 3)
            ys = np.linspace(float(arr[0, 1]), float(arr[1, 1]), 3)
            for x in xs:
                for y in ys:
                    row, col = world_xy_to_grid(float(x), float(y), map_info)
                    if is_inside_grid(row, col, map_info):
                        cells.add((int(row), int(col)))
    return sorted(cells)


def _object_clutter_mask(shape: Tuple[int, int], object_memory: Optional[Iterable[object]]) -> Tuple[np.ndarray, Dict[GridCell, str]]:
    out = np.zeros(shape, dtype=bool)
    categories: Dict[GridCell, str] = {}
    if object_memory is None:
        return out, categories
    source = getattr(object_memory, "nodes", object_memory)
    for obj in list(source or []):
        category = str(getattr(obj, "category", getattr(obj, "caption", ""))).strip().lower().replace(" ", "_")
        if category not in MOVABLE_ROOM_CLUTTER_CATEGORIES:
            continue
        cells: set[GridCell] = set()
        center_grid = getattr(obj, "center_grid", None)
        if center_grid is not None:
            cells.add((int(center_grid[0]), int(center_grid[1])))
        mask = getattr(obj, "last_mask", None)
        if mask is not None:
            arr = np.asarray(mask, dtype=bool)
            if arr.shape == tuple(shape):
                for row, col in zip(*np.nonzero(arr)):
                    cells.add((int(row), int(col)))
        for row, col in cells:
            for dr, dc in _disk_offsets(2):
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    out[rr, cc] = True
                    categories[(rr, cc)] = category
    return out, categories


def _proposal_adjacency(labels: np.ndarray) -> Dict[Tuple[int, int], List[GridCell]]:
    arr = np.asarray(labels, dtype=np.int32)
    adjacency: Dict[Tuple[int, int], List[GridCell]] = {}
    h, w = arr.shape
    for row in range(h):
        for col in range(w):
            a = int(arr[row, col])
            if a <= 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = row + dr, col + dc
                if rr >= h or cc >= w:
                    continue
                b = int(arr[rr, cc])
                if b <= 0 or b == a:
                    continue
                key = tuple(sorted((a, b)))
                adjacency.setdefault(key, []).append((row, col))
                adjacency.setdefault(key, []).append((rr, cc))
    return adjacency


def _proposal_masks_debug(labels: np.ndarray) -> List[dict]:
    arr = np.asarray(labels, dtype=np.int32)
    out = []
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        mask = arr == label
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            continue
        out.append(
            {
                "label_id": int(label),
                "cell_count": int(rr.size),
                "bbox": [int(np.min(rr)), int(np.min(cc)), int(np.max(rr)) + 1, int(np.max(cc)) + 1],
                "mask": mask.astype(np.uint8).tolist(),
            }
        )
    return out


def _empty_room_segmentation_debug() -> dict:
    return {
        "proposal_mode": "distance_watershed",
        "finalization_mode": "doorway_constrained_merge",
        "proposal_room_count": 0,
        "final_room_count": 0,
        "proposal_room_masks": [],
        "merge_operations": [],
        "doorway_edges": [],
        "functional_split_edges": [],
        "adjacency_evidence": [],
        "adjacency_decisions": [],
    }


def _boundary_endpoints(cells: Sequence[GridCell]) -> Tuple[GridCell, GridCell]:
    if not cells:
        return (0, 0), (0, 0)
    if len(cells) == 1:
        return cells[0], cells[0]
    arr = np.asarray(cells, dtype=np.float32)
    span_r = float(np.max(arr[:, 0]) - np.min(arr[:, 0]))
    span_c = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
    order_axis = 0 if span_r >= span_c else 1
    order = np.argsort(arr[:, order_axis])
    first = cells[int(order[0])]
    last = cells[int(order[-1])]
    return first, last


def _endpoint_wall_support(endpoint: GridCell, structural_obstacle_mask: np.ndarray, radius_cells: int) -> float:
    row, col = int(endpoint[0]), int(endpoint[1])
    h, w = structural_obstacle_mask.shape
    cells = 0
    hits = 0
    for dr, dc in _disk_offsets(max(1, int(radius_cells))):
        rr, cc = row + dr, col + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells += 1
            if structural_obstacle_mask[rr, cc]:
                hits += 1
    if cells <= 0 or hits <= 0:
        return 0.0
    return 1.0


def _closing_boundary_separates(
    labels: np.ndarray,
    a: int,
    b: int,
    boundary_cells: Sequence[GridCell],
    structural_free_mask: np.ndarray,
    config: RoomSegmentationConfig,
) -> bool:
    region = (labels == int(a)) | (labels == int(b))
    if not np.any(region):
        return False
    closed = np.asarray(structural_free_mask, dtype=bool) & region
    for row, col in boundary_cells:
        if 0 <= row < closed.shape[0] and 0 <= col < closed.shape[1]:
            closed[row, col] = False
    components = _connected_components(closed)
    min_cells = max(1, int(round(float(config.small_segment_merge_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    large = [comp for comp in components if len(comp) >= min_cells]
    return len(large) >= 2


def _sample_cells(cells: Sequence[GridCell], max_cells: int) -> List[GridCell]:
    src = list(cells)
    if len(src) <= int(max_cells):
        return src
    stride = max(1, len(src) // int(max_cells))
    return src[::stride][: int(max_cells)]


def _connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    src = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    components: List[List[GridCell]] = []
    for row, col in zip(*np.nonzero(src)):
        start = (int(row), int(col))
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[GridCell] = deque([start])
        comp: List[GridCell] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = cur[0] + dr, cur[1] + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or visited[rr, cc] or not src[rr, cc]:
                    continue
                visited[rr, cc] = True
                queue.append((rr, cc))
        components.append(comp)
    return components


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    try:
        from skimage.morphology import binary_closing, disk

        return binary_closing(mask, disk(radius)).astype(bool)
    except Exception:
        return _erode(_dilate(mask, radius), radius)


def _binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    try:
        from skimage.morphology import binary_opening, disk

        return binary_opening(mask, disk(radius)).astype(bool)
    except Exception:
        return _dilate(_erode(mask, radius), radius)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    src = np.asarray(mask, dtype=bool)
    out = np.zeros_like(src, dtype=bool)
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = _disk_offsets(radius)
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return src.copy()
    return ~_dilate(~src, radius)


def _disk_offsets(radius: int) -> List[GridCell]:
    return [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= radius * radius
    ]


def _radius_cells(radius_m: float, resolution_m: float) -> int:
    return max(0, int(round(float(radius_m) / max(float(resolution_m), 1e-9))))


def _local_maxima(mask: np.ndarray, distance_m: np.ndarray, threshold: float) -> List[GridCell]:
    out: List[GridCell] = []
    h, w = mask.shape
    for row, col in zip(*np.nonzero(mask)):
        value = float(distance_m[row, col])
        if value < threshold:
            continue
        local = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < h and 0 <= cc < w and float(distance_m[rr, cc]) > value:
                    local = False
                    break
            if not local:
                break
        if local:
            out.append((int(row), int(col)))
    return out


def _seeded_region_grow(mask: np.ndarray, distance_m: np.ndarray, seeds: Sequence[GridCell]) -> np.ndarray:
    labels = np.zeros_like(mask, dtype=np.int32)
    heap: List[Tuple[float, int, int, int]] = []
    for idx, (row, col) in enumerate(seeds, start=1):
        if not mask[row, col]:
            continue
        labels[row, col] = idx
        heapq.heappush(heap, (-float(distance_m[row, col]), idx, int(row), int(col)))
    h, w = mask.shape
    while heap:
        _neg_dist, label, row, col = heapq.heappop(heap)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + dr, col + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or not mask[rr, cc] or labels[rr, cc] > 0:
                continue
            labels[rr, cc] = label
            heapq.heappush(heap, (-float(distance_m[rr, cc]), label, rr, cc))
    return labels


def _distance_transform_fallback(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    obstacles = np.asarray(~src, dtype=bool)
    rows, cols = np.nonzero(obstacles)
    if rows.size == 0:
        return np.ones_like(src, dtype=np.float32) * max(h, w)
    out = np.zeros_like(src, dtype=np.float32)
    obstacle_points = np.stack([rows, cols], axis=1).astype(np.float32)
    for row, col in zip(*np.nonzero(src)):
        d2 = np.min(np.sum((obstacle_points - np.asarray([row, col], dtype=np.float32)) ** 2, axis=1))
        out[row, col] = math.sqrt(float(d2))
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    eroded = _erode(src, 1)
    return src & ~eroded


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return 0.0 if union <= 0 else float(inter) / float(union)


def _centroid_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:2], dtype=np.float32) - np.asarray(b[:2], dtype=np.float32)))
