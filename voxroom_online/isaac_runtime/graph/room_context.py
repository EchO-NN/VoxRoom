from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from voxroom_online.isaac_runtime.graph.room_semantics import (
    DEFAULT_ROOM_CATEGORIES,
    RoomSemanticLabel,
    VLMRoomLabeler,
    room_semantics_debug,
    summarize_room_object_evidence,
)
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.room_segmentation import (
    ObjectRoomAssignment,
    OnlineRoomSegmenter,
    RoomMask,
    assign_objects_to_room_masks,
    room_segmentation_debug,
)


SCORING_ROOM_CALL_ORDER_PREFIX = [
    "frontier_extraction",
    "room_context_for_frontier_scoring",
]

SCORING_ROOM_CALL_ORDER_FULL = [
    "frontier_extraction",
    "room_context_for_frontier_scoring",
    "scenegraph_update",
    "hcot_subgraph_scoring",
    "frontier_interpolation",
    "frontier_selection",
]


@dataclass
class RoomContextResult:
    room_masks: list[RoomMask] = field(default_factory=list)
    room_semantic_labels: Dict[str, RoomSemanticLabel] = field(default_factory=dict)
    room_segmentation_debug: dict = field(default_factory=dict)
    room_semantics_debug: dict = field(default_factory=dict)
    object_room_assignments: Dict[str, ObjectRoomAssignment] = field(default_factory=dict)
    cache_hit: bool = False
    segmentation_ran: bool = False
    labeling_ran: bool = False
    label_requests: int = 0
    label_cache_hits: int = 0
    call_order_trace: list[str] = field(default_factory=lambda: list(SCORING_ROOM_CALL_ORDER_PREFIX))
    map_observed_hash: str = ""
    object_evidence_hash: str = ""
    room_mask_geometry_hash: str = ""
    room_object_evidence_hash: str = ""
    source: str = "voxel_occupancy_door_wall_v33_vlm"

    def metadata(self, *, full_order: bool = True) -> dict:
        trace = list(SCORING_ROOM_CALL_ORDER_FULL if full_order else self.call_order_trace)
        return {
            "room_context_source": self.source,
            "room_update_invoked_for_frontier_scoring": True,
            "room_segmentation_ran": bool(self.segmentation_ran),
            "room_labeling_ran": bool(self.labeling_ran),
            "room_context_cache_hit": bool(self.cache_hit),
            "room_mask_count": int(len([room for room in self.room_masks if not getattr(room, "stale", False)])),
            "room_label_count": int(len(self.room_semantic_labels)),
            "room_label_requests": int(self.label_requests),
            "room_label_cache_hits": int(self.label_cache_hits),
            "room_call_order_trace": trace,
            "room_segmentation_called_for": "frontier_scoring_pre_hook",
            "room_segmentation_algorithm": str(self.room_segmentation_debug.get("algorithm", "voxel_occupancy_door_wall_v33")),
            "room_segmentation_step_index": self.room_segmentation_debug.get("step"),
            "room_vlm_called": bool(self.labeling_ran),
            "scenegraph_updated_after_room_context": True,
            "frontier_scoring_after_room_context": True,
            "room_map_observed_hash": self.map_observed_hash,
            "room_object_evidence_hash": self.object_evidence_hash,
            "room_mask_geometry_hash": self.room_mask_geometry_hash,
            "room_object_assignment_count": int(len(self.object_room_assignments)),
        }


@dataclass
class RoomContextCache:
    last_result: Optional[RoomContextResult] = None
    label_cache: Dict[str, RoomSemanticLabel] = field(default_factory=dict)
    label_evidence_hash_by_room: Dict[str, str] = field(default_factory=dict)


def prepare_room_context_for_frontier_scoring(
    *,
    step_idx: int,
    mapper: object | None = None,
    object_memory,
    room_segmenter: Optional[OnlineRoomSegmenter],
    room_labeler: Optional[VLMRoomLabeler],
    map_info: MapInfo,
    previous_room_context: Optional[RoomContextCache],
    strict_benchmark: bool,
    occupancy: Optional[np.ndarray] = None,
    observed_free_mask: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    unknown_mask: Optional[np.ndarray] = None,
    allowed_categories: Optional[Sequence[str]] = None,
) -> RoomContextResult:
    """Prepare online room masks and optional semantic labels before frontier scoring."""
    cache = previous_room_context or RoomContextCache()
    source = str(getattr(room_segmenter, "context_source", "voxel_occupancy_door_wall_v33_vlm") or "voxel_occupancy_door_wall_v33_vlm") if room_segmenter is not None else "voxel_occupancy_door_wall_v33_vlm"
    allowed = list(allowed_categories or DEFAULT_ROOM_CATEGORIES)
    if room_segmenter is None:
        if strict_benchmark:
            raise RuntimeError("strict benchmark requires online room context before VoxRoom frontier scoring")
        result = RoomContextResult(
            room_segmentation_debug={"source": "unavailable", "room_count": 0, "rooms": []},
            room_semantics_debug={"backend": "unavailable", "allowed_categories": allowed, "labels": []},
            source="unavailable",
        )
        cache.last_result = result
        return result

    raw_free_arr = _resolve_array("free", observed_free_mask, mapper)
    raw_obstacle_arr = _resolve_array("occupancy", obstacle_mask, mapper)
    if unknown_mask is None:
        observed_arr = _resolve_array("observed", None, mapper)
        raw_unknown_arr = ~np.asarray(observed_arr, dtype=bool)
    else:
        raw_unknown_arr = np.asarray(unknown_mask, dtype=bool)
    nav_free_arr, nav_obstacle_arr, nav_unknown_arr, nav_source = resolve_replay_style_navigation_masks(
        mapper=mapper,
        shape=raw_obstacle_arr.shape,
        fallback_free=raw_free_arr,
        fallback_obstacle=raw_obstacle_arr,
        fallback_unknown=raw_unknown_arr,
    )
    # Final replay reconstructs roomseg from voxel_nav_* snapshots:
    # occupancy_map == obstacle_mask == voxel_nav_occupied_xy,
    # observed_free_mask == voxel_nav_free_xy, and
    # unknown_mask == voxel_nav_unknown_xy. Mirror that contract online.
    occupancy_arr = nav_obstacle_arr
    free_arr = nav_free_arr
    obstacle_arr = nav_obstacle_arr
    unknown_arr = nav_unknown_arr
    static_structural_arr = _resolve_optional_mapper_array("roomseg_static_structural_occupied", mapper, obstacle_arr.shape)
    roomseg_ray_evidence = _resolve_roomseg_ray_evidence(mapper, obstacle_arr.shape)
    height_profile = _resolve_height_profile(mapper, obstacle_arr.shape)
    voxel_grid = _resolve_voxel_grid(mapper, obstacle_arr.shape)

    map_hash = _hash_arrays(
        free_arr,
        obstacle_arr,
        unknown_arr,
        nav_free_arr,
        nav_obstacle_arr,
        nav_unknown_arr,
        static_structural_arr,
        *roomseg_ray_evidence.values(),
        *_height_profile_hash_arrays(height_profile),
        *_voxel_grid_hash_arrays(voxel_grid),
    )
    object_hash = _hash_object_memory(object_memory)
    if (
        cache.last_result is not None
        and cache.last_result.map_observed_hash == map_hash
        and cache.last_result.object_evidence_hash == object_hash
    ):
        cached = cache.last_result
        result = RoomContextResult(
            room_masks=list(cached.room_masks),
            room_semantic_labels=dict(cached.room_semantic_labels),
            room_segmentation_debug=dict(cached.room_segmentation_debug),
            room_semantics_debug=dict(cached.room_semantics_debug),
            object_room_assignments=dict(cached.object_room_assignments),
            cache_hit=True,
            segmentation_ran=False,
            labeling_ran=False,
            label_requests=0,
            label_cache_hits=int(len(cached.room_semantic_labels)),
            call_order_trace=list(SCORING_ROOM_CALL_ORDER_PREFIX),
            map_observed_hash=map_hash,
            object_evidence_hash=object_hash,
            room_mask_geometry_hash=cached.room_mask_geometry_hash,
            room_object_evidence_hash=cached.room_object_evidence_hash,
            source=source,
        )
        cache.last_result = result
        return result

    labels: Dict[str, RoomSemanticLabel] = {}
    label_cache_hits = 0
    label_requests = 0
    labeling_ran = False
    premerge_labels_by_label_id: Dict[int, RoomSemanticLabel] = {}
    if room_labeler is None:
        if strict_benchmark:
            raise RuntimeError("strict benchmark requires room labeling before VoxRoom frontier scoring")
    else:
        use_premerge = hasattr(room_segmenter, "build_proposals") and hasattr(room_segmenter, "finalize_proposals")
        if use_premerge:
            build_kwargs = {
                "step": int(step_idx),
                "object_memory": getattr(object_memory, "nodes", []),
            }
            try:
                build_params = inspect.signature(room_segmenter.build_proposals).parameters
            except (TypeError, ValueError):
                build_params = {}
            if "vertical_profile" in build_params:
                build_kwargs["vertical_profile"] = getattr(mapper, "vertical_profile", None)
            if "height_profile" in build_params:
                build_kwargs["height_profile"] = height_profile
            if "voxel_grid" in build_params:
                build_kwargs["voxel_grid"] = voxel_grid
            if "roomseg_static_structural_occupied" in build_params:
                build_kwargs["roomseg_static_structural_occupied"] = static_structural_arr
            if "roomseg_ray_evidence" in build_params:
                build_kwargs["roomseg_ray_evidence"] = roomseg_ray_evidence
            add_replay_style_navigation_update_kwargs(
                build_kwargs,
                build_params,
                navigation_free_mask=nav_free_arr,
                navigation_obstacle_mask=nav_obstacle_arr,
                navigation_unknown_mask=nav_unknown_arr,
            )
            proposal_rooms, proposal_state = room_segmenter.build_proposals(
                occupancy_arr,
                free_arr,
                obstacle_arr,
                unknown_arr,
                **build_kwargs,
            )
            proposal_assignments = assign_objects_to_room_masks(getattr(object_memory, "nodes", []), proposal_rooms, map_info)
            for proposal in proposal_rooms:
                label_id = int((proposal.metadata or {}).get("label_id", 0) or 0)
                if label_id <= 0:
                    continue
                evidence = summarize_room_object_evidence(getattr(object_memory, "nodes", []), proposal_assignments, proposal.room_id)
                evidence_hash = _hash_room_evidence(proposal, evidence)
                cache_key = "premerge:%s:%s" % (label_id, evidence_hash)
                cached_label = cache.label_cache.get(cache_key)
                if cached_label is not None:
                    premerge_labels_by_label_id[label_id] = cached_label
                    label_cache_hits += 1
                    continue
                before = int(getattr(room_labeler, "request_count", 0))
                premerge_labels_by_label_id[label_id] = room_labeler.label_room(proposal, evidence, None)
                after = int(getattr(room_labeler, "request_count", 0))
                label_requests += max(0, after - before)
                labeling_ran = True
                cache.label_cache[cache_key] = premerge_labels_by_label_id[label_id]
            room_masks = room_segmenter.finalize_proposals(proposal_state, premerge_labels_by_label_id)
        else:
            update_kwargs = {"step": int(step_idx)}
            try:
                update_params = inspect.signature(room_segmenter.update).parameters
            except (TypeError, ValueError):
                update_params = {}
            if "object_memory" in update_params:
                update_kwargs["object_memory"] = getattr(object_memory, "nodes", [])
            if "vertical_profile" in update_params:
                update_kwargs["vertical_profile"] = getattr(mapper, "vertical_profile", None)
            if "height_profile" in update_params:
                update_kwargs["height_profile"] = height_profile
            if "voxel_grid" in update_params:
                update_kwargs["voxel_grid"] = voxel_grid
            if "roomseg_static_structural_occupied" in update_params:
                update_kwargs["roomseg_static_structural_occupied"] = static_structural_arr
            if "roomseg_ray_evidence" in update_params:
                update_kwargs["roomseg_ray_evidence"] = roomseg_ray_evidence
            add_replay_style_navigation_update_kwargs(
                update_kwargs,
                update_params,
                navigation_free_mask=nav_free_arr,
                navigation_obstacle_mask=nav_obstacle_arr,
                navigation_unknown_mask=nav_unknown_arr,
            )
            room_masks = room_segmenter.update(
                occupancy_arr,
                free_arr,
                obstacle_arr,
                unknown_arr,
                **update_kwargs,
            )
    if room_labeler is None:
        update_kwargs = {"step": int(step_idx)}
        try:
            update_params = inspect.signature(room_segmenter.update).parameters
        except (TypeError, ValueError):
            update_params = {}
        if "object_memory" in update_params:
            update_kwargs["object_memory"] = getattr(object_memory, "nodes", [])
        if "vertical_profile" in update_params:
            update_kwargs["vertical_profile"] = getattr(mapper, "vertical_profile", None)
        if "height_profile" in update_params:
            update_kwargs["height_profile"] = height_profile
        if "voxel_grid" in update_params:
            update_kwargs["voxel_grid"] = voxel_grid
        if "roomseg_static_structural_occupied" in update_params:
            update_kwargs["roomseg_static_structural_occupied"] = static_structural_arr
        if "roomseg_ray_evidence" in update_params:
            update_kwargs["roomseg_ray_evidence"] = roomseg_ray_evidence
        add_replay_style_navigation_update_kwargs(
            update_kwargs,
            update_params,
            navigation_free_mask=nav_free_arr,
            navigation_obstacle_mask=nav_obstacle_arr,
            navigation_unknown_mask=nav_unknown_arr,
        )
        room_masks = room_segmenter.update(
            occupancy_arr,
            free_arr,
            obstacle_arr,
            unknown_arr,
            **update_kwargs,
        )
    seg_debug = dict(room_segmenter.last_debug or room_segmentation_debug(room_masks))
    seg_debug["roomseg_online_navigation_mask_source"] = str(nav_source)
    seg_debug["roomseg_online_navigation_free_cells"] = int(np.count_nonzero(nav_free_arr))
    seg_debug["roomseg_online_navigation_occupied_cells"] = int(np.count_nonzero(nav_obstacle_arr))
    seg_debug["roomseg_online_navigation_unknown_cells"] = int(np.count_nonzero(nav_unknown_arr))
    assignments = assign_objects_to_room_masks(getattr(object_memory, "nodes", []), room_masks, map_info)
    if room_labeler is not None:
        for room in room_masks:
            if room.stale:
                continue
            evidence = summarize_room_object_evidence(getattr(object_memory, "nodes", []), assignments, room.room_id)
            evidence_hash = _hash_room_evidence(room, evidence)
            cache_key = "final:%s:%s" % (room.room_id, evidence_hash)
            cached_label = cache.label_cache.get(cache_key)
            if cached_label is not None:
                labels[room.room_id] = cached_label
                label_cache_hits += 1
                continue
            before = int(getattr(room_labeler, "request_count", 0))
            labels[room.room_id] = room_labeler.label_room(room, evidence, None)
            after = int(getattr(room_labeler, "request_count", 0))
            label_requests += max(0, after - before)
            labeling_ran = True
            cache.label_cache[cache_key] = labels[room.room_id]
            cache.label_evidence_hash_by_room[room.room_id] = evidence_hash
    seg_debug["premerge_room_semantics"] = {
        "labels": [label.to_dict() for _label_id, label in sorted(premerge_labels_by_label_id.items())],
        "used_for_open_plan_merge": bool(premerge_labels_by_label_id),
        "final_labels_recomputed_after_merge": bool(premerge_labels_by_label_id and labels),
    }

    semantics_debug = room_semantics_debug(
        labels,
        allowed,
        getattr(room_labeler, "backend", "unavailable") if room_labeler is not None else "unavailable",
    )
    result = RoomContextResult(
        room_masks=list(room_masks),
        room_semantic_labels=labels,
        room_segmentation_debug=seg_debug,
        room_semantics_debug=semantics_debug,
        object_room_assignments=dict(assignments),
        cache_hit=False,
        segmentation_ran=True,
        labeling_ran=labeling_ran,
        label_requests=label_requests,
        label_cache_hits=label_cache_hits,
        call_order_trace=list(SCORING_ROOM_CALL_ORDER_PREFIX),
        map_observed_hash=map_hash,
        object_evidence_hash=object_hash,
        room_mask_geometry_hash=_hash_room_masks(room_masks),
        room_object_evidence_hash=_hash_assignments(assignments),
        source=source,
    )
    cache.last_result = result
    return result


def room_context_not_invoked_metadata() -> dict:
    return {
        "room_context_source": "not_invoked",
        "room_update_invoked_for_frontier_scoring": False,
        "room_segmentation_ran": False,
        "room_labeling_ran": False,
        "room_context_cache_hit": False,
        "room_mask_count": 0,
        "room_label_count": 0,
        "room_label_requests": 0,
        "room_label_cache_hits": 0,
        "room_call_order_trace": [],
        "room_segmentation_called_for": "not_invoked",
        "room_segmentation_algorithm": "not_invoked",
        "room_segmentation_step_index": None,
        "room_vlm_called": False,
        "scenegraph_updated_after_room_context": False,
        "frontier_scoring_after_room_context": False,
    }


def _resolve_array(name: str, explicit: Optional[np.ndarray], mapper: object | None) -> np.ndarray:
    if explicit is not None:
        return np.asarray(explicit)
    if mapper is None:
        raise ValueError("room context requires %s array or mapper" % name)
    grid = getattr(mapper, "grid", None)
    if grid is not None and hasattr(grid, name):
        return np.asarray(getattr(grid, name))
    if hasattr(mapper, name):
        value = getattr(mapper, name)
        return np.asarray(value() if callable(value) else value)
    raise ValueError("room context could not resolve %s from mapper" % name)


def _call_accepts_kwarg(parameters: Mapping[str, inspect.Parameter], name: str) -> bool:
    if name in parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def _shape_checked_bool(value: object, shape: tuple[int, ...]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.shape != tuple(shape):
        return None
    return arr.copy()


def resolve_replay_style_navigation_masks(
    *,
    mapper: object | None,
    shape: tuple[int, ...],
    fallback_free: np.ndarray,
    fallback_obstacle: np.ndarray,
    fallback_unknown: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return the same no-clearance nav masks that replay reads from NPZ.

    Final replay uses voxel_nav_free_xy / voxel_nav_occupied_xy /
    voxel_nav_unknown_xy as the navigation gates for door seed/completion.
    Online runtime should prefer the mapper's latest voxel navigation
    projection over the planner-clearance or generic observed/free masks.
    """
    shape = tuple(shape)
    projection = getattr(mapper, "last_voxel_navigation_projection", None) if mapper is not None else None
    if projection is not None:
        nav_free = _shape_checked_bool(getattr(projection, "free", None), shape)
        nav_obstacle = _shape_checked_bool(getattr(projection, "occupied", None), shape)
        nav_unknown = _shape_checked_bool(getattr(projection, "unknown", None), shape)
        if nav_free is not None and nav_obstacle is not None and nav_unknown is not None:
            return nav_free, nav_obstacle, nav_unknown, "mapper.last_voxel_navigation_projection"

    if mapper is not None and hasattr(mapper, "navigation_debug_layers"):
        try:
            debug = mapper.navigation_debug_layers(include_arrays=True)
        except Exception:
            debug = {}
        nav_free = _shape_checked_bool(debug.get("voxel_nav_final_free_xy"), shape)
        nav_obstacle = _shape_checked_bool(debug.get("voxel_nav_final_occupied_xy"), shape)
        nav_unknown = _shape_checked_bool(debug.get("voxel_nav_final_unknown_xy"), shape)
        if nav_free is not None and nav_obstacle is not None and nav_unknown is not None:
            return nav_free, nav_obstacle, nav_unknown, "mapper.navigation_debug_layers"

    return (
        np.asarray(fallback_free, dtype=bool).copy(),
        np.asarray(fallback_obstacle, dtype=bool).copy(),
        np.asarray(fallback_unknown, dtype=bool).copy(),
        "fallback_observed_free_obstacle_unknown",
    )


def add_replay_style_navigation_update_kwargs(
    update_kwargs: dict[str, object],
    update_params: Mapping[str, inspect.Parameter],
    *,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    navigation_unknown_mask: np.ndarray,
) -> None:
    _ = navigation_unknown_mask
    if _call_accepts_kwarg(update_params, "navigation_free_mask"):
        update_kwargs["navigation_free_mask"] = navigation_free_mask
    if _call_accepts_kwarg(update_params, "navigation_obstacle_mask"):
        update_kwargs["navigation_obstacle_mask"] = navigation_obstacle_mask
    if _call_accepts_kwarg(update_params, "door_seed_no_clearance_free_mask"):
        update_kwargs["door_seed_no_clearance_free_mask"] = navigation_free_mask


def _resolve_optional_mapper_array(name: str, mapper: object | None, shape: tuple[int, ...]) -> np.ndarray:
    if mapper is None or not hasattr(mapper, name):
        return np.zeros(shape, dtype=bool)
    value = getattr(mapper, name)
    arr = np.asarray(value() if callable(value) else value, dtype=bool)
    if arr.shape != tuple(shape):
        return np.zeros(shape, dtype=bool)
    return arr


def _resolve_roomseg_ray_evidence(mapper: object | None, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    out = {
        "ray_covered_count": np.zeros(shape, dtype=np.uint16),
        "terminal_wall_count": np.zeros(shape, dtype=np.uint16),
        "terminal_wall_height_min": np.full(shape, np.inf, dtype=np.float32),
        "terminal_wall_height_max": np.full(shape, -np.inf, dtype=np.float32),
        "terminal_wall_depth_min": np.full(shape, np.inf, dtype=np.float32),
        "terminal_wall_splat": np.zeros(shape, dtype=np.uint8),
    }
    if mapper is None:
        return out
    source = getattr(mapper, "roomseg_ray_evidence", None)
    raw = source() if callable(source) else None
    if not isinstance(raw, Mapping):
        raw = {
            "ray_covered_count": getattr(mapper, "roomseg_ray_covered_count", None),
            "terminal_wall_count": getattr(mapper, "roomseg_terminal_wall_count", None),
            "terminal_wall_height_min": getattr(mapper, "roomseg_terminal_wall_height_min", None),
            "terminal_wall_height_max": getattr(mapper, "roomseg_terminal_wall_height_max", None),
            "terminal_wall_depth_min": getattr(mapper, "roomseg_terminal_wall_depth_min", None),
            "terminal_wall_splat": getattr(mapper, "roomseg_terminal_wall_splat", None),
        }
    for key, default in list(out.items()):
        value = raw.get(key) if isinstance(raw, Mapping) else None
        if value is None:
            continue
        arr = np.asarray(value, dtype=default.dtype)
        if arr.shape == tuple(shape):
            out[key] = arr
    return out


def _resolve_height_profile(mapper: object | None, shape: tuple[int, ...]) -> object | None:
    if mapper is None or not hasattr(mapper, "height_profile"):
        return None
    height_profile = getattr(mapper, "height_profile")
    if callable(height_profile):
        height_profile = height_profile()
    if height_profile is None:
        return None
    profile_shape = getattr(height_profile, "shape", None)
    if tuple(profile_shape or ()) != tuple(shape):
        return None
    for name in ("free_ray_count", "occupied_count", "observed_count"):
        if not hasattr(height_profile, name):
            return None
    return height_profile


def _height_profile_hash_arrays(height_profile: object | None) -> tuple[np.ndarray, ...]:
    if height_profile is None:
        return ()
    return (
        np.asarray(getattr(height_profile, "free_ray_count")),
        np.asarray(getattr(height_profile, "occupied_count")),
        np.asarray(getattr(height_profile, "observed_count")),
    )


def _resolve_voxel_grid(mapper: object | None, shape: tuple[int, ...]) -> object | None:
    if mapper is None or not hasattr(mapper, "voxel_grid"):
        return None
    voxel_grid = getattr(mapper, "voxel_grid")
    if callable(voxel_grid):
        voxel_grid = voxel_grid()
    if voxel_grid is None:
        return None
    grid_shape = getattr(voxel_grid, "shape", None)
    if tuple(grid_shape or ()) != tuple(shape):
        return None
    if not hasattr(voxel_grid, "state") or not hasattr(voxel_grid, "log_odds"):
        return None
    return voxel_grid


def _voxel_grid_hash_arrays(voxel_grid: object | None) -> tuple[np.ndarray, ...]:
    if voxel_grid is None:
        return ()
    return (
        np.asarray(getattr(voxel_grid, "state")),
        np.asarray(getattr(voxel_grid, "log_odds")),
        np.asarray(getattr(voxel_grid, "sensor_range_count", np.zeros_like(getattr(voxel_grid, "state")))),
    )


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        arr = np.asarray(array)
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()


def _hash_object_memory(object_memory) -> str:
    rows = []
    for node in getattr(object_memory, "nodes", []):
        rows.append(
            {
                "id": str(getattr(node, "id", getattr(node, "node_id", ""))),
                "category": str(getattr(node, "category", "")),
                "center": [round(float(v), 3) for v in getattr(node, "center_world", [])[:3]],
                "conf": round(float(getattr(node, "confidence", 0.0)), 3),
                "hits": int(getattr(node, "observed_count", 0)),
                "last": int(getattr(node, "last_seen_step", -1)),
            }
        )
    rows.sort(key=lambda item: item["id"])
    return _hash_json(rows)


def _hash_room_evidence(room: RoomMask, evidence: Sequence[object]) -> str:
    rows = {
        "room_id": room.room_id,
        "objects": [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in evidence],
    }
    return _hash_json(rows)


def _hash_room_masks(room_masks: Sequence[RoomMask]) -> str:
    rows = []
    for room in room_masks:
        if room.stale:
            continue
        rows.append(
            {
                "room_id": room.room_id,
                "area_m2": round(float(room.area_m2), 3),
                "centroid_xy": [round(float(v), 3) for v in room.centroid_xy],
                "boundary_unknown_fraction": round(float(room.boundary_unknown_fraction), 3),
                "mask_confidence": round(float(room.mask_confidence), 3),
            }
        )
    rows.sort(key=lambda item: item["room_id"])
    return _hash_json(rows)


def _hash_assignments(assignments: Mapping[str, ObjectRoomAssignment]) -> str:
    rows = []
    for object_id, assignment in sorted(assignments.items()):
        rows.append(
            {
                "object_id": object_id,
                "room_id": assignment.room_id,
                "overlap": round(float(assignment.mask_overlap_ratio), 3),
                "confidence": round(float(assignment.assignment_confidence), 3),
                "ambiguous": bool(assignment.ambiguous),
            }
        )
    return _hash_json(rows)


def _hash_json(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
