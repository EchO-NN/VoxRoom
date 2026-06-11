from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.config import get_nested, load_config, str_to_bool
from voxroom_online.isaac_runtime.baselines.data_contract import map_info_extra_arrays
from voxroom_online.isaac_runtime.baselines.live_manager import LiveBaselineManager
from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.dataset.episode_generator import point_to_bbox_2d_distance, read_jsonl
from voxroom_online.isaac_runtime.env.habitat_like_env import MapSimHabitatLikeEnv
from voxroom_online.isaac_runtime.debug.graph_debug_dump import save_graph_debug_dump
from voxroom_online.isaac_runtime.debug.roomseg_layer_dump import ROOMSEG_SNAPSHOT_ARRAY_KEYS, save_roomseg_layer_dump
from voxroom_online.isaac_runtime.graph.decision import NavigationDecision, SGNavDecision
from voxroom_online.isaac_runtime.graph.room_context import (
    RoomContextCache,
    RoomContextResult,
    add_replay_style_navigation_update_kwargs,
    prepare_room_context_for_frontier_scoring,
    resolve_replay_style_navigation_masks,
    room_context_not_invoked_metadata,
)
from voxroom_online.isaac_runtime.graph.sgnav_scenegraph_adapter import SGNAV_ROOM_NAMES, SGNavSceneGraphAdapter
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from voxroom_online.isaac_runtime.mapping.frontier import extract_frontiers, extract_frontiers_with_partition, frontier_debug_layers
from voxroom_online.isaac_runtime.mapping.frontier_room_context import assign_frontier_room_context
from voxroom_online.isaac_runtime.mapping.frontier_debug import save_frontier_debug_snapshot
from voxroom_online.isaac_runtime.mapping.height_column_profile import HeightColumnProfileConfig
from voxroom_online.isaac_runtime.mapping.online_mapper import OnlineMapper
from voxroom_online.isaac_runtime.mapping.room_segmentation import (
    room_segmentation_debug,
)
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import (
    VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
    VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
    VOXEL_OCCUPANCY_ROOMSEG_LEGACY_BACKENDS,
    VOXEL_OCCUPANCY_ROOMSEG_LEGACY_CONTEXTS,
    VoxelOccupancyDoorWallRoomSegConfig,
    VoxelOccupancyDoorWallRoomSegmenter,
)
from voxroom_online.isaac_runtime.mapping.voxel_roomseg_evidence import build_voxel_roomseg_evidence
from voxroom_online.isaac_runtime.graph.room_semantics import (
    DEFAULT_ROOM_CATEGORIES,
    VLMRoomLabeler,
)
from voxroom_online.isaac_runtime.metrics.episode_logger import JsonlEpisodeLogger, make_jsonable
from voxroom_online.isaac_runtime.metrics.evaluator import EpisodeEvaluator
from voxroom_online.isaac_runtime.metrics.result_schema import BenchmarkAssetError, complete_result_row, validate_strict_benchmark_assets
from voxroom_online.isaac_runtime.navigation.astar import ClearanceAStarPlanner, GridAStarPlanner, astar_distance_map
from voxroom_online.isaac_runtime.navigation.frontier_execution_state import (
    CommittedFrontierExecutionState,
    frontier_arrival_status,
    make_committed_frontier_arrival_key,
)
from voxroom_online.isaac_runtime.navigation.frontier_exhaustion_audit import FrontierExhaustionAudit, audit_frontier_exhaustion
from voxroom_online.isaac_runtime.navigation.frontier_recovery import (
    FrontierRecoveryConfig,
    FrontierRecoveryTarget,
    find_best_reachable_frontier_approach,
)
from voxroom_online.isaac_runtime.navigation.frontier_terminal_registry import FrontierTerminalRegistry
from voxroom_online.isaac_runtime.navigation.frontier_targeting import FrontierTargetResolution, resolve_frontier_target
from voxroom_online.isaac_runtime.navigation.frontier_commitment import FrontierCommitmentManager
from voxroom_online.isaac_runtime.navigation.waypoint_follower import HolonomicWaypointFollower
from voxroom_online.isaac_runtime.perception.detection_types import (
    MIN_VALID_DETECTION_CONFIDENCE,
    Detection2D,
    Detection3D,
    bbox_touches_image_edge,
    detection_confidence_is_valid,
)
from voxroom_online.isaac_runtime.perception.detector_ipc import SubprocessDetector
from voxroom_online.isaac_runtime.perception.fused_instance_registry import FusedInstanceRegistry
from voxroom_online.isaac_runtime.perception.object_memory import ObjectMemory
from voxroom_online.isaac_runtime.perception.sam2_segmenter import build_sam2_segmenter
from voxroom_online.isaac_runtime.perception.yolo_world_detector import build_detector
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.isaac_runtime.sensors.depth_backproject import detections_to_3d
from voxroom_online.isaac_runtime.visualization.draw_map import save_map_png
from voxroom_online.isaac_runtime.visualization.voxroom_popup import VoxRoomPopupVisualizer


@dataclass
class FrontierRoomsegUpdateGateState:
    initialized: bool = False
    last_update_step: int = -1
    last_update_reason: str | None = None
    last_skip_reason: str | None = None


@dataclass
class FrontierArrivalUpdateState:
    initialized: bool = False
    arrival_pending: bool = False
    arrival_step: int = -1
    arrival_frontier_key: tuple | None = None
    refresh_pending: bool = False
    refresh_reason: str = ""
    refresh_step: int = -1
    refresh_requested_step: int = -1
    refresh_frontier_key: tuple | None = None
    refresh_request_count: int = 0
    duplicate_refresh_requests: int = 0
    last_duplicate_refresh_step: int = -1
    last_duplicate_refresh_reason: str = ""
    block_reselect_until_refresh: bool = False
    update_in_progress: bool = False
    last_consumed_arrival_key: tuple | None = None
    last_consumed_key: tuple | None = None
    last_update_step: int = -1
    last_update_reason: str = ""
    last_skip_reason: str = ""
    cooldown_until_step: int = -1


@dataclass
class ClearanceGoalFilterResult:
    goals: List[Tuple[int, int]]
    rejected_original_count: int
    snapped_count: int
    no_safe_goal_count: int
    min_clearance_m: float
    search_radius_cells: int
    debug_by_input: List[dict] = field(default_factory=list)

    def metadata(self) -> dict:
        return {
            "astar_goal_clearance_filter_enabled": True,
            "astar_goal_cells_after_clearance_filter": int(len(self.goals)),
            "astar_goal_clearance_rejected_original_count": int(self.rejected_original_count),
            "astar_goal_clearance_snapped_count": int(self.snapped_count),
            "astar_goal_clearance_no_safe_goal_count": int(self.no_safe_goal_count),
            "astar_goal_clearance_debug_by_input": list(self.debug_by_input[:32]),
        }


@dataclass
class RoomsegFrontierUpdateToken:
    allowed: bool
    reason: str
    step: int


def make_frontier_arrival_key(nav_decision) -> tuple:
    if nav_decision is None:
        return ("none", None, None)
    meta = dict(getattr(nav_decision, "metadata", None) or {})
    commitment = dict(meta.get("frontier_commitment") or {})
    active_id = meta.get("active_frontier_id", commitment.get("active_frontier_id"))
    target0 = None
    target_cells = list(getattr(nav_decision, "target_cells", None) or [])
    if target_cells:
        try:
            target0 = tuple(int(v) for v in target_cells[0])
        except Exception:
            target0 = tuple(target_cells[0])
    return (str(getattr(nav_decision, "mode", "none")), active_id, target0)


def request_frontier_refresh(
    *,
    refresh_state: FrontierArrivalUpdateState,
    step: int,
    reason: str,
    frontier_key: tuple | None,
    block_reselect: bool = True,
) -> bool:
    stable_key = ("frontier_refresh", frontier_key, str(reason))
    if bool(refresh_state.refresh_pending) and refresh_state.refresh_frontier_key == stable_key:
        refresh_state.duplicate_refresh_requests += 1
        refresh_state.last_duplicate_refresh_step = int(step)
        refresh_state.last_duplicate_refresh_reason = str(reason)
        return False
    if refresh_state.last_consumed_key == stable_key and int(step) < int(refresh_state.cooldown_until_step):
        refresh_state.duplicate_refresh_requests += 1
        refresh_state.last_duplicate_refresh_step = int(step)
        refresh_state.last_duplicate_refresh_reason = str(reason)
        return False
    refresh_state.initialized = True
    refresh_state.arrival_pending = True
    refresh_state.arrival_step = int(step)
    refresh_state.arrival_frontier_key = frontier_key
    refresh_state.refresh_pending = True
    refresh_state.refresh_reason = str(reason)
    refresh_state.refresh_step = int(step)
    refresh_state.refresh_requested_step = int(step)
    refresh_state.refresh_frontier_key = stable_key
    refresh_state.refresh_request_count += 1
    refresh_state.block_reselect_until_refresh = bool(block_reselect)
    refresh_state.update_in_progress = False
    return True


def consume_frontier_refresh(
    *,
    refresh_state: FrontierArrivalUpdateState,
    step: int,
    reason: str,
    cooldown_steps: int,
) -> None:
    refresh_state.last_consumed_arrival_key = refresh_state.arrival_frontier_key
    refresh_state.last_consumed_key = refresh_state.refresh_frontier_key
    refresh_state.arrival_pending = False
    refresh_state.refresh_pending = False
    refresh_state.block_reselect_until_refresh = False
    refresh_state.update_in_progress = False
    refresh_state.last_update_step = int(step)
    refresh_state.last_update_reason = str(reason)
    refresh_state.cooldown_until_step = int(step) + int(cooldown_steps)


def frontier_recovery_config_from_args(args) -> FrontierRecoveryConfig:
    return FrontierRecoveryConfig(
        enabled=bool(getattr(args, "frontier_unreachable_recovery_enabled", True)),
        local_search_radius_m=float(getattr(args, "frontier_unreachable_recovery_local_search_radius_m", 1.50)),
        global_search_enabled=bool(getattr(args, "frontier_unreachable_recovery_global_search_enabled", True)),
        global_max_frontier_distance_m=float(getattr(args, "frontier_unreachable_recovery_global_max_frontier_distance_m", 3.00)),
        min_clearance_m=float(getattr(args, "frontier_unreachable_recovery_min_clearance_m", 0.18)),
        min_approach_improvement_m=float(getattr(args, "frontier_unreachable_recovery_min_approach_improvement_m", 0.20)),
        allow_current_as_partial_arrival=bool(
            getattr(args, "frontier_unreachable_recovery_allow_current_as_partial_arrival", False)
        ),
        current_partial_arrival_radius_m=float(getattr(args, "frontier_unreachable_recovery_current_partial_arrival_radius_m", 0.35)),
        current_partial_arrival_min_motion_m=float(getattr(args, "frontier_unreachable_recovery_current_partial_arrival_min_motion_m", 0.30)),
        current_partial_arrival_min_steps_since_selection=int(
            getattr(args, "frontier_unreachable_recovery_current_partial_arrival_min_steps_since_selection", 6)
        ),
        max_recovery_attempts_per_frontier=int(getattr(args, "frontier_unreachable_recovery_max_attempts_per_frontier", 1)),
        max_debug_candidates=int(getattr(args, "frontier_unreachable_recovery_max_debug_candidates", 64)),
    )


def _as_grid_cell(value) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        if len(value) < 2:
            return None
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


def _frontier_members_for_recovery(nav_decision, execution: CommittedFrontierExecutionState) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
    selected_frontier = None
    if nav_decision is not None and getattr(nav_decision, "frontier_decision", None) is not None:
        selected_frontier = nav_decision.frontier_decision.selected_frontier
    center = None
    members: list[tuple[int, int]] = []
    if selected_frontier is not None:
        center = _as_grid_cell(getattr(selected_frontier, "center_grid", None))
        members = [cell for cell in (_as_grid_cell(v) for v in getattr(selected_frontier, "members", []) or []) if cell is not None]
    if center is None:
        center = _as_grid_cell(getattr(execution, "selected_center_grid", None))
    if not members and center is not None:
        members = [center]
    return center, members


def should_update_roomseg_frontiers(
    *,
    step: int,
    has_current_path: bool,
    gate_state: FrontierRoomsegUpdateGateState,
    arrival_state: FrontierArrivalUpdateState | None = None,
    policy: str = "at_frontier_arrival",
    freeze_during_navigation: bool = True,
    target_reached: bool = False,
    target_invalidated: bool = False,
    no_active_path: bool = False,
    no_progress: bool = False,
    debug_force: bool = False,
    update_on_target_invalidated: bool = False,
    update_on_no_active_path: bool = False,
    update_on_no_progress: bool = False,
) -> tuple[bool, str]:
    _ = step
    normalized_policy = str(policy or "always").strip().lower()
    if not bool(freeze_during_navigation) or normalized_policy in {"always", "every_replan", "legacy"}:
        return True, "legacy_replan"
    if bool(debug_force):
        return True, "debug_force"
    if not bool(gate_state.initialized):
        return True, "initial"
    if arrival_state is not None and bool(getattr(arrival_state, "refresh_pending", False)):
        key = getattr(arrival_state, "refresh_frontier_key", None)
        if (
            key != getattr(arrival_state, "last_consumed_key", None)
            or int(getattr(arrival_state, "refresh_requested_step", -1))
            > int(getattr(arrival_state, "last_update_step", -1))
        ):
            return True, str(getattr(arrival_state, "refresh_reason", "") or "frontier_refresh")
        return False, "frontier_refresh_already_consumed"
    if arrival_state is not None and int(step) < int(arrival_state.cooldown_until_step):
        return False, "frontier_arrival_cooldown"
    if arrival_state is not None and bool(arrival_state.arrival_pending):
        if arrival_state.arrival_frontier_key != arrival_state.last_consumed_arrival_key:
            return True, "frontier_arrival"
        return False, "frontier_arrival_already_consumed"
    if bool(target_reached):
        return True, "target_reached"
    if bool(target_invalidated) and bool(update_on_target_invalidated):
        return True, "target_invalidated"
    if bool(target_invalidated):
        return False, "target_invalidated_cached"
    if bool(no_progress) and bool(update_on_no_progress):
        return True, "no_progress"
    if bool(no_progress):
        return False, "no_progress_cached"
    if (bool(no_active_path) or not bool(has_current_path)) and bool(update_on_no_active_path):
        return True, "no_active_path"
    if bool(no_active_path) or not bool(has_current_path):
        return False, "no_active_path_cached"
    return False, "cached_during_navigation"


def mark_roomseg_frontier_gate_update(gate_state: FrontierRoomsegUpdateGateState, *, step: int, reason: str) -> None:
    gate_state.initialized = True
    gate_state.last_update_step = int(step)
    gate_state.last_update_reason = str(reason)
    gate_state.last_skip_reason = None


def mark_roomseg_frontier_gate_skip(gate_state: FrontierRoomsegUpdateGateState, *, reason: str) -> None:
    gate_state.last_skip_reason = str(reason)


def frontier_selection_require_reachable(*, frontier_mask_probe: bool) -> bool:
    return not bool(frontier_mask_probe)


def should_save_roomseg_snapshot_for_context(
    *,
    save_roomseg_snapshots: bool,
    room_context_updated_this_step: bool,
    save_cached_roomseg_snapshots: bool,
    snapshot_source: str = "actual_roomseg_update",
    agent_grid: Sequence[int] | None = None,
    selected_frontier_key: object | None = None,
    room_mask_geometry_hash: str | None = None,
    roomseg_update_reason: str = "",
    last_snapshot_state: Mapping[str, object] | None = None,
    step: int = 0,
    failure_loop_min_interval_steps: int = 50,
) -> bool:
    if not bool(save_roomseg_snapshots):
        return False
    source = str(snapshot_source or "")
    if not bool(room_context_updated_this_step):
        return bool(save_cached_roomseg_snapshots)
    if source == "cached_roomseg_context" and not bool(save_cached_roomseg_snapshots):
        return False
    state = dict(last_snapshot_state or {})
    if not state:
        return True
    reason = str(roomseg_update_reason or "")
    if not _roomseg_snapshot_failure_loop_reason(reason):
        return True
    previous_step = int(state.get("step", -10**9) or -10**9)
    if int(step) - previous_step >= max(0, int(failure_loop_min_interval_steps)):
        return True
    return not (
        _snapshot_identity(agent_grid) == state.get("agent_grid")
        and _snapshot_identity(selected_frontier_key) == state.get("selected_frontier_key")
        and str(room_mask_geometry_hash or "") == str(state.get("room_mask_geometry_hash") or "")
        and reason == str(state.get("roomseg_update_reason") or "")
    )


def _roomseg_snapshot_failure_loop_reason(reason: str) -> bool:
    lowered = str(reason or "").lower()
    failure_tokens = (
        "no_path",
        "path_exhausted",
        "confirmed_unreachable",
        "guard_blocked",
        "no_progress",
        "no_clearance",
        "failed",
        "failure",
        "unreachable",
    )
    return any(token in lowered for token in failure_tokens)


def _snapshot_identity(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        try:
            if len(value) >= 2:  # type: ignore[arg-type]
                return (int(value[0]), int(value[1]))  # type: ignore[index]
        except Exception:
            return tuple(str(item) for item in value)
    return str(value)


def roomseg_snapshot_geometry_hash(
    room_masks: Sequence[object] | None = None,
    room_debug: Mapping[str, object] | None = None,
    fallback_hash: str | None = None,
) -> str:
    if fallback_hash:
        return str(fallback_hash)
    debug = dict(room_debug or {})
    for key in ("room_mask_geometry_hash", "room_geometry_hash"):
        value = debug.get(key)
        if value:
            return str(value)
    hasher = hashlib.blake2b(digest_size=16)
    for key in (
        "voxel_final_room_label_map",
        "final_room_label_map",
        "context_room_label_map",
        "room_label_map",
        "final_room_labels",
    ):
        value = debug.get(key)
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            hasher.update(str(key).encode("utf-8"))
            hasher.update(str(arr.shape).encode("utf-8"))
            hasher.update(str(arr.dtype).encode("utf-8"))
            hasher.update(arr.view(np.uint8))
            return hasher.hexdigest()
    rooms = list(room_masks or [])
    rooms.sort(key=lambda room: str(getattr(room, "room_id", "")))
    for room in rooms:
        mask = getattr(room, "mask", None)
        if mask is None:
            continue
        arr = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        hasher.update(str(getattr(room, "room_id", "")).encode("utf-8"))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(arr.view(np.uint8))
    return hasher.hexdigest()


def should_stop_explore_until_no_frontiers(
    *,
    explore_until_no_frontiers: bool,
    real_frontiers: Sequence[object],
    filtered_frontiers: Sequence[object],
    near_fallback_frontiers: Sequence[object],
    frontier_near_fallback_counts_as_exploration_frontier: bool = False,
) -> tuple[bool, str]:
    if not bool(explore_until_no_frontiers):
        return False, "not_explore_until_no_frontiers"
    if filtered_frontiers:
        return False, "frontiers_available"
    if real_frontiers:
        return True, "no_reachable_unconsumed_frontiers"
    if near_fallback_frontiers:
        if bool(frontier_near_fallback_counts_as_exploration_frontier):
            return False, "near_fallback_frontiers_available"
        return True, "no_real_frontiers_only_near_fallback"
    return True, "no_frontiers"


def load_preprocessed_for_episode(episode: dict):
    scene_dir = Path(episode["preprocessed_scene_dir"])
    with open(scene_dir / "map_info.json", "r", encoding="utf-8") as handle:
        map_info = MapInfo.from_dict(json.load(handle))
    occupancy = np.load(scene_dir / "occupancy.npy")
    navigable = np.load(scene_dir / "navigable.npy").astype(bool)
    return scene_dir, map_info, occupancy, navigable


def load_passable_opening_mask(scene_dir: Path, map_info: MapInfo) -> np.ndarray:
    objects_path = scene_dir / "objects_all.json"
    if objects_path.exists():
        from voxroom_online.isaac_runtime.dataset.occupancy_builder import rasterize_passable_openings

        with open(objects_path, "r", encoding="utf-8") as handle:
            objects_all = json.load(handle)
        return rasterize_passable_openings(objects_all, map_info).astype(bool)
    mask_path = scene_dir / "passable_openings.npy"
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        if mask.shape == (int(map_info.height), int(map_info.width)):
            return mask
    return np.zeros((int(map_info.height), int(map_info.width)), dtype=bool)


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def apply_voxel_cpu_numba_thread_override(
    room_segmentation_config: Mapping[str, object] | None,
    env: Mapping[str, str] | None = None,
) -> tuple[dict, int | None]:
    cfg = dict(room_segmentation_config or {})
    values = dict(env or os.environ)
    override_raw = (
        values.get("VOXROOM_VOXEL_CPU_NUMBA_THREADS")
        or values.get("VOXROOM_NUMBA_THREADS")
        or values.get("SGNAV_VOXEL_CPU_NUMBA_THREADS")
        or values.get("RUN_NUMBA_NUM_THREADS")
    )
    if not override_raw:
        return cfg, None
    override = _positive_int_or_none(override_raw)
    if override is None:
        raise ValueError(
            "VOXROOM_VOXEL_CPU_NUMBA_THREADS/RUN_NUMBA_NUM_THREADS must be a positive integer, "
            f"got {override_raw!r}"
        )
    voxel_grid_cfg = dict(cfg.get("voxel_grid", {}) or {})
    voxel_grid_cfg["cpu_numba_threads"] = int(override)
    voxel_grid_cfg["cpu_numba_threads_mode"] = "manual"
    voxel_grid_cfg["cpu_numba_autotune_candidates"] = [int(override)]
    cfg["voxel_grid"] = voxel_grid_cfg
    return cfg, int(override)


def _resolve_roomseg_depth_stride_px(room_segmentation_config: Mapping[str, object] | None, default_stride_px: int) -> int:
    """Return the depth stride used to build vertical-profile roomseg evidence.

    The vertical-free room-segmentation map is generated during OnlineMapper's
    depth ray pass. If roomseg asks for denser sampling, that request has to
    feed the mapper pass instead of only the later segmentation stage.
    """

    default_stride = max(1, int(default_stride_px))
    strides = [default_stride]
    cfg = dict(room_segmentation_config or {})
    candidate_blocks: list[Mapping[str, object]] = []
    top_level_depth = cfg.get("depth")
    if isinstance(top_level_depth, Mapping):
        candidate_blocks.append(top_level_depth)
    for section in ("voxel_grid", "voxel_occupancy_door_wall"):
        section_cfg = cfg.get(section)
        if not isinstance(section_cfg, Mapping):
            continue
        depth_cfg = section_cfg.get("depth")
        if isinstance(depth_cfg, Mapping):
            candidate_blocks.append(depth_cfg)
    for block in candidate_blocks:
        parsed = _positive_int_or_none(block.get("roomseg_depth_stride_px"))
        if parsed is None:
            parsed = _positive_int_or_none(block.get("depth_stride_px"))
        if parsed is not None:
            strides.append(parsed)
    return min(strides)


def resolve_frontier_source_layers(
    *,
    room_debug: Mapping[str, object] | None,
    mapper: object | None,
    navigation_free: np.ndarray,
    navigation_observed: np.ndarray,
    navigation_occupancy: np.ndarray,
    frontier_traversible: np.ndarray,
    source: str = "navigation",
    require_navigation_reachable: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    nav_free = np.asarray(navigation_free, dtype=bool)
    nav_observed = np.asarray(navigation_observed, dtype=bool)
    nav_occupancy = np.asarray(navigation_occupancy, dtype=bool)
    traversible = np.asarray(frontier_traversible, dtype=bool)
    if nav_free.shape != nav_observed.shape or nav_free.shape != nav_occupancy.shape or nav_free.shape != traversible.shape:
        raise ValueError("frontier source masks must share one HxW shape")
    src = str(source or "navigation").strip().lower()
    vertical_sources = {"vertical_free", "height_profile_vertical_free", "voxel_vertical_free"}
    if src not in vertical_sources:
        return nav_free & traversible, nav_observed, nav_occupancy, {
            "frontier_source": "navigation",
            "frontier_cells_from_navigation_free": int(np.count_nonzero(nav_free & traversible)),
        }

    debug = dict(room_debug or {})
    vfree = None
    vobserved = None
    vwall = None
    source_detail = "roomseg_debug"
    if src in {"vertical_free", "voxel_vertical_free"}:
        vfree = _debug_bool_array(debug, "voxel_vertical_free_xy", nav_free.shape)
        vobserved = _debug_bool_array(debug, "voxel_vertical_observed_xy", nav_free.shape)
        if vobserved is None:
            vobserved = _debug_bool_array(debug, "voxel_active_observed_xy", nav_free.shape)
        vwall = _debug_bool_array(debug, "voxel_wall_xy", nav_free.shape)
        if vfree is not None and vobserved is not None and vwall is not None:
            source_detail = "roomseg_debug_voxel"
    if (vfree is None or vobserved is None or vwall is None) and src in {"vertical_free", "height_profile_vertical_free"}:
        vfree = _debug_bool_array(debug, "height_profile_vertical_free_xy", nav_free.shape)
        vobserved = _debug_bool_array(debug, "height_profile_vertical_observed_xy", nav_free.shape)
        vwall = _debug_bool_array(debug, "height_profile_wall_xy", nav_free.shape)
        if vfree is not None and vobserved is not None and vwall is not None:
            source_detail = "roomseg_debug_height_profile"
    if vfree is None or vobserved is None or vwall is None:
        voxel_grid = getattr(mapper, "voxel_grid", None)
        if src in {"vertical_free", "voxel_vertical_free"} and voxel_grid is not None:
            try:
                evidence = build_voxel_roomseg_evidence(
                    voxel_grid=voxel_grid,
                    navigation_free_mask=nav_free,
                    navigation_obstacle_mask=nav_occupancy,
                    unknown_mask_from_navigation=~nav_observed,
                    resolution_m=float(getattr(getattr(mapper, "grid", None), "map_info", None).resolution_m),
                    config=None,
                )
                vfree = np.asarray(evidence.vertical_free_xy, dtype=bool)
                vobserved = np.asarray(evidence.active_observed_xy, dtype=bool)
                vwall = np.asarray(evidence.wall_xy, dtype=bool)
                source_detail = "mapper_voxel_grid"
            except Exception:
                vfree = vobserved = vwall = None
    if vfree is None or vobserved is None or vwall is None:
        height_profile = getattr(mapper, "height_profile", None)
        if src in {"vertical_free", "height_profile_vertical_free"} and height_profile is not None:
            hp_cfg = getattr(mapper, "height_profile_config", HeightColumnProfileConfig())
            cls = height_profile.classify_columns(navigation_free_mask=nav_free, cfg=hp_cfg)
            vfree = np.asarray(cls.vertical_free_xy, dtype=bool)
            vobserved = np.asarray(cls.observed_z_bin_count_xy > 0, dtype=bool)
            vwall = np.asarray(cls.wall_xy, dtype=bool)
            source_detail = "mapper_height_profile"
    if vfree is None or vobserved is None or vwall is None:
        fallback = nav_free & traversible
        return fallback, nav_observed, nav_occupancy, {
            "frontier_source": "navigation",
            "frontier_source_requested": src,
            "frontier_source_fallback_reason": "%s_unavailable" % src,
            "frontier_cells_from_navigation_free": int(np.count_nonzero(fallback)),
        }
    vertical_cells = int(np.count_nonzero(vfree))
    frontier_free = np.asarray(vfree, dtype=bool)
    removed = int(np.count_nonzero(frontier_free & ~traversible)) if bool(require_navigation_reachable) else 0
    if bool(require_navigation_reachable):
        frontier_free = frontier_free & traversible
    outside_mask = _debug_bool_array(room_debug, "voxel_outside_xy", frontier_free.shape)
    if outside_mask is None and hasattr(mapper, "voxel_grid"):
        grid = getattr(mapper, "voxel_grid", None)
        outside_candidate = getattr(grid, "outside_xy", None)
        if outside_candidate is not None:
            arr = np.asarray(outside_candidate, dtype=bool)
            if arr.shape == frontier_free.shape:
                outside_mask = arr
    outside_removed = 0
    outside_bool = None
    if outside_mask is not None:
        outside_bool = np.asarray(outside_mask, dtype=bool)
        outside_removed = int(np.count_nonzero(frontier_free & outside_bool))
        frontier_free = frontier_free & ~outside_bool
    source_name = "voxel_vertical_free" if source_detail in {"roomseg_debug_voxel", "mapper_voxel_grid"} or src == "voxel_vertical_free" else "vertical_free"
    if source_name == "voxel_vertical_free":
        frontier_observed = np.asarray(vobserved, dtype=bool) | np.asarray(vfree, dtype=bool) | np.asarray(vwall, dtype=bool)
    else:
        frontier_observed = np.asarray(vobserved, dtype=bool)
    frontier_occupancy = np.asarray(vwall, dtype=bool)
    if outside_bool is not None:
        frontier_observed = frontier_observed | outside_bool
        frontier_occupancy = frontier_occupancy & ~outside_bool
    return frontier_free.astype(bool), frontier_observed.astype(bool), frontier_occupancy.astype(bool), {
        "frontier_source": source_name,
        "frontier_source_requested": src,
        "frontier_source_detail": source_detail,
        "frontier_vertical_free_cells": vertical_cells,
        "frontier_vertical_observed_cells": int(np.count_nonzero(frontier_observed)),
        "frontier_vertical_wall_cells": int(np.count_nonzero(vwall)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(frontier_free)),
        "frontier_cells_removed_by_navigation_unreachable": int(removed),
        "frontier_cells_removed_by_voxel_outside": int(outside_removed),
        "frontier_vertical_free_require_navigation_reachable": bool(require_navigation_reachable),
    }


def _debug_bool_array(debug: Mapping[str, object], key: str, shape: tuple[int, int]) -> np.ndarray | None:
    value = debug.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        return None
    return arr


def _compact_json_debug(value: object, *, max_list_items: int = 64, _depth: int = 0) -> object:
    """Keep result JSON small; full debug arrays belong in snapshot npz files."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        summary: dict[str, object] = {
            "array_summary": True,
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "size": int(arr.size),
        }
        if arr.size:
            try:
                summary["nonzero"] = int(np.count_nonzero(arr))
            except Exception:
                pass
        if arr.size <= 16:
            summary["values"] = make_jsonable(arr.tolist())
        return summary
    if isinstance(value, Mapping):
        return {
            str(key): _compact_json_debug(item, max_list_items=max_list_items, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) <= int(max_list_items):
            return [
                _compact_json_debug(item, max_list_items=max_list_items, _depth=_depth + 1)
                for item in seq
            ]
        return {
            "list_summary": True,
            "length": int(len(seq)),
            "items": [
                _compact_json_debug(item, max_list_items=max_list_items, _depth=_depth + 1)
                for item in seq[: int(max_list_items)]
            ],
            "truncated_count": int(len(seq) - int(max_list_items)),
        }
    return value


def _roomseg_debug_for_layer_dump(room_debug: Mapping[str, object], room_segmenter: object | None) -> dict:
    debug = dict(room_debug or {})
    result = getattr(room_segmenter, "last_result", None)
    layers = getattr(result, "layers", None)
    if isinstance(layers, Mapping):
        aliases = {
            "vertical_free_room_domain": "vertical_free_raw",
            "vertical_occupied_0p2_2p0": "vertical_occupied_raw",
            "vertical_observed_map": "vertical_observed_raw",
            "vertical_observed_0p2_2p0": "vertical_observed_raw",
            "vertical_unknown_before_overlay": "vertical_unknown_raw",
            "navigation_free_room_domain": "free_clean",
            "initial_roomseg_free": "free_clean",
            "initial_roomseg_occupied": "wall_candidate_clean",
            "initial_roomseg_unknown": "unknown_clean",
            "repaired_roomseg_free": "free_clean",
            "repaired_roomseg_occupied": "wall_candidate_clean",
            "repaired_roomseg_unknown": "unknown_clean",
            "pass2_extension_intersection_targets": "pass2_extension_intersection_targets",
            "pass2_line_extension_completion": "pass2_line_extension_completion",
            "wall_target_after_line_extension": "wall_target_after_line_extension",
            "completed_wall_after_line_extension": "completed_wall_after_line_extension",
            "height_profile_vertical_free_xy": "height_profile_vertical_free_xy",
            "height_profile_wall_xy": "height_profile_wall_xy",
            "height_profile_unknown_xy": "height_profile_unknown_xy",
            "height_profile_door_seed_mask": "height_profile_door_seed_mask",
            "height_profile_door_cut_mask": "height_profile_door_cut_mask",
            "height_profile_step1_wall_gap_fill_map": "height_profile_step1_wall_gap_fill_map",
            "height_profile_step2_extension_separator_map": "height_profile_step2_extension_separator_map",
            "height_profile_final_room_label_map": "height_profile_final_room_label_map",
            "height_profile_boundary_source_map": "height_profile_boundary_source_map",
            "vertical_free_clipped_outside_navigation_map": "vertical_free_clipped_outside_navigation_map",
            "free_wall_conflict_map_before_sanitize": "free_wall_conflict_map_before_sanitize",
            "roomseg_sanitized_free": "roomseg_sanitized_free",
            "roomseg_sanitized_wall": "roomseg_sanitized_wall",
            "terminal_wall_roomseg_mask": "terminal_wall_roomseg_mask",
            "pre_extension_door_detected_map": "pre_extension_door_detected_map",
            "pre_extension_door_cut_mask": "pre_extension_door_cut_mask",
            "pre_extension_door_pattern_type_map": "pre_extension_door_pattern_type_map",
            "strict_pre_extension_door_cut_mask": "strict_pre_extension_door_cut_mask",
            "partial_door_seed_mask": "partial_door_seed_mask",
            "partial_door_line_mask": "partial_door_line_mask",
            "partial_door_extension_cut_mask": "partial_door_extension_cut_mask",
            "rejected_door_extension_mask": "rejected_door_extension_mask",
            "original_step1_step2_virtual_boundary_map": "original_step1_step2_virtual_boundary_map",
            "wall_extension_boundary_mask": "wall_extension_boundary_mask",
            "door_completion_boundary_mask": "door_completion_boundary_mask",
            "virtual_boundary_source_map": "virtual_boundary_source_map",
            "pre_extension_partition_free": "pre_extension_partition_free",
            "pre_extension_room_label_map": "pre_extension_room_label_map",
            "step1_step2_accepted_closure_map": "step1_step2_accepted_separator_map",
            "step1_step2_accepted_separator_map": "step1_step2_accepted_separator_map",
            "structural_free_mask": "free_clean",
            "wall_boundary_map": "wall_candidate_clean",
            "candidate_wall": "wall_candidate_clean",
            "final_room_label_map": "room_labels_after_corridor_merge",
            "room_labels_after_merge": "room_labels_after_corridor_merge",
            "room_proposal_labels_before_merge": "room_labels_before_separators",
        }
        for out_key, layer_key in aliases.items():
            if out_key not in debug and layer_key in layers:
                debug[out_key] = np.asarray(layers[layer_key])
    if "final_room_label_map" not in debug and result is not None and hasattr(result, "room_label_map"):
        debug["final_room_label_map"] = np.asarray(getattr(result, "room_label_map"))
    return debug


def _roomseg_voxel_snapshot_arrays(mapper: object) -> dict[str, np.ndarray]:
    if not hasattr(mapper, "roomseg_voxel_evidence"):
        return {}
    evidence = getattr(mapper, "roomseg_voxel_evidence")()
    state = np.asarray(evidence.get("state"), dtype=np.uint8)
    if state.ndim != 3:
        return {}
    log_odds = np.asarray(evidence.get("log_odds", np.zeros_like(state, dtype=np.int16)), dtype=np.int16)
    if log_odds.shape != state.shape:
        log_odds = np.zeros_like(state, dtype=np.int16)
    sensor_range_count = np.asarray(evidence.get("sensor_range_count", np.zeros_like(state, dtype=np.uint8)), dtype=np.uint8)
    if sensor_range_count.shape != state.shape:
        sensor_range_count = np.zeros_like(state, dtype=np.uint8)
    z_min = float(evidence.get("z_min_m", 0.0))
    z_res = float(evidence.get("z_resolution_m", 1.0))
    z_centers = z_min + (np.arange(state.shape[0], dtype=np.float32) + 0.5) * z_res
    nav_endpoint = np.asarray(evidence.get("voxel_nav_occupied_endpoint_count_xy", np.zeros(state.shape[1:], dtype=np.uint16)), dtype=np.uint16)
    nav_free_ray = np.asarray(evidence.get("voxel_nav_free_ray_count_xy", np.zeros(state.shape[1:], dtype=np.uint16)), dtype=np.uint16)
    if nav_endpoint.shape != state.shape[1:]:
        nav_endpoint = np.zeros(state.shape[1:], dtype=np.uint16)
    if nav_free_ray.shape != state.shape[1:]:
        nav_free_ray = np.zeros(state.shape[1:], dtype=np.uint16)

    def scalar(value: object, default: float = np.nan) -> np.ndarray:
        if value is None:
            return np.asarray(default, dtype=np.float32)
        return np.asarray(float(value), dtype=np.float32)

    return {
        "voxel_occupancy_state_zyx": state,
        "voxel_occupancy_log_odds_zyx": log_odds,
        "voxel_sensor_range_count_zyx": sensor_range_count,
        "voxel_nav_occupied_endpoint_count_xy": nav_endpoint,
        "voxel_nav_free_ray_count_xy": nav_free_ray,
        "voxel_occupancy_z_centers_m": z_centers.astype(np.float32),
        "voxel_occupancy_z_min_m": scalar(evidence.get("z_min_m")),
        "voxel_occupancy_z_max_m": scalar(evidence.get("z_max_m")),
        "voxel_occupancy_z_resolution_m": scalar(evidence.get("z_resolution_m")),
        "voxel_occupancy_active_z_min_m": scalar(evidence.get("active_z_min_m")),
        "voxel_occupancy_active_z_max_m": scalar(evidence.get("active_z_max_m")),
        "voxel_occupancy_ceiling_height_estimate_m": scalar(evidence.get("ceiling_height_estimate_m")),
        "voxel_occupancy_ceiling_estimate_status": np.asarray(str(evidence.get("ceiling_estimate_status", "snapshot"))),
        "voxel_ceiling_estimate_status": np.asarray(str(evidence.get("ceiling_estimate_status", "snapshot"))),
    }


def _roomseg_memory_snapshot_arrays(room_segmenter: object | None) -> dict[str, np.ndarray]:
    if room_segmenter is None:
        return {}
    debug = dict(getattr(room_segmenter, "last_debug", {}) or {})
    out: dict[str, np.ndarray] = {}
    for key in (
        "voxel_roomseg_memory_before_json",
        "voxel_roomseg_memory_after_json",
        "voxel_door_memory_before_roomseg_json",
        "voxel_door_memory_after_roomseg_json",
        "voxel_separator_memory_before_roomseg_json",
        "voxel_separator_memory_after_roomseg_json",
    ):
        if key in debug:
            out[key] = np.asarray(str(debug[key]))
    return out


def apply_episode_planning_clearance(
    scene_dir: Path,
    map_info: MapInfo,
    navigable: np.ndarray,
    episode: dict,
    runtime_planning_clearance_m: Optional[float] = None,
) -> np.ndarray:
    if runtime_planning_clearance_m is None:
        min_clearance = float(episode.get("metadata", {}).get("min_planning_clearance_m", 0.0))
    else:
        min_clearance = float(runtime_planning_clearance_m)
    if min_clearance <= 0.0:
        return navigable
    objects_all_path = scene_dir / "objects_all.json"
    if not objects_all_path.exists():
        return navigable
    from voxroom_online.isaac_runtime.dataset.episode_generator import build_clearance_mask, filter_start_clearance_objects

    with open(objects_all_path, "r", encoding="utf-8") as handle:
        clearance_objects = filter_start_clearance_objects(json.load(handle))
    return build_clearance_mask(navigable, map_info, clearance_objects, min_clearance)


def apply_dynamic_astar_edge_clearance(
    traversible: np.ndarray,
    occupied_walls: np.ndarray,
    resolution_m: float,
    extra_clearance_m: float,
    current_grid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Keep A* an extra margin away from raw occupied wall/obstacle cells."""
    out = np.asarray(traversible, dtype=bool).copy()
    clearance = max(0.0, float(extra_clearance_m))
    resolution = max(1e-6, float(resolution_m))
    radius_cells = int(math.ceil(clearance / resolution))
    if radius_cells > 0:
        out &= ~_disk_dilate_bool(np.asarray(occupied_walls, dtype=bool), radius_cells)
    if current_grid is not None:
        rr, cc = int(current_grid[0]), int(current_grid[1])
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and bool(np.asarray(traversible, dtype=bool)[rr, cc]):
            out[rr, cc] = True
    return out


def clearance_map_m(traversible: np.ndarray, resolution_m: float) -> np.ndarray:
    free = np.asarray(traversible, dtype=bool)
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(free).astype(np.float32) * float(resolution_m)
    except Exception:
        planner = ClearanceAStarPlanner(free, resolution_m, allow_diagonal=True)
        return np.asarray(planner.clearance_m, dtype=np.float32)


def effective_lookahead_min_clearance_m(
    configured_min_clearance_m: float,
    *,
    robot_radius_m: float,
    runtime_planning_clearance_m: float = 0.0,
    astar_clearance_hard_min_m: float = 0.0,
) -> float:
    """Keep lookahead validation at least as strict as config and A* policy."""
    configured = max(0.0, float(configured_min_clearance_m))
    planner_floor = max(
        0.0,
        float(robot_radius_m) + max(0.0, float(runtime_planning_clearance_m)),
        float(astar_clearance_hard_min_m),
    )
    if configured <= 0.0:
        return planner_floor
    return max(configured, planner_floor)


def clearance_policy_debug(
    configured_min_clearance_m: float,
    *,
    robot_radius_m: float,
    runtime_planning_clearance_m: float = 0.0,
    astar_clearance_hard_min_m: float = 0.0,
) -> dict[str, object]:
    configured = max(0.0, float(configured_min_clearance_m))
    planner_floor = max(
        0.0,
        float(robot_radius_m) + max(0.0, float(runtime_planning_clearance_m)),
        float(astar_clearance_hard_min_m),
    )
    effective = planner_floor if configured <= 0.0 else max(configured, planner_floor)
    return {
        "lookahead_clearance_configured_m": float(configured),
        "lookahead_clearance_planner_floor_m": float(planner_floor),
        "lookahead_clearance_effective_m": float(effective),
        "lookahead_clearance_policy": "max(configured, planner_floor)",
    }


def make_runtime_nav_planner(args, traversible: np.ndarray, occupancy: np.ndarray, resolution_m: float):
    if bool(getattr(args, "astar_clearance_cost_enabled", False)):
        planner = ClearanceAStarPlanner(
            traversible,
            resolution_m,
            occupied=occupancy,
            allow_diagonal=bool(getattr(args, "astar_allow_diagonal", True)),
            clearance_desired_m=float(getattr(args, "astar_clearance_desired_m", 0.25)),
            clearance_weight=float(getattr(args, "astar_clearance_weight", 3.0)),
            clearance_power=float(getattr(args, "astar_clearance_power", 2.0)),
            clearance_hard_min_m=float(getattr(args, "astar_clearance_hard_min_m", 0.0)),
        )
        return planner, np.asarray(planner.clearance_m, dtype=np.float32), True
    planner = GridAStarPlanner(traversible, resolution_m, allow_diagonal=bool(getattr(args, "astar_allow_diagonal", True)))
    return planner, clearance_map_m(traversible, resolution_m), False


def snap_to_free_local(navigable: np.ndarray, current_cell: tuple[int, int], radius_cells: int = 3) -> tuple[int, int] | None:
    nav = np.asarray(navigable, dtype=bool)
    if nav.ndim != 2:
        return None
    row, col = int(current_cell[0]), int(current_cell[1])
    h, w = nav.shape
    if 0 <= row < h and 0 <= col < w and bool(nav[row, col]):
        return (row, col)
    radius = max(0, int(radius_cells))
    best: tuple[float, int, int] | None = None
    for dr in range(-radius, radius + 1):
        rr = row + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(-radius, radius + 1):
            cc = col + dc
            if cc < 0 or cc >= w or not bool(nav[rr, cc]):
                continue
            dist2 = float(dr * dr + dc * dc)
            if best is None or dist2 < best[0]:
                best = (dist2, rr, cc)
    if best is None:
        return None
    return (int(best[1]), int(best[2]))


def _disk_dilate_bool(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    try:
        from scipy import ndimage

        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure = (yy * yy + xx * xx) <= radius * radius
        return ndimage.binary_dilation(src, structure=structure).astype(bool)
    except Exception:
        out = np.array(src, copy=True)
        rows, cols = np.nonzero(src)
        h, w = src.shape
        offsets = [
            (dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius
        ]
        for row, col in zip(rows, cols):
            for dr, dc in offsets:
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
        return out


def apply_success_distance_override(episode: dict, args) -> dict:
    success_distance_m = getattr(args, "success_distance_m", None)
    if success_distance_m is None:
        return dict(episode)
    success_distance = float(success_distance_m)
    updated = dict(episode)
    scene_dir, map_info, _occupancy, navigable = load_preprocessed_for_episode(updated)
    navigable = apply_episode_planning_clearance(
        scene_dir,
        map_info,
        navigable,
        updated,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    goal_objects = load_episode_goal_objects(scene_dir, updated)
    goal_cells = build_exact_goal_cells(goal_objects, navigable, map_info, success_distance)
    if goal_cells:
        start = (int(updated["start_grid"][0]), int(updated["start_grid"][1]))
        planner = GridAStarPlanner(navigable, map_info.resolution_m, allow_diagonal=True)
        shortest = planner.distance(start, goal_cells)
        if math.isfinite(shortest):
            updated["goal_regions_grid"] = [[int(r), int(c)] for r, c in goal_cells]
            updated["shortest_path_distance_m"] = float(shortest)
    updated["success_distance_m"] = success_distance
    metadata = dict(updated.get("metadata", {}))
    metadata["runtime_success_distance_override_m"] = success_distance
    updated["metadata"] = metadata
    return updated


def effective_perception_every_steps(detector_name: str | None, requested_steps: int | None) -> int:
    requested = max(1, int(requested_steps or 1))
    if str(detector_name or "").strip().lower() in {"yolo_world", "grounding_dino"}:
        return 1
    return requested


def load_episode_goal_objects(scene_dir: Path, episode: Mapping[str, object]) -> List[Mapping[str, object]]:
    objects_path = scene_dir / "objects.json"
    with open(objects_path, "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    goal_ids = {str(value) for value in episode.get("goal_instance_ids", []) or []}
    if goal_ids:
        matched = [obj for obj in objects if str(obj.get("instance_id")) in goal_ids]
        if matched:
            return matched
    goal_category = normalize_category(str(episode.get("goal_category", "")))
    return [obj for obj in objects if normalize_category(str(obj.get("category", ""))) == goal_category]


def build_exact_goal_cells(
    goal_objects: Sequence[Mapping[str, object]],
    navigable: np.ndarray,
    map_info: MapInfo,
    success_distance_m: float,
) -> List[Tuple[int, int]]:
    cells = []
    fallback: List[Tuple[int, int]] = []
    best_distance = math.inf
    for r, c in np.argwhere(np.asarray(navigable).astype(bool)):
        rr, cc = int(r), int(c)
        wx, wy = grid_to_world_xy(rr, cc, map_info)
        distance = min_goal_bbox_distance(wx, wy, goal_objects)
        if not math.isfinite(distance):
            continue
        if distance <= float(success_distance_m):
            cells.append((rr, cc))
            continue
        if distance + 1e-6 < best_distance:
            best_distance = distance
            fallback = [(rr, cc)]
        elif abs(distance - best_distance) <= 1e-6:
            fallback.append((rr, cc))
    return cells if cells else fallback


def min_goal_bbox_distance(x: float, y: float, goal_objects: Sequence[Mapping[str, object]]) -> float:
    distances = []
    for obj in goal_objects:
        bbox_min = obj.get("bbox_min_world")
        bbox_max = obj.get("bbox_max_world")
        if bbox_min is None or bbox_max is None:
            continue
        distances.append(point_to_bbox_2d_distance(x, y, bbox_min, bbox_max))
    return min(distances) if distances else math.inf


def maybe_build_detector(
    detector_name: str,
    model_path: str,
    categories: List[str],
    conf: float = 0.7,
    iou: float = 0.5,
    allow_ipc_fallback: bool = False,
):
    try:
        detector = build_detector(
            detector_name,
            model_path,
            conf=conf,
            iou=iou,
            grounding_dino_config=getattr(maybe_build_detector, "grounding_dino_config", None),
            grounding_dino_text_threshold=float(getattr(maybe_build_detector, "grounding_dino_text_threshold", 0.25)),
            grounding_dino_device=str(getattr(maybe_build_detector, "grounding_dino_device", "cuda")),
        )
    except Exception as exc:
        if detector_name not in {"yolo_world", "grounding_dino"} or not allow_ipc_fallback:
            raise
        print(
            "[detector-ipc] direct %s load failed in this process; using external auxiliary detector env worker: %s"
            % (detector_name, exc),
            flush=True,
        )
        detector = SubprocessDetector(
            detector_name,
            model_path,
            conf=conf,
            iou=iou,
            grounding_dino_config=str(getattr(maybe_build_detector, "grounding_dino_config", "") or ""),
            grounding_dino_text_threshold=float(getattr(maybe_build_detector, "grounding_dino_text_threshold", 0.25)),
            grounding_dino_device=str(getattr(maybe_build_detector, "grounding_dino_device", "cuda")),
        )
    detector.set_vocabulary(categories)
    return detector


def load_scene_categories(scene_dir: Path) -> List[str]:
    with open(scene_dir / "objects.json", "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    return sorted({str(obj.get("category", "unknown")) for obj in objects})


def ensure_detector_loaded(args, scene_dir: Path, allow_ipc_fallback: bool = False) -> None:
    if args.detector == "none":
        old_detector = getattr(args, "_detector_instance", None)
        if old_detector is not None and hasattr(old_detector, "close"):
            old_detector.close()
        args._detector_instance = None
        args._detector_key = None
        return
    categories = load_scene_categories(scene_dir)
    conf = max(float(getattr(args, "detector_conf", 0.7)), float(getattr(args, "min_valid_detection_confidence", MIN_VALID_DETECTION_CONFIDENCE)))
    iou = float(getattr(args, "detector_iou", 0.5))
    model_path = detector_model_path(args)
    grounding_dino_config = str(getattr(args, "grounding_dino_config", "") or "")
    grounding_dino_text_threshold = float(getattr(args, "grounding_dino_text_threshold", 0.25))
    grounding_dino_device = str(getattr(args, "grounding_dino_device", "cuda") or "cuda")
    key = (
        str(args.detector),
        str(model_path),
        conf,
        iou,
        grounding_dino_config,
        grounding_dino_text_threshold,
        grounding_dino_device,
        bool(allow_ipc_fallback),
        tuple(categories),
    )
    if getattr(args, "_detector_key", None) == key:
        return
    old_detector = getattr(args, "_detector_instance", None)
    if old_detector is not None and hasattr(old_detector, "close"):
        old_detector.close()
    args._detector_instance = None
    args._detector_key = None
    maybe_build_detector.grounding_dino_config = grounding_dino_config
    maybe_build_detector.grounding_dino_text_threshold = grounding_dino_text_threshold
    maybe_build_detector.grounding_dino_device = grounding_dino_device
    args._detector_instance = maybe_build_detector(
        args.detector,
        model_path,
        categories,
        conf=conf,
        iou=iou,
        allow_ipc_fallback=allow_ipc_fallback,
    )
    args._detector_key = key


def detector_model_path(args) -> str:
    detector = str(getattr(args, "detector", "") or "").strip().lower()
    if detector == "grounding_dino":
        return str(getattr(args, "grounding_dino_checkpoint", "") or "")
    return str(getattr(args, "yolo_world_model", "") or "")


def frontier_room_contexts_for_debug(
    *,
    frontiers: Sequence[object],
    room_debug: Mapping[str, object],
    room_masks: Sequence[object],
    room_semantic_labels: Mapping[str, object],
    observed_free: np.ndarray,
    unknown: np.ndarray,
    agent_grid: Tuple[int, int],
    resolution_m: float,
    config: Mapping[str, object],
) -> list[dict]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", True)) or not bool(cfg.get("use_known_free_side", True)):
        return []
    label_map_key = "context_room_label_map" if bool(cfg.get("use_context_overlay_labels", True)) else "final_room_label_map"
    labels = np.asarray(room_debug.get(label_map_key, room_debug.get("final_room_label_map", [])), dtype=np.int32)
    if labels.shape != np.asarray(observed_free).shape:
        return []
    label_to_room = _label_id_to_room_metadata(room_masks, room_semantic_labels)
    out: list[dict] = []
    for index, frontier in enumerate(frontiers):
        members = getattr(frontier, "members", [])
        context = assign_frontier_room_context(
            members,
            labels,
            observed_free,
            unknown,
            agent_grid,
            resolution_m,
            local_radius_m=float(cfg.get("local_radius_m", 0.35)),
            nearest_fallback_radius_m=float(cfg.get("nearest_fallback_radius_m", 1.25)),
            min_label_ratio=float(cfg.get("min_label_ratio", 0.20)),
        )
        meta = label_to_room.get(int(context["room_id"])) if context.get("room_id") is not None else None
        out.append(
            {
                "frontier_id": int(index),
                "center_grid": [int(v) for v in getattr(frontier, "center_grid", (-1, -1))],
                "room_context": {
                    **context,
                    "room_id": meta.get("room_id") if meta else context.get("room_id"),
                    "room_label_id": int(context["room_id"]) if context.get("room_id") is not None else None,
                    "room_label": meta.get("category", "unknown") if meta else "unknown",
                },
            }
        )
    return out


def _label_id_to_room_metadata(room_masks: Sequence[object], room_semantic_labels: Mapping[str, object]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for room in room_masks:
        label_id = int((getattr(room, "metadata", {}) or {}).get("label_id", 0) or 0)
        if label_id <= 0:
            continue
        room_id = str(getattr(room, "room_id", ""))
        semantic = room_semantic_labels.get(room_id)
        out[label_id] = {
            "room_id": room_id,
            "category": str(getattr(semantic, "category", "unknown") if semantic is not None else "unknown"),
        }
    return out


def ensure_segmenter_loaded(args) -> None:
    mode = str(getattr(args, "segmenter", "none") or "none").strip().lower()
    if mode in {"none", "false", "0", ""}:
        args._segmenter_instance = None
        args._segmenter_key = None
        return
    key = (
        mode,
        str(getattr(args, "sam2_checkpoint", "") or ""),
        str(getattr(args, "sam2_model_cfg", "") or ""),
        str(getattr(args, "sam2_device", "cuda") or "cuda"),
    )
    if getattr(args, "_segmenter_key", None) == key:
        return
    args._segmenter_instance = build_sam2_segmenter(
        mode=mode,
        checkpoint=str(getattr(args, "sam2_checkpoint", "") or ""),
        model_cfg=str(getattr(args, "sam2_model_cfg", "") or ""),
        device=str(getattr(args, "sam2_device", "cuda") or "cuda"),
        required=mode == "sam2",
    )
    args._segmenter_key = key


def observed_room_map(full_room_map: Optional[np.ndarray], observed: np.ndarray) -> Optional[np.ndarray]:
    if full_room_map is None:
        return None
    return full_room_map * observed.astype(np.float32)[None, None, :, :]


def path_cells_to_world(path_cells: Iterable[Tuple[int, int]], map_info: MapInfo) -> List[Tuple[float, float]]:
    return [grid_to_world_xy(int(r), int(c), map_info) for r, c in path_cells]


def distance_to_target_cells_m(
    current_grid: Tuple[int, int],
    target_cells: Iterable[Tuple[int, int]],
    resolution_m: float,
) -> float:
    cells = [tuple(int(v) for v in cell) for cell in target_cells]
    if not cells:
        return float("inf")
    cur = np.asarray(current_grid, dtype=np.float32)
    arr = np.asarray(cells, dtype=np.float32)
    return float(np.min(np.linalg.norm(arr - cur[None, :], axis=1)) * float(resolution_m))


def navigation_target_reached(
    current_grid: Tuple[int, int],
    nav_decision: Optional[NavigationDecision],
    resolution_m: float,
    reached_radius_m: float,
) -> Tuple[bool, float]:
    if nav_decision is None or nav_decision.mode not in {"frontier", "candidate", "reperception"}:
        return False, float("inf")
    distance_m = distance_to_target_cells_m(current_grid, nav_decision.target_cells or [], resolution_m)
    return bool(distance_m <= float(reached_radius_m)), float(distance_m)


def planning_target_cells_within_radius(
    target_cells: Iterable[Tuple[int, int]],
    traversible: np.ndarray,
    resolution_m: float,
    radius_m: float,
) -> List[Tuple[int, int]]:
    cells = [tuple(int(v) for v in cell) for cell in target_cells]
    if not cells:
        return []
    nav = np.asarray(traversible, dtype=bool)
    h, w = nav.shape
    radius_cells = max(0, int(math.ceil(float(radius_m) / max(float(resolution_m), 1e-6))))
    candidates: dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for target_idx, (row, col) in enumerate(cells):
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                rr, cc = int(row + dr), int(col + dc)
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not bool(nav[rr, cc]):
                    continue
                key = (rr, cc)
                rank = (int(dr * dr + dc * dc), int(target_idx), int(abs(dr) + abs(dc)))
                if key not in candidates or rank < candidates[key]:
                    candidates[key] = rank
    return [cell for cell, _rank in sorted(candidates.items(), key=lambda item: item[1])]


def filter_goal_cells_by_clearance_result(
    goal_cells: Iterable[Tuple[int, int]],
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    *,
    min_clearance_m: float,
    search_radius_cells: int,
) -> ClearanceGoalFilterResult:
    cells = [tuple(int(v) for v in cell) for cell in goal_cells]
    if not cells:
        return ClearanceGoalFilterResult([], 0, 0, 0, max(0.0, float(min_clearance_m)), max(0, int(search_radius_cells)), [])
    nav = np.asarray(traversible, dtype=bool)
    clearance = np.asarray(clearance_m, dtype=np.float32)
    h, w = nav.shape
    min_clearance = max(0.0, float(min_clearance_m))
    radius = max(0, int(search_radius_cells))
    out: list[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    rejected_original_count = 0
    snapped_count = 0
    no_safe_goal_count = 0
    debug_by_input: list[dict] = []
    for row, col in cells:
        input_cell = (int(row), int(col))
        original_clearance = float(clearance[row, col]) if 0 <= row < h and 0 <= col < w else 0.0
        if 0 <= row < h and 0 <= col < w and bool(nav[row, col]) and float(clearance[row, col]) >= min_clearance:
            candidate = (int(row), int(col))
            reason = "original_clearance_ok"
        else:
            rejected_original_count += 1
            best_cell = None
            best_rank = None
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc > radius * radius:
                        continue
                    rr, cc = int(row + dr), int(col + dc)
                    if rr < 0 or rr >= h or cc < 0 or cc >= w or not bool(nav[rr, cc]):
                        continue
                    clear = float(clearance[rr, cc])
                    if clear < min_clearance:
                        continue
                    rank = (-clear, int(dr * dr + dc * dc), int(abs(dr) + abs(dc)))
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_cell = (rr, cc)
            if best_cell is None:
                no_safe_goal_count += 1
                debug_by_input.append(
                    {
                        "input_cell": [int(input_cell[0]), int(input_cell[1])],
                        "input_clearance_m": float(original_clearance),
                        "selected_cell": None,
                        "reason": "no_clearance_safe_goal_nearby",
                    }
                )
                continue
            candidate = best_cell
            snapped_count += int(candidate != input_cell)
            reason = "snapped_to_clearance_safe_goal" if candidate != input_cell else "same_cell_after_search"
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
        debug_by_input.append(
            {
                "input_cell": [int(input_cell[0]), int(input_cell[1])],
                "input_clearance_m": float(original_clearance),
                "selected_cell": [int(candidate[0]), int(candidate[1])],
                "selected_clearance_m": float(clearance[int(candidate[0]), int(candidate[1])]),
                "reason": reason,
            }
        )
    return ClearanceGoalFilterResult(
        out,
        int(rejected_original_count),
        int(snapped_count),
        int(no_safe_goal_count),
        float(min_clearance),
        int(radius),
        debug_by_input,
    )


def filter_goal_cells_by_clearance(
    goal_cells: Iterable[Tuple[int, int]],
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    *,
    min_clearance_m: float,
    search_radius_cells: int,
) -> List[Tuple[int, int]]:
    return filter_goal_cells_by_clearance_result(
        goal_cells,
        traversible,
        clearance_m,
        min_clearance_m=min_clearance_m,
        search_radius_cells=search_radius_cells,
    ).goals


def path_clearance_debug(path: Sequence[Tuple[int, int]], clearance_m: np.ndarray, robot_radius_m: float) -> dict[str, object]:
    if not path:
        return {
            "astar_path_min_clearance_m": None,
            "astar_path_mean_clearance_m": None,
            "astar_path_too_close_cells": 0,
        }
    clearance = np.asarray(clearance_m, dtype=np.float32)
    values: list[float] = []
    too_close = 0
    h, w = clearance.shape
    for row, col in path:
        rr, cc = int(row), int(col)
        if 0 <= rr < h and 0 <= cc < w:
            value = float(clearance[rr, cc])
            values.append(value)
            if value < float(robot_radius_m):
                too_close += 1
    if not values:
        return {
            "astar_path_min_clearance_m": None,
            "astar_path_mean_clearance_m": None,
            "astar_path_too_close_cells": 0,
        }
    out: dict[str, object] = {
        "astar_path_min_clearance_m": float(min(values)),
        "astar_path_mean_clearance_m": float(sum(values) / len(values)),
        "astar_path_too_close_cells": int(too_close),
    }
    if float(out["astar_path_min_clearance_m"]) < float(robot_radius_m):
        out["astar_path_warning"] = "astar_path_below_robot_radius_clearance"
    return out


def segment_is_clear_world(
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    map_info: MapInfo,
    min_clearance_m: float,
) -> bool:
    nav = np.asarray(traversible, dtype=bool)
    clearance = np.asarray(clearance_m, dtype=np.float32)
    dx = float(end_xy[0]) - float(start_xy[0])
    dy = float(end_xy[1]) - float(start_xy[1])
    step_m = max(0.02, float(map_info.resolution_m) * 0.5)
    samples = max(1, int(math.ceil(math.hypot(dx, dy) / step_m)))
    h, w = nav.shape
    for idx in range(samples + 1):
        t = float(idx) / float(samples)
        x = float(start_xy[0]) + dx * t
        y = float(start_xy[1]) + dy * t
        row, col = world_xy_to_grid(x, y, map_info)
        if row < 0 or row >= h or col < 0 or col >= w or not bool(nav[row, col]):
            return False
        if float(clearance[row, col]) < float(min_clearance_m):
            return False
    return True


def collision_checked_path_world(
    pose: Sequence[float],
    path_cells: Sequence[Tuple[int, int]],
    map_info: MapInfo,
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    *,
    lookahead_m: float,
    min_clearance_m: float,
    max_skip_cells: int,
) -> List[Tuple[float, float]]:
    candidates = list(path_cells[1 : min(len(path_cells), max(2, int(max_skip_cells) + 2))])
    if not candidates:
        return []
    start_xy = (float(pose[0]), float(pose[1]))
    fallback: list[Tuple[float, float]] = []
    for idx, cell in enumerate(candidates):
        target = grid_to_world_xy(int(cell[0]), int(cell[1]), map_info)
        if not fallback:
            fallback = [target]
        if math.hypot(float(target[0]) - start_xy[0], float(target[1]) - start_xy[1]) < float(lookahead_m) and idx < len(candidates) - 1:
            continue
        if segment_is_clear_world(start_xy, target, traversible, clearance_m, map_info, float(min_clearance_m)):
            tail = path_cells[1 + idx + 1 : min(len(path_cells), 20)]
            return [target] + path_cells_to_world(tail, map_info)
    if fallback and segment_is_clear_world(start_xy, fallback[0], traversible, clearance_m, map_info, float(min_clearance_m)):
        return fallback
    return []


@dataclass
class LongTermGoalState:
    mode: str = "none"
    target_cells: List[Tuple[int, int]] = field(default_factory=list)
    center_grid: Optional[Tuple[int, int]] = None
    selected_step: int = -1
    reached: bool = False
    invalid_reason: str = ""
    nav_decision: Optional[NavigationDecision] = None

    def exists(self) -> bool:
        return self.mode not in {"", "none"} and bool(self.target_cells) and not self.invalid_reason

    def set_from(self, nav_decision: NavigationDecision, step: int) -> None:
        self.mode = str(nav_decision.mode or "none")
        self.target_cells = [tuple(int(v) for v in cell) for cell in (nav_decision.target_cells or [])]
        self.center_grid = self._center_from_decision(nav_decision)
        self.selected_step = int(step)
        self.reached = False
        self.invalid_reason = ""
        self.nav_decision = nav_decision

    def clear(self, reason: str = "cleared") -> None:
        self.mode = "none"
        self.target_cells = []
        self.center_grid = None
        self.selected_step = -1
        self.reached = reason == "reached"
        self.invalid_reason = ""
        self.nav_decision = None

    def invalidate(self, reason: str) -> None:
        self.invalid_reason = str(reason)
        self.mode = "none"
        self.target_cells = []
        self.center_grid = None
        self.nav_decision = None

    def to_navigation_decision(self) -> NavigationDecision:
        if self.nav_decision is None:
            return NavigationDecision(
                self.mode,
                list(self.target_cells),
                False,
                None,
                None,
                "continue_long_term_goal",
                metadata={
                    "long_term_goal_locked": True,
                    "long_term_goal_lock_reason": "locked_long_term_goal",
                    "long_term_goal_selected_step": int(self.selected_step),
                    "long_term_goal_mode": self.mode,
                },
            )
        metadata = {
            **dict(self.nav_decision.metadata or {}),
            "long_term_goal_locked": True,
            "long_term_goal_lock_reason": "locked_long_term_goal",
            "long_term_goal_selected_step": int(self.selected_step),
            "long_term_goal_mode": self.mode,
        }
        return NavigationDecision(
            self.nav_decision.mode,
            list(self.target_cells),
            bool(self.nav_decision.stop),
            self.nav_decision.selected_candidate,
            self.nav_decision.frontier_decision,
            self.nav_decision.reason,
            state=self.nav_decision.state,
            metadata=metadata,
        )

    @staticmethod
    def _center_from_decision(nav_decision: NavigationDecision) -> Optional[Tuple[int, int]]:
        if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
            return tuple(int(v) for v in nav_decision.frontier_decision.selected_frontier.center_grid)
        if nav_decision.selected_candidate is not None:
            return tuple(int(v) for v in nav_decision.selected_candidate.center_grid)
        if nav_decision.target_cells:
            return tuple(int(v) for v in nav_decision.target_cells[0])
        return None


def trim_path_to_current(path: List[Tuple[int, int]], current_grid: Tuple[int, int]) -> List[Tuple[int, int]]:
    for idx, cell in enumerate(path):
        if cell == current_grid:
            return path[idx:]
    return []


@dataclass(frozen=True)
class PathTrimResult:
    suffix: List[Tuple[int, int]]
    success: bool
    nearest_index: Optional[int]
    nearest_distance_cells: Optional[float]
    reason: str


def trim_path_to_nearest_with_info(
    path: List[Tuple[int, int]],
    current_grid: Tuple[int, int],
    max_dist_cells: int = 3,
) -> PathTrimResult:
    if not path:
        return PathTrimResult([], False, None, None, "empty_path")
    cur = np.asarray(current_grid, dtype=np.float32)
    arr = np.asarray(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return PathTrimResult([], False, None, None, "invalid_path_shape")
    dists = np.linalg.norm(arr - cur[None, :], axis=1)
    idx = int(np.argmin(dists))
    nearest_dist = float(dists[idx])
    if nearest_dist <= float(max_dist_cells):
        exact = bool(tuple(int(v) for v in path[idx]) == tuple(int(v) for v in current_grid))
        return PathTrimResult(list(path[idx:]), True, idx, nearest_dist, "exact_match" if exact else "nearest_within_threshold")
    return PathTrimResult([], False, idx, nearest_dist, "nearest_too_far")


def trim_path_to_nearest(path: List[Tuple[int, int]], current_grid: Tuple[int, int], max_dist_cells: int = 3) -> List[Tuple[int, int]]:
    return trim_path_to_nearest_with_info(path, current_grid, max_dist_cells=max_dist_cells).suffix


def path_trim_debug_metadata(
    *,
    trim: PathTrimResult,
    before_head: Tuple[int, int] | None,
    after_head: Tuple[int, int] | None,
    replan_requested: bool,
    replan_reason: str | None,
    cleared_stale_path: bool,
) -> dict[str, object]:
    return {
        "path_trim_success": bool(trim.success),
        "path_trim_succeeded": bool(trim.success),
        "path_trim_failed": not bool(trim.success),
        "path_trim_reason": str(trim.reason),
        "path_trim_nearest_index": None if trim.nearest_index is None else int(trim.nearest_index),
        "path_trim_nearest_distance_cells": None if trim.nearest_distance_cells is None else float(trim.nearest_distance_cells),
        "current_grid_on_path_or_near_path": bool(trim.success),
        "current_path_head_before_trim": None if before_head is None else [int(before_head[0]), int(before_head[1])],
        "current_path_head_after_trim": None if after_head is None else [int(after_head[0]), int(after_head[1])],
        "current_path_head_grid": None if after_head is None else [int(after_head[0]), int(after_head[1])],
        "path_replan_requested": bool(replan_requested),
        "path_replan_reason": None if replan_reason is None else str(replan_reason),
        "path_replan_current_path_cleared_as_stale": bool(cleared_stale_path),
        "path_replan_suppressed_by_roomseg_freeze": False,
    }


def active_target_replan_trace_extra(path_replan_debug: Mapping[str, object]) -> dict[str, object]:
    """Fields that must be serialized into runtime_decision_trace for frozen roomseg replans."""
    return {
        **dict(path_replan_debug),
        "path_replan_only_active_target": True,
        "target_reselection_suppressed_by_roomseg_freeze": True,
        "path_replan_suppressed_by_roomseg_freeze": False,
        "path_replan_reused_committed_target": True,
    }


def astar_no_path_failure_reason(*, path_replan_only_active_target: bool) -> str:
    if bool(path_replan_only_active_target):
        return "astar_no_path_to_active_target"
    return "astar_no_path"


def active_navigation_decision_for_path_replan(
    last_nav_decision: NavigationDecision | None,
    *,
    update_roomseg_frontiers: bool,
    allow_path_replan_during_roomseg_freeze: bool = True,
) -> NavigationDecision | None:
    """Return the current active target when only A* replan is allowed."""
    if bool(update_roomseg_frontiers):
        return None
    if not bool(allow_path_replan_during_roomseg_freeze):
        return None
    if last_nav_decision is None:
        return None
    if not bool(getattr(last_nav_decision, "target_cells", None)):
        return None
    return last_nav_decision


def frontier_exact_target_locked_for_planning(
    nav_decision: NavigationDecision | None,
    committed_frontier_execution: object | None,
) -> bool:
    """Do not let radius-goal planning move an already committed frontier target."""
    if nav_decision is None or str(getattr(nav_decision, "mode", "")) != "frontier":
        return False
    if committed_frontier_execution is None or not bool(getattr(committed_frontier_execution, "exists", lambda: False)()):
        return False
    metadata = dict(getattr(nav_decision, "metadata", {}) or {})
    if bool(metadata.get("frontier_execution_locked", False)):
        return True
    return str(metadata.get("frontier_commitment_reason", "")) == "continue_committed_frontier_execution"


def mark_observed_disc(observed: np.ndarray, center: Tuple[int, int], radius_cells: int) -> None:
    r0, c0 = int(center[0]), int(center[1])
    h, w = observed.shape
    r_min, r_max = max(0, r0 - radius_cells), min(h - 1, r0 + radius_cells)
    c_min, c_max = max(0, c0 - radius_cells), min(w - 1, c0 + radius_cells)
    rr, cc = np.ogrid[r_min : r_max + 1, c_min : c_max + 1]
    mask = (rr - r0) ** 2 + (cc - c0) ** 2 <= radius_cells**2
    observed[r_min : r_max + 1, c_min : c_max + 1][mask] = True


def filter_detections_by_confidence(detections: List[Detection2D], min_confidence: float) -> List[Detection2D]:
    threshold = float(min_confidence)
    return [det for det in detections if detection_confidence_is_valid(float(det.confidence), threshold)]


def filter_edge_touching_detections(
    detections: List[Detection2D],
    *,
    image_width: int,
    image_height: int,
    step_idx: int,
    reject_edge_touching_bboxes: bool,
    margin_px: float,
    margin_ratio: float,
    min_confidence: Optional[float] = None,
    raw_log: Optional[List[dict]] = None,
) -> List[Detection2D]:
    kept: List[Detection2D] = []
    for idx, det in enumerate(detections):
        touches = bbox_touches_image_edge(
            det.bbox_xyxy,
            int(image_width),
            int(image_height),
            margin_px=float(margin_px),
            margin_ratio=float(margin_ratio),
        )
        low_confidence = (
            min_confidence is not None
            and not detection_confidence_is_valid(float(det.confidence), float(min_confidence))
        )
        det.bbox_touches_edge = bool(touches)
        # Edge-touching detector/masked detections are partial visual evidence, not a
        # discard condition. The legacy flag is kept for CLI compatibility and
        # recorded below, but strict object tracking now decides policy use from
        # mask/depth association and track stability.
        rejected = bool(low_confidence)
        det.used_for_object_track = not rejected
        if low_confidence:
            det.reject_reason = "low_confidence"
        else:
            det.reject_reason = None
        record = {
            "raw_detection_id": "frame_%04d_det_%04d" % (int(step_idx), int(idx)),
            "step": int(step_idx),
            "category": normalize_category(det.category),
            "raw_label": str(det.raw_label),
            "confidence": float(det.confidence),
            "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
            "bbox_touches_edge": bool(touches),
            "legacy_reject_edge_touching_bboxes_requested": bool(reject_edge_touching_bboxes),
            "visibility_status": "partial_edge" if touches else "unknown",
            "used_for_object_track": not rejected,
            "reject_reason": det.reject_reason,
        }
        if raw_log is not None:
            raw_log.append(record)
        if not rejected:
            kept.append(det)
    return kept


def goal_candidate_pair_distances(object_memory: ObjectMemory, goal_category: str, max_distance_m: float = 1.0) -> List[dict]:
    goal = normalize_category(goal_category)
    nodes = [node for node in object_memory.nodes if normalize_category(node.category) == goal]
    out = []
    for idx, a in enumerate(nodes):
        for b in nodes[idx + 1 :]:
            dist = float(np.linalg.norm(np.asarray(a.center_world[:2], dtype=np.float32) - np.asarray(b.center_world[:2], dtype=np.float32)))
            if dist <= float(max_distance_m):
                out.append({"a": int(a.node_id), "b": int(b.node_id), "dist": dist})
    return out


def candidate_center_payload(object_memory: ObjectMemory, goal_category: str, selected_candidate_id: Optional[int] = None) -> List[dict]:
    goal = normalize_category(goal_category)
    payload = []
    for node in object_memory.nodes:
        if normalize_category(node.category) != goal:
            continue
        status = "selected" if selected_candidate_id is not None and int(node.node_id) == int(selected_candidate_id) else "candidate"
        payload.append(
            {
                "node_id": int(node.node_id),
                "center_grid": tuple(int(v) for v in node.center_grid),
                "confidence": float(node.confidence),
                "observed_count": int(node.observed_count),
                "status": status,
            }
        )
    return payload


def build_reperception_state_payload(decision_metadata: Mapping[str, object]) -> dict:
    reperception = dict(decision_metadata.get("reperception") or {})
    candidate_id = reperception.get("candidate_id", decision_metadata.get("selected_candidate_id"))
    decision = reperception.get("decision")
    if decision is None:
        if bool(decision_metadata.get("candidate_accepted", False)):
            decision = "ACCEPT_GOAL"
        elif bool(decision_metadata.get("candidate_rejected", False)):
            decision = "REJECT_GOAL"
        elif candidate_id is not None:
            decision = "CONTINUE_OBSERVING"
    return {
        "candidate_id": candidate_id,
        "num_reperception_steps": int(
            reperception.get(
                "num_reperception_steps",
                decision_metadata.get("candidate_reperception_steps", 0),
            )
            or 0
        ),
        "accumulated_credibility": float(
            reperception.get(
                "accumulated_credibility",
                decision_metadata.get("candidate_credibility", 0.0),
            )
            or 0.0
        ),
        "last_s_k": float(reperception.get("last_s_k", reperception.get("s_k", 0.0)) or 0.0),
        "detector_confidence": float(reperception.get("detector_confidence", 0.0) or 0.0),
        "supporting_subgraphs": list(reperception.get("supporting_subgraphs") or []),
        "decision": decision,
    }


def build_stop_state_payload(nav_decision: Optional[NavigationDecision], row: Mapping[str, object]) -> dict:
    metadata = dict(getattr(nav_decision, "metadata", {}) or {}) if nav_decision is not None else {}
    stop_allowed = bool(getattr(nav_decision, "stop", False)) if nav_decision is not None else False
    candidate_confirmed = bool(metadata.get("candidate_accepted", False)) or str(getattr(nav_decision, "mode", "")) == "stop"
    if row.get("stop_blocked_reason"):
        reason = str(row["stop_blocked_reason"])
    elif not stop_allowed and not candidate_confirmed:
        reason = "candidate_not_confirmed"
    else:
        reason = str(row.get("stop_reason") or metadata.get("stop_reason") or getattr(nav_decision, "reason", ""))
    return {
        "stop_allowed": bool(stop_allowed),
        "stop_reason": reason,
        "candidate_confirmed": bool(candidate_confirmed),
        "policy_stop_confirmed": bool(row.get("policy_stop_confirmed", False)),
        "success_requires_sgnav_stop": bool(row.get("success_requires_sgnav_stop", False)),
        "gt_success_region_reached": bool(row.get("gt_success_region_reached", False)),
        "stop_blocked_reason": row.get("stop_blocked_reason"),
        "mode": getattr(nav_decision, "mode", None) if nav_decision is not None else None,
        "success": bool(row.get("success", False)),
        "distance_to_goal": row.get("distance_to_goal"),
    }


def success_region_can_finish(
    distance_to_goal: float,
    success_distance: float,
    *,
    require_sgnav_stop: bool,
    policy_stop_confirmed: bool,
    ignore_goal_success: bool = False,
) -> bool:
    if bool(ignore_goal_success):
        return False
    inside_success_region = float(distance_to_goal) <= float(success_distance)
    if not inside_success_region:
        return False
    return bool(policy_stop_confirmed or not require_sgnav_stop)


def update_metric_evaluator_pose_or_mark_invalid(
    evaluator: EpisodeEvaluator,
    metric_planner: GridAStarPlanner,
    pose_world: Sequence[float],
    map_info: MapInfo,
    *,
    collided: bool = False,
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    metric_grid = metric_planner.snap_to_free(
        world_xy_to_grid(float(pose_world[0]), float(pose_world[1]), map_info)
    )
    if metric_grid is None:
        evaluator.num_steps += 1
        evaluator.path_accum.update((float(pose_world[0]), float(pose_world[1])))
        if collided:
            evaluator.num_collisions += 1
        return False, None
    evaluator.update_pose(pose_world, metric_grid, collided=collided)
    return True, metric_grid


def final_log_row(row: dict) -> dict:
    if row.get("failure_reason"):
        stop_reason = row["failure_reason"]
    elif bool(row.get("success", False)):
        stop_reason = "success"
    elif row.get("terminal_reason"):
        stop_reason = row["terminal_reason"]
    else:
        stop_reason = "not_success"
    out = {
        "goal_category": row.get("goal_category"),
        "success": bool(row.get("success", False)),
        "distance_to_goal": row.get("distance_to_goal"),
        "spl": row.get("spl"),
        "stop_reason": stop_reason,
        "sgnav_decision_mode": row.get("sgnav_decision_mode"),
        "sgnav_decision_reason": row.get("sgnav_decision_reason"),
    }
    for key in (
        "frontier_target_mode",
        "frontier_center_grid",
        "frontier_actual_target_grid",
        "frontier_unreachable_recovery",
        "frontier_unreachable_reason",
        "frontier_refresh_pending",
        "frontier_refresh_reason",
        "frontier_refresh_consumed_step",
        "frontier_reselect_blocked_until_refresh",
        "frontier_direct_target_unreachable_count",
        "frontier_recovery",
        "frontier_execution_debug",
        "frontier_stop_at_current_grid",
        "frontier_blacklisted",
        "active_long_term_goal_mode",
        "active_long_term_goal_age",
        "long_term_goal",
        "paper_llm_enabled",
        "paper_llm_requests",
        "hcot_llm_enabled",
        "hcot_llm_attempts",
        "hcot_llm_failures",
        "hcot_llm_fallbacks",
        "hcot_llm_last_error",
        "hc_p_num_subgraphs_total",
        "hc_p_num_subgraphs_scored",
        "stop_called",
        "policy_stop_confirmed",
        "success_requires_sgnav_stop",
        "gt_success_region_reached",
        "gt_success_without_sgnav_stop_steps",
        "explore_until_no_frontiers",
        "goal_success_ignored_steps",
        "stop_blocked_reason",
        "metric_pose_valid",
        "metric_pose_invalid_steps",
        "metric_pose_last_invalid_step",
        "metric_pose_last_invalid_reason",
        "mapping_latency_ms",
        "mapping_latency_breakdown_avg_ms",
        "mapping_latency_breakdown_counts",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def detections_to_3d_static_map_ray(
    detections: List[Detection2D],
    camera_pose_world: Tuple[float, float, float, float],
    image_width: int,
    camera_hfov_deg: float,
    map_info: MapInfo,
    occupancy: np.ndarray,
    navigable: np.ndarray,
    max_range_m: float = 6.0,
    min_range_m: float = 0.20,
) -> List[Detection3D]:
    """Approximate RGB-only detections on the static map without depth reads."""

    if image_width <= 0:
        return []
    cam_x, cam_y, cam_z, cam_yaw = [float(v) for v in camera_pose_world]
    hfov = math.radians(float(camera_hfov_deg))
    fx = (float(image_width) * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    cx = float(image_width) * 0.5
    step_m = max(float(map_info.resolution_m) * 0.5, 0.02)
    max_range = max(float(max_range_m), float(min_range_m) + step_m)
    out: List[Detection3D] = []

    for det in detections:
        x1, _y1, x2, _y2 = [float(v) for v in det.bbox_xyxy]
        u = 0.5 * (x1 + x2)
        ray_yaw = cam_yaw + math.atan2(u - cx, fx)
        cos_yaw = math.cos(ray_yaw)
        sin_yaw = math.sin(ray_yaw)
        hit_cell = None
        last_free_cell = None
        dist = float(min_range_m)
        while dist <= max_range:
            row, col = world_xy_to_grid(cam_x + cos_yaw * dist, cam_y + sin_yaw * dist, map_info)
            if not is_inside_grid(row, col, map_info):
                break
            if bool(occupancy[row, col]):
                hit_cell = (row, col)
                break
            if bool(navigable[row, col]):
                last_free_cell = (row, col)
            dist += step_m
        cell = hit_cell or last_free_cell
        if cell is None:
            continue
        wx, wy = grid_to_world_xy(cell[0], cell[1], map_info)
        out.append(
            Detection3D(
                category=det.category,
                raw_label=det.raw_label,
                confidence=float(det.confidence),
                center_world=(float(wx), float(wy), float(cam_z)),
                bbox_xyxy=det.bbox_xyxy,
            )
        )
    return out


def predict_kinematic_pose(pose: Tuple[float, float, float, float], cmd: Tuple[float, float, float], dt: float) -> Tuple[float, float, float, float]:
    x, y, z, yaw = [float(v) for v in pose]
    vx, vy, wz = [float(v) for v in cmd]
    dx = math.cos(yaw) * vx - math.sin(yaw) * vy
    dy = math.sin(yaw) * vx + math.cos(yaw) * vy
    yaw = yaw + wz * float(dt)
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return x + dx * float(dt), y + dy * float(dt), z, yaw


def pose_is_grid_safe(
    pose: Tuple[float, float, float, float],
    navigable: np.ndarray,
    map_info: MapInfo,
    camera_forward_offset_m: float = 0.0,
) -> bool:
    x, y, _z, yaw = [float(v) for v in pose]
    samples = [(x, y)]
    if abs(float(camera_forward_offset_m)) > 1e-6:
        samples.append((x + math.cos(yaw) * float(camera_forward_offset_m), y + math.sin(yaw) * float(camera_forward_offset_m)))
    h, w = navigable.shape
    for sx, sy in samples:
        r, c = world_xy_to_grid(sx, sy, map_info)
        if not (0 <= r < h and 0 <= c < w) or not bool(navigable[r, c]):
            return False
    return True


def pose_swept_is_grid_safe(
    pose: Tuple[float, float, float, float],
    cmd: Tuple[float, float, float],
    dt: float,
    navigable: np.ndarray,
    map_info: MapInfo,
    camera_forward_offset_m: float = 0.0,
    sample_step_m: float | None = None,
) -> bool:
    start = tuple(float(v) for v in pose)
    end = predict_kinematic_pose(start, cmd, float(dt))
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    dyaw = float(end[3]) - float(start[3])
    while dyaw > math.pi:
        dyaw -= 2.0 * math.pi
    while dyaw < -math.pi:
        dyaw += 2.0 * math.pi
    step_m = float(sample_step_m) if sample_step_m is not None else max(0.02, float(map_info.resolution_m) * 0.5)
    linear_steps = int(math.ceil(math.hypot(dx, dy) / max(step_m, 1e-6)))
    angular_steps = int(math.ceil(abs(dyaw) / math.radians(6.0)))
    samples = max(1, linear_steps, angular_steps)
    for idx in range(samples + 1):
        t = float(idx) / float(samples)
        yaw = float(start[3]) + dyaw * t
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        while yaw < -math.pi:
            yaw += 2.0 * math.pi
        interp = (
            float(start[0]) + dx * t,
            float(start[1]) + dy * t,
            float(start[2]),
            yaw,
        )
        if not pose_is_grid_safe(interp, navigable, map_info, camera_forward_offset_m):
            return False
    return True


def _pose_clearance_status(
    pose: Tuple[float, float, float, float],
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    map_info: MapInfo,
    min_clearance_m: float,
    camera_forward_offset_m: float = 0.0,
) -> tuple[bool, float, tuple[int, int] | None, str | None]:
    nav = np.asarray(traversible, dtype=bool)
    clearance = np.asarray(clearance_m, dtype=np.float32)
    x, y, _z, yaw = [float(v) for v in pose]
    samples = [(x, y)]
    if abs(float(camera_forward_offset_m)) > 1e-6:
        samples.append((x + math.cos(yaw) * float(camera_forward_offset_m), y + math.sin(yaw) * float(camera_forward_offset_m)))
    h, w = nav.shape
    min_seen = float("inf")
    for sx, sy in samples:
        row, col = world_xy_to_grid(float(sx), float(sy), map_info)
        if row < 0 or row >= h or col < 0 or col >= w:
            return False, 0.0 if not math.isfinite(min_seen) else min_seen, (int(row), int(col)), "outside_grid"
        if not bool(nav[row, col]):
            return False, float(clearance[row, col]) if clearance.shape == nav.shape else 0.0, (int(row), int(col)), "non_traversible"
        clear = float(clearance[row, col]) if clearance.shape == nav.shape else 0.0
        min_seen = min(min_seen, clear)
        if clear + 1e-6 < float(min_clearance_m):
            return False, clear, (int(row), int(col)), "low_clearance"
    if not math.isfinite(min_seen):
        min_seen = 0.0
    return True, float(min_seen), None, None


def pose_has_clearance(
    pose: Tuple[float, float, float, float],
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    map_info: MapInfo,
    min_clearance_m: float,
    camera_forward_offset_m: float = 0.0,
) -> bool:
    ok, _min_clearance, _failed_grid, _reason = _pose_clearance_status(
        pose,
        traversible,
        clearance_m,
        map_info,
        min_clearance_m,
        camera_forward_offset_m,
    )
    return bool(ok)


def pose_swept_has_clearance(
    pose: Tuple[float, float, float, float],
    cmd: Tuple[float, float, float],
    dt: float,
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    map_info: MapInfo,
    min_clearance_m: float,
    camera_forward_offset_m: float = 0.0,
    sample_step_m: float | None = None,
) -> tuple[bool, dict[str, object]]:
    start = tuple(float(v) for v in pose)
    end = predict_kinematic_pose(start, cmd, float(dt))
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    dyaw = float(end[3]) - float(start[3])
    while dyaw > math.pi:
        dyaw -= 2.0 * math.pi
    while dyaw < -math.pi:
        dyaw += 2.0 * math.pi
    step_m = float(sample_step_m) if sample_step_m is not None else max(0.02, float(map_info.resolution_m) * 0.5)
    linear_steps = int(math.ceil(math.hypot(dx, dy) / max(step_m, 1e-6)))
    angular_steps = int(math.ceil(abs(dyaw) / math.radians(6.0)))
    samples = max(1, linear_steps, angular_steps)
    min_swept_clearance = float("inf")
    failed_grid = None
    block_reason = None
    for idx in range(samples + 1):
        t = float(idx) / float(samples)
        yaw = float(start[3]) + dyaw * t
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        while yaw < -math.pi:
            yaw += 2.0 * math.pi
        interp = (
            float(start[0]) + dx * t,
            float(start[1]) + dy * t,
            float(start[2]),
            yaw,
        )
        ok, clear, grid, reason = _pose_clearance_status(
            interp,
            traversible,
            clearance_m,
            map_info,
            min_clearance_m,
            camera_forward_offset_m,
        )
        min_swept_clearance = min(min_swept_clearance, float(clear))
        if not ok:
            failed_grid = grid
            block_reason = reason
            break
    if not math.isfinite(min_swept_clearance):
        min_swept_clearance = 0.0
    debug = {
        "guard_min_clearance_m": float(min_clearance_m),
        "guard_min_swept_clearance_m": float(min_swept_clearance),
        "guard_failed_sample_grid": [int(failed_grid[0]), int(failed_grid[1])] if failed_grid is not None else None,
        "guard_block_reason": block_reason,
        "guard_sweep_samples": int(samples + 1),
    }
    return bool(block_reason is None), debug


def guard_kinematic_cmd_with_clearance(
    pose: Tuple[float, float, float, float],
    cmd: Tuple[float, float, float],
    dt: float,
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    map_info: MapInfo,
    min_clearance_m: float,
    camera_forward_offset_m: float,
) -> Tuple[Tuple[float, float, float], bool, dict[str, object]]:
    last_debug: dict[str, object] = {
        "guard_min_clearance_m": float(min_clearance_m),
        "guard_min_swept_clearance_m": None,
        "guard_failed_sample_grid": None,
        "guard_block_reason": None,
        "guard_accepted_scale": None,
    }
    for scale in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
        scaled = (float(cmd[0]) * scale, float(cmd[1]) * scale, float(cmd[2]))
        ok, debug = pose_swept_has_clearance(
            pose,
            scaled,
            dt,
            traversible,
            clearance_m,
            map_info,
            min_clearance_m,
            camera_forward_offset_m,
        )
        last_debug = {**debug, "guard_accepted_scale": float(scale) if ok else None}
        if ok:
            return scaled, False, last_debug
    rotate_only = (0.0, 0.0, float(cmd[2]))
    if abs(rotate_only[2]) > 1e-6:
        ok, debug = pose_swept_has_clearance(
            pose,
            rotate_only,
            dt,
            traversible,
            clearance_m,
            map_info,
            min_clearance_m,
            camera_forward_offset_m,
        )
        if ok:
            linear_requested = math.hypot(float(cmd[0]), float(cmd[1])) > 1e-6
            return rotate_only, bool(linear_requested), {**debug, "guard_accepted_scale": 0.0, "guard_rotation_only": True}
        last_debug = {**debug, "guard_accepted_scale": None, "guard_rotation_only": True}
    stopped = (0.0, 0.0, 0.0)
    ok, debug = pose_swept_has_clearance(
        pose,
        stopped,
        dt,
        traversible,
        clearance_m,
        map_info,
        min_clearance_m,
        camera_forward_offset_m,
    )
    if ok:
        return stopped, True, {
            **debug,
            "guard_block_reason": last_debug.get("guard_block_reason"),
            "guard_failed_sample_grid": last_debug.get("guard_failed_sample_grid"),
            "guard_min_swept_clearance_m": last_debug.get("guard_min_swept_clearance_m", debug.get("guard_min_swept_clearance_m")),
            "guard_accepted_scale": 0.0,
        }
    return stopped, True, {**last_debug, "guard_fallback_stop_unsafe": True}


def guard_kinematic_cmd(
    pose: Tuple[float, float, float, float],
    cmd: Tuple[float, float, float],
    dt: float,
    navigable: np.ndarray,
    map_info: MapInfo,
    camera_forward_offset_m: float,
) -> Tuple[Tuple[float, float, float], bool]:
    for scale in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
        scaled = (float(cmd[0]) * scale, float(cmd[1]) * scale, float(cmd[2]))
        if pose_swept_is_grid_safe(pose, scaled, dt, navigable, map_info, camera_forward_offset_m):
            return scaled, False
    rotate_only = (0.0, 0.0, float(cmd[2]))
    if abs(rotate_only[2]) > 1e-6 and pose_swept_is_grid_safe(pose, rotate_only, dt, navigable, map_info, camera_forward_offset_m):
        linear_requested = math.hypot(float(cmd[0]), float(cmd[1])) > 1e-6
        return rotate_only, bool(linear_requested)
    stopped = (0.0, 0.0, 0.0)
    if pose_swept_is_grid_safe(pose, stopped, dt, navigable, map_info, camera_forward_offset_m):
        return stopped, True
    return stopped, True


def seed_object_memory_from_preprocessed(scene_dir: Path, map_info: MapInfo, object_memory: ObjectMemory) -> int:
    objects_path = scene_dir / "objects.json"
    if not objects_path.exists():
        return 0
    with open(objects_path, "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    detections: List[Detection3D] = []
    for obj in objects:
        center = obj.get("center_world")
        if not center or len(center) < 3:
            continue
        detections.append(
            Detection3D(
                category=str(obj.get("category", "unknown")),
                raw_label=str(obj.get("raw_label", obj.get("category", "unknown"))),
                confidence=1.0,
                center_world=(float(center[0]), float(center[1]), float(center[2])),
                bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            )
        )
    object_memory.update(detections, step_id=0, map_info=map_info)
    return len(detections)


def detector_requires_rgb(detector, detector_name: str) -> bool:
    return detector is not None and str(detector_name) not in {"dry_run", "none"}


def detector_can_use_cuda_rgb(detector, detector_name: str, camera_annotator_device: str) -> bool:
    return (
        detector_requires_rgb(detector, detector_name)
        and not isinstance(detector, SubprocessDetector)
        and str(detector_name) == "yolo_world"
        and str(camera_annotator_device).lower() == "cuda"
    )


def run_episode_isaac_closed_loop(episode: dict, args) -> dict:
    from voxroom_online.isaac_runtime.env.isaac_process import IsaacSimServer

    scene_dir, static_map_info, static_occupancy, static_navigable = load_preprocessed_for_episode(episode)
    static_openings = load_passable_opening_mask(scene_dir, static_map_info)
    static_navigable = apply_episode_planning_clearance(
        scene_dir,
        static_map_info,
        static_navigable,
        episode,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    ensure_detector_loaded(args, scene_dir, allow_ipc_fallback=True)
    detector = getattr(args, "_detector_instance", None)
    if detector_requires_rgb(detector, args.detector):
        ensure_segmenter_loaded(args)
        segmenter = getattr(args, "_segmenter_instance", None)
    else:
        segmenter = None
    camera_annotator_device = str(getattr(args, "camera_annotator_device", "cuda")).strip().lower()
    if camera_annotator_device not in {"cpu", "cuda"}:
        camera_annotator_device = "cpu"

    metric_planner = GridAStarPlanner(static_navigable, static_map_info.resolution_m, allow_diagonal=True)
    evaluator = EpisodeEvaluator(episode, metric_planner)
    object_memory = ObjectMemory(
        merge_radius_m=float(args.object_merge_radius_m),
        min_valid_confidence=float(args.min_valid_detection_confidence),
    )
    fused_instance_registry = FusedInstanceRegistry(
        merge_distance_m=float(getattr(args, "instance_merge_distance_m", args.object_merge_radius_m)),
        merge_iou_3d=float(getattr(args, "instance_merge_iou_3d", 0.15)),
        min_valid_confidence=float(args.min_valid_detection_confidence),
        reject_edge_touching_bboxes=bool(args.reject_edge_touching_bboxes),
        bbox_edge_margin_px=float(args.bbox_edge_margin_px),
        bbox_edge_margin_ratio=float(args.bbox_edge_margin_ratio),
        partial_class_weight=float(args.partial_class_weight),
        min_geometry_confidence=float(args.min_geometry_confidence),
        partial_stability_min_observations=int(args.partial_stability_min_observations),
        mask_iou_association_threshold=float(args.mask_iou_association_threshold),
        mask_containment_track_match_threshold=float(args.mask_containment_track_match_threshold),
        footprint_iou_association_threshold=float(args.footprint_iou_association_threshold),
        child_containment_threshold=float(args.child_containment_threshold),
        child_area_ratio_threshold=float(args.child_area_ratio_threshold),
    )
    if args.seed_gt_object_memory:
        print("[voxroom-loop] seed_gt_object_memory ignored for depth-online mapping", flush=True)
    seeded = 0
    scenegraph = SGNavSceneGraphAdapter(
        args.sgnav_repo,
        use_original=args.use_original_scenegraph,
        semantic_priors_path=getattr(args, "semantic_priors_path", None),
        vllm_config={
            "enabled": bool(getattr(args, "vllm_frontier_scoring", False)),
            "base_url": getattr(args, "vllm_base_url", None),
            "model": getattr(args, "vllm_model", None),
            "timeout_s": float(getattr(args, "vllm_timeout_s", 8.0)),
            "temperature": float(getattr(args, "vllm_temperature", 0.0)),
            "max_frontiers": int(getattr(args, "vllm_max_frontiers", 32)),
            "include_image": bool(getattr(args, "vllm_image_scoring", True)),
            "image_max_width": int(getattr(args, "vllm_image_max_width", 640)),
            "image_jpeg_quality": int(getattr(args, "vllm_image_jpeg_quality", 75)),
        },
        llm_config={
            "enabled": bool(getattr(args, "llm_enabled", False)),
            "base_url": getattr(args, "llm_base_url", None),
            "model": getattr(args, "llm_model", None),
            "api_key": getattr(args, "llm_api_key", None),
            "timeout_s": float(getattr(args, "llm_timeout_s", 30.0)),
            "temperature": float(getattr(args, "llm_temperature", 0.0)),
            "max_tokens": int(getattr(args, "llm_max_tokens", 512)),
            "max_hcot_subgraphs_per_decision": int(getattr(args, "max_hcot_subgraphs_per_decision", 8)),
            "strict_benchmark": bool(getattr(args, "strict_benchmark", False)),
        },
        sgnav_mode=str(getattr(args, "sgnav_mode", "legacy")),
    )
    if bool(getattr(args, "vllm_frontier_scoring", False)):
        print(
            "[voxroom-vllm] frontier scoring enabled: base_url=%s model=%s image=%s"
            % (
                getattr(args, "vllm_base_url", "http://127.0.0.1:8000/v1"),
                getattr(args, "vllm_model", "qwen3-vl-8b-instruct"),
                bool(getattr(args, "vllm_image_scoring", True)),
            ),
            flush=True,
        )
    scenegraph.reset(episode["goal_category"])
    full_room_map = None
    scenegraph.update(object_memory, room_map=full_room_map)
    evaluator.num_scenegraph_updates += 1
    decision_policy = SGNavDecision(
        scenegraph,
        frontier_distance_weight=float(args.frontier_distance_weight),
        frontier_min_select_distance_m=float(args.frontier_min_distance_m),
        frontier_allow_near_fallback=bool(args.frontier_allow_near_fallback),
        candidate_min_hits=int(args.candidate_min_detector_hits),
        candidate_start_min_confidence=float(args.candidate_start_min_confidence),
        candidate_start_min_hits=int(args.candidate_start_min_hits),
        candidate_recent_max_age_steps=int(args.candidate_recent_max_age_steps),
        candidate_match_substring=bool(args.candidate_match_substring),
        candidate_accept_requires_reperception=bool(args.candidate_accept_requires_reperception),
        candidate_reject_ttl_steps=int(args.candidate_reject_ttl_steps),
        candidate_accept_threshold=float(args.candidate_accept_threshold),
        candidate_stop_distance_m=float(args.candidate_stop_distance_m),
        candidate_standoff_min_m=float(args.candidate_standoff_min_m),
        candidate_standoff_max_m=float(args.candidate_standoff_max_m),
        candidate_standoff_max_cells=int(args.candidate_standoff_max_cells),
        candidate_standoff_ideal_m=float(args.candidate_standoff_ideal_m),
        reperception_enabled=bool(args.reperception_enabled),
        reperception_min_observations=int(args.reperception_min_observations),
        reperception_max_steps=int(args.reperception_max_steps),
        reperception_same_goal_radius_m=float(args.reperception_same_goal_radius_m),
        stop_verification_steps=int(args.stop_verification_steps),
        stop_verification_min_hits=int(args.stop_verification_min_hits),
        found_goal_stop_distance_m=float(args.found_goal_stop_distance_m),
        score_frontiers_before_candidate=bool(args.score_frontiers_before_candidate),
        frontier_scenegraph_score_norm=str(args.frontier_scenegraph_score_norm),
        frontier_selection_mode=str(args.frontier_selection_mode),
        frontier_random_seed=int(args.frontier_random_seed),
    )
    follower = HolonomicWaypointFollower(
        max_vx=float(args.max_vx_mps),
        max_vy=float(args.max_vy_mps),
        max_wz=float(args.max_wz_radps),
        lookahead_m=float(args.lookahead_m),
    )
    vertical_or_free_cfg = dict(getattr(args, "room_segmentation_config", {}).get("vertical_or_free", {}) or {})
    height_profile_cfg = dict(getattr(args, "room_segmentation_config", {}).get("height_profile", {}) or {})
    voxel_grid_cfg = dict(getattr(args, "room_segmentation_config", {}).get("voxel_grid", {}) or {})
    voxel_outside_cfg = dict(getattr(args, "room_segmentation_config", {}).get("voxel_outside", {}) or {})
    voxel_nav_cfg = dict(getattr(args, "room_segmentation_config", {}).get("voxel_navigation_projection", {}) or {})
    voxel_blind_zone_cfg = dict(getattr(args, "room_segmentation_config", {}).get("voxel_navigation_blind_zone", {}) or {})
    voxel_runtime_cfg = dict(getattr(args, "voxel_runtime_config", {}) or {})
    if bool(voxel_grid_cfg.get("enabled", True)) and bool(voxel_grid_cfg.get("voxel_grid_drives_navigation", True)):
        required_skip_flags = (
            "skip_legacy_vertical_profile_when_voxel_backend",
            "skip_legacy_height_profile_when_voxel_backend",
            "skip_legacy_roomseg_ray_evidence_when_voxel_backend",
        )
        missing_skip_flags = [key for key in required_skip_flags if not bool(voxel_runtime_cfg.get(key, False))]
        if missing_skip_flags:
            raise RuntimeError(
                "voxel grid drives navigation, but legacy mapping recomputation is enabled: %s"
                % ", ".join(missing_skip_flags)
            )
    roomseg_depth_stride_px = _resolve_roomseg_depth_stride_px(
        getattr(args, "room_segmentation_config", {}),
        int(args.depth_stride_px),
    )
    if int(roomseg_depth_stride_px) != int(args.depth_stride_px):
        print(
            "[voxroom-roomseg] vertical-free depth stride: mapping=%d roomseg_effective=%d"
            % (int(args.depth_stride_px), int(roomseg_depth_stride_px)),
            flush=True,
        )
    mapper = OnlineMapper(
        size_m=float(args.online_map_size_m),
        resolution_m=float(args.online_resolution_m),
        depth_max_m=float(args.depth_max_m),
        depth_min_m=float(args.depth_min_m),
        depth_stride_px=int(roomseg_depth_stride_px),
        obstacle_min_height_m=float(args.obstacle_min_height_m),
        obstacle_max_height_m=float(args.obstacle_max_height_m),
        free_min_height_m=float(args.free_min_height_m),
        free_max_height_m=float(args.free_max_height_m),
        vertical_profile_free_min_height_m=float(vertical_or_free_cfg.get("z_min_m", 0.20)),
        vertical_profile_free_max_height_m=float(vertical_or_free_cfg.get("z_max_m", 2.00)),
        splat_point_threshold=int(args.splat_point_threshold),
        free_splat_point_threshold=int(args.free_splat_point_threshold),
        robot_radius_m=float(args.robot_radius_m),
        inflation_radius_m=float(args.online_inflation_radius_m),
        height_profile_enabled=bool(height_profile_cfg.get("enabled", True)),
        height_profile_z_min_m=float(height_profile_cfg.get("z_min_m", 0.10)),
        height_profile_z_max_m=float(height_profile_cfg.get("z_max_m", height_profile_cfg.get("storage_z_max_m", 3.20))),
        height_profile_storage_z_max_m=float(height_profile_cfg.get("storage_z_max_m", height_profile_cfg.get("z_max_m", 3.20))),
        height_profile_active_z_max_fallback_m=float(height_profile_cfg.get("active_z_max_fallback_m", height_profile_cfg.get("active_z_max_cap_m", 2.80))),
        height_profile_active_z_max_ceiling_ratio=float(height_profile_cfg.get("active_z_max_ceiling_ratio", 0.85)),
        height_profile_active_z_max_cap_m=float(height_profile_cfg.get("active_z_max_cap_m", 2.80)),
        height_profile_z_bin_size_m=float(height_profile_cfg.get("z_bin_size_m", 0.05)),
        ceiling_height_estimator_config=dict(height_profile_cfg.get("ceiling_estimator", {}) or {}),
        voxel_grid_enabled=bool(voxel_grid_cfg.get("enabled", True)),
        voxel_grid_z_min_m=float(voxel_grid_cfg.get("z_min_m", -0.10)),
        voxel_grid_z_max_m=float(voxel_grid_cfg.get("z_max_m", 4.00)),
        voxel_grid_z_resolution_m=float(voxel_grid_cfg.get("z_resolution_m", 0.05)),
        voxel_grid_active_z_min_m=float(voxel_grid_cfg.get("active_z_min_m", 0.10)),
        voxel_grid_active_z_max_fallback_m=float(voxel_grid_cfg.get("active_z_max_fallback_m", voxel_grid_cfg.get("active_z_max_cap_m", 2.80))),
        voxel_grid_active_z_max_ceiling_ratio=float(voxel_grid_cfg.get("active_z_max_ceiling_ratio", 0.85)),
        voxel_grid_active_z_max_cap_m=float(voxel_grid_cfg.get("active_z_max_cap_m", 2.80)),
        voxel_grid_config=voxel_grid_cfg,
        voxel_outside_config=voxel_outside_cfg,
        voxel_navigation_projection_config=voxel_nav_cfg,
        voxel_navigation_blind_zone_config=voxel_blind_zone_cfg,
        voxel_runtime_config=voxel_runtime_cfg,
    )
    voxel_perf_recent_path = Path(args.output).expanduser().with_name("voxel_perf_log.jsonl")
    runtime_perf_path = Path(args.output).expanduser().with_name("runtime_perf.jsonl")
    runtime_control_trace_path = Path(args.output).expanduser().with_name("runtime_control_trace.jsonl")
    runtime_decision_trace_path = Path(args.output).expanduser().with_name("runtime_decision_trace.jsonl")
    voxel_perf_recent_rows: list[dict[str, object]] = []

    def record_voxel_perf_trace(step_idx: int | None) -> None:
        if not bool(voxel_grid_cfg.get("enabled", True)):
            return
        voxel_grid = getattr(mapper, "voxel_grid", None)
        if voxel_grid is None:
            return
        stats = dict(getattr(voxel_grid, "last_integration_stats", {}).to_dict())
        nav_debug = {
            str(key): value
            for key, value in dict(getattr(voxel_grid, "last_navigation_debug", {}) or {}).items()
            if not isinstance(value, np.ndarray)
        }
        row = make_jsonable(
            {
                "step": None if step_idx is None else int(step_idx),
                "backend": stats.get("voxel_integration_backend"),
                "thread_count": stats.get("voxel_integrate_backend_thread_count"),
                "effective_thread_count": stats.get("voxel_integrate_backend_effective_thread_count"),
                "requested_thread_count": stats.get("voxel_integrate_numba_requested_thread_count"),
                "threads_mode": stats.get("voxel_integrate_numba_threads_mode"),
                "threading_layer": stats.get("voxel_numba_threading_layer"),
                "integrate_total_ms": stats.get("voxel_integrate_total_ms"),
                "pass1_ms": stats.get("voxel_integrate_pass1_ms"),
                "event_prefix_ms": stats.get("voxel_integrate_event_prefix_ms"),
                "pass2_ms": stats.get("voxel_integrate_pass2_ms"),
                "event_bucket_ms": stats.get("voxel_integrate_event_bucket_ms"),
                "bucket_free_ms": stats.get("voxel_integrate_bucket_free_ms"),
                "bucket_occ_ms": stats.get("voxel_integrate_bucket_occ_ms"),
                "bucket_sensor_ms": stats.get("voxel_integrate_bucket_sensor_ms"),
                "apply_logodds_ms": stats.get("voxel_integrate_apply_logodds_ms"),
                "apply_sensor_ms": stats.get("voxel_integrate_apply_sensor_ms"),
                "endpoint_column_ms": stats.get("voxel_integrate_endpoint_column_ms"),
                "changed_scan_ms": stats.get("voxel_integrate_changed_scan_ms"),
                "refresh_mode": stats.get("voxel_refresh_mode"),
                "refresh_state_ms": stats.get("voxel_refresh_state_ms"),
                "project_navigation_ms": nav_debug.get("voxel_project_navigation_ms"),
                "free_events": stats.get("voxel_integrate_total_samples"),
                "sensor_events": stats.get("voxel_integrate_total_sensor_events"),
                "occ_events": stats.get("voxel_integrate_total_occ_events"),
                "stats": stats,
                "navigation": nav_debug,
            }
        )
        voxel_perf_recent_rows.append(row)
        if len(voxel_perf_recent_rows) > 200:
            del voxel_perf_recent_rows[:-200]
        numeric_step = None if step_idx is None else int(step_idx)
        should_flush = numeric_step is None or numeric_step < 0 or numeric_step % 50 == 0
        if should_flush:
            try:
                voxel_perf_recent_path.parent.mkdir(parents=True, exist_ok=True)
                voxel_perf_recent_path.write_text(
                    "\n".join(json.dumps(item, ensure_ascii=False) for item in voxel_perf_recent_rows) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                print("[VOXEL PERF] failed to write %s: %s" % (str(voxel_perf_recent_path), exc), flush=True)
        if numeric_step is not None and numeric_step >= 0 and numeric_step % 50 == 0:
            print(
                "[VOXEL PERF] step=%d backend=%s th=%s/%s mode=%s layer=%s total=%.2fms p1=%.2f p2=%.2f bucket=%.2f f/o/s=%.2f/%.2f/%.2f log=%.2f sensor=%.2f endpoint=%.2f changed_scan=%.2f refresh=%s nav=%.2f"
                % (
                    int(numeric_step),
                    str(row.get("backend")),
                    str(row.get("effective_thread_count") or row.get("thread_count")),
                    str(row.get("requested_thread_count")),
                    str(row.get("threads_mode")),
                    str(row.get("threading_layer")),
                    float(row.get("integrate_total_ms") or 0.0),
                    float(row.get("pass1_ms") or 0.0),
                    float(row.get("pass2_ms") or 0.0),
                    float(row.get("event_bucket_ms") or 0.0),
                    float(row.get("bucket_free_ms") or 0.0),
                    float(row.get("bucket_occ_ms") or 0.0),
                    float(row.get("bucket_sensor_ms") or 0.0),
                    float(row.get("apply_logodds_ms") or 0.0),
                    float(row.get("apply_sensor_ms") or 0.0),
                    float(row.get("endpoint_column_ms") or 0.0),
                    float(row.get("changed_scan_ms") or 0.0),
                    str(row.get("refresh_mode")),
                    float(row.get("project_navigation_ms") or 0.0),
                ),
                flush=True,
            )

    def write_runtime_perf_trace(
        step_idx: int,
        map_state_local: Mapping[str, object],
        *,
        roomseg_update_due: bool,
        roomseg_update_reason: str,
        roomseg_update_allowed_by_token: bool | None = None,
        roomseg_update_token_reason: str | None = None,
        roomseg_update_call_count_this_step: int = 0,
        roomseg_update_called_sites: Sequence[str] | None = None,
        frontier_execution_state: CommittedFrontierExecutionState | None = None,
        current_grid_for_frontier: Tuple[int, int] | None = None,
    ) -> None:
        voxel_grid = getattr(mapper, "voxel_grid", None)
        stats_obj = getattr(voxel_grid, "last_integration_stats", None) if voxel_grid is not None else None
        stats = dict(stats_obj.to_dict()) if hasattr(stats_obj, "to_dict") else {}
        nav_debug = dict(getattr(voxel_grid, "last_navigation_debug", {}) or {}) if voxel_grid is not None else {}
        nav_debug = {str(k): v for k, v in nav_debug.items() if not isinstance(v, np.ndarray)}
        mapper_timing = dict(getattr(mapper, "last_timing_stats", {}) or {})
        state_timing = dict(map_state_local.get("_state_timing_ms") or {})
        planner_timing = dict(map_state_local.get("_planner_timing_ms") or {})
        frontier_exec_debug = (
            frontier_execution_state.debug_metadata(
                step=int(step_idx),
                current_grid=current_grid_for_frontier,
                resolution_m=float(dynamic_map_info.resolution_m),
            )
            if frontier_execution_state is not None
            else {}
        )
        row = make_jsonable(
            {
                "frontier_refresh_pending": bool(frontier_arrival_update_state.refresh_pending),
                "frontier_refresh_reason": str(frontier_arrival_update_state.refresh_reason),
                "frontier_reselect_blocked_until_refresh": bool(frontier_arrival_update_state.block_reselect_until_refresh),
                "frontier_refresh_consumed_step": int(frontier_arrival_update_state.last_update_step),
                "frontier_refresh_requested_step": int(frontier_arrival_update_state.refresh_requested_step),
                "frontier_refresh_request_count": int(frontier_arrival_update_state.refresh_request_count),
                "frontier_duplicate_refresh_requests": int(frontier_arrival_update_state.duplicate_refresh_requests),
                "step": int(step_idx),
                "voxel_backend": stats.get("voxel_integration_backend"),
                "voxel_threads_requested": stats.get("voxel_integrate_numba_requested_thread_count"),
                "voxel_threads_effective": stats.get("voxel_integrate_backend_effective_thread_count"),
                "voxel_threading_layer": stats.get("voxel_integrate_numba_threading_layer"),
                "voxel_integrate_total_ms": stats.get("voxel_integrate_total_ms"),
                "voxel_pass1_ms": stats.get("voxel_integrate_pass1_ms"),
                "voxel_pass2_ms": stats.get("voxel_integrate_pass2_ms"),
                "voxel_bucket_free_ms": stats.get("voxel_integrate_bucket_free_ms"),
                "voxel_bucket_occ_ms": stats.get("voxel_integrate_bucket_occ_ms"),
                "voxel_apply_logodds_ms": stats.get("voxel_integrate_apply_logodds_ms"),
                "voxel_endpoint_column_ms": stats.get("voxel_integrate_endpoint_column_ms"),
                "voxel_project_navigation_ms": nav_debug.get("voxel_project_navigation_ms"),
                "navigation_projection_mode": nav_debug.get("voxel_project_navigation_mode"),
                "navigation_dirty_rc_count": nav_debug.get("voxel_project_navigation_dirty_rc_count"),
                "navigation_dirty_projected_rc_count": nav_debug.get("voxel_project_navigation_dirty_projected_rc_count"),
                "array_export_ms": state_timing.get("array_export_ms", mapper_timing.get("array_export_ms")),
                "traversible_ms": state_timing.get("traversible_ms", mapper_timing.get("traversible_ms")),
                "base_planner_init_ms": planner_timing.get("base_planner_init_ms", 0.0),
                "astar_clearance_ms": planner_timing.get("astar_clearance_ms", 0.0),
                "planner_init_ms": planner_timing.get("planner_init_ms", 0.0),
                "roomseg_update_due": bool(roomseg_update_due),
                "roomseg_update_reason": str(roomseg_update_reason),
                "roomseg_update_allowed_by_token": bool(roomseg_update_due if roomseg_update_allowed_by_token is None else roomseg_update_allowed_by_token),
                "roomseg_update_token_reason": str(roomseg_update_reason if roomseg_update_token_reason is None else roomseg_update_token_reason),
                "roomseg_update_called_sites": list(roomseg_update_called_sites or []),
                "roomseg_update_call_count_this_step": int(roomseg_update_call_count_this_step),
                "frontier_raw_count": int(last_frontier_raw_cells),
                "frontier_real_count": int(last_frontier_real_count),
                "frontier_near_fallback_count": int(last_frontier_near_fallback_count),
                "frontier_terminal_suppressed_count": int(last_frontier_terminal_suppressed_count),
                "frontier_filtered_count": int(last_frontier_filtered_count),
                "explore_until_no_frontiers_stop_candidate": bool(exploration_complete),
                "explore_until_no_frontiers_stop_reason": str(exploration_complete_reason),
                **frontier_exec_debug,
                **frontier_terminal_registry.debug_metadata(),
            }
        )
        try:
            runtime_perf_path.parent.mkdir(parents=True, exist_ok=True)
            with runtime_perf_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            print("[runtime-perf] failed to write %s: %s" % (str(runtime_perf_path), exc), flush=True)

    def write_runtime_control_trace(
        *,
        step_idx: int,
        pose_local: Sequence[float],
        current_grid_local: Tuple[int, int] | None,
        current_path_local: Sequence[Tuple[int, int]],
        path_world_local: Sequence[Tuple[float, float]],
        raw_cmd: Tuple[float, float, float],
        guarded_cmd: Tuple[float, float, float],
        blocked_by_guard: bool,
        target_distance_m: float,
        nav_decision_local: NavigationDecision | None,
        frontier_execution_state: CommittedFrontierExecutionState | None,
        blocked_by_online_guard: bool = False,
        guard_debug: Mapping[str, object] | None = None,
    ) -> None:
        path_cells = [tuple(int(v) for v in cell) for cell in list(current_path_local)]
        path_world_cells = [tuple(float(v) for v in cell) for cell in list(path_world_local)]
        exec_debug = (
            frontier_execution_state.debug_metadata(
                step=int(step_idx),
                current_grid=current_grid_local,
                resolution_m=float(dynamic_map_info.resolution_m),
            )
            if frontier_execution_state is not None
            else {}
        )
        if bool(frontier_arrival_update_state.refresh_pending):
            execution_phase = "refresh_pending"
        elif bool(exec_debug.get("frontier_exec_partial_arrival_pending", False)):
            execution_phase = "partial_arrival"
        elif bool(exec_debug.get("frontier_exec_recovery_active", False)):
            execution_phase = "recovery"
        elif bool(exec_debug.get("frontier_exec_active", False)):
            execution_phase = "direct"
        else:
            execution_phase = "selecting"
        row = make_jsonable(
            {
                "step": int(step_idx),
                "frontier_execution_phase": execution_phase,
                "frontier_refresh_pending": bool(frontier_arrival_update_state.refresh_pending),
                "frontier_refresh_reason": str(frontier_arrival_update_state.refresh_reason),
                "frontier_reselect_blocked_until_refresh": bool(frontier_arrival_update_state.block_reselect_until_refresh),
                "pose_world": [float(v) for v in pose_local],
                "current_grid": list(current_grid_local) if current_grid_local is not None else None,
                "nav_mode": None if nav_decision_local is None else str(nav_decision_local.mode),
                "nav_reason": None if nav_decision_local is None else str(nav_decision_local.reason),
                "target_cells_count": int(len(getattr(nav_decision_local, "target_cells", []) or [])) if nav_decision_local is not None else 0,
                "target_distance_m": float(target_distance_m) if np.isfinite(float(target_distance_m)) else None,
                "path_len": int(len(path_cells)),
                "path_head": [list(cell) for cell in path_cells[:6]],
                "path_tail": [list(cell) for cell in path_cells[-3:]],
                "path_world_len": int(len(path_world_cells)),
                "path_world_head": [[float(x), float(y)] for x, y in path_world_cells[:3]],
                "lookahead_selected_world": (
                    [float(v) for v in follower.select_lookahead(pose_local, list(path_world_cells))]
                    if path_world_cells
                    else None
                ),
                "raw_cmd": [float(v) for v in raw_cmd],
                "guarded_cmd": [float(v) for v in guarded_cmd],
                "blocked_by_guard": bool(blocked_by_guard),
                "blocked_by_online_guard": bool(blocked_by_online_guard),
                "robot_radius_m": float(args.robot_radius_m),
                "runtime_planning_clearance_m": float(args.runtime_planning_clearance_m),
                "lookahead_min_clearance_m_configured": float(getattr(args, "lookahead_min_clearance_m", 0.14)),
                "lookahead_min_clearance_m_effective": float(
                    getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
                ),
                **clearance_policy_debug(
                    float(getattr(args, "lookahead_min_clearance_m", 0.14)),
                    robot_radius_m=float(args.robot_radius_m),
                    runtime_planning_clearance_m=float(args.runtime_planning_clearance_m),
                    astar_clearance_hard_min_m=float(getattr(args, "astar_clearance_hard_min_m", 0.0)),
                ),
                "guard_min_clearance_m_effective": float(
                    getattr(args, "guard_effective_min_clearance_m", getattr(args, "guard_min_clearance_m", 0.14))
                ),
                "target_clearance_m": (
                    dict(getattr(nav_decision_local, "metadata", {}) or {}).get("frontier_target_clearance_m")
                    if nav_decision_local is not None
                    else None
                ),
                "astar_reached_goal_grid": (
                    dict(getattr(nav_decision_local, "metadata", {}) or {}).get("astar_reached_goal_grid")
                    if nav_decision_local is not None
                    else None
                ),
                "frontier_center_grid": (
                    dict(getattr(nav_decision_local, "metadata", {}) or {}).get("frontier_center_grid")
                    if nav_decision_local is not None
                    else None
                ),
                "frontier_actual_target_grid": (
                    dict(getattr(nav_decision_local, "metadata", {}) or {}).get("frontier_actual_target_grid")
                    if nav_decision_local is not None
                    else None
                ),
                "frontier_target_belongs_to_committed_frontier": bool(
                    dict(getattr(nav_decision_local, "metadata", {}) or {}).get("frontier_commitment_target_consistent", True)
                )
                if nav_decision_local is not None
                else None,
                "commit_center_target_mismatch": bool(
                    dict(getattr(nav_decision_local, "metadata", {}) or {}).get("commit_center_target_mismatch", False)
                )
                if nav_decision_local is not None
                else None,
                **dict(guard_debug or {}),
                "predicted_pose_world": [float(v) for v in predict_kinematic_pose(tuple(float(v) for v in pose_local), guarded_cmd, float(args.control_dt))],
                **exec_debug,
            }
        )
        try:
            runtime_control_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with runtime_control_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            print("[runtime-control] failed to write %s: %s" % (str(runtime_control_trace_path), exc), flush=True)

    def write_runtime_decision_trace(
        *,
        step_idx: int,
        event_type: str,
        event_reason: str,
        current_grid_local: Tuple[int, int] | None = None,
        nav_decision_local: NavigationDecision | None = None,
        recovery_action: str | None = None,
        recovery_target: FrontierRecoveryTarget | None = None,
        roomseg_update_allowed: bool | None = None,
        roomseg_update_reason: str | None = None,
        roomseg_context_updated_this_step: bool | None = None,
        snapshot_saved: bool | None = None,
        current_path_len: int | None = None,
        zero_velocity_continue: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        meta = dict(getattr(nav_decision_local, "metadata", None) or {})
        selected_frontier = (
            nav_decision_local.frontier_decision.selected_frontier
            if nav_decision_local is not None
            and nav_decision_local.frontier_decision is not None
            else None
        )
        exec_debug = committed_frontier_execution.debug_metadata(
            step=int(step_idx),
            current_grid=current_grid_local,
            resolution_m=float(dynamic_map_info.resolution_m),
        )
        row = make_jsonable(
            {
                "step": int(step_idx),
                "event_type": str(event_type),
                "event_reason": str(event_reason),
                "current_grid": list(current_grid_local) if current_grid_local is not None else None,
                "selected_frontier_center_grid": (
                    [int(v) for v in selected_frontier.center_grid]
                    if selected_frontier is not None
                    else meta.get("frontier_center_grid")
                ),
                "frontier_actual_target_grid": meta.get("frontier_actual_target_grid"),
                "committed_frontier_id": committed_frontier_execution.frontier_id,
                "committed_frontier_exists": bool(committed_frontier_execution.exists()),
                "recovery_action": recovery_action,
                "recovery_target_grid": (
                    [int(recovery_target.target_cell[0]), int(recovery_target.target_cell[1])]
                    if recovery_target is not None and recovery_target.target_cell is not None
                    else None
                ),
                "recovery_mode": None if recovery_target is None else str(recovery_target.mode),
                "recovery_reason": None if recovery_target is None else str(recovery_target.reason),
                "roomseg_update_allowed": roomseg_update_allowed,
                "roomseg_update_reason": roomseg_update_reason,
                "roomseg_context_updated_this_step": roomseg_context_updated_this_step,
                "snapshot_saved": snapshot_saved,
                "current_path_len": current_path_len,
                "zero_velocity_continue": bool(zero_velocity_continue),
                "frontier_refresh_pending": bool(frontier_arrival_update_state.refresh_pending),
                "frontier_refresh_reason": str(frontier_arrival_update_state.refresh_reason),
                "frontier_refresh_key": frontier_arrival_update_state.refresh_frontier_key,
                "frontier_refresh_request_count": int(frontier_arrival_update_state.refresh_request_count),
                "frontier_duplicate_refresh_requests": int(frontier_arrival_update_state.duplicate_refresh_requests),
                "frontier_last_duplicate_refresh_step": int(frontier_arrival_update_state.last_duplicate_refresh_step),
                "frontier_reselect_blocked_until_refresh": bool(frontier_arrival_update_state.block_reselect_until_refresh),
                "frontier_raw_count": int(last_frontier_raw_cells),
                "frontier_real_count": int(last_frontier_real_count),
                "frontier_near_fallback_count": int(last_frontier_near_fallback_count),
                "frontier_terminal_suppressed_count": int(last_frontier_terminal_suppressed_count),
                "frontier_filtered_count": int(last_frontier_filtered_count),
                "explore_until_no_frontiers_stop_candidate": bool(exploration_complete),
                "explore_until_no_frontiers_stop_reason": str(exploration_complete_reason),
                "nav_mode": None if nav_decision_local is None else str(nav_decision_local.mode),
                "nav_reason": None if nav_decision_local is None else str(nav_decision_local.reason),
                "target_cells_count": int(len(getattr(nav_decision_local, "target_cells", []) or [])) if nav_decision_local is not None else 0,
                **exec_debug,
                **frontier_terminal_registry.debug_metadata(),
                **dict(extra or {}),
            }
        )
        try:
            runtime_decision_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with runtime_decision_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            print("[runtime-decision] failed to write %s: %s" % (str(runtime_decision_trace_path), exc), flush=True)

    static_goal_cells = [(int(r), int(c)) for r, c in episode["goal_regions_grid"]]
    start_pose = tuple(float(v) for v in episode["start_pose_world"])
    mapper.reset((float(start_pose[0]), float(start_pose[1])))
    dynamic_map_info = mapper.grid.map_info
    frontier_commitment = (
        FrontierCommitmentManager(
            resolution_m=dynamic_map_info.resolution_m,
            match_radius_m=float(args.frontier_commit_match_radius_m),
            reached_radius_m=float(args.frontier_commit_reached_radius_m),
            min_commit_steps=int(args.frontier_commit_min_steps),
            max_commit_steps=int(args.frontier_commit_max_steps),
            switch_margin=float(args.frontier_commit_switch_margin),
            switch_ratio=float(args.frontier_commit_switch_ratio),
            no_progress_steps=int(args.frontier_commit_no_progress_steps),
            progress_min_delta_m=float(args.frontier_commit_progress_min_delta_m),
            blacklist_ttl_steps=int(args.frontier_blacklist_ttl_steps),
            switch_requires_refresh=bool(getattr(args, "frontier_switch_requires_refresh", True)),
        )
        if bool(args.frontier_commitment_enabled)
        else None
    )
    room_map_mode = str(getattr(args, "room_map_mode", "observed_rooms_json") or "none").strip().lower()
    print(
        "[voxroom-loop] actual_room_map_mode=%s actual_roomseg_backend=%s"
        % (room_map_mode, str(getattr(args, "roomseg_backend", ""))),
        flush=True,
    )
    room_segmenter = None
    room_labeler = None
    room_semantic_labels = {}
    last_room_masks = []
    room_context_cache = RoomContextCache()
    last_room_context_result: Optional[RoomContextResult] = None
    last_room_context_metadata = room_context_not_invoked_metadata()
    last_room_segmentation_debug = {
        "source": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
        "algorithm": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "room_count": 0,
        "rooms": [],
    }
    last_room_semantics_debug = {
        "backend": "unavailable",
        "allowed_categories": list(getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES)),
        "labels": [],
    }
    if room_map_mode in {
        VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
        *VOXEL_OCCUPANCY_ROOMSEG_LEGACY_BACKENDS,
        *VOXEL_OCCUPANCY_ROOMSEG_LEGACY_CONTEXTS,
        "voxel_occupancy_door_wall",
        "voxel_occupancy_door_wall_vlm",
    }:
        roomseg_backend = str(
            getattr(args, "room_segmentation_config", {}).get("backend", VOXEL_OCCUPANCY_ROOMSEG_BACKEND)
            or VOXEL_OCCUPANCY_ROOMSEG_BACKEND
        ).strip().lower()
        allowed_voxel_backends = {VOXEL_OCCUPANCY_ROOMSEG_BACKEND, *VOXEL_OCCUPANCY_ROOMSEG_LEGACY_BACKENDS}
        if roomseg_backend not in allowed_voxel_backends:
            raise ValueError(
                "%s room_map_mode requires --roomseg-backend in %s"
                % (VOXEL_OCCUPANCY_ROOMSEG_BACKEND, sorted(allowed_voxel_backends))
            )
        voxel_cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
            getattr(args, "room_segmentation_config", {}),
            resolution_m=float(dynamic_map_info.resolution_m),
            map_info=dynamic_map_info,
        )
        room_segmenter = VoxelOccupancyDoorWallRoomSegmenter(voxel_cfg, map_info=dynamic_map_info)
        room_label_client = (
            getattr(scenegraph, "paper_llm_client", None)
            if str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm"
            else None
        )
        room_labeler = VLMRoomLabeler(
            client=room_label_client,
            allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            min_confidence=float(getattr(args, "room_label_min_confidence", 0.60)),
            ambiguity_margin=float(getattr(args, "room_label_ambiguity_margin", 0.15)),
            min_reliable_objects=int(getattr(args, "room_label_min_reliable_objects", 2)),
            unknown_category=str(getattr(args, "room_label_unknown_category", "unknown")),
            require_backend=bool(getattr(args, "strict_benchmark", False))
            and str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm",
            max_room_objects_in_prompt=int(getattr(args, "max_room_objects_in_prompt", 25)),
        )
        last_room_semantics_debug["backend"] = room_labeler.backend
    elif room_map_mode in {"none", "disabled", ""}:
        pass
    else:
        raise ValueError(
            "VoxRoom-Online supports only voxel occupancy room segmentation modes; unsupported mapping.room_map_mode: %s"
            % room_map_mode
        )
    goal_cells = []
    for goal_r, goal_c in static_goal_cells:
        gx, gy = grid_to_world_xy(goal_r, goal_c, static_map_info)
        dyn_goal = world_xy_to_grid(gx, gy, dynamic_map_info)
        if is_inside_grid(dyn_goal[0], dyn_goal[1], dynamic_map_info):
            goal_cells.append(dyn_goal)
    max_steps = int(args.max_control_steps)
    replan_every = max(1, int(args.replan_every_steps))
    requested_perception_every = max(1, int(args.perception_every_steps))
    perception_every = effective_perception_every_steps(args.detector, requested_perception_every)
    if perception_every != requested_perception_every:
        print(
            "[voxroom-loop] open-vocabulary detector requires every-frame perception; overriding perception_every_steps "
            "%d -> %d" % (requested_perception_every, perception_every),
            flush=True,
        )
    success_distance = float(episode.get("success_distance_m", 1.0))
    explore_until_no_frontiers = bool(getattr(args, "explore_until_no_frontiers", False))
    frontier_mask_probe = str(getattr(args, "policy", "") or "") in {
        "random_frontier_mask_probe",
        "nearest_frontier_mask_probe",
    }
    current_path: List[Tuple[int, int]] = []
    full_path: List[Tuple[int, int]] = []
    failure_reason = None
    stop_called = False
    policy_stop_confirmed = False
    gt_success_region_reached = False
    metric_pose_valid = True
    metric_pose_invalid_steps = 0
    metric_pose_last_invalid_step = -1
    metric_pose_last_invalid_reason = None
    logged_metric_pose_invalid = False
    frontier_target_resolution_failures = 0
    frontier_target_no_clearance_safe_count = 0
    frontier_commit_target_mismatch_count = 0
    guard_low_clearance_blocked_steps = 0
    last_guard_debug: dict[str, object] = {}
    gt_success_without_sgnav_stop_steps = 0
    gt_success_ignored_steps = 0
    target_radius_reached_steps = 0
    nav_execution_progress_key = None
    nav_execution_best_distance_m = float("inf")
    nav_execution_no_progress_steps = 0
    stop_blocked_reason = None
    logged_gt_success_without_sgnav_stop = False
    last_detections_2d: List[Detection2D] = []
    detection_category_counts = Counter()
    goal_detection_history = []
    last_frontiers = []
    roomseg_frontier_update_gate = FrontierRoomsegUpdateGateState()
    frontier_arrival_update_state = FrontierArrivalUpdateState()
    committed_frontier_execution = CommittedFrontierExecutionState()
    roomseg_update_call_counts_by_step: dict[int, int] = {}
    roomseg_update_called_sites_by_step: dict[int, list[str]] = {}
    last_nav_decision = None
    last_frontier_commitment_metadata = {}
    last_frontier_commitment_reason = ""
    last_selected_candidate = None
    logged_selected_candidate_id = None
    raw_detection_debug_log: List[dict] = []
    last_dynamic_occupancy = mapper.grid.occupied.astype(bool)
    last_dynamic_free = mapper.grid.free.astype(bool)
    last_dynamic_navigable = mapper.traversible(unknown_is_obstacle=True)
    last_dynamic_astar_navigable = apply_dynamic_astar_edge_clearance(
        last_dynamic_navigable,
        mapper.grid.occupied,
        mapper.grid.map_info.resolution_m,
        float(args.runtime_planning_clearance_m),
    )
    last_dynamic_observed = mapper.grid.observed.astype(bool)
    last_frontier_raw_cells = 0
    last_frontier_clusters = 0
    last_roomseg_snapshot_frontier_key = None
    last_roomseg_snapshot_agent_grid = None
    last_roomseg_snapshot_selected_key = None
    last_roomseg_snapshot_geometry_hash = None
    last_roomseg_snapshot_update_reason = ""
    last_roomseg_snapshot_step = -1
    frontier_target_mode = None
    frontier_center_grid = None
    frontier_actual_target_grid = None
    frontier_unreachable_recovery = False
    frontier_unreachable_reason = None
    frontier_stop_at_current_grid = None
    frontier_blacklisted = False
    frontier_direct_no_path_count = 0
    last_frontier_recovery_metadata: dict[str, object] = {}
    frontier_recovery_cfg = frontier_recovery_config_from_args(args)
    initial_frontier_loop_grid: tuple[int, int] | None = None
    roomseg_updates_since_start = 0
    selected_frontier_changes_since_start = 0
    control_steps_with_nonzero_motion = 0
    frontier_initial_refresh_loop_detected = False
    last_initial_loop_selected_frontier_key = None
    frontier_terminal_registry = FrontierTerminalRegistry(
        match_radius_cells=max(1, int(round(float(getattr(args, "frontier_terminal_match_radius_m", 0.25)) / max(float(dynamic_map_info.resolution_m), 1e-6)))),
        target_match_radius_cells=max(1, int(round(float(getattr(args, "frontier_terminal_target_match_radius_m", 0.10)) / max(float(dynamic_map_info.resolution_m), 1e-6)))),
        require_target_match_when_record_has_target=bool(getattr(args, "frontier_terminal_require_target_match_when_record_has_target", True)),
        allow_center_only_match_for_targetless_records=bool(getattr(args, "frontier_terminal_allow_center_only_match_for_targetless_records", True)),
        ignore_targetless_query_for_targeted_record=bool(getattr(args, "frontier_terminal_ignore_targetless_query_for_targeted_record", True)),
    )
    frontier_refresh_loop_prevented_count = 0
    frontier_repeated_same_target_refresh_count = 0
    exploration_complete = False
    exploration_complete_reason = ""
    last_frontier_real_count = 0
    last_frontier_near_fallback_count = 0
    last_frontier_terminal_suppressed_count = 0
    last_frontier_filtered_count = 0
    last_frontier_exhaustion_audit = FrontierExhaustionAudit(
        reachable_frontier_components=0,
        recovery_approach_candidates=0,
        navigation_unknown_adjacent_to_free_cells=0,
        stop_allowed=False,
        stop_blocked_reason="not_audited",
    )
    exploration_stop_reason_detailed = ""

    def _frontier_center_target_for_terminal(
        nav_decision_local: NavigationDecision | None,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        meta = dict(getattr(nav_decision_local, "metadata", None) or {}) if nav_decision_local is not None else {}
        center = _as_grid_cell(getattr(committed_frontier_execution, "selected_center_grid", None))
        target = _as_grid_cell(getattr(committed_frontier_execution, "actual_target_grid", None))
        if center is None:
            center = _as_grid_cell(meta.get("frontier_center_grid"))
        if target is None:
            target = _as_grid_cell(meta.get("frontier_actual_target_grid"))
        if center is None and nav_decision_local is not None and nav_decision_local.frontier_decision is not None:
            selected = nav_decision_local.frontier_decision.selected_frontier
            if selected is not None:
                center = _as_grid_cell(getattr(selected, "center_grid", None))
        if target is None and nav_decision_local is not None and nav_decision_local.target_cells:
            target = _as_grid_cell(nav_decision_local.target_cells[0])
        return center, target

    def mark_frontier_terminal_for_refresh(
        *,
        nav_decision_local: NavigationDecision | None,
        current_grid_local: tuple[int, int],
        reason: str,
        status: str,
    ) -> tuple | None:
        nonlocal frontier_repeated_same_target_refresh_count
        center, target = _frontier_center_target_for_terminal(nav_decision_local)
        if center is None:
            return None
        key = frontier_terminal_registry.make_key(center, target)
        if key in frontier_terminal_registry.records:
            frontier_repeated_same_target_refresh_count += 1
        return frontier_terminal_registry.mark_terminal(
            center_grid=center,
            target_grid=target,
            robot_grid=tuple(int(v) for v in current_grid_local),
            reason=reason,
            status=status,
            step=int(step),
            suppress_steps=_frontier_terminal_suppress_steps_for_status(str(status)),
        )

    def _frontier_terminal_suppress_steps_for_status(status: str) -> int:
        normalized = str(status)
        if normalized == "reached":
            return int(getattr(args, "frontier_terminal_reached_suppress_steps", getattr(args, "frontier_terminal_suppress_steps", 10**9)))
        if normalized == "partial_reached":
            return int(getattr(args, "frontier_terminal_partial_reached_suppress_steps", getattr(args, "frontier_terminal_suppress_steps", 10**9)))
        if normalized == "failed_unreachable":
            return int(getattr(args, "frontier_terminal_failed_unreachable_suppress_steps", getattr(args, "frontier_terminal_suppress_steps", 10**9)))
        if normalized == "stale_near_fallback":
            return int(getattr(args, "frontier_terminal_stale_near_fallback_suppress_steps", getattr(args, "frontier_terminal_suppress_steps", 10**9)))
        return int(getattr(args, "frontier_terminal_suppress_steps", 10**9))

    def _frontier_refresh_key_or_fallback(key: tuple | None, nav_decision_local: NavigationDecision | None) -> tuple | None:
        if key is not None:
            return key
        if committed_frontier_execution.exists():
            return make_committed_frontier_arrival_key(committed_frontier_execution)
        if nav_decision_local is not None:
            return make_frontier_arrival_key(nav_decision_local)
        return None

    def terminal_trace_metadata(*, selected_key: tuple | None = None, target_equals_current: bool = False) -> dict[str, object]:
        out = dict(frontier_terminal_registry.debug_metadata())
        if selected_key is not None:
            out["frontier_selected_key"] = list(selected_key)
        out["frontier_selected_target_equals_current"] = bool(target_equals_current)
        return out

    def compute_frontier_recovery_target(
        *,
        nav_decision_local: NavigationDecision,
        current_grid_local: Tuple[int, int],
        traversible_local: np.ndarray,
        clearance_local: np.ndarray,
        planner_local: GridAStarPlanner | None,
        trigger_reason: str,
    ) -> FrontierRecoveryTarget | None:
        if not bool(getattr(args, "frontier_unreachable_recovery_enabled", True)):
            return None
        if not committed_frontier_execution.exists():
            return None
        if (
            int(committed_frontier_execution.recovery_attempts)
            >= int(frontier_recovery_cfg.max_recovery_attempts_per_frontier)
            and not bool(committed_frontier_execution.recovery_active)
        ):
            return None
        center, members = _frontier_members_for_recovery(nav_decision_local, committed_frontier_execution)
        if center is None:
            return None
        original_target = (
            committed_frontier_execution.original_actual_target_grid
            or committed_frontier_execution.actual_target_grid
            or _as_grid_cell(dict(getattr(nav_decision_local, "metadata", {}) or {}).get("frontier_actual_target_grid"))
        )
        recovery_traversible = np.asarray(traversible_local, dtype=bool)
        recovery_planner = planner_local
        if recovery_planner is None:
            recovery_planner = GridAStarPlanner(
                recovery_traversible,
                float(dynamic_map_info.resolution_m),
                allow_diagonal=True,
            )
        recovery = find_best_reachable_frontier_approach(
            current_grid=tuple(int(v) for v in current_grid_local),
            frontier_center=center,
            frontier_members=members,
            original_target=original_target,
            traversible=recovery_traversible,
            clearance_m=np.asarray(clearance_local, dtype=np.float32),
            planner=recovery_planner,
            resolution_m=float(dynamic_map_info.resolution_m),
            config=frontier_recovery_cfg,
            selected_step=int(committed_frontier_execution.selected_step),
            current_step=int(step),
            selected_grid=committed_frontier_execution.selected_robot_grid,
        )
        recovery.reason = str(trigger_reason) if recovery.reachable else str(recovery.reason)
        return recovery

    def apply_frontier_recovery_or_refresh(
        *,
        nav_decision_local: NavigationDecision,
        current_grid_local: Tuple[int, int],
        traversible_local: np.ndarray,
        clearance_local: np.ndarray,
        planner_local: GridAStarPlanner | None,
        trigger_reason: str,
        refresh_reason: str,
        partial_refresh_reason: str | None = None,
        mark_failed_if_no_recovery: bool = False,
    ) -> str:
        nonlocal current_path, force_perception_step, frontier_unreachable_recovery, frontier_unreachable_reason
        nonlocal frontier_stop_at_current_grid, frontier_blacklisted, last_frontier_commitment_reason
        nonlocal last_frontier_commitment_metadata, last_nav_decision, nav_execution_progress_key
        nonlocal nav_execution_best_distance_m, nav_execution_no_progress_steps, last_frontier_recovery_metadata
        recovery = compute_frontier_recovery_target(
            nav_decision_local=nav_decision_local,
            current_grid_local=current_grid_local,
            traversible_local=traversible_local,
            clearance_local=clearance_local,
            planner_local=planner_local,
            trigger_reason=trigger_reason,
        )
        recovery_metadata = dict(recovery.metadata()) if recovery is not None else {
            "frontier_target_planning_stage": "recovery_unavailable",
            "frontier_recovery_reason": str(trigger_reason),
            "frontier_unreachable_recovery": False,
        }
        last_frontier_recovery_metadata = dict(recovery_metadata)
        nav_decision_local.metadata = {
            **dict(nav_decision_local.metadata or {}),
            **recovery_metadata,
            "frontier_refresh_required_before_reselect": True,
        }
        if recovery is not None and recovery.reachable and recovery.target_cell is not None:
            target = tuple(int(v) for v in recovery.target_cell)
            if recovery.mode == "current_cell_partial_arrival" or target == tuple(int(v) for v in current_grid_local) or len(recovery.path) <= 1:
                effective_refresh_reason = str(partial_refresh_reason or refresh_reason)
                key = mark_frontier_terminal_for_refresh(
                    nav_decision_local=nav_decision_local,
                    current_grid_local=tuple(int(v) for v in current_grid_local),
                    reason=effective_refresh_reason,
                    status="partial_reached",
                )
                refresh_key = _frontier_refresh_key_or_fallback(key, nav_decision_local)
                committed_frontier_execution.mark_partial_arrival_pending(int(step), refresh_key, effective_refresh_reason)
                refresh_requested = request_frontier_refresh(
                    refresh_state=frontier_arrival_update_state,
                    step=int(step),
                    reason=effective_refresh_reason,
                    frontier_key=refresh_key,
                    block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
                )
                write_runtime_decision_trace(
                    step_idx=int(step),
                    event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
                    event_reason=effective_refresh_reason,
                    current_grid_local=tuple(int(v) for v in current_grid_local),
                    nav_decision_local=nav_decision_local,
                    recovery_action="refresh_pending" if refresh_requested else "duplicate_refresh_skipped",
                    recovery_target=recovery,
                    current_path_len=0,
                    zero_velocity_continue=True,
                    extra=terminal_trace_metadata(selected_key=key, target_equals_current=target == tuple(int(v) for v in current_grid_local)),
                )
                if frontier_commitment is not None:
                    frontier_commitment.mark_active_reached(step, effective_refresh_reason)
                current_path = []
                force_perception_step = True
                full_path.append(tuple(int(v) for v in current_grid_local))
                frontier_unreachable_recovery = True
                frontier_unreachable_reason = effective_refresh_reason
                frontier_stop_at_current_grid = [int(current_grid_local[0]), int(current_grid_local[1])]
                last_frontier_commitment_reason = effective_refresh_reason
                last_frontier_commitment_metadata = {
                    **dict(last_frontier_commitment_metadata),
                    **committed_frontier_execution.debug_metadata(
                        step=int(step),
                        current_grid=tuple(int(v) for v in current_grid_local),
                        resolution_m=float(dynamic_map_info.resolution_m),
                    ),
                    **recovery_metadata,
                    "frontier_commitment_reason": effective_refresh_reason,
                    "frontier_refresh_pending": True,
                    "frontier_refresh_reason": effective_refresh_reason,
                }
                nav_decision_local.reason = effective_refresh_reason
                nav_decision_local.metadata = {
                    **dict(nav_decision_local.metadata or {}),
                    "frontier_refresh_pending": True,
                    "frontier_refresh_reason": effective_refresh_reason,
                }
                last_nav_decision = nav_decision_local
                return "refresh_pending"
            committed_frontier_execution.start_recovery(
                step=int(step),
                target_grid=target,
                path=list(recovery.path),
                reason=trigger_reason,
            )
            nav_decision_local.target_cells = [target]
            nav_decision_local.reason = str(trigger_reason)
            nav_decision_local.metadata = {
                **dict(nav_decision_local.metadata or {}),
                **recovery_metadata,
                **committed_frontier_execution.debug_metadata(
                    step=int(step),
                    current_grid=tuple(int(v) for v in current_grid_local),
                    resolution_m=float(dynamic_map_info.resolution_m),
                ),
                "frontier_actual_target_grid": [int(target[0]), int(target[1])],
                "frontier_target_planning_stage": "recovery",
            }
            current_path = list(recovery.path)
            write_runtime_decision_trace(
                step_idx=int(step),
                event_type="frontier_recovery_started",
                event_reason=str(trigger_reason),
                current_grid_local=tuple(int(v) for v in current_grid_local),
                nav_decision_local=nav_decision_local,
                recovery_action="recovery_started",
                recovery_target=recovery,
                current_path_len=len(current_path),
            )
            force_perception_step = False
            frontier_unreachable_recovery = True
            frontier_unreachable_reason = trigger_reason
            frontier_blacklisted = False
            last_frontier_commitment_reason = trigger_reason
            last_frontier_commitment_metadata = {
                **dict(last_frontier_commitment_metadata),
                **dict(nav_decision_local.metadata or {}),
                "frontier_commitment_reason": trigger_reason,
                "frontier_blacklisted": False,
            }
            last_nav_decision = nav_decision_local
            nav_execution_progress_key = None
            nav_execution_best_distance_m = float("inf")
            nav_execution_no_progress_steps = 0
            return "recovery_started"
        if mark_failed_if_no_recovery and committed_frontier_execution.exists():
            committed_frontier_execution.mark_failed(refresh_reason, blacklist=False)
        if frontier_commitment is not None:
            frontier_commitment.mark_active_failed(step, refresh_reason, blacklist=False)
        key = mark_frontier_terminal_for_refresh(
            nav_decision_local=nav_decision_local,
            current_grid_local=tuple(int(v) for v in current_grid_local),
            reason=refresh_reason,
            status="failed_unreachable",
        )
        refresh_key = _frontier_refresh_key_or_fallback(key, nav_decision_local)
        refresh_requested = request_frontier_refresh(
            refresh_state=frontier_arrival_update_state,
            step=int(step),
            reason=refresh_reason,
            frontier_key=refresh_key,
            block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
        )
        write_runtime_decision_trace(
            step_idx=int(step),
            event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
            event_reason=refresh_reason,
            current_grid_local=tuple(int(v) for v in current_grid_local),
            nav_decision_local=nav_decision_local,
            recovery_action="failed_pending_refresh" if refresh_requested else "duplicate_refresh_skipped",
            recovery_target=recovery,
            current_path_len=0,
            zero_velocity_continue=True,
            extra=terminal_trace_metadata(selected_key=key),
        )
        current_path = []
        force_perception_step = True
        full_path.append(tuple(int(v) for v in current_grid_local))
        frontier_unreachable_recovery = True
        frontier_unreachable_reason = refresh_reason
        frontier_stop_at_current_grid = [int(current_grid_local[0]), int(current_grid_local[1])]
        frontier_blacklisted = False
        last_frontier_commitment_reason = refresh_reason
        last_frontier_commitment_metadata = {
            **dict(last_frontier_commitment_metadata),
            **recovery_metadata,
            **committed_frontier_execution.debug_metadata(
                step=int(step),
                current_grid=tuple(int(v) for v in current_grid_local),
                resolution_m=float(dynamic_map_info.resolution_m),
            ),
            "frontier_commitment_reason": refresh_reason,
            "frontier_refresh_pending": True,
            "frontier_refresh_reason": refresh_reason,
            "frontier_blacklisted": False,
        }
        last_nav_decision = nav_decision_local
        nav_execution_progress_key = None
        nav_execution_best_distance_m = float("inf")
        nav_execution_no_progress_steps = 0
        return "refresh_pending"

    def request_frontier_refresh_at_current(
        *,
        nav_decision_local: NavigationDecision | None,
        current_grid_local: Tuple[int, int],
        reason: str,
        status_reason: str | None = None,
        reached: bool = False,
    ) -> None:
        nonlocal current_path, force_perception_step, frontier_unreachable_recovery, frontier_unreachable_reason
        nonlocal frontier_stop_at_current_grid, frontier_blacklisted, last_frontier_commitment_reason
        nonlocal last_frontier_commitment_metadata, last_nav_decision, nav_execution_progress_key
        nonlocal nav_execution_best_distance_m, nav_execution_no_progress_steps
        key = mark_frontier_terminal_for_refresh(
            nav_decision_local=nav_decision_local,
            current_grid_local=tuple(int(v) for v in current_grid_local),
            reason=reason,
            status="partial_reached" if reached else "failed_unreachable",
        )
        refresh_key = _frontier_refresh_key_or_fallback(key, nav_decision_local)
        if committed_frontier_execution.exists():
            committed_frontier_execution.mark_partial_arrival_pending(int(step), refresh_key, reason)
        refresh_requested = request_frontier_refresh(
            refresh_state=frontier_arrival_update_state,
            step=int(step),
            reason=reason,
            frontier_key=refresh_key,
            block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
        )
        write_runtime_decision_trace(
            step_idx=int(step),
            event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
            event_reason=reason,
            current_grid_local=tuple(int(v) for v in current_grid_local),
            nav_decision_local=nav_decision_local,
            recovery_action="refresh_pending" if refresh_requested else "duplicate_refresh_skipped",
            current_path_len=0,
            zero_velocity_continue=True,
            extra={**terminal_trace_metadata(selected_key=key), "frontier_status_reason": status_reason},
        )
        if frontier_commitment is not None:
            if reached:
                frontier_commitment.mark_active_reached(step, reason)
            else:
                frontier_commitment.mark_active_failed(step, reason, blacklist=False)
        current_path = []
        force_perception_step = True
        full_path.append(tuple(int(v) for v in current_grid_local))
        frontier_unreachable_recovery = True
        frontier_unreachable_reason = reason
        frontier_stop_at_current_grid = [int(current_grid_local[0]), int(current_grid_local[1])]
        frontier_blacklisted = False
        last_frontier_commitment_reason = reason
        debug_metadata = (
            committed_frontier_execution.debug_metadata(
                step=int(step),
                current_grid=tuple(int(v) for v in current_grid_local),
                resolution_m=float(dynamic_map_info.resolution_m),
            )
            if committed_frontier_execution is not None
            else {}
        )
        last_frontier_commitment_metadata = {
            **dict(last_frontier_commitment_metadata),
            **debug_metadata,
            **dict(last_frontier_recovery_metadata),
            "frontier_commitment_reason": reason,
            "frontier_refresh_pending": True,
            "frontier_refresh_reason": reason,
            "frontier_refresh_status_reason": status_reason,
            "frontier_blacklisted": False,
            "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
        }
        if nav_decision_local is not None:
            nav_decision_local.reason = reason
            nav_decision_local.metadata = {
                **dict(nav_decision_local.metadata or {}),
                **debug_metadata,
                **dict(last_frontier_recovery_metadata),
                "frontier_refresh_pending": True,
                "frontier_refresh_reason": reason,
                "frontier_refresh_status_reason": status_reason,
                "frontier_blacklisted": False,
                "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
            }
            last_nav_decision = nav_decision_local
        nav_execution_progress_key = None
        nav_execution_best_distance_m = float("inf")
        nav_execution_no_progress_steps = 0

    paper_mode = str(getattr(args, "sgnav_mode", "legacy")).strip().lower() == "paper"
    long_term_goal = LongTermGoalState()
    voxroom_viz_enabled = bool(getattr(args, "sgnav_viz", False))
    voxroom_viz_save_dir = getattr(args, "voxroom_viz_save_dir", None)
    detection_localization = str(getattr(args, "detection_localization", "static_map_ray")).strip().lower()
    if detection_localization in {"static_map_ray", "map_ray", "rgb_map_ray"}:
        print("[voxroom-loop] static-map detection localization disabled; using depth projection", flush=True)
        detection_localization = "depth"
    args.read_depth = True
    roomseg_debug_only = bool(getattr(args, "roomseg_debug_only", False))
    viz = None
    viz_requested = bool(voxroom_viz_enabled or voxroom_viz_save_dir)
    viz_every = max(1, int(getattr(args, "voxroom_viz_every_steps", 1)))
    detector_cuda_rgb = detector_can_use_cuda_rgb(detector, args.detector, camera_annotator_device)
    vllm_needs_cpu_rgb = bool(getattr(args, "vllm_frontier_scoring", False) and getattr(args, "vllm_image_scoring", True))
    logged_detector_rgb_device = False
    latency_totals_ms = {
        "perception": 0.0,
        "mapping": 0.0,
        "graph": 0.0,
        "llm": 0.0,
        "planning": 0.0,
    }
    latency_counts = {key: 0 for key in latency_totals_ms}
    mapping_breakdown_totals_ms: dict[str, float] = {}
    mapping_breakdown_counts: dict[str, int] = {}
    loop_timing_enabled = str(os.environ.get("SGNAV_LOOP_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}
    loop_timing_last = time.perf_counter()

    if str(os.environ.get("SGNAV_FAULTHANDLER", "")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            import faulthandler
            import signal

            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
            print("[sgnav-loop-timing] faulthandler SIGUSR1 enabled", flush=True)
        except Exception as exc:
            print("[sgnav-loop-timing] faulthandler setup failed: %s" % exc, flush=True)

    def record_latency(name: str, started_at: float) -> None:
        latency_totals_ms[name] += max(0.0, (time.perf_counter() - started_at) * 1000.0)
        latency_counts[name] += 1

    def loop_tick(step_idx: int, label: str, *, reset: bool = False) -> None:
        nonlocal loop_timing_last
        if not loop_timing_enabled:
            return
        now = time.perf_counter()
        elapsed_ms = 0.0 if reset else max(0.0, (now - loop_timing_last) * 1000.0)
        loop_timing_last = now
        print("[sgnav-loop-timing] step=%s %s +%.1fms" % (step_idx, str(label), elapsed_ms), flush=True)

    def record_mapping_timing(name: str, elapsed_ms: float) -> None:
        key = str(name)
        value = max(0.0, float(elapsed_ms))
        mapping_breakdown_totals_ms[key] = mapping_breakdown_totals_ms.get(key, 0.0) + value
        mapping_breakdown_counts[key] = mapping_breakdown_counts.get(key, 0) + 1

    def record_mapping_breakdown(breakdown: Mapping[str, object]) -> None:
        for key, value in dict(breakdown).items():
            if key == "reason" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                record_mapping_timing(str(key), float(value))

    def total_llm_requests() -> int:
        vllm_scorer = getattr(scenegraph, "vllm_scorer", None)
        paper_llm_client = getattr(scenegraph, "paper_llm_client", None)
        hcot_scorer = getattr(scenegraph, "hcot_scorer", None)
        return (
            int(getattr(vllm_scorer, "request_count", 0))
            + int(getattr(paper_llm_client, "request_count", 0))
            + int(getattr(hcot_scorer, "llm_request_count", 0))
        )

    run_dir = Path(args.output).expanduser().parent
    scene_id = str(episode.get("scene_id") or episode.get("scene_name") or run_dir.name)
    live_baseline_manager = LiveBaselineManager(args, scene_id=scene_id, run_dir=run_dir)

    def needs_viz_frame(step_idx: int) -> bool:
        return viz_requested and step_idx < max_steps and step_idx % viz_every == 0

    def rgb_request_device(step_idx: int) -> Optional[str]:
        if step_idx >= max_steps:
            return None
        if bool(live_baseline_manager.enabled):
            return "cpu"
        if detector_requires_rgb(detector, args.detector) and step_idx % perception_every == 0:
            return "cuda" if detector_cuda_rgb else "cpu"
        if needs_viz_frame(step_idx):
            return "cpu"
        return None

    server = IsaacSimServer(
        headless=args.headless,
        width=int(args.isaac_width),
        height=int(args.isaac_height),
        camera_hfov_deg=float(args.camera_hfov_deg),
        mast_height_m=float(args.camera_mast_height_m),
        forward_offset_m=float(args.camera_forward_offset_m),
        camera_pitch_deg=float(args.camera_pitch_deg),
        camera_near_m=float(args.camera_near_m),
        camera_far_m=float(args.camera_far_m),
        enable_depth=bool(getattr(args, "read_depth", False)),
        camera_annotator_device=camera_annotator_device,
        enable_nearfield_depth=bool(getattr(args, "nearfield_depth", False)),
        nearfield_width=int(args.nearfield_width),
        nearfield_height=int(args.nearfield_height),
        nearfield_hfov_deg=float(args.nearfield_hfov_deg),
        nearfield_height_m=float(args.nearfield_height_m),
        nearfield_near_m=float(args.nearfield_near_m),
        nearfield_far_m=float(args.nearfield_far_m),
    )
    try:
        first_rgb_device = rgb_request_device(0)
        obs = server.reset_episode(
            episode["usd_path"],
            start_pose,
            read_rgb=first_rgb_device is not None,
            rgb_device=first_rgb_device,
        )
        print("[voxroom-loop] first Isaac observation received", flush=True)
        viz = VoxRoomPopupVisualizer(
            enabled=voxroom_viz_enabled,
            save_dir=voxroom_viz_save_dir,
            panel_size=(int(args.voxroom_viz_width), int(args.voxroom_viz_height)),
            save_every_steps=int(args.voxroom_viz_save_every_steps),
            ipc_jpeg_quality=int(args.voxroom_viz_jpeg_quality),
            debug_overlay_layers=bool(args.debug_overlay_layers),
            save_overlay_layer_metadata=bool(args.save_overlay_layer_metadata),
            show_gt_goal_cells=bool(args.show_gt_goal_cells),
            show_room_proposals=bool(args.show_room_proposals),
            show_room_masks=bool(args.show_room_masks),
            show_room_labels=bool(args.show_room_labels),
            show_frontier_member_cells=bool(args.show_frontier_member_cells),
            show_object_nodes=bool(args.show_object_nodes),
            show_candidate_markers=bool(args.show_candidate_markers),
            min_valid_detection_confidence=float(args.min_valid_detection_confidence),
            max_green_like_primitives_before_warning=int(args.max_green_like_primitives_before_warning),
        ) if (voxroom_viz_enabled or voxroom_viz_save_dir) else None
        intr = CameraIntrinsics.from_hfov(int(args.isaac_width), int(args.isaac_height), float(args.camera_hfov_deg))
        nearfield_intr = CameraIntrinsics.from_hfov(
            int(args.nearfield_width),
            int(args.nearfield_height),
            float(args.nearfield_hfov_deg),
        )
        live_baseline_manager.on_episode_start(
            {
                "scene_id": scene_id,
                "episode_file": getattr(args, "episode_file", None),
                "episode_index": int(getattr(args, "episode_index", 0)),
                "frontier_selection_mode": str(getattr(args, "frontier_selection_mode", "")),
                "frontier_source": str(getattr(args, "frontier_source", "")),
                "policy_control": str(getattr(args, "live_baseline_policy_control", "never")),
            }
        )

        def viz_rgb(current_obs: dict) -> np.ndarray:
            if current_obs.get("has_rgb") and current_obs.get("rgb_device") == "cpu":
                return current_obs["rgb"]
            return server.get_observation(read_rgb=True, read_depth=False, rgb_device="cpu")["rgb"]

        last_decision_mode = "init"
        last_decision_reason = ""
        goal_candidate_count = 0
        sam2_failure_logged = False
        force_perception_step = False
        panorama_frames = 0

        goal_norm = normalize_category(episode["goal_category"])

        def category_matches_goal(category: str) -> bool:
            cat_norm = normalize_category(category)
            return cat_norm == goal_norm or (goal_norm and (goal_norm in cat_norm or cat_norm in goal_norm))

        def summarize_detection(det: Detection2D, step_idx: int) -> dict:
            return {
                "step": int(step_idx),
                "category": normalize_category(det.category),
                "raw_label": str(det.raw_label),
                "confidence": float(det.confidence),
                "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
            }

        def update_mapper_state(current_obs: dict, step_idx: int | None = None, *, build_planners: bool = False):
            nonlocal dynamic_map_info, last_dynamic_occupancy, last_dynamic_free, last_dynamic_navigable, last_dynamic_astar_navigable, last_dynamic_observed
            pose_local = current_obs["pose_world"]
            if not current_obs.get("has_depth"):
                return None
            started_at = time.perf_counter()
            state_timing_local: dict[str, float] = {}
            mapper.update(current_obs["depth"], intr, pose_local, current_obs["camera_pose_world"])
            record_mapping_breakdown(getattr(mapper, "last_timing_stats", {}))
            record_voxel_perf_trace(step_idx)
            mapper.last_debug_stats["depth_source"] = str(current_obs.get("depth_source", "unknown"))
            mapper.last_debug_stats["depth_semantics"] = str(current_obs.get("depth_semantics", "unknown"))
            mapper.last_debug_stats["camera_frame_sync_updates"] = int(current_obs.get("camera_frame_sync_updates", 0) or 0)
            mapper.last_debug_stats["camera_rendering_time"] = current_obs.get("camera_rendering_time")
            if bool(getattr(args, "nearfield_depth", False)) and current_obs.get("has_nearfield_depth"):
                nearfield_started_at = time.perf_counter()
                nearfield_stats = mapper.update_nearfield_topdown(
                    current_obs["nearfield_depth"],
                    nearfield_intr,
                    pose_local,
                    current_obs["nearfield_camera_pose_world"],
                    radius_m=float(args.nearfield_radius_m),
                    ignore_radius_m=float(args.nearfield_ignore_radius_m),
                    depth_stride_px=int(args.nearfield_depth_stride_px),
                    floor_tolerance_m=float(args.nearfield_floor_tolerance_m),
                    obstacle_min_height_m=float(args.nearfield_obstacle_min_height_m),
                    obstacle_max_height_m=float(args.nearfield_obstacle_max_height_m),
                    splat_point_threshold=int(args.nearfield_splat_point_threshold),
                    free_splat_point_threshold=int(args.nearfield_free_splat_point_threshold),
                )
                record_mapping_timing("nearfield_update_ms", (time.perf_counter() - nearfield_started_at) * 1000.0)
                nearfield_stats["depth_source"] = str(current_obs.get("nearfield_depth_source", "unknown"))
                nearfield_stats["depth_semantics"] = str(current_obs.get("nearfield_depth_semantics", "unknown"))
                mapper.last_debug_stats["nearfield"] = nearfield_stats
            if bool(getattr(args, "static_nearfield_map", False)):
                static_nearfield_started_at = time.perf_counter()
                static_nearfield_stats = mapper.update_static_nearfield(
                    static_occupancy,
                    static_navigable,
                    static_map_info,
                    pose_local,
                    radius_m=float(args.static_nearfield_radius_m),
                    static_openings=static_openings,
                )
                record_mapping_timing("static_nearfield_update_ms", (time.perf_counter() - static_nearfield_started_at) * 1000.0)
                mapper.last_debug_stats["static_nearfield"] = static_nearfield_stats
            if bool(getattr(args, "mapping_debug", False)):
                stats = mapper.last_debug_stats
                rel = stats.get("rel_z_m_percentiles", {})
                bands = stats.get("image_bands", {})
                near = stats.get("nearfield", {})
                static_near = stats.get("static_nearfield", {})
                print(
                    "[mapping-debug] step=%s depth=%s sync=%s mode=%s valid=%s rays=%s skip_h=%s free_ray=%s occ_end=%s free_pts=%s obs_pts=%s "
                    "rel_z_p5/50/95=%s/%s/%s top_obs=%s mid_obs=%s bottom_obs=%s ceiling=%s negative=%s "
                    "nearfield=%s/%s/%s static_near=%s/%s/%s"
                    % (
                        "?" if step_idx is None else int(step_idx),
                        stats.get("depth_source", "unknown"),
                        stats.get("camera_frame_sync_updates", 0),
                        stats.get("mapping_mode", "unknown"),
                        stats.get("valid_points", 0),
                        stats.get("ray_count", 0),
                        stats.get("skipped_height_rays", 0),
                        stats.get("free_ray_cells", stats.get("free_splat_cells", 0)),
                        stats.get("occupied_endpoint_cells", stats.get("obstacle_splat_cells", 0)),
                        stats.get("free_band_points", 0),
                        stats.get("obstacle_band_points", 0),
                        rel.get("p5"),
                        rel.get("p50"),
                        rel.get("p95"),
                        (bands.get("top") or {}).get("obstacle_band_points", 0),
                        (bands.get("middle") or {}).get("obstacle_band_points", 0),
                        (bands.get("bottom") or {}).get("obstacle_band_points", 0),
                        stats.get("ceiling_like_points", 0),
                        stats.get("negative_height_points", 0),
                        near.get("reason", "off"),
                        near.get("free_splat_cells", 0),
                        near.get("obstacle_splat_cells", 0),
                        static_near.get("reason", "off"),
                        static_near.get("free_cells", 0),
                        static_near.get("occupied_cells", 0),
                    ),
                    flush=True,
                )
            dynamic_map_info = mapper.grid.map_info
            array_export_started_at = time.perf_counter()
            occupancy_local = mapper.grid.occupied.astype(bool)
            free_local = mapper.grid.free.astype(bool)
            observed_local = mapper.grid.observed.astype(bool)
            elapsed = (time.perf_counter() - array_export_started_at) * 1000.0
            state_timing_local["array_export_ms"] = float(elapsed)
            record_mapping_timing("array_export_ms", elapsed)
            traversible_started_at = time.perf_counter()
            navigable_local = mapper.traversible(unknown_is_obstacle=True)
            elapsed = (time.perf_counter() - traversible_started_at) * 1000.0
            state_timing_local["traversible_ms"] = float(elapsed)
            record_mapping_timing("traversible_ms", elapsed)
            base_nav_planner_local = None
            snap_started_at = time.perf_counter()
            pose_grid_local = world_xy_to_grid(float(pose_local[0]), float(pose_local[1]), dynamic_map_info)
            current_grid_local = snap_to_free_local(
                navigable_local,
                pose_grid_local,
                radius_cells=max(3, int(round(float(args.robot_radius_m) / max(float(dynamic_map_info.resolution_m), 1e-6))) + 2),
            )
            if current_grid_local is None:
                planner_init_started_at = time.perf_counter()
                base_nav_planner_local = GridAStarPlanner(navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
                record_mapping_timing("base_planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
                current_grid_local = base_nav_planner_local.snap_to_free(pose_grid_local)
            else:
                record_mapping_timing("base_planner_init_ms", 0.0)
            record_mapping_timing("snap_to_free_ms", (time.perf_counter() - snap_started_at) * 1000.0)
            astar_navigable_local = navigable_local
            nav_planner_local = None
            astar_clearance_m_local = None
            astar_used_clearance_cost_local = False
            if bool(build_planners):
                clearance_started_at = time.perf_counter()
                astar_navigable_local = apply_dynamic_astar_edge_clearance(
                    navigable_local,
                    occupancy_local,
                    dynamic_map_info.resolution_m,
                    float(args.runtime_planning_clearance_m),
                    current_grid=current_grid_local,
                )
                record_mapping_timing("astar_clearance_ms", (time.perf_counter() - clearance_started_at) * 1000.0)
                planner_init_started_at = time.perf_counter()
                nav_planner_local, astar_clearance_m_local, astar_used_clearance_cost_local = make_runtime_nav_planner(
                    args,
                    astar_navigable_local,
                    occupancy_local,
                    dynamic_map_info.resolution_m,
                )
                if base_nav_planner_local is None:
                    base_nav_planner_local = GridAStarPlanner(navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
                record_mapping_timing("planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
            else:
                record_mapping_timing("astar_clearance_ms", 0.0)
                record_mapping_timing("planner_init_ms", 0.0)
            if current_grid_local is None:
                recovery_started_at = time.perf_counter()
                mapper.update_simple_radius(pose_local, radius_m=max(float(args.robot_radius_m), float(args.online_resolution_m)))
                record_mapping_timing("simple_radius_recovery_ms", (time.perf_counter() - recovery_started_at) * 1000.0)
                array_export_started_at = time.perf_counter()
                occupancy_local = mapper.grid.occupied.astype(bool)
                free_local = mapper.grid.free.astype(bool)
                observed_local = mapper.grid.observed.astype(bool)
                elapsed = (time.perf_counter() - array_export_started_at) * 1000.0
                state_timing_local["array_export_ms"] = float(elapsed)
                record_mapping_timing("array_export_ms", elapsed)
                traversible_started_at = time.perf_counter()
                navigable_local = mapper.traversible(unknown_is_obstacle=True)
                elapsed = (time.perf_counter() - traversible_started_at) * 1000.0
                state_timing_local["traversible_ms"] = float(elapsed)
                record_mapping_timing("traversible_ms", elapsed)
                snap_started_at = time.perf_counter()
                pose_grid_local = world_xy_to_grid(float(pose_local[0]), float(pose_local[1]), dynamic_map_info)
                current_grid_local = snap_to_free_local(navigable_local, pose_grid_local, radius_cells=5)
                if current_grid_local is None:
                    planner_init_started_at = time.perf_counter()
                    base_nav_planner_local = GridAStarPlanner(navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
                    record_mapping_timing("base_planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
                    current_grid_local = base_nav_planner_local.snap_to_free(pose_grid_local)
                record_mapping_timing("snap_to_free_ms", (time.perf_counter() - snap_started_at) * 1000.0)
                if bool(build_planners):
                    clearance_started_at = time.perf_counter()
                    astar_navigable_local = apply_dynamic_astar_edge_clearance(
                        navigable_local,
                        occupancy_local,
                        dynamic_map_info.resolution_m,
                        float(args.runtime_planning_clearance_m),
                        current_grid=current_grid_local,
                    )
                    record_mapping_timing("astar_clearance_ms", (time.perf_counter() - clearance_started_at) * 1000.0)
                    planner_init_started_at = time.perf_counter()
                    nav_planner_local, astar_clearance_m_local, astar_used_clearance_cost_local = make_runtime_nav_planner(
                        args,
                        astar_navigable_local,
                        occupancy_local,
                        dynamic_map_info.resolution_m,
                    )
                    if base_nav_planner_local is None:
                        base_nav_planner_local = GridAStarPlanner(navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
                    record_mapping_timing("planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
            last_dynamic_occupancy = occupancy_local
            last_dynamic_free = free_local
            last_dynamic_navigable = navigable_local
            last_dynamic_astar_navigable = astar_navigable_local
            last_dynamic_observed = observed_local
            record_mapping_timing("state_total_ms", (time.perf_counter() - started_at) * 1000.0)
            record_latency("mapping", started_at)
            return {
                "pose": pose_local,
                "map_info": dynamic_map_info,
                "occupancy": occupancy_local,
                "free": free_local,
                "observed": observed_local,
                "navigable": navigable_local,
                "astar_navigable": astar_navigable_local,
                "base_nav_planner": base_nav_planner_local,
                "nav_planner": nav_planner_local,
                "current_grid": current_grid_local,
                "astar_clearance_m": astar_clearance_m_local,
                "astar_used_clearance_cost": bool(astar_used_clearance_cost_local),
                "astar_clearance_desired_m": float(getattr(args, "astar_clearance_desired_m", 0.25)),
                "_planner_timing_ms": {
                    "base_planner_init_ms": 0.0,
                    "astar_clearance_ms": 0.0,
                    "planner_init_ms": 0.0,
                },
                "_state_timing_ms": state_timing_local,
            }

        def ensure_runtime_planners(map_state_local: dict) -> dict:
            if map_state_local.get("nav_planner") is not None and map_state_local.get("base_nav_planner") is not None:
                return map_state_local
            navigable_local = np.asarray(map_state_local["navigable"], dtype=bool)
            occupancy_local = np.asarray(map_state_local["occupancy"], dtype=bool)
            map_info_local = map_state_local["map_info"]
            current_grid_local = map_state_local.get("current_grid")
            planner_timing: dict[str, float] = {}
            planner_init_started_at = time.perf_counter()
            base_nav_planner_local = GridAStarPlanner(navigable_local, map_info_local.resolution_m, allow_diagonal=True)
            elapsed = (time.perf_counter() - planner_init_started_at) * 1000.0
            planner_timing["base_planner_init_ms"] = float(elapsed)
            record_mapping_timing("base_planner_init_ms", elapsed)
            if current_grid_local is None:
                snap_started_at = time.perf_counter()
                pose_local = map_state_local["pose"]
                current_grid_local = base_nav_planner_local.snap_to_free(
                    world_xy_to_grid(float(pose_local[0]), float(pose_local[1]), map_info_local)
                )
                elapsed = (time.perf_counter() - snap_started_at) * 1000.0
                planner_timing["snap_to_free_ms"] = float(elapsed)
                record_mapping_timing("snap_to_free_ms", elapsed)
            clearance_started_at = time.perf_counter()
            astar_navigable_local = apply_dynamic_astar_edge_clearance(
                navigable_local,
                occupancy_local,
                map_info_local.resolution_m,
                float(args.runtime_planning_clearance_m),
                current_grid=current_grid_local,
            )
            elapsed = (time.perf_counter() - clearance_started_at) * 1000.0
            planner_timing["astar_clearance_ms"] = float(elapsed)
            record_mapping_timing("astar_clearance_ms", elapsed)
            planner_init_started_at = time.perf_counter()
            nav_planner_local, astar_clearance_m_local, astar_used_clearance_cost_local = make_runtime_nav_planner(
                args,
                astar_navigable_local,
                occupancy_local,
                map_info_local.resolution_m,
            )
            elapsed = (time.perf_counter() - planner_init_started_at) * 1000.0
            planner_timing["planner_init_ms"] = float(elapsed)
            record_mapping_timing("planner_init_ms", elapsed)
            map_state_local.update(
                {
                    "astar_navigable": astar_navigable_local,
                    "base_nav_planner": base_nav_planner_local,
                    "nav_planner": nav_planner_local,
                    "current_grid": current_grid_local,
                    "astar_clearance_m": astar_clearance_m_local,
                    "astar_used_clearance_cost": bool(astar_used_clearance_cost_local),
                    "astar_clearance_desired_m": float(getattr(args, "astar_clearance_desired_m", 0.25)),
                    "_planner_timing_ms": planner_timing,
                }
            )
            return map_state_local

        def runtime_navigation_debug_layers(map_state_local: Mapping[str, object], *, include_arrays: bool = False) -> dict[str, object]:
            debug = mapper.navigation_debug_layers(include_arrays=bool(include_arrays))
            clearance = map_state_local.get("astar_clearance_m")
            if isinstance(clearance, np.ndarray):
                clearance_arr = np.asarray(clearance, dtype=np.float32)
                lookahead_min_clearance = float(getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14)))
                if bool(include_arrays):
                    debug["astar_clearance_m"] = clearance_arr.copy()
                    debug["astar_clearance_low_cells"] = clearance_arr < lookahead_min_clearance
                finite = clearance_arr[np.isfinite(clearance_arr)]
                debug["astar_clearance_min_m"] = None if finite.size == 0 else float(np.min(finite))
                debug["astar_clearance_mean_m"] = None if finite.size == 0 else float(np.mean(finite))
                debug["astar_clearance_low_cells_count"] = int(np.count_nonzero(clearance_arr < lookahead_min_clearance))
            debug["astar_clearance_desired_m"] = float(getattr(args, "astar_clearance_desired_m", 0.25))
            debug["astar_used_clearance_cost"] = bool(map_state_local.get("astar_used_clearance_cost", False))
            debug["lookahead_min_clearance_m_configured"] = float(getattr(args, "lookahead_min_clearance_m", 0.14))
            debug["lookahead_min_clearance_m_effective"] = float(
                getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
            )
            return debug

        def run_detector_update(
            current_obs: dict,
            step_idx: int,
            map_info_local: MapInfo,
            occupancy_local: np.ndarray,
            navigable_local: np.ndarray,
        ) -> dict:
            nonlocal detector_cuda_rgb, logged_detector_rgb_device, segmenter, sam2_failure_logged, last_detections_2d
            if detector is None:
                last_detections_2d = []
                return current_obs
            started_at = time.perf_counter()
            obs_local = current_obs
            detector_rgb = obs_local["rgb"]
            used_cuda_rgb = False
            if detector_cuda_rgb:
                if not (
                    obs_local.get("has_rgb")
                    and obs_local.get("rgb_device") == "cuda"
                    and obs_local.get("rgb_gpu") is not None
                ):
                    obs_local = server.get_observation(read_rgb=True, read_depth=False, rgb_device="cuda")
                if obs_local.get("has_rgb") and obs_local.get("rgb_device") == "cuda" and obs_local.get("rgb_gpu") is not None:
                    detector_rgb = obs_local["rgb_gpu"]
                    used_cuda_rgb = True
                else:
                    detector_cuda_rgb = False
                    detector_rgb = obs_local["rgb"]
            if used_cuda_rgb:
                if not logged_detector_rgb_device:
                    print("[voxroom-loop] detector RGB input: Isaac CUDA annotator -> detector tensor", flush=True)
                    logged_detector_rgb_device = True
            elif detector_requires_rgb(detector, args.detector) and not logged_detector_rgb_device:
                print("[voxroom-loop] detector RGB input: CPU fallback", flush=True)
                logged_detector_rgb_device = True
            detections_2d = detector.detect(detector_rgb)
            if int(args.max_detections_per_frame) > 0:
                detections_2d = detections_2d[: int(args.max_detections_per_frame)]
            detections_2d = filter_edge_touching_detections(
                list(detections_2d),
                image_width=int(args.isaac_width),
                image_height=int(args.isaac_height),
                step_idx=int(step_idx),
                reject_edge_touching_bboxes=bool(args.reject_edge_touching_bboxes),
                margin_px=float(args.bbox_edge_margin_px),
                margin_ratio=float(args.bbox_edge_margin_ratio),
                min_confidence=float(args.detector_conf),
                raw_log=raw_detection_debug_log,
            )
            if segmenter is not None and detections_2d:
                try:
                    segment_rgb = obs_local["rgb"] if obs_local.get("rgb_device") == "cpu" else viz_rgb(obs_local)
                    detections_2d = segmenter.segment(segment_rgb, list(detections_2d))
                except Exception as exc:
                    if str(getattr(args, "segmenter", "none")).strip().lower() == "sam2":
                        raise
                    if not sam2_failure_logged:
                        print("[sam2] segmentation failed; continuing with detector boxes only: %s" % exc, flush=True)
                        sam2_failure_logged = True
                    segmenter = None
            last_detections_2d = list(detections_2d)
            for det in detections_2d:
                cat_norm = normalize_category(det.category)
                detection_category_counts[cat_norm] += 1
                if cat_norm == goal_norm:
                    goal_detection_history.append(summarize_detection(det, step_idx))
            if goal_detection_history and goal_detection_history[-1]["step"] == int(step_idx):
                recent = [row for row in goal_detection_history if row["step"] == int(step_idx)]
                print(
                    "[voxroom-loop] detector goal detections step=%s: %s"
                    % (
                        step_idx,
                        ", ".join(
                            "%s raw=%s conf=%.3f"
                            % (row["category"], row["raw_label"], row["confidence"])
                            for row in recent
                        ),
                    ),
                    flush=True,
                )
            if detection_localization == "depth":
                detections_3d = detections_to_3d(
                    detections_2d,
                    obs_local["depth"],
                    intr,
                    obs_local["camera_pose_world"],
                    depth_max_m=float(args.depth_max_m),
                    min_points=int(args.min_depth_points_per_detection),
                )
            elif detection_localization in {"static_map_ray", "map_ray", "rgb_map_ray"}:
                detections_3d = detections_to_3d_static_map_ray(
                    detections_2d,
                    obs_local["camera_pose_world"],
                    int(args.isaac_width),
                    float(args.camera_hfov_deg),
                    map_info_local,
                    occupancy_local,
                    navigable_local,
                    max_range_m=float(args.depth_max_m),
                )
            elif detection_localization == "none":
                detections_3d = []
            else:
                raise ValueError("Unsupported detection localization mode: %s" % detection_localization)
            if str(getattr(args, "sgnav_mode", "legacy")).strip().lower() == "paper" and detection_localization == "depth":
                fused_instances = fused_instance_registry.update(
                    detections_2d,
                    obs_local["depth"],
                    intr,
                    obs_local["camera_pose_world"],
                    step_id=step_idx,
                    depth_max_m=float(args.depth_max_m),
                    min_points=int(args.min_depth_points_per_detection),
                    stride=int(args.depth_stride_px),
                )
                object_memory.update_fused_instances(
                    [instance for instance in fused_instances if int(instance.last_seen_step) == int(step_idx)],
                    step_id=step_idx,
                    map_info=map_info_local,
                )
            else:
                object_memory.update(detections_3d, step_id=step_idx, map_info=map_info_local)
            if paper_mode:
                object_memory.dedupe_goal_candidates(
                    goal_category=episode["goal_category"],
                    merge_radius_m=max(float(args.object_merge_radius_m), 0.75),
                    map_info=map_info_local,
                )
            evaluator.num_detector_calls += 1
            evaluator.num_yolo_calls += 1
            record_latency("perception", started_at)
            return obs_local

        def update_scenegraph_frame(current_obs: dict, step_idx: int, map_state: dict) -> None:
            started_at = time.perf_counter()
            llm_requests_before = total_llm_requests()
            room_map = None if full_room_map is None else observed_room_map(full_room_map, map_state["observed"])
            rgb_for_graph = current_obs["rgb"] if current_obs.get("has_rgb") and current_obs.get("rgb_device") == "cpu" else None
            if rgb_for_graph is None and vllm_needs_cpu_rgb:
                rgb_for_graph = viz_rgb(current_obs)
            scenegraph.update_from_frame(
                object_memory,
                room_map=room_map,
                room_masks=last_room_masks,
                room_semantic_labels=room_semantic_labels,
                room_segmentation_debug=last_room_segmentation_debug,
                room_semantics_debug=last_room_semantics_debug,
                rgb=rgb_for_graph,
                depth=current_obs.get("depth"),
                detections_2d=last_detections_2d,
                map_info=map_state["map_info"],
                occupancy=map_state["occupancy"],
                free=map_state["free"],
                navigable=map_state["free"],
                observed=map_state["observed"],
                pose_world=map_state["pose"],
                camera_pose_world=current_obs.get("camera_pose_world"),
                step_id=step_idx,
            )
            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
            evaluator.num_scenegraph_updates += 1
            record_latency("graph", started_at)
            if total_llm_requests() > llm_requests_before:
                record_latency("llm", started_at)

        def save_roomseg_snapshot_for_frontier_scoring(
            step_idx: int,
            map_state: dict,
            *,
            frontier_map: np.ndarray | None = None,
            selected_frontier_members: Sequence[Sequence[int]] | None = None,
            selected_frontier_center_rc: Sequence[int] | None = None,
        ) -> dict | None:
            if not bool(getattr(args, "save_roomseg_snapshots", False)):
                return None
            voxel_snapshot_arrays = (
                _roomseg_voxel_snapshot_arrays(mapper)
                if bool(getattr(args, "save_roomseg_voxel_evidence", False))
                else None
            )
            memory_snapshot_arrays = _roomseg_memory_snapshot_arrays(room_segmenter)
            extra_npz_arrays = {
                **dict(voxel_snapshot_arrays or {}),
                **dict(memory_snapshot_arrays or {}),
                **map_info_extra_arrays(map_state.get("map_info")),
            }
            nav_free_mask, nav_obstacle_mask, nav_unknown_mask, _nav_source = resolve_replay_style_navigation_masks(
                mapper=mapper,
                shape=np.asarray(map_state["occupancy"]).shape,
                fallback_free=map_state["free"],
                fallback_obstacle=map_state["occupancy"],
                fallback_unknown=~np.asarray(map_state["observed"], dtype=bool),
            )
            return save_roomseg_layer_dump(
                out_dir=str(getattr(args, "roomseg_snapshot_dir", "result/roomseg_snapshots")),
                step=int(step_idx),
                room_debug=_roomseg_debug_for_layer_dump(last_room_segmentation_debug, room_segmenter),
                occupancy_map=nav_obstacle_mask,
                observed_free_mask=nav_free_mask,
                obstacle_mask=nav_obstacle_mask,
                unknown_mask=nav_unknown_mask,
                frontier_map=frontier_map,
                selected_frontier_members=selected_frontier_members,
                selected_frontier_center_rc=selected_frontier_center_rc,
                agent_rc=map_state["current_grid"],
                max_saves=int(getattr(args, "roomseg_snapshot_max_saves", 500)),
                save_npz=True,
                save_png=False,
                save_summary_json=True,
                save_overlay_png=False,
                save_layers_png=False,
                save_navigation_room_masks_png=True,
                npz_keys=ROOMSEG_SNAPSHOT_ARRAY_KEYS,
                extra_npz_arrays=extra_npz_arrays,
                include_selected_frontier_sector=False,
            )

        def update_room_context_for_frontier_scoring(
            step_idx: int,
            map_state: dict,
            *,
            call_site: str = "direct",
        ) -> RoomContextResult:
            nonlocal last_room_masks, room_semantic_labels, last_room_segmentation_debug
            nonlocal last_room_semantics_debug, last_room_context_result, last_room_context_metadata
            step_key = int(step_idx)
            roomseg_update_call_counts_by_step[step_key] = int(roomseg_update_call_counts_by_step.get(step_key, 0)) + 1
            roomseg_update_called_sites_by_step.setdefault(step_key, []).append(str(call_site))
            result = prepare_room_context_for_frontier_scoring(
                step_idx=int(step_idx),
                mapper=mapper,
                object_memory=object_memory,
                room_segmenter=room_segmenter,
                room_labeler=room_labeler,
                map_info=map_state["map_info"],
                previous_room_context=room_context_cache,
                strict_benchmark=bool(getattr(args, "strict_benchmark", False)),
                occupancy=map_state["occupancy"],
                observed_free_mask=map_state["free"],
                obstacle_mask=map_state["occupancy"],
                unknown_mask=~np.asarray(map_state["observed"], dtype=bool),
                allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            )
            last_room_context_result = result
            last_room_masks = list(result.room_masks)
            room_semantic_labels = dict(result.room_semantic_labels)
            last_room_segmentation_debug = {
                **dict(result.room_segmentation_debug),
                **runtime_navigation_debug_layers(map_state),
            }
            last_room_semantics_debug = dict(result.room_semantics_debug)
            last_room_context_metadata = result.metadata(full_order=False)
            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
            return result

        def maybe_update_room_context_for_frontier_scoring(
            *,
            token: RoomsegFrontierUpdateToken,
            step_idx: int,
            map_state: dict,
            call_site: str,
        ) -> RoomContextResult | None:
            if not bool(token.allowed):
                return None
            return update_room_context_for_frontier_scoring(
                step_idx,
                map_state,
                call_site=call_site,
            )

        def save_selected_roomseg_snapshot(
            step_idx: int,
            map_state: dict,
            frontier_layers: Mapping[str, np.ndarray],
            selected_frontier,
            *,
            snapshot_source: str = "actual_roomseg_update",
            roomseg_update_reason: str = "",
        ) -> None:
            nonlocal last_room_segmentation_debug, last_room_context_metadata
            nonlocal last_roomseg_snapshot_agent_grid, last_roomseg_snapshot_selected_key
            nonlocal last_roomseg_snapshot_geometry_hash, last_roomseg_snapshot_update_reason, last_roomseg_snapshot_step
            selected_members = getattr(selected_frontier, "members", None) if selected_frontier is not None else None
            selected_center = getattr(selected_frontier, "center_grid", None) if selected_frontier is not None else None
            agent_grid = tuple(int(v) for v in map_state.get("current_grid")) if map_state.get("current_grid") is not None else None
            selected_key = tuple(int(v) for v in selected_center) if selected_center is not None else None
            geometry_hash = roomseg_snapshot_geometry_hash(
                last_room_masks,
                last_room_segmentation_debug,
                fallback_hash=str(last_room_context_metadata.get("room_mask_geometry_hash") or ""),
            )
            moved_cells = None
            if agent_grid is not None and last_roomseg_snapshot_agent_grid is not None:
                moved_cells = int(
                    abs(int(agent_grid[0]) - int(last_roomseg_snapshot_agent_grid[0]))
                    + abs(int(agent_grid[1]) - int(last_roomseg_snapshot_agent_grid[1]))
                )
            selected_changed = (
                None
                if last_roomseg_snapshot_selected_key is None
                else bool(selected_key != last_roomseg_snapshot_selected_key)
            )
            last_room_segmentation_debug = {
                **dict(last_room_segmentation_debug),
                "snapshot_source": str(snapshot_source),
                "roomseg_update_reason": str(roomseg_update_reason),
                "roomseg_snapshot_geometry_hash": str(geometry_hash),
                "agent_moved_since_last_snapshot_cells": moved_cells,
                "selected_frontier_changed_from_last_snapshot": selected_changed,
            }
            live_baseline_obs = obs if isinstance(obs, dict) else None
            if (
                getattr(live_baseline_manager, "baseline_name", "") == "tvars_original_isaac"
                and not (
                    live_baseline_obs is not None
                    and live_baseline_obs.get("has_rgb")
                    and live_baseline_obs.get("rgb_device") == "cpu"
                )
            ):
                live_baseline_obs = dict(live_baseline_obs or {})
                live_baseline_obs["rgb"] = viz_rgb(live_baseline_obs)
                live_baseline_obs["has_rgb"] = True
                live_baseline_obs["rgb_device"] = "cpu"
            live_baseline_manager.on_step(
                step=int(step_idx),
                obs=live_baseline_obs,
                map_state=map_state,
                mapper=mapper,
                room_segmenter=room_segmenter,
                frontier_map=frontier_layers.get("frontier") if frontier_layers is not None else None,
                selected_frontier_center_rc=selected_center,
                camera_intrinsics=intr,
            )
            snapshot = save_roomseg_snapshot_for_frontier_scoring(
                step_idx,
                map_state,
                frontier_map=frontier_layers.get("frontier") if frontier_layers is not None else None,
                selected_frontier_members=selected_members,
                selected_frontier_center_rc=selected_center,
            )
            if snapshot is None:
                return
            last_roomseg_snapshot_agent_grid = agent_grid
            last_roomseg_snapshot_selected_key = selected_key
            last_roomseg_snapshot_geometry_hash = str(geometry_hash)
            last_roomseg_snapshot_update_reason = str(roomseg_update_reason)
            last_roomseg_snapshot_step = int(step_idx)
            snapshot_paths = dict(snapshot.get("paths", {}))
            if snapshot_paths.get("npz"):
                baseline_snapshot = live_baseline_manager.on_snapshot_saved(
                    step=int(step_idx),
                    source_snapshot_npz=Path(snapshot_paths["npz"]),
                    source_summary_json=Path(snapshot_paths["summary_json"]) if snapshot_paths.get("summary_json") else None,
                )
                if baseline_snapshot is not None:
                    baseline_key = str(getattr(live_baseline_manager, "baseline_name", "live_baseline") or "live_baseline")
                    snapshot_paths[f"{baseline_key}_npz"] = str(baseline_snapshot)
            last_room_segmentation_debug = {
                **dict(last_room_segmentation_debug),
                "roomseg_snapshot_paths": snapshot_paths,
                "roomseg_snapshot_summary": dict(snapshot.get("summary", {})),
            }
            last_room_context_metadata = {
                **dict(last_room_context_metadata),
                "roomseg_snapshot_paths": snapshot_paths,
            }
            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)

        def update_roomseg_debug_only(step_idx: int, map_state: dict) -> None:
            nonlocal last_room_masks, room_semantic_labels, last_room_segmentation_debug
            nonlocal last_room_semantics_debug, last_room_context_metadata, last_room_context_result
            if room_segmenter is None:
                last_room_masks = []
                room_semantic_labels = {}
                last_room_context_result = None
                last_room_context_metadata = {
                    **room_context_not_invoked_metadata(),
                    "roomseg_debug_only": True,
                    "room_context_source": "roomseg_debug_only",
                    "room_update_invoked_for_frontier_scoring": False,
                    "room_segmentation_ran": False,
                    "room_labeling_ran": False,
                    "room_segmentation_called_for": "debug_only",
                    "room_vlm_called": False,
                    "scenegraph_updated_after_room_context": False,
                    "frontier_scoring_after_room_context": False,
                    "room_mask_count": 0,
                }
                last_room_segmentation_debug = {
                    "roomseg_debug_only": True,
                    "enabled": False,
                    "source": str(room_map_mode),
                    "algorithm": str(getattr(args, "room_segmentation_config", {}).get("algorithm", room_map_mode)),
                    "room_count": 0,
                    "rooms": [],
                    "reason": "room_segmenter_not_configured",
                }
                if viz is not None:
                    viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
                return

            started_at = time.perf_counter()
            update_kwargs = {
                "step": int(step_idx),
                "object_memory": [],
            }
            vertical_profile = getattr(mapper, "vertical_profile", None)
            if vertical_profile is not None:
                update_kwargs["vertical_profile"] = vertical_profile
            height_profile = getattr(mapper, "height_profile", None)
            if height_profile is not None:
                update_kwargs["height_profile"] = height_profile
            voxel_grid = getattr(mapper, "voxel_grid", None)
            if voxel_grid is not None:
                update_kwargs["voxel_grid"] = voxel_grid
            roomseg_static_structural = getattr(mapper, "roomseg_static_structural_occupied", None)
            if roomseg_static_structural is not None:
                update_kwargs["roomseg_static_structural_occupied"] = roomseg_static_structural
            roomseg_ray_evidence = getattr(mapper, "roomseg_ray_evidence", None)
            if callable(roomseg_ray_evidence):
                update_kwargs["roomseg_ray_evidence"] = roomseg_ray_evidence()
            nav_free_mask, nav_obstacle_mask, nav_unknown_mask, nav_source = resolve_replay_style_navigation_masks(
                mapper=mapper,
                shape=np.asarray(map_state["occupancy"]).shape,
                fallback_free=map_state["free"],
                fallback_obstacle=map_state["occupancy"],
                fallback_unknown=~np.asarray(map_state["observed"], dtype=bool),
            )
            try:
                update_params = inspect.signature(room_segmenter.update).parameters
            except (TypeError, ValueError):
                update_params = {}
            add_replay_style_navigation_update_kwargs(
                update_kwargs,
                update_params,
                navigation_free_mask=nav_free_mask,
                navigation_obstacle_mask=nav_obstacle_mask,
                navigation_unknown_mask=nav_unknown_mask,
            )
            try:
                masks = room_segmenter.update(
                    nav_obstacle_mask,
                    nav_free_mask,
                    nav_obstacle_mask,
                    nav_unknown_mask,
                    **update_kwargs,
                )
            except TypeError:
                if str(getattr(room_segmenter, "context_source", "")).startswith("voxel_occupancy"):
                    raise
                update_kwargs.pop("vertical_profile", None)
                update_kwargs.pop("height_profile", None)
                update_kwargs.pop("voxel_grid", None)
                update_kwargs.pop("roomseg_static_structural_occupied", None)
                update_kwargs.pop("roomseg_ray_evidence", None)
                update_kwargs.pop("navigation_free_mask", None)
                update_kwargs.pop("navigation_obstacle_mask", None)
                update_kwargs.pop("door_seed_no_clearance_free_mask", None)
                masks = room_segmenter.update(
                    map_state["occupancy"],
                    map_state["free"],
                    map_state["occupancy"],
                    ~np.asarray(map_state["observed"], dtype=bool),
                    **update_kwargs,
                )
            last_room_masks = list(masks)
            room_semantic_labels = {}
            debug = dict(getattr(room_segmenter, "last_debug", {}) or {})
            if not debug:
                debug = room_segmentation_debug(
                    last_room_masks,
                    map_state["free"],
                    map_state["occupancy"],
                    ~np.asarray(map_state["observed"], dtype=bool),
                    step=int(step_idx),
                    config=None,
                    source=str(room_map_mode),
                )
            debug = {
                **debug,
                "roomseg_debug_only": True,
                "roomseg_debug_view": "vertical_free_room_domain",
                "source": str(debug.get("source") or room_map_mode),
                "algorithm": str(debug.get("algorithm") or getattr(args, "room_segmentation_config", {}).get("algorithm", room_map_mode)),
                "room_count": int(len(last_room_masks)),
                "room_vlm_called": False,
                "scenegraph_updated_after_room_context": False,
                "frontier_scoring_after_room_context": False,
                "room_segmentation_called_for": "debug_only",
                "room_segmentation_step_index": int(step_idx),
                "room_segmentation_runtime_ms": max(0.0, (time.perf_counter() - started_at) * 1000.0),
                "roomseg_online_navigation_mask_source": str(nav_source),
            }
            last_room_segmentation_debug = debug
            last_room_semantics_debug = {
                "backend": "disabled_roomseg_debug_only",
                "labels": [],
                "request_count": 0,
                "failure_count": 0,
            }
            last_room_context_result = None
            last_room_context_metadata = {
                **room_context_not_invoked_metadata(),
                "roomseg_debug_only": True,
                "room_context_source": "roomseg_debug_only",
                "room_update_invoked_for_frontier_scoring": False,
                "room_segmentation_ran": True,
                "room_labeling_ran": False,
                "room_context_cache_hit": False,
                "room_label_count": 0,
                "room_label_requests": 0,
                "room_label_cache_hits": 0,
                "room_call_order_trace": ["mapping", "room_segmentation_debug_only", "visualization"],
                "room_segmentation_called_for": "debug_only",
                "room_segmentation_algorithm": str(last_room_segmentation_debug.get("algorithm", "")),
                "room_segmentation_step_index": int(step_idx),
                "room_vlm_called": False,
                "scenegraph_updated_after_room_context": False,
                "frontier_scoring_after_room_context": False,
                "room_mask_count": int(len(last_room_masks)),
            }
            if bool(getattr(args, "debug_roomseg_layers", False)):
                dump = save_roomseg_layer_dump(
                    out_dir=str(getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")),
                    step=int(step_idx),
                    room_debug=_roomseg_debug_for_layer_dump(last_room_segmentation_debug, room_segmenter),
                    occupancy_map=nav_obstacle_mask,
                    observed_free_mask=nav_free_mask,
                    obstacle_mask=nav_obstacle_mask,
                    unknown_mask=nav_unknown_mask,
                    frontier_map=np.zeros_like(map_state["occupancy"], dtype=bool),
                    selected_frontier_members=None,
                    selected_frontier_center_rc=None,
                    agent_rc=map_state["current_grid"],
                    max_saves=int(getattr(args, "debug_roomseg_max_saves", 50)),
                    save_npz=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_npz", True)),
                    save_png=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_png", True)),
                    save_summary_json=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_summary_json", True)),
                    include_selected_frontier_sector=False,
                )
                last_room_segmentation_debug = {
                    **dict(last_room_segmentation_debug),
                    "roomseg_debug_layers": dict(dump.get("paths", {})),
                    "roomseg_debug_summary": dict(dump.get("summary", {})),
                }
                last_room_context_metadata = {
                    **dict(last_room_context_metadata),
                    "roomseg_debug_layers": dict(dump.get("paths", {})),
                    "roomseg_debug_likely_cause": dict(dump.get("summary", {})).get("likely_cause"),
                }
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)

        panorama_steps = max(0, int(getattr(args, "panorama_steps", 0)))
        if panorama_steps > 0:
            print("[voxroom-loop] opening panorama: %d RGB-D views" % panorama_steps, flush=True)
        for pano_idx in range(panorama_steps):
            map_state = update_mapper_state(obs, -panorama_steps + pano_idx)
            if map_state is None or map_state["current_grid"] is None:
                failure_reason = "panorama_agent_off_navigable_map"
                break
            pano_step = -panorama_steps + pano_idx
            if roomseg_debug_only:
                update_roomseg_debug_only(pano_step, map_state)
                panorama_frames += 1
                if viz is not None:
                    viz.update(
                        step=pano_step,
                        rgb=viz_rgb(obs),
                        detections_2d=[],
                        occupancy=map_state["occupancy"],
                        navigable=map_state["free"],
                        observed=map_state["observed"],
                        goal_cells=goal_cells,
                        current_grid=map_state["current_grid"],
                        pose=map_state["pose"],
                        frontiers=[],
                        nav_decision=None,
                        current_path=[],
                        full_path=[],
                        object_memory=object_memory,
                        goal_category=episode["goal_category"],
                        distance_to_goal=evaluator.final_distance_to_goal,
                        path_length=float(evaluator.path_accum.total_m),
                        scenegraph_backend="roomseg_debug_only",
                        score_debug={},
                        failure_reason=failure_reason,
                    )
                if pano_idx + 1 < panorama_steps:
                    yaw_delta = (2.0 * math.pi) / float(panorama_steps)
                    pano_wz = max(1e-3, abs(float(args.panorama_wz_radps)))
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        pano_wz,
                        dt=yaw_delta / pano_wz,
                        render_updates=int(args.panorama_render_updates_per_step),
                        read_rgb=viz_requested,
                        read_depth=True,
                        rgb_device="cpu",
                    )
                continue
            obs = run_detector_update(obs, -panorama_steps + pano_idx, map_state["map_info"], map_state["occupancy"], map_state["navigable"])
            update_scenegraph_frame(obs, -panorama_steps + pano_idx, map_state)
            panorama_frames += 1
            has_goal_detection = any(row["step"] == int(pano_step) for row in goal_detection_history)
            if viz is not None and (has_goal_detection or pano_idx + 1 == panorama_steps):
                viz.update(
                    step=pano_step,
                    rgb=viz_rgb(obs),
                    detections_2d=last_detections_2d,
                    occupancy=map_state["occupancy"],
                    navigable=map_state["free"],
                    observed=map_state["observed"],
                    goal_cells=goal_cells,
                    current_grid=map_state["current_grid"],
                    pose=map_state["pose"],
                    frontiers=last_frontiers,
                    nav_decision=last_nav_decision,
                    current_path=current_path,
                    full_path=full_path,
                    object_memory=object_memory,
                    goal_category=episode["goal_category"],
                    distance_to_goal=evaluator.final_distance_to_goal,
                    path_length=float(evaluator.path_accum.total_m),
                    scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                    score_debug=scenegraph.last_score_debug,
                    failure_reason=failure_reason,
                )
            if pano_idx + 1 < panorama_steps:
                yaw_delta = (2.0 * math.pi) / float(panorama_steps)
                pano_wz = max(1e-3, abs(float(args.panorama_wz_radps)))
                obs = server.step_kinematic_velocity(
                    0.0,
                    0.0,
                    pano_wz,
                    dt=yaw_delta / pano_wz,
                    render_updates=int(args.panorama_render_updates_per_step),
                    read_rgb=detector_requires_rgb(detector, args.detector),
                    read_depth=True,
                    rgb_device="cuda" if detector_cuda_rgb else "cpu",
                )
        for step in range(0 if failure_reason is not None else max_steps):
            loop_tick(step, "step_start", reset=True)
            map_state = update_mapper_state(obs, step)
            loop_tick(step, "after_update_mapper_state")
            if map_state is None:
                failure_reason = "depth_unavailable_for_online_mapping"
                break
            pose = map_state["pose"]
            dynamic_map_info = map_state["map_info"]
            occupancy = map_state["occupancy"]
            free = map_state["free"]
            observed = map_state["observed"]
            navigable = map_state["navigable"]
            nav_planner = map_state["nav_planner"]
            base_nav_planner = map_state.get("base_nav_planner", nav_planner)
            current_grid = map_state["current_grid"]
            if current_grid is None:
                failure_reason = "agent_off_navigable_map"
                break
            include_runtime_debug_arrays = bool(getattr(args, "runtime_debug_full_arrays", False))
            include_runtime_debug_arrays = include_runtime_debug_arrays or bool(
                viz is not None
                and step % viz_every == 0
                and bool(getattr(args, "runtime_debug_full_arrays_on_viz", True))
            )
            last_room_segmentation_debug = {
                **dict(last_room_segmentation_debug),
                **runtime_navigation_debug_layers(map_state, include_arrays=include_runtime_debug_arrays),
            }
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
            metric_pose_valid, metric_grid = update_metric_evaluator_pose_or_mark_invalid(
                evaluator,
                metric_planner,
                pose,
                static_map_info,
                collided=bool(obs.get("collided", False)),
            )
            if not metric_pose_valid:
                metric_pose_invalid_steps += 1
                metric_pose_last_invalid_step = int(step)
                metric_pose_last_invalid_reason = "agent_off_static_metric_map"
                if not logged_metric_pose_invalid:
                    print(
                        "[voxroom-loop] agent outside static metric map; metric row will be marked invalid, "
                        "continuing online voxel navigation",
                        flush=True,
                    )
                    logged_metric_pose_invalid = True
            policy_distance_to_goal = evaluator.final_distance_to_goal if metric_pose_valid else float("inf")
            if roomseg_debug_only:
                update_roomseg_debug_only(step, map_state)
                if viz is not None and step % viz_every == 0:
                    viz.update(
                        step=step,
                        rgb=viz_rgb(obs),
                        detections_2d=[],
                        occupancy=occupancy,
                        navigable=free,
                        observed=observed,
                        goal_cells=goal_cells,
                        current_grid=current_grid,
                        pose=pose,
                        frontiers=[],
                        nav_decision=None,
                        current_path=[],
                        full_path=[],
                        object_memory=object_memory,
                        goal_category=episode["goal_category"],
                        distance_to_goal=evaluator.final_distance_to_goal,
                        path_length=float(evaluator.path_accum.total_m),
                        scenegraph_backend="roomseg_debug_only",
                        score_debug={},
                        failure_reason=failure_reason,
                    )
                if step + 1 < max_steps:
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        float(args.panorama_wz_radps),
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=viz_requested and (step + 1) % viz_every == 0,
                        read_depth=True,
                        rgb_device="cpu",
                    )
                    continue
                break
            if policy_distance_to_goal <= success_distance:
                gt_success_region_reached = True
                if explore_until_no_frontiers:
                    gt_success_ignored_steps += 1
            if success_region_can_finish(
                policy_distance_to_goal,
                success_distance,
                require_sgnav_stop=bool(args.require_sgnav_stop),
                policy_stop_confirmed=False,
                ignore_goal_success=explore_until_no_frontiers,
            ):
                stop_called = True
                break
            if policy_distance_to_goal <= success_distance and bool(args.require_sgnav_stop):
                gt_success_without_sgnav_stop_steps += 1
                stop_blocked_reason = "sgnav_stop_required"
                if not logged_gt_success_without_sgnav_stop:
                    print(
                        "[voxroom-loop] inside GT success radius, but strict VoxRoom STOP is not confirmed; continuing perception/replanning",
                        flush=True,
                    )
                    logged_gt_success_without_sgnav_stop = True

            perception_due = detector is not None and (step % perception_every == 0 or force_perception_step)
            if perception_due:
                obs = run_detector_update(obs, step, dynamic_map_info, occupancy, navigable)
                force_perception_step = False
            if step % perception_every == 0 or perception_due:
                update_scenegraph_frame(obs, step, map_state)
                loop_tick(step, "after_update_scenegraph_frame")

            path_replan_reason: str | None = None
            path_replan_executed = False
            path_replan_reused_committed_target = False
            path_replan_target_cells_count = 0
            path_replan_current_path_cleared_as_stale = False
            current_path_head_before_trim = tuple(int(v) for v in current_path[0]) if current_path else None
            current_path_head_after_trim = current_path_head_before_trim
            needs_replan = not current_path or step % replan_every == 0
            if not current_path:
                path_replan_reason = "no_current_path"
            elif step % replan_every == 0:
                path_replan_reason = "periodic_replan"
            trim_result = PathTrimResult(
                list(current_path),
                bool(current_path),
                0 if current_path else None,
                0.0 if current_path else None,
                "not_checked" if current_path else "empty_path",
            )
            if current_path:
                trim_result = trim_path_to_nearest_with_info(
                    current_path,
                    current_grid,
                    max_dist_cells=int(getattr(args, "path_trim_max_distance_cells", 3)),
                )
                if trim_result.success:
                    current_path = list(trim_result.suffix)
                    current_path_head_after_trim = tuple(int(v) for v in current_path[0]) if current_path else None
                else:
                    current_path = []
                    needs_replan = True
                    path_replan_reason = "path_trim_failed_%s" % str(trim_result.reason)
                    path_replan_current_path_cleared_as_stale = True
                    current_path_head_after_trim = None
            path_replan_debug = path_trim_debug_metadata(
                trim=trim_result,
                before_head=current_path_head_before_trim,
                after_head=current_path_head_after_trim,
                replan_requested=bool(needs_replan),
                replan_reason=path_replan_reason,
                cleared_stale_path=path_replan_current_path_cleared_as_stale,
            )
            last_room_segmentation_debug = {
                **dict(last_room_segmentation_debug),
                **path_replan_debug,
            }
            committed_frontier_active = (
                bool(getattr(args, "frontier_execution_state_enabled", True))
                and committed_frontier_execution.exists()
                and not bool(committed_frontier_execution.arrival_pending_update)
            )
            if committed_frontier_active:
                arrival_confirmed, arrival_distance_m, arrival_reason = frontier_arrival_status(
                    step=int(step),
                    current_grid=tuple(int(v) for v in current_grid),
                    execution=committed_frontier_execution,
                    resolution_m=float(dynamic_map_info.resolution_m),
                    reached_radius_m=float(args.frontier_commit_reached_radius_m),
                    min_steps_since_selection=int(getattr(args, "frontier_arrival_min_steps_since_selection", 6)),
                    confirm_steps=int(getattr(args, "frontier_arrival_confirm_steps", 2)),
                    current_path=current_path,
                )
                last_frontier_commitment_metadata = {
                    **dict(last_frontier_commitment_metadata),
                    **committed_frontier_execution.debug_metadata(
                        step=int(step),
                        current_grid=tuple(int(v) for v in current_grid),
                        resolution_m=float(dynamic_map_info.resolution_m),
                    ),
                    "frontier_arrival_status_reason": str(arrival_reason),
                }
                if arrival_confirmed:
                    reason = "frontier_arrival_confirmed"
                    target_radius_reached_steps += 1
                    arrival_key = mark_frontier_terminal_for_refresh(
                        nav_decision_local=last_nav_decision,
                        current_grid_local=tuple(int(v) for v in current_grid),
                        reason="frontier_arrival",
                        status="reached",
                    )
                    refresh_key = _frontier_refresh_key_or_fallback(arrival_key, last_nav_decision)
                    committed_frontier_execution.mark_arrival_pending(int(step), refresh_key)
                    refresh_requested = request_frontier_refresh(
                        refresh_state=frontier_arrival_update_state,
                        step=int(step),
                        reason="frontier_arrival",
                        frontier_key=refresh_key,
                        block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
                    )
                    write_runtime_decision_trace(
                        step_idx=int(step),
                        event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
                        event_reason="frontier_arrival",
                        current_grid_local=tuple(int(v) for v in current_grid),
                        nav_decision_local=last_nav_decision,
                        recovery_action="arrival_pending_update" if refresh_requested else "duplicate_refresh_skipped",
                        current_path_len=0,
                        zero_velocity_continue=True,
                        extra=terminal_trace_metadata(selected_key=arrival_key),
                    )
                    try:
                        mapper.force_full_navigation_projection_once = True
                        mapper.voxel_grid.invalidate_navigation_projection_cache(reason="frontier_arrival")
                    except Exception:
                        pass
                    if frontier_commitment is not None and last_decision_mode == "frontier":
                        if bool(getattr(args, "frontier_arrival_blacklist_on_reached", False)):
                            frontier_commitment.mark_active_failed(step, reason, blacklist=True)
                        else:
                            frontier_commitment.mark_active_reached(step, reason)
                    current_path = []
                    force_perception_step = True
                    full_path.append(tuple(int(v) for v in current_grid))
                    last_decision_reason = reason
                    if last_nav_decision is not None:
                        last_nav_decision.reason = reason
                        last_nav_decision.metadata = {
                            **dict(last_nav_decision.metadata or {}),
                            "target_radius_reached": True,
                            "frontier_arrival_confirmed": True,
                            "target_reached_distance_m": float(arrival_distance_m),
                            "target_reached_radius_m": float(args.frontier_commit_reached_radius_m),
                            **committed_frontier_execution.debug_metadata(
                                step=int(step),
                                current_grid=tuple(int(v) for v in current_grid),
                                resolution_m=float(dynamic_map_info.resolution_m),
                            ),
                        }
                    last_frontier_commitment_reason = reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": reason,
                        "frontier_refresh_pending": True,
                        "frontier_refresh_reason": "frontier_arrival",
                        "target_radius_reached": True,
                        "frontier_arrival_confirmed": True,
                        "target_reached_distance_m": float(arrival_distance_m),
                        "target_reached_radius_m": float(args.frontier_commit_reached_radius_m),
                    }
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
            update_roomseg_frontiers, roomseg_frontier_update_reason = should_update_roomseg_frontiers(
                step=int(step),
                has_current_path=bool(current_path or committed_frontier_active),
                gate_state=roomseg_frontier_update_gate,
                arrival_state=frontier_arrival_update_state,
                policy=str(getattr(args, "roomseg_frontier_update_policy", "at_frontier_arrival")),
                freeze_during_navigation=bool(getattr(args, "freeze_roomseg_and_frontiers_during_navigation", True)),
                target_invalidated=bool(needs_replan and not current_path and roomseg_frontier_update_gate.initialized and not committed_frontier_active),
                no_active_path=bool(not current_path and not committed_frontier_active),
                debug_force=bool(getattr(args, "force_roomseg_frontier_update", False)),
                update_on_target_invalidated=bool(getattr(args, "roomseg_frontier_update_on_target_invalidated", False)),
                update_on_no_active_path=bool(getattr(args, "roomseg_frontier_update_on_no_active_path", False)),
                update_on_no_progress=bool(getattr(args, "roomseg_frontier_update_on_no_progress", False)),
            )
            roomseg_update_token = RoomsegFrontierUpdateToken(
                allowed=bool(update_roomseg_frontiers),
                reason=str(roomseg_frontier_update_reason),
                step=int(step),
            )
            if (
                bool(frontier_arrival_update_state.refresh_pending)
                and bool(frontier_arrival_update_state.block_reselect_until_refresh)
                and not bool(update_roomseg_frontiers)
            ):
                write_runtime_decision_trace(
                    step_idx=int(step),
                    event_type="frontier_reselect_blocked_until_refresh",
                    event_reason=str(roomseg_frontier_update_reason),
                    current_grid_local=tuple(int(v) for v in current_grid) if current_grid is not None else None,
                    nav_decision_local=last_nav_decision,
                    roomseg_update_allowed=bool(update_roomseg_frontiers),
                    roomseg_update_reason=str(roomseg_frontier_update_reason),
                    current_path_len=len(current_path),
                )
                raise RuntimeError("frontier_refresh_pending_but_gate_denied")
            if needs_replan and not update_roomseg_frontiers:
                mark_roomseg_frontier_gate_skip(roomseg_frontier_update_gate, reason=roomseg_frontier_update_reason)
                last_room_segmentation_debug = {
                    **dict(last_room_segmentation_debug),
                    "roomseg_frontier_update_due": False,
                    "roomseg_frontier_update_reason": roomseg_frontier_update_reason,
                    "roomseg_frontier_update_policy": str(getattr(args, "roomseg_frontier_update_policy", "at_frontier_arrival")),
                    "roomseg_frontier_last_update_step": int(roomseg_frontier_update_gate.last_update_step),
                    "roomseg_frontier_last_update_reason": roomseg_frontier_update_gate.last_update_reason,
                    "frontiers_frozen_during_navigation": True,
                    "frontier_arrival_pending": bool(frontier_arrival_update_state.arrival_pending),
                    "frontier_arrival_last_update_step": int(frontier_arrival_update_state.last_update_step),
                    "frontier_arrival_cooldown_until_step": int(frontier_arrival_update_state.cooldown_until_step),
                    "path_replan_suppressed_by_roomseg_freeze": False,
                    "path_replan_allowed_during_roomseg_freeze": True,
                    "path_replan_only_active_target": bool(
                        getattr(args, "allow_path_replan_during_roomseg_freeze", True)
                        and last_nav_decision is not None
                        and bool(getattr(last_nav_decision, "target_cells", None))
                    ),
                    "target_reselection_suppressed_by_roomseg_freeze": bool(
                        last_nav_decision is not None and bool(getattr(last_nav_decision, "target_cells", None))
                    ),
                }

            if needs_replan:
                consume_frontier_arrival_after_roomseg = bool(
                    update_roomseg_frontiers
                    and (
                        bool(frontier_arrival_update_state.refresh_pending)
                        or roomseg_frontier_update_reason == "frontier_arrival"
                    )
                )
                if update_roomseg_frontiers:
                    mark_roomseg_frontier_gate_update(roomseg_frontier_update_gate, step=int(step), reason=roomseg_frontier_update_reason)
                else:
                    mark_roomseg_frontier_gate_skip(roomseg_frontier_update_gate, reason=roomseg_frontier_update_reason)
                    frontier_arrival_update_state.last_skip_reason = str(roomseg_frontier_update_reason)
                planning_started_at = time.perf_counter()
                map_state = ensure_runtime_planners(map_state)
                nav_planner = map_state["nav_planner"]
                base_nav_planner = map_state.get("base_nav_planner", nav_planner)
                current_grid = map_state["current_grid"]
                if current_grid is None:
                    failure_reason = "agent_off_navigable_map"
                    break
                if initial_frontier_loop_grid is None:
                    initial_frontier_loop_grid = tuple(int(v) for v in current_grid)
                write_runtime_decision_trace(
                    step_idx=int(step),
                    event_type="initial_replan" if not bool(roomseg_frontier_update_gate.initialized) else "replan_start",
                    event_reason=str(roomseg_frontier_update_reason),
                    current_grid_local=tuple(int(v) for v in current_grid),
                    roomseg_update_allowed=bool(update_roomseg_frontiers),
                    roomseg_update_reason=str(roomseg_frontier_update_reason),
                    roomseg_context_updated_this_step=False,
                    current_path_len=len(current_path),
                )
                llm_requests_before = total_llm_requests()
                astar_traversible = np.asarray(map_state.get("astar_navigable", navigable), dtype=bool)
                frontier_traversible = navigable.astype(bool)
                room_context_result = None
                room_context_updated_this_step = False
                room_context_source = "none"
                snapshot_saved_this_step = False
                refresh_consumed_this_step = False
                roomseg_context_needed = (
                    update_roomseg_frontiers
                    and room_segmenter is not None
                    and (
                        str(getattr(args, "frontier_source", "navigation")).strip().lower()
                        in {"vertical_free", "height_profile_vertical_free", "voxel_vertical_free"}
                        or bool(getattr(args, "save_roomseg_snapshots", False))
                        or bool(getattr(args, "debug_roomseg_layers", False))
                        or viz is not None
                    )
                    and (
                        not paper_mode
                        or str(getattr(args, "frontier_selection_mode", "sgnav")).strip().lower() in {"nearest", "random"}
                    )
                )
                if roomseg_context_needed:
                    loop_tick(step, "before_room_context")
                    room_context_result = maybe_update_room_context_for_frontier_scoring(
                        token=roomseg_update_token,
                        step_idx=step,
                        map_state=map_state,
                        call_site="frontier_replan_room_context",
                    )
                    if room_context_result is not None:
                        room_context_updated_this_step = True
                        room_context_source = "actual_roomseg_update"
                        roomseg_updates_since_start += 1
                    loop_tick(step, "after_room_context")
                if consume_frontier_arrival_after_roomseg:
                    consumed_reason = str(roomseg_frontier_update_reason)
                    consume_frontier_refresh(
                        refresh_state=frontier_arrival_update_state,
                        step=int(step),
                        reason=consumed_reason,
                        cooldown_steps=int(getattr(args, "roomseg_frontier_arrival_update_cooldown_steps", 3)),
                    )
                    committed_frontier_execution.consume_arrival_update(int(step))
                    frontier_terminal_registry.mark_refresh_consumed(
                        frontier_arrival_update_state.last_consumed_arrival_key,
                        int(step),
                    )
                    refresh_consumed_this_step = True
                    write_runtime_decision_trace(
                        step_idx=int(step),
                        event_type="frontier_refresh_consumed",
                        event_reason=consumed_reason,
                        current_grid_local=tuple(int(v) for v in current_grid),
                        roomseg_update_allowed=bool(update_roomseg_frontiers),
                        roomseg_update_reason=str(roomseg_frontier_update_reason),
                        roomseg_context_updated_this_step=bool(room_context_updated_this_step),
                        current_path_len=len(current_path),
                    )
                    if paper_mode:
                        long_term_goal.clear("%s_consumed" % consumed_reason)
                frontier_free, frontier_observed, frontier_occupancy, frontier_source_metadata = resolve_frontier_source_layers(
                    room_debug=last_room_segmentation_debug,
                    mapper=mapper,
                    navigation_free=free,
                    navigation_observed=observed,
                    navigation_occupancy=occupancy,
                    frontier_traversible=frontier_traversible,
                    source=str(getattr(args, "frontier_source", "navigation")),
                    require_navigation_reachable=bool(getattr(args, "frontier_vertical_free_require_navigation_reachable", True)),
                )
                loop_tick(step, "after_resolve_frontier_source")
                last_room_segmentation_debug = {
                    **dict(last_room_segmentation_debug),
                    **dict(frontier_source_metadata),
                    "roomseg_frontier_update_due": bool(update_roomseg_frontiers),
                    "roomseg_frontier_update_reason": roomseg_frontier_update_reason,
                    "roomseg_frontier_update_policy": str(getattr(args, "roomseg_frontier_update_policy", "at_frontier_arrival")),
                    "roomseg_frontier_last_update_step": int(roomseg_frontier_update_gate.last_update_step),
                    "roomseg_frontier_last_update_reason": roomseg_frontier_update_gate.last_update_reason,
                    "frontiers_frozen_during_navigation": not bool(update_roomseg_frontiers),
                    "frontier_arrival_pending": bool(frontier_arrival_update_state.arrival_pending),
                    "frontier_arrival_last_update_step": int(frontier_arrival_update_state.last_update_step),
                    "frontier_arrival_cooldown_until_step": int(frontier_arrival_update_state.cooldown_until_step),
                }
                distance_traversible = frontier_traversible.copy()
                rr, cc = int(current_grid[0]), int(current_grid[1])
                if 0 <= rr < distance_traversible.shape[0] and 0 <= cc < distance_traversible.shape[1]:
                    distance_traversible[rr, cc] = True
                static_only_nearfield = getattr(mapper, "static_nearfield_mask", None)
                depth_free_mask = getattr(mapper, "depth_free_mask", None)
                if bool(getattr(args, "static_nearfield_map", False)) and static_only_nearfield is not None and depth_free_mask is not None:
                    static_only_nearfield = np.asarray(static_only_nearfield).astype(bool) & ~np.asarray(depth_free_mask).astype(bool)
                else:
                    static_only_nearfield = None
                frontier_layers = frontier_debug_layers(
                    frontier_free,
                    observed=frontier_observed,
                    occupancy=frontier_occupancy,
                    obstacle_dilation_radius_cells=int(args.frontier_obstacle_dilation_radius_cells),
                    unknown_dilation_radius_cells=int(args.frontier_unknown_dilation_radius_cells),
                    exclude_mask=static_only_nearfield,
                    unknown_source="observed" if str(frontier_source_metadata.get("frontier_source")) in {"vertical_free", "voxel_vertical_free"} else str(args.frontier_unknown_source),
                )
                if bool(getattr(args, "frontier_debug_dump", False)):
                    assert frontier_free.shape == frontier_observed.shape == frontier_occupancy.shape == distance_traversible.shape
                    assert int(np.count_nonzero(frontier_layers["frontier"] & ~frontier_free)) == 0
                frontiers = list(last_frontiers)
                locked_goal_used = False
                candidate_override = None
                active_path_replan_decision = None
                path_replan_only_active_target = False
                target_reselection_suppressed_by_roomseg_freeze = False
                committed_goal_locked = (
                    bool(getattr(args, "frontier_execution_state_enabled", True))
                    and committed_frontier_execution.exists()
                    and not bool(committed_frontier_execution.arrival_pending_update)
                )
                if not bool(committed_goal_locked):
                    active_path_replan_decision = active_navigation_decision_for_path_replan(
                        last_nav_decision,
                        update_roomseg_frontiers=bool(update_roomseg_frontiers),
                        allow_path_replan_during_roomseg_freeze=bool(
                            getattr(args, "allow_path_replan_during_roomseg_freeze", True)
                        ),
                    )
                if active_path_replan_decision is not None:
                    path_replan_only_active_target = True
                    target_reselection_suppressed_by_roomseg_freeze = True
                if committed_goal_locked:
                    nav_decision = committed_frontier_execution.to_navigation_decision(last_nav_decision)
                    locked_goal_used = True
                    path_replan_reused_committed_target = True
                    last_frontier_commitment_reason = "continue_committed_frontier_execution"
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        **committed_frontier_execution.debug_metadata(
                            step=int(step),
                            current_grid=tuple(int(v) for v in current_grid),
                            resolution_m=float(dynamic_map_info.resolution_m),
                        ),
                        "frontier_commitment_reason": "continue_committed_frontier_execution",
                        "frontier_execution_locked": True,
                    }
                elif paper_mode and long_term_goal.exists() and long_term_goal.mode == "frontier":
                    candidate_override = decision_policy.choose_navigation_target(
                        object_memory,
                        episode["goal_category"],
                        current_grid,
                        [],
                        nav_planner,
                        dynamic_map_info,
                        pose,
                        allow_frontier=False,
                        current_step=step,
                    )
                    if candidate_override.mode not in {"candidate", "reperception", "stop"}:
                        candidate_override = None
                if committed_goal_locked:
                    pass
                elif active_path_replan_decision is not None:
                    nav_decision = active_path_replan_decision
                    locked_goal_used = True
                    path_replan_reused_committed_target = True
                    nav_decision.metadata = {
                        **dict(getattr(nav_decision, "metadata", {}) or {}),
                        "path_replan_only_active_target": True,
                        "target_reselection_suppressed_by_roomseg_freeze": True,
                        "roomseg_frontier_update_due": bool(update_roomseg_frontiers),
                        "roomseg_frontier_update_reason": str(roomseg_frontier_update_reason),
                    }
                    write_runtime_decision_trace(
                        step_idx=int(step),
                        event_type="active_target_replan_during_roomseg_freeze",
                        event_reason=str(path_replan_reason or "active_target_replan"),
                        current_grid_local=tuple(int(v) for v in current_grid),
                        nav_decision_local=nav_decision,
                        roomseg_update_allowed=bool(update_roomseg_frontiers),
                        roomseg_update_reason=str(roomseg_frontier_update_reason),
                        roomseg_context_updated_this_step=bool(room_context_updated_this_step),
                        current_path_len=len(current_path),
                        extra=active_target_replan_trace_extra(path_replan_debug),
                    )
                elif paper_mode and long_term_goal.exists() and candidate_override is None:
                    nav_decision = long_term_goal.to_navigation_decision()
                    locked_goal_used = True
                    if nav_decision.mode == "frontier":
                        last_frontier_commitment_reason = "continue_committed_frontier"
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": "continue_committed_frontier",
                            "long_term_goal_locked": True,
                            "long_term_goal_selected_step": int(long_term_goal.selected_step),
                        }
                else:
                    if candidate_override is not None:
                        long_term_goal.clear("candidate_override")
                    last_frontier_raw_cells = int(np.count_nonzero(frontier_layers["frontier"]))
                    frontier_partition = extract_frontiers_with_partition(
                        free=frontier_free,
                        observed=frontier_observed,
                        traversible=distance_traversible,
                        map_info=dynamic_map_info,
                        agent_grid=current_grid,
                        min_cluster_size=int(args.frontier_min_cluster_size),
                        min_distance_m=float(args.frontier_min_distance_m),
                        max_count=int(args.frontier_max_count),
                        occupancy=frontier_occupancy,
                        obstacle_dilation_radius_cells=int(args.frontier_obstacle_dilation_radius_cells),
                        unknown_dilation_radius_cells=int(args.frontier_unknown_dilation_radius_cells),
                        exclude_mask=static_only_nearfield,
                        unknown_source="observed" if str(frontier_source_metadata.get("frontier_source")) in {"vertical_free", "voxel_vertical_free"} else str(args.frontier_unknown_source),
                        cluster_distance_mode=str(args.frontier_cluster_distance_mode),
                        allow_near_frontier_fallback=bool(args.frontier_allow_near_fallback),
                        require_reachable=frontier_selection_require_reachable(frontier_mask_probe=bool(frontier_mask_probe)),
                    )
                    real_frontiers = list(frontier_partition.real_frontiers)
                    near_fallback_frontiers = list(frontier_partition.near_fallback_frontiers)
                    raw_frontiers_for_filter = list(frontier_partition.filtered_frontiers)
                    terminal_target_cache: dict[tuple[int, int], tuple[int, int] | None] = {}

                    def resolve_terminal_target_for_frontier(frontier: object) -> tuple[int, int] | None:
                        center = _as_grid_cell(getattr(frontier, "center_grid", None))
                        if center is None:
                            return None
                        if center in terminal_target_cache:
                            return terminal_target_cache[center]
                        target: tuple[int, int] | None = None
                        try:
                            astar_clearance_for_target = map_state.get("astar_clearance_m")
                            if not isinstance(astar_clearance_for_target, np.ndarray):
                                astar_clearance_for_target = clearance_map_m(astar_traversible, dynamic_map_info.resolution_m)
                            resolution = resolve_frontier_target(
                                frontier,
                                tuple(int(v) for v in current_grid),
                                nav_planner,
                                astar_traversible,
                                np.asarray(astar_clearance_for_target, dtype=np.float32),
                                float(dynamic_map_info.resolution_m),
                                min_clearance_m=float(
                                    getattr(args, "frontier_target_min_goal_clearance_m", getattr(args, "astar_goal_min_clearance_m", 0.18))
                                ),
                                search_radius_m=float(getattr(args, "frontier_target_search_radius_m", 0.45)),
                                reached_radius_m=float(args.frontier_commit_reached_radius_m),
                                max_candidates=int(getattr(args, "frontier_target_max_candidates", 128)),
                                require_reachable=bool(getattr(args, "frontier_target_require_reachable", True)),
                            )
                            if bool(resolution.reachable) and resolution.target_cells:
                                target = _as_grid_cell(resolution.target_cells[0])
                        except Exception:
                            target = None
                        terminal_target_cache[center] = target
                        return target

                    raw_alive_radius_cells = max(
                        1,
                        int(round(float(getattr(args, "frontier_terminal_raw_alive_radius_m", 0.30)) / max(float(dynamic_map_info.resolution_m), 1e-6))),
                    )

                    def raw_frontier_alive_checker(record: object) -> bool:
                        record_center = _as_grid_cell(getattr(record, "center_grid", None))
                        if record_center is None:
                            return False
                        for frontier in raw_frontiers_for_filter:
                            center = _as_grid_cell(getattr(frontier, "center_grid", None))
                            if center is None:
                                continue
                            dr = int(center[0]) - int(record_center[0])
                            dc = int(center[1]) - int(record_center[1])
                            if dr * dr + dc * dc <= int(raw_alive_radius_cells) * int(raw_alive_radius_cells):
                                return True
                        return False

                    filter_result = frontier_terminal_registry.filter_frontiers_with_debug(
                        raw_frontiers_for_filter,
                        target_resolver=resolve_terminal_target_for_frontier,
                        step=int(step),
                        raw_alive_checker=raw_frontier_alive_checker
                        if bool(getattr(args, "frontier_terminal_unsuppress_if_raw_frontier_still_exists", True))
                        else None,
                    )
                    filtered_frontiers = list(filter_result.filtered)
                    suppressed_this_step = int(len(raw_frontiers_for_filter) - len(filtered_frontiers))
                    terminal_filter_relaxed = False
                    terminal_filter_fail_safe_release = False
                    relaxed_filter_result = None
                    if len(filtered_frontiers) == 0 and len(real_frontiers) > 0 and suppressed_this_step > 0:
                        ignore_statuses: set[str] = {"stale_near_fallback"}
                        if bool(getattr(args, "frontier_terminal_unsuppress_partial_when_all_filtered", True)):
                            ignore_statuses.add("partial_reached")
                        if bool(getattr(args, "frontier_terminal_unsuppress_failed_when_all_filtered", True)):
                            ignore_statuses.add("failed_unreachable")
                        relaxed_filter_result = frontier_terminal_registry.filter_frontiers_with_debug(
                            raw_frontiers_for_filter,
                            target_resolver=resolve_terminal_target_for_frontier,
                            step=int(step),
                            ignore_statuses=ignore_statuses,
                            raw_alive_checker=raw_frontier_alive_checker
                            if bool(getattr(args, "frontier_terminal_unsuppress_if_raw_frontier_still_exists", True))
                            else None,
                        )
                        if relaxed_filter_result.filtered:
                            filtered_frontiers = list(relaxed_filter_result.filtered)
                            terminal_filter_relaxed = True
                        else:
                            filtered_frontiers = list(real_frontiers)
                            terminal_filter_fail_safe_release = True
                    frontier_partition.filtered_frontiers = list(filtered_frontiers)
                    frontier_partition.suppressed_count = int(suppressed_this_step)
                    if suppressed_this_step > 0:
                        frontier_refresh_loop_prevented_count += int(suppressed_this_step)
                    frontiers = list(filtered_frontiers)
                    loop_tick(step, "after_extract_frontiers")
                    last_frontier_real_count = len(real_frontiers)
                    last_frontier_near_fallback_count = len(near_fallback_frontiers)
                    last_frontier_terminal_suppressed_count = int(suppressed_this_step)
                    last_frontier_filtered_count = len(frontiers)
                    last_frontier_clusters = len(frontiers)
                    last_frontiers = list(frontiers)
                    stop_due_to_frontiers, frontier_stop_reason = should_stop_explore_until_no_frontiers(
                        explore_until_no_frontiers=bool(explore_until_no_frontiers),
                        real_frontiers=real_frontiers,
                        filtered_frontiers=frontiers,
                        near_fallback_frontiers=near_fallback_frontiers,
                        frontier_near_fallback_counts_as_exploration_frontier=bool(
                            getattr(args, "frontier_near_fallback_counts_as_exploration_frontier", False)
                        ),
                    )
                    last_frontier_exhaustion_audit = audit_frontier_exhaustion(
                        real_frontiers=real_frontiers,
                        filtered_frontiers=frontiers,
                        near_fallback_frontiers=near_fallback_frontiers,
                        navigation_unknown_mask=np.asarray(frontier_layers.get("unknown"), dtype=bool),
                        free_mask=np.asarray(frontier_free, dtype=bool),
                        frontier_near_fallback_counts_as_exploration_frontier=bool(
                            getattr(args, "frontier_near_fallback_counts_as_exploration_frontier", False)
                        ),
                    )
                    if stop_due_to_frontiers and not bool(last_frontier_exhaustion_audit.stop_allowed):
                        frontier_stop_reason = str(last_frontier_exhaustion_audit.stop_blocked_reason or "stop_blocked_frontier_exhaustion_audit")
                        stop_due_to_frontiers = False
                        exploration_stop_reason_detailed = str(frontier_stop_reason)
                        refresh_requested = request_frontier_refresh(
                            refresh_state=frontier_arrival_update_state,
                            step=int(step),
                            reason=str(frontier_stop_reason),
                            frontier_key=("frontier_exhaustion_audit", tuple(int(v) for v in current_grid)),
                            block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
                        )
                        if bool(refresh_requested):
                            frontier_refresh_loop_prevented_count += 1
                        update_roomseg_frontiers = bool(update_roomseg_frontiers or refresh_requested)
                        roomseg_frontier_update_reason = (
                            str(frontier_stop_reason) if bool(refresh_requested) else str(roomseg_frontier_update_reason)
                        )
                        roomseg_update_token = RoomsegFrontierUpdateToken(
                            allowed=bool(update_roomseg_frontiers),
                            reason=str(roomseg_frontier_update_reason),
                            step=int(step),
                        )
                        write_runtime_decision_trace(
                            step_idx=int(step),
                            event_type="frontier_exhaustion_stop_blocked",
                            event_reason=str(frontier_stop_reason),
                            current_grid_local=tuple(int(v) for v in current_grid),
                            nav_decision_local=last_nav_decision,
                            roomseg_update_allowed=bool(update_roomseg_frontiers),
                            roomseg_update_reason=str(roomseg_frontier_update_reason),
                            roomseg_context_updated_this_step=bool(room_context_updated_this_step),
                            current_path_len=len(current_path),
                            extra={
                                "frontier_refresh_requested": bool(refresh_requested),
                                "frontier_exhaustion_audit": last_frontier_exhaustion_audit.to_dict(),
                            },
                        )
                    if stop_due_to_frontiers:
                        exploration_complete = True
                        exploration_complete_reason = str(frontier_stop_reason)
                        exploration_stop_reason_detailed = str(frontier_stop_reason)
                        stop_called = True
                        policy_stop_confirmed = True
                        failure_reason = None
                        last_decision_mode = "stop"
                        last_decision_reason = str(frontier_stop_reason)
                        last_nav_decision = NavigationDecision(
                            mode="none",
                            target_cells=[],
                            stop=True,
                            selected_candidate=None,
                            frontier_decision=None,
                            reason=str(frontier_stop_reason),
                            metadata={
                                "exploration_complete": True,
                                "exploration_complete_reason": str(frontier_stop_reason),
                                "frontier_raw_count": int(last_frontier_raw_cells),
                                "frontier_real_count": int(last_frontier_real_count),
                                "frontier_near_fallback_count": int(last_frontier_near_fallback_count),
                                "frontier_terminal_suppressed_count": int(last_frontier_terminal_suppressed_count),
                                "frontier_terminal_filter_relaxed": bool(terminal_filter_relaxed),
                                "frontier_terminal_fail_safe_release": bool(terminal_filter_fail_safe_release),
                                "frontier_terminal_suppressed_by_status": dict(filter_result.suppressed_by_status),
                                "frontier_terminal_target_resolved_count": int(filter_result.target_resolved_count),
                                "frontier_terminal_target_missing_count": int(filter_result.target_missing_count),
                                "frontier_filtered_count": int(last_frontier_filtered_count),
                                "frontier_exhaustion_audit": last_frontier_exhaustion_audit.to_dict(),
                                **frontier_terminal_registry.debug_metadata(),
                            },
                        )
                        write_runtime_decision_trace(
                            step_idx=int(step),
                            event_type="explore_until_no_frontiers_stop",
                            event_reason=str(frontier_stop_reason),
                            current_grid_local=tuple(int(v) for v in current_grid),
                            nav_decision_local=last_nav_decision,
                            roomseg_update_allowed=bool(update_roomseg_frontiers),
                            roomseg_update_reason=str(roomseg_frontier_update_reason),
                            roomseg_context_updated_this_step=bool(room_context_updated_this_step),
                            current_path_len=len(current_path),
                            extra={
                                "exploration_complete": True,
                                "exploration_complete_reason": str(frontier_stop_reason),
                                "frontier_exhaustion_audit": last_frontier_exhaustion_audit.to_dict(),
                            },
                        )
                        break
                    candidate_preview = None
                    if paper_mode and frontiers and candidate_override is None and not frontier_mask_probe:
                        candidate_preview = decision_policy.select_goal_candidate(
                            object_memory,
                            episode["goal_category"],
                            pose,
                            current_step=step,
                        )
                        if room_context_result is None and (bool(args.score_frontiers_before_candidate) or candidate_preview is None):
                            if update_roomseg_frontiers:
                                room_context_result = maybe_update_room_context_for_frontier_scoring(
                                    token=roomseg_update_token,
                                    step_idx=step,
                                    map_state=map_state,
                                    call_site="candidate_preview_room_context",
                                )
                                if room_context_result is not None:
                                    room_context_updated_this_step = True
                                    room_context_source = "actual_roomseg_update"
                                    roomseg_updates_since_start += 1
                                update_scenegraph_frame(obs, step, map_state)
                            else:
                                room_context_result = last_room_context_result
                                room_context_source = "cached_roomseg_context" if room_context_result is not None else "none"
                    nav_decision = candidate_override or decision_policy.choose_navigation_target(
                        object_memory,
                        episode["goal_category"],
                        current_grid,
                        frontiers,
                        base_nav_planner if bool(frontier_mask_probe) else nav_planner,
                        dynamic_map_info,
                        pose,
                        allow_frontier=True,
                        current_step=step,
                    )
                    loop_tick(step, "after_choose_navigation_target")
                    selected_frontier_key_for_loop = None
                    if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
                        selected_frontier_key_for_loop = tuple(
                            int(v) for v in nav_decision.frontier_decision.selected_frontier.center_grid
                        )
                    if (
                        selected_frontier_key_for_loop is not None
                        and last_initial_loop_selected_frontier_key is not None
                        and selected_frontier_key_for_loop != last_initial_loop_selected_frontier_key
                    ):
                        selected_frontier_changes_since_start += 1
                    if selected_frontier_key_for_loop is not None:
                        last_initial_loop_selected_frontier_key = selected_frontier_key_for_loop
                    same_as_initial_grid = (
                        initial_frontier_loop_grid is not None
                        and tuple(int(v) for v in current_grid) == initial_frontier_loop_grid
                    )
                    if (
                        not bool(frontier_initial_refresh_loop_detected)
                        and int(step) > 5
                        and bool(same_as_initial_grid)
                        and int(roomseg_updates_since_start) > 3
                        and int(control_steps_with_nonzero_motion) == 0
                    ):
                        frontier_initial_refresh_loop_detected = True
                        write_runtime_decision_trace(
                            step_idx=int(step),
                            event_type="frontier_initial_refresh_loop_detected",
                            event_reason="initial_grid_repeated_roomseg_refresh_without_motion",
                            current_grid_local=tuple(int(v) for v in current_grid),
                            nav_decision_local=nav_decision,
                            roomseg_update_allowed=bool(update_roomseg_frontiers),
                            roomseg_update_reason=str(roomseg_frontier_update_reason),
                            roomseg_context_updated_this_step=bool(room_context_updated_this_step),
                            current_path_len=len(current_path),
                            extra={
                                "initial_grid": list(initial_frontier_loop_grid),
                                "roomseg_updates_since_start": int(roomseg_updates_since_start),
                                "selected_frontier_changes_since_start": int(selected_frontier_changes_since_start),
                                "control_steps_with_nonzero_motion": int(control_steps_with_nonzero_motion),
                            },
                        )
                        if bool(getattr(args, "frontier_initial_refresh_loop_raise", False)):
                            raise RuntimeError("frontier_initial_refresh_loop_detected")
                    write_runtime_decision_trace(
                        step_idx=int(step),
                        event_type="frontier_selected_after_refresh" if refresh_consumed_this_step else "frontier_selected",
                        event_reason=str(nav_decision.reason),
                        current_grid_local=tuple(int(v) for v in current_grid),
                        nav_decision_local=nav_decision,
                        roomseg_update_allowed=bool(update_roomseg_frontiers),
                        roomseg_update_reason=str(roomseg_frontier_update_reason),
                        roomseg_context_updated_this_step=bool(room_context_updated_this_step),
                        current_path_len=len(current_path),
                    )
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        **dict(frontier_source_metadata),
                        "frontier_terminal_filter_relaxed": bool(terminal_filter_relaxed),
                        "frontier_terminal_fail_safe_release": bool(terminal_filter_fail_safe_release),
                        "frontier_terminal_suppressed_by_status": dict(filter_result.suppressed_by_status),
                        "frontier_terminal_target_resolved_count": int(filter_result.target_resolved_count),
                        "frontier_terminal_target_missing_count": int(filter_result.target_missing_count),
                        "frontier_terminal_targetless_query_ignored_count": int(filter_result.targetless_query_ignored_count),
                        "room_context_used_cached_due_to_update_gate": bool(
                            room_context_result is last_room_context_result and not bool(update_roomseg_frontiers)
                        ),
                        "room_context_source": str(room_context_source),
                        "room_context_updated_this_step": bool(room_context_updated_this_step),
                        "roomseg_frontier_update_reason": str(roomseg_frontier_update_reason),
                    }
                    terminal_suppressed_all_real = bool(
                        last_frontier_real_count > 0
                        and int(last_frontier_terminal_suppressed_count) >= int(last_frontier_real_count)
                        and int(last_frontier_filtered_count) == 0
                    )
                    if (
                        nav_decision.mode == "none"
                        and nav_decision.reason == "no_frontiers"
                        and last_frontier_raw_cells > 0
                        and not terminal_suppressed_all_real
                    ):
                        nav_decision.reason = "no_selectable_frontiers"
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_selectable_failure_reason": "raw_frontiers_filtered_out",
                            "frontier_raw_cells": int(last_frontier_raw_cells),
                            "frontier_clusters": int(last_frontier_clusters),
                            "frontier_min_distance_m": float(args.frontier_min_distance_m),
                            "frontier_allow_near_fallback": bool(args.frontier_allow_near_fallback),
                            "frontier_cluster_distance_mode": str(args.frontier_cluster_distance_mode),
                        }
                    if room_context_result is not None:
                        last_room_context_metadata = room_context_result.metadata(full_order=True)
                        frontier_room_contexts = frontier_room_contexts_for_debug(
                            frontiers=frontiers,
                            room_debug=last_room_segmentation_debug,
                            room_masks=last_room_masks,
                            room_semantic_labels=room_semantic_labels,
                            observed_free=free,
                            unknown=~np.asarray(observed, dtype=bool),
                            agent_grid=current_grid,
                            resolution_m=float(dynamic_map_info.resolution_m),
                            config=dict(getattr(args, "room_segmentation_config", {}).get("frontier_room_context", {}) or {}),
                        )
                        selected_frontier = (
                            nav_decision.frontier_decision.selected_frontier
                            if nav_decision.frontier_decision is not None
                            else None
                        )
                        selected_index = (
                            nav_decision.frontier_decision.selected_index
                            if nav_decision.frontier_decision is not None
                            else None
                        )
                        selected_room_context = None
                        if selected_index is not None:
                            for item in frontier_room_contexts:
                                if int(item.get("frontier_id", -1)) == int(selected_index):
                                    selected_room_context = dict(item.get("room_context") or {})
                                    break
                        selected_center_for_snapshot = (
                            getattr(selected_frontier, "center_grid", None)
                            if selected_frontier is not None
                            else None
                        )
                        selected_key_for_snapshot = (
                            tuple(int(v) for v in selected_center_for_snapshot)
                            if selected_center_for_snapshot is not None
                            else None
                        )
                        geometry_hash_for_snapshot = roomseg_snapshot_geometry_hash(
                            last_room_masks,
                            last_room_segmentation_debug,
                            fallback_hash=str(last_room_context_metadata.get("room_mask_geometry_hash") or ""),
                        )
                        last_room_context_metadata = {
                            **dict(last_room_context_metadata),
                            "frontier_room_contexts": frontier_room_contexts,
                            "selected_frontier_room_context": selected_room_context,
                        }
                        setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            **dict(last_room_context_metadata),
                        }
                        if should_save_roomseg_snapshot_for_context(
                            save_roomseg_snapshots=bool(getattr(args, "save_roomseg_snapshots", False)),
                            room_context_updated_this_step=bool(room_context_updated_this_step),
                            save_cached_roomseg_snapshots=bool(getattr(args, "save_cached_roomseg_snapshots", False)),
                            snapshot_source=str(room_context_source),
                            agent_grid=tuple(int(v) for v in current_grid),
                            selected_frontier_key=selected_key_for_snapshot,
                            room_mask_geometry_hash=str(geometry_hash_for_snapshot),
                            roomseg_update_reason=str(roomseg_frontier_update_reason),
                            last_snapshot_state={
                                "agent_grid": last_roomseg_snapshot_agent_grid,
                                "selected_frontier_key": last_roomseg_snapshot_selected_key,
                                "room_mask_geometry_hash": last_roomseg_snapshot_geometry_hash,
                                "roomseg_update_reason": last_roomseg_snapshot_update_reason,
                                "step": last_roomseg_snapshot_step,
                            },
                            step=int(step),
                            failure_loop_min_interval_steps=int(getattr(args, "roomseg_snapshot_failure_loop_min_interval", 50)),
                        ):
                            save_selected_roomseg_snapshot(
                                step,
                                map_state,
                                frontier_layers,
                                selected_frontier,
                                snapshot_source=str(room_context_source),
                                roomseg_update_reason=str(roomseg_frontier_update_reason),
                            )
                            snapshot_saved_this_step = True
                        if bool(getattr(args, "debug_roomseg_layers", False)):
                            selected_members = getattr(selected_frontier, "members", None) if selected_frontier is not None else None
                            selected_center = getattr(selected_frontier, "center_grid", None) if selected_frontier is not None else None
                            nav_free_for_dump, nav_obstacle_for_dump, nav_unknown_for_dump, _nav_source_for_dump = resolve_replay_style_navigation_masks(
                                mapper=mapper,
                                shape=np.asarray(occupancy).shape,
                                fallback_free=free,
                                fallback_obstacle=occupancy,
                                fallback_unknown=~np.asarray(observed, dtype=bool),
                            )
                            dump = save_roomseg_layer_dump(
                                out_dir=str(getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")),
                                step=int(step),
                                room_debug=_roomseg_debug_for_layer_dump(last_room_segmentation_debug, room_segmenter),
                                occupancy_map=nav_obstacle_for_dump,
                                observed_free_mask=nav_free_for_dump,
                                obstacle_mask=nav_obstacle_for_dump,
                                unknown_mask=nav_unknown_for_dump,
                                frontier_map=frontier_layers["frontier"],
                                selected_frontier_members=selected_members,
                                selected_frontier_center_rc=selected_center,
                                agent_rc=current_grid,
                                max_saves=int(getattr(args, "debug_roomseg_max_saves", 50)),
                                save_npz=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_npz", True)),
                                save_png=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_png", True)),
                                save_summary_json=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_summary_json", True)),
                                include_selected_frontier_sector=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("include_selected_frontier_sector", True)),
                            )
                            last_room_segmentation_debug = {
                                **dict(last_room_segmentation_debug),
                                "roomseg_debug_layers": dict(dump.get("paths", {})),
                                "roomseg_debug_summary": dict(dump.get("summary", {})),
                            }
                            last_room_context_metadata["roomseg_debug_layers"] = dict(dump.get("paths", {}))
                            last_room_context_metadata["roomseg_debug_likely_cause"] = dict(dump.get("summary", {})).get("likely_cause")
                            nav_decision.metadata = {
                                **dict(nav_decision.metadata or {}),
                                "roomseg_debug_layers": dict(dump.get("paths", {})),
                                "roomseg_debug_likely_cause": dict(dump.get("summary", {})).get("likely_cause"),
                            }
                            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
                            if viz is not None:
                                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
                target_resolution: FrontierTargetResolution | None = None
                if (
                    bool(getattr(args, "frontier_targeting_enabled", True))
                    and not bool(locked_goal_used)
                    and nav_decision.mode == "frontier"
                    and nav_decision.frontier_decision is not None
                    and nav_decision.frontier_decision.selected_frontier is not None
                ):
                    selected_frontier_for_target = nav_decision.frontier_decision.selected_frontier
                    astar_clearance_for_target = map_state.get("astar_clearance_m")
                    if not isinstance(astar_clearance_for_target, np.ndarray):
                        astar_clearance_for_target = clearance_map_m(astar_traversible, dynamic_map_info.resolution_m)
                    target_resolution = resolve_frontier_target(
                        selected_frontier_for_target,
                        tuple(int(v) for v in current_grid),
                        nav_planner,
                        astar_traversible,
                        np.asarray(astar_clearance_for_target, dtype=np.float32),
                        float(dynamic_map_info.resolution_m),
                        min_clearance_m=float(getattr(args, "frontier_target_min_goal_clearance_m", getattr(args, "astar_goal_min_clearance_m", 0.18))),
                        search_radius_m=float(getattr(args, "frontier_target_search_radius_m", 0.45)),
                        reached_radius_m=float(args.frontier_commit_reached_radius_m),
                        max_candidates=int(getattr(args, "frontier_target_max_candidates", 128)),
                        require_reachable=bool(getattr(args, "frontier_target_require_reachable", True)),
                    )
                    nav_decision.target_cells = list(target_resolution.target_cells)
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        **target_resolution.metadata(),
                        "frontier_unreachable_recovery": not bool(target_resolution.reachable),
                        "frontier_unreachable_reason": None if target_resolution.reachable else str(target_resolution.reason),
                    }
                    if not target_resolution.reachable:
                        frontier_target_resolution_failures += 1
                        if str(target_resolution.reason) == "frontier_no_clearance_safe_target":
                            frontier_target_no_clearance_safe_count += 1
                if (
                    nav_decision.mode == "frontier"
                    and not locked_goal_used
                    and nav_decision.target_cells
                    and tuple(int(v) for v in nav_decision.target_cells[0]) == tuple(int(v) for v in current_grid)
                ):
                    reason = "frontier_current_cell_is_best_reachable_approach"
                    key = mark_frontier_terminal_for_refresh(
                        nav_decision_local=nav_decision,
                        current_grid_local=tuple(int(v) for v in current_grid),
                        reason=reason,
                        status="partial_reached",
                    )
                    refresh_key = _frontier_refresh_key_or_fallback(key, nav_decision)
                    refresh_requested = request_frontier_refresh(
                        refresh_state=frontier_arrival_update_state,
                        step=int(step),
                        reason=reason,
                        frontier_key=refresh_key,
                        block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
                    )
                    nav_decision.reason = reason
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        "frontier_refresh_pending": bool(refresh_requested),
                        "frontier_refresh_reason": reason,
                        "frontier_selected_target_equals_current": True,
                    }
                    write_runtime_decision_trace(
                        step_idx=int(step),
                        event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
                        event_reason=reason,
                        current_grid_local=tuple(int(v) for v in current_grid),
                        nav_decision_local=nav_decision,
                        recovery_action="target_equals_current_refresh" if refresh_requested else "duplicate_refresh_skipped",
                        current_path_len=0,
                        zero_velocity_continue=True,
                        extra=terminal_trace_metadata(selected_key=key, target_equals_current=True),
                    )
                    if frontier_commitment is not None:
                        frontier_commitment.mark_active_reached(step, reason)
                    current_path = []
                    force_perception_step = True
                    full_path.append(tuple(int(v) for v in current_grid))
                    last_nav_decision = nav_decision
                    last_decision_reason = reason
                    last_frontier_commitment_reason = reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": reason,
                        "frontier_refresh_pending": bool(refresh_requested),
                        "frontier_refresh_reason": reason,
                    }
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                if frontier_commitment is not None and nav_decision.mode == "frontier" and not locked_goal_used:
                    frontier_decision = nav_decision.frontier_decision
                    proposed_frontier = frontier_decision.selected_frontier if frontier_decision is not None else None
                    proposed_score = 0.0
                    scores_by_index = []
                    if frontier_decision is not None:
                        scores_by_index = list(frontier_decision.total_scores)
                        if frontier_decision.selected_index is not None and frontier_decision.selected_index < len(scores_by_index):
                            proposed_score = float(scores_by_index[int(frontier_decision.selected_index)])
                    commit_decision = frontier_commitment.select(
                        frontiers,
                        proposed_frontier,
                        proposed_score,
                        current_grid,
                        step,
                        planner=None if bool(frontier_mask_probe) else nav_planner,
                        target_cells=nav_decision.target_cells,
                        target_frontier=proposed_frontier,
                        target_metadata=dict(nav_decision.metadata or {}),
                        scores_by_index=scores_by_index,
                    )
                    last_frontier_commitment_metadata = dict(commit_decision.metadata)
                    last_frontier_commitment_metadata["frontier_commitment_reason"] = commit_decision.reason
                    last_frontier_commitment_reason = commit_decision.reason
                    nav_decision.target_cells = list(commit_decision.target_cells)
                    nav_decision.reason = commit_decision.reason
                    if commit_decision.reason == "frontier_switch_requires_refresh":
                        pending_switch_reason = str(
                            last_frontier_commitment_metadata.get("pending_switch_reason") or "frontier_switch"
                        )
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_commitment": last_frontier_commitment_metadata,
                            "frontier_refresh_required_before_reselect": True,
                            "pending_switch_reason": pending_switch_reason,
                        }
                        request_frontier_refresh_at_current(
                            nav_decision_local=nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            reason="frontier_switch_refresh",
                            status_reason=pending_switch_reason,
                            reached=False,
                        )
                        long_term_goal.invalidate("frontier_switch_refresh")
                        obs = server.step_kinematic_velocity(
                            0.0,
                            0.0,
                            0.0,
                            dt=float(args.control_dt),
                            render_updates=int(args.render_updates_per_step),
                            read_rgb=detector_requires_rgb(detector, args.detector),
                            read_depth=True,
                            rgb_device="cuda" if detector_cuda_rgb else "cpu",
                        )
                        continue
                    if commit_decision.keep_existing:
                        active_center = last_frontier_commitment_metadata.get("active_frontier_center_grid")
                        active_target = last_frontier_commitment_metadata.get("active_frontier_actual_target_grid")
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_center_grid": active_center,
                            "frontier_actual_target_grid": active_target,
                            "frontier_target_mode": "committed_target",
                            "frontier_commitment_target_consistent": bool(
                                last_frontier_commitment_metadata.get("frontier_commitment_target_consistent", True)
                            ),
                        }
                    if frontier_decision is not None:
                        frontier_decision.selected_frontier = commit_decision.frontier
                        if commit_decision.frontier is not None:
                            try:
                                frontier_decision.selected_index = frontiers.index(commit_decision.frontier)
                            except ValueError:
                                frontier_decision.selected_index = None
                        else:
                            frontier_decision.selected_index = None
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        "frontier_commitment": last_frontier_commitment_metadata,
                    }
                    if bool(getattr(args, "frontier_execution_state_enabled", True)) and nav_decision.target_cells:
                        committed_frontier_execution.start_from_decision(
                            nav_decision,
                            int(step),
                            last_frontier_commitment_metadata,
                            current_grid=tuple(int(v) for v in current_grid),
                        )
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            **committed_frontier_execution.debug_metadata(
                                step=int(step),
                                current_grid=tuple(int(v) for v in current_grid),
                                resolution_m=float(dynamic_map_info.resolution_m),
                            ),
                        }
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            **committed_frontier_execution.debug_metadata(
                                step=int(step),
                                current_grid=tuple(int(v) for v in current_grid),
                                resolution_m=float(dynamic_map_info.resolution_m),
                            ),
                        }
                if nav_decision.mode == "frontier":
                    nav_meta = dict(nav_decision.metadata or {})
                    frontier_target_mode = nav_meta.get("frontier_target_mode")
                    frontier_center_grid = nav_meta.get("frontier_center_grid")
                    frontier_actual_target_grid = nav_meta.get("frontier_actual_target_grid")
                    frontier_unreachable_recovery = bool(nav_meta.get("frontier_unreachable_recovery", False))
                    frontier_unreachable_reason = nav_meta.get("frontier_unreachable_reason")
                    active_center = dict(nav_meta.get("frontier_commitment") or {}).get("active_frontier_center_grid")
                    active_target = dict(nav_meta.get("frontier_commitment") or {}).get("active_frontier_actual_target_grid")
                    commit_mismatch = bool(
                        active_center is not None
                        and frontier_center_grid is not None
                        and [int(v) for v in active_center] != [int(v) for v in frontier_center_grid]
                    )
                    if commit_mismatch:
                        frontier_commit_target_mismatch_count += 1
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "commit_center_target_mismatch": True,
                            "active_frontier_center_grid": active_center,
                            "active_frontier_actual_target_grid": active_target,
                        }
                    if nav_decision.frontier_decision is not None and not nav_decision.target_cells:
                        reason = str(frontier_unreachable_reason or "frontier_empty_target")
                        if (
                            bool(getattr(args, "frontier_execution_state_enabled", True))
                            and committed_frontier_execution.exists()
                        ):
                            astar_clearance_for_recovery = map_state.get("astar_clearance_m")
                            if not isinstance(astar_clearance_for_recovery, np.ndarray):
                                astar_clearance_for_recovery = clearance_map_m(astar_traversible, dynamic_map_info.resolution_m)
                            apply_frontier_recovery_or_refresh(
                                nav_decision_local=nav_decision,
                                current_grid_local=tuple(int(v) for v in current_grid),
                                traversible_local=astar_traversible,
                                clearance_local=np.asarray(astar_clearance_for_recovery, dtype=np.float32),
                                planner_local=nav_planner,
                                trigger_reason="frontier_empty_target_recovery",
                                refresh_reason="frontier_confirmed_unreachable_empty_target",
                                partial_refresh_reason="frontier_partial_arrival_empty_target",
                                mark_failed_if_no_recovery=True,
                            )
                            obs = server.step_kinematic_velocity(
                                0.0,
                                0.0,
                                0.0,
                                dt=float(args.control_dt),
                                render_updates=int(args.render_updates_per_step),
                                read_rgb=detector_requires_rgb(detector, args.detector),
                                read_depth=True,
                                rgb_device="cuda" if detector_cuda_rgb else "cpu",
                            )
                            continue
                        selected_frontier = nav_decision.frontier_decision.selected_frontier
                        if frontier_commitment is not None and selected_frontier is not None:
                            frontier_commitment.blacklist_frontier(
                                selected_frontier,
                                step,
                                reason,
                            )
                        frontier_blacklisted = True
                        long_term_goal.invalidate(reason)
                        current_path = []
                        force_perception_step = True
                        last_frontier_commitment_reason = reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": last_frontier_commitment_reason,
                            "frontier_blacklisted": True,
                        }
                        continue
                if paper_mode and not locked_goal_used and nav_decision.mode in {"frontier", "candidate"} and nav_decision.target_cells:
                    long_term_goal.set_from(nav_decision, step)
                last_nav_decision = nav_decision
                path_replan_target_cells_count = int(len(nav_decision.target_cells or []))
                nav_target_key = (
                    str(nav_decision.mode),
                    tuple(int(v) for v in nav_decision.target_cells[0]) if nav_decision.target_cells else None,
                    int(len(nav_decision.target_cells or [])),
                    last_frontier_commitment_metadata.get("active_frontier_id"),
                )
                if nav_target_key != nav_execution_progress_key:
                    nav_execution_progress_key = nav_target_key
                    nav_execution_best_distance_m = float("inf")
                    nav_execution_no_progress_steps = 0
                if not locked_goal_used:
                    evaluator.num_frontier_decisions += 1
                last_decision_mode = nav_decision.mode
                last_decision_reason = nav_decision.reason
                if bool(getattr(args, "debug_graph_dump", False)):
                    save_graph_debug_dump(
                        args.debug_graph_dump_dir,
                        step=step,
                        goal=episode["goal_category"],
                        scenegraph=scenegraph,
                        frontiers=frontiers,
                        frontier_decision=nav_decision.frontier_decision,
                        nav_decision=nav_decision,
                        commitment_metadata=last_frontier_commitment_metadata,
                    )
                if bool(getattr(args, "frontier_debug_dump", False)):
                    selected_frontier_cell = None
                    if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
                        selected_frontier_cell = nav_decision.frontier_decision.selected_frontier.center_grid
                    save_frontier_debug_snapshot(
                        args.frontier_debug_dir,
                        step,
                        free=frontier_free,
                        occupied=occupancy,
                        observed=observed,
                        unknown=frontier_layers["unknown"],
                        unknown_dilated=frontier_layers["unknown_dilated"],
                        frontier=frontier_layers["frontier"],
                        traversible=distance_traversible,
                        dist_map=astar_distance_map(
                            distance_traversible,
                            current_grid,
                            dynamic_map_info.resolution_m,
                            allow_diagonal=True,
                        ),
                        agent_grid=current_grid,
                        clusters=frontiers,
                        selected_frontier=selected_frontier_cell,
                        candidate_centers=candidate_center_payload(
                            object_memory,
                            episode["goal_category"],
                            selected_candidate_id=(
                                int(nav_decision.selected_candidate.node_id)
                                if nav_decision.selected_candidate is not None
                                else None
                            ),
                        ),
                        candidate_target_cells=nav_decision.target_cells if nav_decision.mode == "candidate" else [],
                        selected_candidate=(
                            nav_decision.selected_candidate.center_grid
                            if nav_decision.selected_candidate is not None
                            else None
                        ),
                        decision_mode=nav_decision.mode,
                    )
                if nav_decision.selected_candidate is not None:
                    last_selected_candidate = nav_decision.selected_candidate
                    candidate_id = int(last_selected_candidate.node_id)
                    if logged_selected_candidate_id != candidate_id:
                        logged_selected_candidate_id = candidate_id
                        print(
                            "[voxroom-loop] selected candidate id=%d category=%s raw=%s conf=%.3f observed=%d center=%s"
                            % (
                                candidate_id,
                                normalize_category(last_selected_candidate.category),
                                str(last_selected_candidate.raw_label),
                                float(last_selected_candidate.confidence),
                                int(last_selected_candidate.observed_count),
                                tuple(float(v) for v in last_selected_candidate.center_world),
                            ),
                            flush=True,
                        )
                goal_candidate_count = len(
                    [
                        node
                        for node in object_memory.nodes
                        if category_matches_goal(node.category)
                    ]
                )
                if nav_decision.stop:
                    policy_stop_confirmed = True
                    stop_blocked_reason = None
                    if policy_distance_to_goal <= success_distance:
                        gt_success_region_reached = True
                    if success_region_can_finish(
                        policy_distance_to_goal,
                        success_distance,
                        require_sgnav_stop=bool(args.require_sgnav_stop),
                        policy_stop_confirmed=True,
                        ignore_goal_success=explore_until_no_frontiers,
                    ):
                        stop_called = True
                    else:
                        failure_reason = (
                            "goal_success_ignored_for_frontier_exploration"
                            if explore_until_no_frontiers
                            else "sgnav_stop_outside_goal_region"
                        )
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                        )
                    break
                if nav_decision.mode == "reperception":
                    current_path = []
                    full_path.append(tuple(int(v) for v in current_grid))
                    if viz is not None and step % viz_every == 0:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                        )
                    force_perception_step = True
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        float(args.reperception_turn_wz_radps),
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                nav_goals = nav_decision.target_cells
                if paper_mode and nav_decision.mode == "frontier" and not nav_goals:
                    reason = str(
                        (nav_decision.metadata or {}).get("frontier_unreachable_reason")
                        or nav_decision.reason
                        or "frontier_empty_target"
                    )
                    if frontier_commitment is not None:
                        frontier_commitment.mark_active_failed(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    current_path = []
                    force_perception_step = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    frontier_blacklisted = True
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                    }
                    continue
                if not nav_goals and args.allow_gt_goal_fallback:
                    nav_goals = goal_cells
                    last_decision_mode = "gt_goal_fallback"
                    last_decision_reason = "explicit_gt_goal_fallback"
                if not nav_goals:
                    failure_reason = nav_decision.reason or "sgnav_no_target"
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                    )
                    break
                planning_goals = nav_goals
                locked_frontier_exact_goal = frontier_exact_target_locked_for_planning(
                    nav_decision,
                    committed_frontier_execution,
                )
                if nav_decision.mode == "frontier" and locked_frontier_exact_goal:
                    planning_goals = list(nav_goals)
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        "frontier_planning_goal_mode": "locked_exact_target",
                        "frontier_planning_goal_cells_count": int(len(planning_goals)),
                        "frontier_planning_goal_radius_m": 0.0,
                        "frontier_planning_goal_radius_expansion_skipped": True,
                    }
                elif nav_decision.mode == "frontier":
                    radius_goals = planning_target_cells_within_radius(
                        nav_goals,
                        astar_traversible,
                        dynamic_map_info.resolution_m,
                        float(args.frontier_commit_reached_radius_m),
                    )
                    if radius_goals:
                        planning_goals = radius_goals
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_planning_goal_mode": "within_target_radius",
                            "frontier_planning_goal_cells_count": int(len(radius_goals)),
                            "frontier_planning_goal_radius_m": float(args.frontier_commit_reached_radius_m),
                        }
                astar_clearance_for_goals = map_state.get("astar_clearance_m")
                if not isinstance(astar_clearance_for_goals, np.ndarray):
                    astar_clearance_for_goals = clearance_map_m(astar_traversible, dynamic_map_info.resolution_m)
                astar_goal_search_radius_cells = (
                    0
                    if locked_frontier_exact_goal
                    else int(
                        math.ceil(float(getattr(args, "astar_goal_search_radius_m", 0.35)) / max(float(dynamic_map_info.resolution_m), 1e-6))
                    )
                )
                goal_filter = filter_goal_cells_by_clearance_result(
                    planning_goals,
                    astar_traversible,
                    np.asarray(astar_clearance_for_goals, dtype=np.float32),
                    min_clearance_m=float(getattr(args, "astar_goal_min_clearance_m", 0.18)),
                    search_radius_cells=astar_goal_search_radius_cells,
                )
                nav_decision.metadata = {
                    **dict(nav_decision.metadata or {}),
                    **goal_filter.metadata(),
                    "astar_goal_min_clearance_m": float(getattr(args, "astar_goal_min_clearance_m", 0.18)),
                    "astar_goal_search_radius_m": (
                        0.0 if locked_frontier_exact_goal else float(getattr(args, "astar_goal_search_radius_m", 0.35))
                    ),
                    "astar_goal_search_radius_cells": int(astar_goal_search_radius_cells),
                    "astar_goal_clearance_filter_locked_exact_target": bool(locked_frontier_exact_goal),
                    "astar_goal_cells_before_clearance_filter": int(len(planning_goals)),
                    "astar_goal_cells_after_clearance_filter": int(len(goal_filter.goals)),
                    "astar_goal_clearance_reject_reason": None if goal_filter.goals else "frontier_no_clearance_safe_planning_goal",
                }
                if not goal_filter.goals:
                    reason = "frontier_no_clearance_safe_planning_goal" if nav_decision.mode == "frontier" else "no_clearance_safe_planning_goal"
                    if nav_decision.mode == "frontier":
                        frontier_target_resolution_failures += 1
                        frontier_target_no_clearance_safe_count += 1
                        if (
                            bool(getattr(args, "frontier_execution_state_enabled", True))
                            and committed_frontier_execution.exists()
                        ):
                            apply_frontier_recovery_or_refresh(
                                nav_decision_local=nav_decision,
                                current_grid_local=tuple(int(v) for v in current_grid),
                                traversible_local=astar_traversible,
                                clearance_local=np.asarray(astar_clearance_for_goals, dtype=np.float32),
                                planner_local=nav_planner,
                                trigger_reason="frontier_no_clearance_target_recovery",
                                refresh_reason="frontier_confirmed_unreachable_no_clearance_target",
                                partial_refresh_reason="frontier_partial_arrival_no_clearance_target",
                                mark_failed_if_no_recovery=True,
                            )
                            obs = server.step_kinematic_velocity(
                                0.0,
                                0.0,
                                0.0,
                                dt=float(args.control_dt),
                                render_updates=int(args.render_updates_per_step),
                                read_rgb=detector_requires_rgb(detector, args.detector),
                                read_depth=True,
                                rgb_device="cuda" if detector_cuda_rgb else "cpu",
                            )
                            continue
                        if frontier_commitment is not None:
                            selected_frontier = (
                                nav_decision.frontier_decision.selected_frontier
                                if nav_decision.frontier_decision is not None
                                else None
                            )
                            if selected_frontier is not None:
                                frontier_commitment.blacklist_frontier(selected_frontier, step, reason)
                            frontier_commitment.mark_active_failed(step, reason, blacklist=True)
                        committed_frontier_execution.mark_failed(reason, blacklist=True)
                        long_term_goal.invalidate(reason)
                        current_path = []
                        force_perception_step = True
                        frontier_unreachable_recovery = True
                        frontier_unreachable_reason = reason
                        frontier_blacklisted = True
                        frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                        last_frontier_commitment_reason = "%s_blacklisted" % reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": last_frontier_commitment_reason,
                            "frontier_blacklisted": True,
                            "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                            "astar_goal_clearance_no_safe_goal_count": int(goal_filter.no_safe_goal_count),
                        }
                        continue
                    failure_reason = reason
                    break
                planning_goals = list(goal_filter.goals)
                path_replan_target_cells_count = int(len(planning_goals))
                result = nav_planner.plan(current_grid, planning_goals)
                path_replan_executed = True
                loop_tick(step, "after_nav_plan")
                if not result.path:
                    astar_failure_reason = astar_no_path_failure_reason(
                        path_replan_only_active_target=bool(path_replan_only_active_target)
                    )
                    last_room_segmentation_debug = {
                        **dict(last_room_segmentation_debug),
                        **path_replan_debug,
                        "path_replan_executed": True,
                        "path_replan_reused_committed_target": bool(path_replan_reused_committed_target),
                        "path_replan_only_active_target": bool(path_replan_only_active_target),
                        "target_reselection_suppressed_by_roomseg_freeze": bool(target_reselection_suppressed_by_roomseg_freeze),
                        "path_replan_target_cells": int(path_replan_target_cells_count),
                        "astar_success": False,
                        "astar_failure_reason": str(astar_failure_reason),
                    }
                    if (
                        bool(getattr(args, "frontier_execution_state_enabled", True))
                        and nav_decision.mode == "frontier"
                        and committed_frontier_execution.exists()
                    ):
                        frontier_direct_no_path_count += 1
                        recovery_action = apply_frontier_recovery_or_refresh(
                            nav_decision_local=nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            traversible_local=astar_traversible,
                            clearance_local=np.asarray(astar_clearance_for_goals, dtype=np.float32),
                            planner_local=nav_planner,
                            trigger_reason="frontier_direct_target_no_path_recovery",
                            refresh_reason="frontier_confirmed_unreachable_no_path",
                            partial_refresh_reason="frontier_partial_arrival_no_path",
                            mark_failed_if_no_recovery=True,
                        )
                        if paper_mode and recovery_action == "refresh_pending":
                            long_term_goal.invalidate(str(last_frontier_commitment_reason))
                        obs = server.step_kinematic_velocity(
                            0.0,
                            0.0,
                            0.0,
                            dt=float(args.control_dt),
                            render_updates=int(args.render_updates_per_step),
                            read_rgb=detector_requires_rgb(detector, args.detector),
                            read_depth=True,
                            rgb_device="cuda" if detector_cuda_rgb else "cpu",
                        )
                        continue
                    if paper_mode and nav_decision.mode == "frontier":
                        reason = "frontier_center_unreachable"
                        long_term_goal.invalidate(reason)
                        if frontier_commitment is not None:
                            frontier_commitment.mark_active_failed(step, reason, blacklist=True)
                        current_path = []
                        force_perception_step = True
                        frontier_unreachable_recovery = True
                        frontier_unreachable_reason = reason
                        frontier_blacklisted = True
                        frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                        last_frontier_commitment_reason = "%s_blacklisted" % reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": last_frontier_commitment_reason,
                            "frontier_blacklisted": True,
                            "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                        }
                        continue
                    failure_reason = str(astar_failure_reason)
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                        )
                    break
                if frontier_mask_probe and nav_decision.mode == "frontier" and len(result.path) <= 1:
                    reason = "frontier_trivial_path"
                    selected_frontier = (
                        nav_decision.frontier_decision.selected_frontier
                        if nav_decision.frontier_decision is not None
                        else None
                    )
                    if frontier_commitment is not None:
                        if selected_frontier is not None:
                            frontier_commitment.blacklist_frontier(selected_frontier, step, reason)
                        frontier_commitment.mark_active_failed(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    current_path = []
                    force_perception_step = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    frontier_blacklisted = True
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                    }
                    continue
                if (
                    frontier_mask_probe
                    and nav_decision.mode == "frontier"
                    and nav_decision.frontier_decision is not None
                    and nav_decision.frontier_decision.selected_frontier is not None
                ):
                    selected_frontier = nav_decision.frontier_decision.selected_frontier
                    selected_center = tuple(int(v) for v in selected_frontier.center_grid)
                    selected_target = tuple(int(v) for v in (nav_decision.target_cells[0] if nav_decision.target_cells else selected_center))
                    snapshot_key = (selected_center, selected_target)
                    if snapshot_key != last_roomseg_snapshot_frontier_key:
                        save_selected_roomseg_snapshot(
                            step,
                            map_state,
                            frontier_layers,
                            selected_frontier,
                        )
                        last_roomseg_snapshot_frontier_key = snapshot_key
                if result.reached_goal is not None:
                    reached_goal = tuple(int(v) for v in result.reached_goal)
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        "astar_reached_goal_grid": [int(reached_goal[0]), int(reached_goal[1])],
                    }
                    if nav_decision.mode == "frontier" and locked_frontier_exact_goal:
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_astar_reached_goal_sync_skipped": True,
                            "frontier_astar_reached_goal_sync_skip_reason": "locked_committed_frontier_exact_target",
                        }
                    elif nav_decision.mode == "frontier":
                        nav_decision.target_cells = [reached_goal]
                        frontier_actual_target_grid = [int(reached_goal[0]), int(reached_goal[1])]
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_actual_target_grid": frontier_actual_target_grid,
                        }
                        if (
                            bool(getattr(args, "frontier_execution_state_enabled", True))
                            and committed_frontier_execution.exists()
                        ):
                            committed_frontier_execution.sync_actual_target(reached_goal, int(step))
                            last_frontier_commitment_metadata = {
                                **dict(last_frontier_commitment_metadata),
                                **committed_frontier_execution.debug_metadata(
                                    step=int(step),
                                    current_grid=tuple(int(v) for v in current_grid),
                                    resolution_m=float(dynamic_map_info.resolution_m),
                                ),
                                "active_frontier_actual_target_grid": frontier_actual_target_grid,
                                "frontier_actual_target_grid": frontier_actual_target_grid,
                                "frontier_astar_reached_goal_synced": True,
                            }
                current_path = result.path
                path_replan_debug = {
                    **dict(path_replan_debug),
                    "path_replan_executed": bool(path_replan_executed),
                    "path_replan_reused_committed_target": bool(path_replan_reused_committed_target),
                    "path_replan_only_active_target": bool(path_replan_only_active_target),
                    "target_reselection_suppressed_by_roomseg_freeze": bool(target_reselection_suppressed_by_roomseg_freeze),
                    "path_replan_target_cells": int(path_replan_target_cells_count),
                    "astar_success": True,
                    "astar_failure_reason": None,
                }
                if (
                    bool(getattr(args, "frontier_execution_state_enabled", True))
                    and nav_decision.mode == "frontier"
                    and committed_frontier_execution.exists()
                ):
                    if result.reached_goal is None:
                        committed_frontier_execution.mark_replanned(int(step))
                astar_clearance_for_path = map_state.get("astar_clearance_m")
                if not isinstance(astar_clearance_for_path, np.ndarray):
                    astar_clearance_for_path = clearance_map_m(astar_traversible, dynamic_map_info.resolution_m)
                path_debug = path_clearance_debug(
                    current_path,
                    np.asarray(astar_clearance_for_path, dtype=np.float32),
                    float(args.robot_radius_m),
                )
                map_state.update(path_debug)
                last_room_segmentation_debug = {
                    **dict(last_room_segmentation_debug),
                    **path_replan_debug,
                    **path_debug,
                    "astar_used_clearance_cost": bool(map_state.get("astar_used_clearance_cost", False)),
                    "astar_clearance_desired_m": float(getattr(args, "astar_clearance_desired_m", 0.25)),
                    "lookahead_min_clearance_m_configured": float(getattr(args, "lookahead_min_clearance_m", 0.14)),
                    "lookahead_min_clearance_m_effective": float(
                        getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
                    ),
                }
                if last_nav_decision is not None:
                    last_nav_decision.metadata = {
                        **dict(last_nav_decision.metadata or {}),
                        **path_replan_debug,
                        **path_debug,
                        "astar_used_clearance_cost": bool(map_state.get("astar_used_clearance_cost", False)),
                        "lookahead_min_clearance_m_configured": float(getattr(args, "lookahead_min_clearance_m", 0.14)),
                        "lookahead_min_clearance_m_effective": float(
                            getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
                        ),
                    }
                full_path.extend(result.path)
                record_latency("planning", planning_started_at)
                if total_llm_requests() > llm_requests_before:
                    record_latency("llm", planning_started_at)

            if viz is not None and step % viz_every == 0:
                loop_tick(step, "before_viz_update")
                viz.update(
                    step=step,
                    rgb=viz_rgb(obs),
                    detections_2d=last_detections_2d,
                    occupancy=occupancy,
                    navigable=free,
                    observed=observed,
                    goal_cells=goal_cells,
                    current_grid=current_grid,
                    pose=pose,
                    frontiers=last_frontiers,
                    nav_decision=last_nav_decision,
                    current_path=current_path,
                    full_path=full_path,
                    object_memory=object_memory,
                    goal_category=episode["goal_category"],
                    distance_to_goal=evaluator.final_distance_to_goal,
                    path_length=float(evaluator.path_accum.total_m),
                    scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                    score_debug=scenegraph.last_score_debug,
                )
                loop_tick(step, "after_viz_update")

            roomseg_update_call_count_this_step = int(roomseg_update_call_counts_by_step.get(int(step), 0))
            roomseg_update_called_sites_this_step = list(roomseg_update_called_sites_by_step.get(int(step), []))
            if (
                bool(getattr(args, "frontier_roomseg_update_assert_no_bypass", True))
                and not bool(update_roomseg_frontiers)
                and roomseg_update_call_count_this_step > 0
            ):
                raise RuntimeError(
                    "roomseg_frontier_update_bypass step=%d reason=%s call_sites=%s"
                    % (int(step), str(roomseg_frontier_update_reason), ",".join(roomseg_update_called_sites_this_step))
                )
            write_runtime_perf_trace(
                step,
                map_state,
                roomseg_update_due=bool(update_roomseg_frontiers),
                roomseg_update_reason=str(roomseg_frontier_update_reason),
                roomseg_update_allowed_by_token=bool(roomseg_update_token.allowed),
                roomseg_update_token_reason=str(roomseg_update_token.reason),
                roomseg_update_call_count_this_step=roomseg_update_call_count_this_step,
                roomseg_update_called_sites=roomseg_update_called_sites_this_step,
                frontier_execution_state=committed_frontier_execution,
                current_grid_for_frontier=tuple(int(v) for v in current_grid) if current_grid is not None else None,
            )

            target_reached, target_distance_m = navigation_target_reached(
                current_grid,
                last_nav_decision,
                dynamic_map_info.resolution_m,
                float(args.frontier_commit_reached_radius_m),
            )
            frontier_target_reached_waits_for_confirm = (
                bool(getattr(args, "frontier_execution_state_enabled", True))
                and last_nav_decision is not None
                and last_nav_decision.mode == "frontier"
                and committed_frontier_execution.exists()
            )
            if target_reached and not frontier_target_reached_waits_for_confirm:
                reason = "target_radius_reached"
                target_radius_reached_steps += 1
                arrival_key = mark_frontier_terminal_for_refresh(
                    nav_decision_local=last_nav_decision,
                    current_grid_local=tuple(int(v) for v in current_grid),
                    reason="frontier_arrival",
                    status="reached" if last_nav_decision is not None and last_nav_decision.mode == "frontier" else "partial_reached",
                )
                refresh_key = _frontier_refresh_key_or_fallback(arrival_key, last_nav_decision)
                refresh_requested = request_frontier_refresh(
                    refresh_state=frontier_arrival_update_state,
                    step=int(step),
                    reason="frontier_arrival",
                    frontier_key=refresh_key,
                    block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
                )
                write_runtime_decision_trace(
                    step_idx=int(step),
                    event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
                    event_reason="frontier_arrival",
                    current_grid_local=tuple(int(v) for v in current_grid),
                    nav_decision_local=last_nav_decision,
                    recovery_action="arrival_pending_update" if refresh_requested else "duplicate_refresh_skipped",
                    current_path_len=0,
                    zero_velocity_continue=True,
                    extra=terminal_trace_metadata(selected_key=arrival_key if isinstance(arrival_key, tuple) else None),
                )
                try:
                    mapper.force_full_navigation_projection_once = True
                    mapper.voxel_grid.invalidate_navigation_projection_cache(reason="frontier_arrival")
                except Exception:
                    pass
                if frontier_commitment is not None and last_decision_mode == "frontier":
                    frontier_commitment.mark_active_reached(step, reason)
                if paper_mode:
                    long_term_goal.clear("reached")
                current_path = []
                force_perception_step = True
                full_path.append(tuple(int(v) for v in current_grid))
                last_decision_reason = reason
                if last_nav_decision is not None:
                    last_nav_decision.reason = reason
                    last_nav_decision.metadata = {
                        **dict(last_nav_decision.metadata or {}),
                        "target_radius_reached": True,
                        "target_reached_distance_m": float(target_distance_m),
                        "target_reached_radius_m": float(args.frontier_commit_reached_radius_m),
                        "frontier_refresh_pending": True,
                        "frontier_refresh_reason": "frontier_arrival",
                    }
                last_frontier_commitment_reason = reason
                last_frontier_commitment_metadata = {
                    **dict(last_frontier_commitment_metadata),
                    "frontier_commitment_reason": reason,
                    "target_radius_reached": True,
                    "target_reached_distance_m": float(target_distance_m),
                    "target_reached_radius_m": float(args.frontier_commit_reached_radius_m),
                    "frontier_refresh_pending": True,
                    "frontier_refresh_reason": "frontier_arrival",
                }
                obs = server.step_kinematic_velocity(
                    0.0,
                    0.0,
                    0.0,
                    dt=float(args.control_dt),
                    render_updates=int(args.render_updates_per_step),
                    read_rgb=detector_requires_rgb(detector, args.detector),
                    read_depth=True,
                    rgb_device="cuda" if detector_cuda_rgb else "cpu",
                )
                continue

            if (
                last_nav_decision is not None
                and last_nav_decision.mode == "frontier"
                and current_path
                and np.isfinite(float(target_distance_m))
            ):
                if bool(getattr(args, "frontier_execution_state_enabled", True)) and committed_frontier_execution.exists():
                    target_distance_m = committed_frontier_execution.update_progress(
                        tuple(int(v) for v in current_grid),
                        float(dynamic_map_info.resolution_m),
                        float(args.frontier_commit_progress_min_delta_m),
                    )
                    nav_execution_best_distance_m = float(committed_frontier_execution.best_distance_m)
                    nav_execution_no_progress_steps = int(committed_frontier_execution.no_progress_steps)
                else:
                    progress_delta = float(args.frontier_commit_progress_min_delta_m)
                    if float(target_distance_m) + progress_delta < float(nav_execution_best_distance_m):
                        nav_execution_best_distance_m = float(target_distance_m)
                        nav_execution_no_progress_steps = 0
                    else:
                        nav_execution_no_progress_steps += 1
                if nav_execution_no_progress_steps >= int(args.frontier_commit_no_progress_steps):
                    reason = "frontier_no_progress_during_execution"
                    execution_best_distance_m = float(nav_execution_best_distance_m)
                    if (
                        bool(getattr(args, "frontier_execution_state_enabled", True))
                        and committed_frontier_execution.exists()
                    ):
                        astar_clearance_for_recovery = map_state.get("astar_clearance_m")
                        if not isinstance(astar_clearance_for_recovery, np.ndarray):
                            astar_clearance_for_recovery = clearance_map_m(
                                np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                                dynamic_map_info.resolution_m,
                            )
                        if bool(committed_frontier_execution.recovery_active):
                            request_frontier_refresh_at_current(
                                nav_decision_local=last_nav_decision,
                                current_grid_local=tuple(int(v) for v in current_grid),
                                reason="frontier_recovery_no_progress_refresh",
                                status_reason=reason,
                                reached=True,
                            )
                        else:
                            apply_frontier_recovery_or_refresh(
                                nav_decision_local=last_nav_decision,
                                current_grid_local=tuple(int(v) for v in current_grid),
                                traversible_local=np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                                clearance_local=np.asarray(astar_clearance_for_recovery, dtype=np.float32),
                                planner_local=nav_planner,
                                trigger_reason="frontier_no_progress_recovery",
                                refresh_reason="frontier_no_progress_refresh",
                                partial_refresh_reason="frontier_partial_arrival_no_progress",
                                mark_failed_if_no_recovery=True,
                            )
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_execution_no_progress_steps": int(args.frontier_commit_no_progress_steps),
                            "frontier_execution_best_distance_m": float(execution_best_distance_m),
                            "frontier_execution_distance_m": float(target_distance_m),
                            "frontier_execution_no_progress_threshold": int(args.frontier_commit_no_progress_steps),
                        }
                        if paper_mode:
                            long_term_goal.invalidate(str(last_frontier_commitment_reason))
                    else:
                        long_term_goal.invalidate(reason)
                        current_path = []
                        full_path.append(tuple(int(v) for v in current_grid))
                        force_perception_step = True
                        nav_execution_progress_key = None
                        nav_execution_best_distance_m = float("inf")
                        nav_execution_no_progress_steps = 0
                        frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                        frontier_blacklisted = True
                        frontier_unreachable_recovery = True
                        frontier_unreachable_reason = reason
                        last_frontier_commitment_reason = "%s_blacklisted" % reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": last_frontier_commitment_reason,
                            "frontier_blacklisted": True,
                            "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                        }
                    last_decision_reason = str(last_frontier_commitment_reason or reason)
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue

            if bool(getattr(args, "lookahead_collision_check_enabled", True)):
                astar_clearance_for_lookahead = map_state.get("astar_clearance_m")
                if not isinstance(astar_clearance_for_lookahead, np.ndarray):
                    astar_clearance_for_lookahead = clearance_map_m(
                        np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                        dynamic_map_info.resolution_m,
                    )
                path_world = collision_checked_path_world(
                    pose,
                    current_path,
                    dynamic_map_info,
                    np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                    np.asarray(astar_clearance_for_lookahead, dtype=np.float32),
                    lookahead_m=float(getattr(args, "lookahead_m", 0.15)),
                    min_clearance_m=float(
                        getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
                    ),
                    max_skip_cells=int(getattr(args, "smoothing_max_skip_cells", 8)),
                )
                loop_tick(step, "after_collision_checked_path")
            else:
                path_world = path_cells_to_world(current_path[1: min(len(current_path), 20)], dynamic_map_info)
                loop_tick(step, "after_path_cells_to_world")
            if not path_world:
                if policy_distance_to_goal <= success_distance:
                    gt_success_region_reached = True
                    if explore_until_no_frontiers:
                        gt_success_ignored_steps += 1
                if success_region_can_finish(
                    policy_distance_to_goal,
                    success_distance,
                    require_sgnav_stop=bool(args.require_sgnav_stop),
                    policy_stop_confirmed=policy_stop_confirmed,
                    ignore_goal_success=explore_until_no_frontiers,
                ):
                    stop_called = True
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                        )
                    break
                if policy_distance_to_goal <= success_distance and bool(args.require_sgnav_stop):
                    gt_success_without_sgnav_stop_steps += 1
                    stop_blocked_reason = "sgnav_stop_required"
                    if not logged_gt_success_without_sgnav_stop:
                        print(
                            "[voxroom-loop] local path ended inside GT success radius, but VoxRoom STOP is not confirmed; continuing",
                            flush=True,
                        )
                        logged_gt_success_without_sgnav_stop = True
                if (
                    bool(getattr(args, "frontier_execution_state_enabled", True))
                    and last_decision_mode == "frontier"
                    and committed_frontier_execution.exists()
                ):
                    arrived, path_exhausted_distance_m, arrival_reason = frontier_arrival_status(
                        step=int(step),
                        current_grid=tuple(int(v) for v in current_grid),
                        execution=committed_frontier_execution,
                        resolution_m=float(dynamic_map_info.resolution_m),
                        reached_radius_m=float(args.frontier_commit_reached_radius_m),
                        min_steps_since_selection=int(getattr(args, "frontier_arrival_min_steps_since_selection", 6)),
                        confirm_steps=int(getattr(args, "frontier_arrival_confirm_steps", 2)),
                        current_path=[],
                    )
                    if arrived:
                        reason = "frontier_arrival_confirmed"
                        target_radius_reached_steps += 1
                        arrival_key = mark_frontier_terminal_for_refresh(
                            nav_decision_local=last_nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            reason="frontier_arrival",
                            status="reached",
                        )
                        refresh_key = _frontier_refresh_key_or_fallback(arrival_key, last_nav_decision)
                        committed_frontier_execution.mark_arrival_pending(int(step), refresh_key)
                        refresh_requested = request_frontier_refresh(
                            refresh_state=frontier_arrival_update_state,
                            step=int(step),
                            reason="frontier_arrival",
                            frontier_key=refresh_key,
                            block_reselect=bool(getattr(args, "frontier_refresh_blocks_reselect", True)),
                        )
                        write_runtime_decision_trace(
                            step_idx=int(step),
                            event_type="frontier_refresh_requested" if refresh_requested else "frontier_duplicate_refresh_skipped",
                            event_reason="frontier_arrival",
                            current_grid_local=tuple(int(v) for v in current_grid),
                            nav_decision_local=last_nav_decision,
                            recovery_action="arrival_pending_update" if refresh_requested else "duplicate_refresh_skipped",
                            current_path_len=0,
                            zero_velocity_continue=True,
                            extra=terminal_trace_metadata(selected_key=arrival_key),
                        )
                        if frontier_commitment is not None:
                            frontier_commitment.mark_active_reached(step, reason)
                        current_path = []
                        force_perception_step = True
                        full_path.append(tuple(int(v) for v in current_grid))
                        last_decision_reason = reason
                        last_frontier_commitment_reason = reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            **committed_frontier_execution.debug_metadata(
                                step=int(step),
                                current_grid=tuple(int(v) for v in current_grid),
                                resolution_m=float(dynamic_map_info.resolution_m),
                            ),
                            "frontier_commitment_reason": reason,
                            "frontier_arrival_status_reason": str(arrival_reason),
                            "target_reached_distance_m": float(path_exhausted_distance_m),
                            "frontier_refresh_pending": True,
                            "frontier_refresh_reason": "frontier_arrival",
                        }
                        if last_nav_decision is not None:
                            last_nav_decision.metadata = {
                                **dict(last_nav_decision.metadata or {}),
                                "frontier_refresh_pending": True,
                                "frontier_refresh_reason": "frontier_arrival",
                            }
                        obs = server.step_kinematic_velocity(
                            0.0,
                            0.0,
                            0.0,
                            dt=float(args.control_dt),
                            render_updates=int(args.render_updates_per_step),
                            read_rgb=detector_requires_rgb(detector, args.detector),
                            read_depth=True,
                            rgb_device="cuda" if detector_cuda_rgb else "cpu",
                        )
                        continue
                    if bool(committed_frontier_execution.recovery_active):
                        request_frontier_refresh_at_current(
                            nav_decision_local=last_nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            reason="frontier_recovery_arrival_path_exhausted",
                            status_reason=str(arrival_reason),
                            reached=True,
                        )
                    else:
                        astar_clearance_for_recovery = map_state.get("astar_clearance_m")
                        if not isinstance(astar_clearance_for_recovery, np.ndarray):
                            astar_clearance_for_recovery = clearance_map_m(
                                np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                                dynamic_map_info.resolution_m,
                            )
                        apply_frontier_recovery_or_refresh(
                            nav_decision_local=last_nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            traversible_local=np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                            clearance_local=np.asarray(astar_clearance_for_recovery, dtype=np.float32),
                            planner_local=nav_planner,
                            trigger_reason="frontier_path_exhausted_recovery",
                            refresh_reason="frontier_path_exhausted_confirmed_no_path",
                            partial_refresh_reason="frontier_partial_arrival_path_exhausted",
                            mark_failed_if_no_recovery=True,
                        )
                    if paper_mode:
                        long_term_goal.invalidate(str(last_frontier_commitment_reason))
                    last_decision_reason = str(last_frontier_commitment_reason)
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_arrival_status_reason": str(arrival_reason),
                        "lookahead_min_clearance_m_configured": float(getattr(args, "lookahead_min_clearance_m", 0.14)),
                        "lookahead_min_clearance_m_effective": float(
                            getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
                        ),
                    }
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                if last_decision_mode in {"candidate", "frontier"}:
                    if paper_mode:
                        long_term_goal.clear("reached")
                    current_path = []
                    force_perception_step = True
                    full_path.append(tuple(int(v) for v in current_grid))
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                        )
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        float(args.reperception_turn_wz_radps),
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                else:
                    failure_reason = "path_exhausted_without_success"
                if viz is not None:
                    viz.update(
                        step=step,
                        rgb=viz_rgb(obs),
                        detections_2d=last_detections_2d,
                        occupancy=occupancy,
                        navigable=free,
                        observed=observed,
                        goal_cells=goal_cells,
                        current_grid=current_grid,
                        pose=pose,
                        frontiers=last_frontiers,
                        nav_decision=last_nav_decision,
                        current_path=current_path,
                        full_path=full_path,
                        object_memory=object_memory,
                        goal_category=episode["goal_category"],
                        distance_to_goal=evaluator.final_distance_to_goal,
                        path_length=float(evaluator.path_accum.total_m),
                        scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                        score_debug=scenegraph.last_score_debug,
                        failure_reason=failure_reason,
                    )
                break
            raw_cmd = follower.compute_cmd(pose, path_world)
            loop_tick(step, "after_compute_cmd")
            astar_clearance_for_guard = map_state.get("astar_clearance_m")
            if not isinstance(astar_clearance_for_guard, np.ndarray):
                astar_clearance_for_guard = clearance_map_m(
                    np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                    dynamic_map_info.resolution_m,
                )
            cmd, blocked_by_online_guard, guard_debug = guard_kinematic_cmd_with_clearance(
                tuple(float(v) for v in pose),
                raw_cmd,
                float(args.control_dt),
                np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                np.asarray(astar_clearance_for_guard, dtype=np.float32),
                dynamic_map_info,
                float(getattr(args, "guard_effective_min_clearance_m", getattr(args, "guard_min_clearance_m", 0.14))),
                float(args.camera_forward_offset_m),
            )
            blocked_by_guard = bool(blocked_by_online_guard)
            last_guard_debug = dict(guard_debug or {})
            if blocked_by_online_guard and str(last_guard_debug.get("guard_block_reason")) == "low_clearance":
                guard_low_clearance_blocked_steps += 1
            loop_tick(step, "after_guard_kinematic_cmd")
            if abs(float(cmd[0])) > 1e-6 or abs(float(cmd[1])) > 1e-6 or abs(float(cmd[2])) > 1e-6:
                control_steps_with_nonzero_motion += 1
            write_runtime_control_trace(
                step_idx=int(step),
                pose_local=pose,
                current_grid_local=tuple(int(v) for v in current_grid) if current_grid is not None else None,
                current_path_local=current_path,
                path_world_local=path_world,
                raw_cmd=raw_cmd,
                guarded_cmd=cmd,
                blocked_by_guard=bool(blocked_by_guard),
                blocked_by_online_guard=bool(blocked_by_online_guard),
                guard_debug=last_guard_debug,
                target_distance_m=float(target_distance_m),
                nav_decision_local=last_nav_decision,
                frontier_execution_state=committed_frontier_execution,
            )
            write_runtime_decision_trace(
                step_idx=int(step),
                event_type="control_command_issued",
                event_reason=str(last_decision_reason),
                current_grid_local=tuple(int(v) for v in current_grid) if current_grid is not None else None,
                nav_decision_local=last_nav_decision,
                current_path_len=len(current_path),
                zero_velocity_continue=False,
                extra={
                    "raw_cmd": [float(v) for v in raw_cmd],
                    "guarded_cmd": [float(v) for v in cmd],
                    "blocked_by_guard": bool(blocked_by_guard),
                },
            )
            if blocked_by_guard:
                if (
                    bool(getattr(args, "frontier_execution_state_enabled", True))
                    and last_decision_mode == "frontier"
                    and committed_frontier_execution.exists()
                ):
                    committed_frontier_execution.guard_blocked_steps += 1
                    guard_threshold = max(1, int(getattr(args, "frontier_guard_blocked_confirm_steps", 5)))
                    if committed_frontier_execution.guard_blocked_steps < guard_threshold:
                        reason = "frontier_guard_blocked_replan_same_target"
                        current_path = []
                        full_path.append(tuple(int(v) for v in current_grid))
                        force_perception_step = False
                        last_decision_reason = reason
                        last_frontier_commitment_reason = reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            **committed_frontier_execution.debug_metadata(
                                step=int(step),
                                current_grid=tuple(int(v) for v in current_grid),
                                resolution_m=float(dynamic_map_info.resolution_m),
                            ),
                            "frontier_commitment_reason": reason,
                            "frontier_guard_blocked_confirm_steps": int(guard_threshold),
                        }
                        obs = server.step_kinematic_velocity(
                            0.0,
                            0.0,
                            0.0,
                            dt=float(args.control_dt),
                            render_updates=int(args.render_updates_per_step),
                            read_rgb=detector_requires_rgb(detector, args.detector),
                            read_depth=True,
                            rgb_device="cuda" if detector_cuda_rgb else "cpu",
                        )
                        continue
                    reason = "frontier_guard_blocked_confirmed"
                    if bool(committed_frontier_execution.recovery_active):
                        request_frontier_refresh_at_current(
                            nav_decision_local=last_nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            reason="frontier_recovery_guard_blocked_refresh",
                            status_reason=reason,
                            reached=True,
                        )
                    else:
                        astar_clearance_for_recovery = map_state.get("astar_clearance_m")
                        if not isinstance(astar_clearance_for_recovery, np.ndarray):
                            astar_clearance_for_recovery = clearance_map_m(
                                np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                                dynamic_map_info.resolution_m,
                            )
                        apply_frontier_recovery_or_refresh(
                            nav_decision_local=last_nav_decision,
                            current_grid_local=tuple(int(v) for v in current_grid),
                            traversible_local=np.asarray(map_state.get("astar_navigable", navigable), dtype=bool),
                            clearance_local=np.asarray(astar_clearance_for_recovery, dtype=np.float32),
                            planner_local=nav_planner,
                            trigger_reason="frontier_guard_blocked_recovery",
                            refresh_reason="frontier_guard_blocked_refresh",
                            partial_refresh_reason="frontier_partial_arrival_guard_blocked",
                            mark_failed_if_no_recovery=True,
                        )
                    if paper_mode:
                        long_term_goal.invalidate(str(last_frontier_commitment_reason))
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_guard_blocked_confirm_steps": int(guard_threshold),
                    }
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                if paper_mode and last_decision_mode == "frontier":
                    reason = "frontier_unreachable_stop_at_current_pose"
                    if frontier_commitment is not None:
                        frontier_commitment.mark_active_failed(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    frontier_blacklisted = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                    }
                    current_path = []
                    full_path.append(tuple(int(v) for v in current_grid))
                    force_perception_step = True
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                failure_reason = "kinematic_collision_guard"
                break
            next_step = step + 1
            next_rgb_device = rgb_request_device(next_step)
            if (
                bool(getattr(args, "frontier_execution_state_enabled", True))
                and last_decision_mode == "frontier"
                and committed_frontier_execution.exists()
            ):
                committed_frontier_execution.no_path_replans = 0
                committed_frontier_execution.guard_blocked_steps = 0
            obs = server.step_kinematic_velocity(
                cmd[0],
                cmd[1],
                cmd[2],
                dt=float(args.control_dt),
                render_updates=int(args.render_updates_per_step),
                read_rgb=next_rgb_device is not None,
                read_depth=True,
                rgb_device=next_rgb_device,
            )
            loop_tick(step, "after_step_kinematic_velocity")
        else:
            if failure_reason is None:
                failure_reason = "max_control_steps"

        loop_tick(evaluator.num_steps, "after_control_loop", reset=True)
        row = evaluator.finish(stop_called=stop_called, planner=args.planner, detector=args.detector, failure_reason=failure_reason)
        loop_tick(evaluator.num_steps, "after_evaluator_finish")
        row["sim_backend"] = "isaac"
        row["closed_loop"] = True
        row["control_mode"] = "kinematic_holonomic"
        row["stop_called"] = bool(stop_called)
        row["policy_stop_confirmed"] = bool(policy_stop_confirmed)
        row["success_requires_sgnav_stop"] = bool(args.require_sgnav_stop)
        row["gt_success_region_reached"] = bool(gt_success_region_reached)
        row["metric_pose_valid"] = bool(metric_pose_valid)
        row["metric_pose_invalid_steps"] = int(metric_pose_invalid_steps)
        row["metric_pose_last_invalid_step"] = (
            int(metric_pose_last_invalid_step) if int(metric_pose_last_invalid_step) >= 0 else None
        )
        row["metric_pose_last_invalid_reason"] = metric_pose_last_invalid_reason
        if int(metric_pose_invalid_steps) > 0:
            row["metric_valid"] = False
        row["gt_success_without_sgnav_stop_steps"] = int(gt_success_without_sgnav_stop_steps)
        row["explore_until_no_frontiers"] = bool(explore_until_no_frontiers)
        row["goal_success_ignored_steps"] = int(gt_success_ignored_steps)
        row["target_radius_reached_steps"] = int(target_radius_reached_steps)
        row["target_reached_radius_m"] = float(args.frontier_commit_reached_radius_m)
        row["stop_blocked_reason"] = stop_blocked_reason
        row["seeded_object_memory_count"] = int(seeded)
        row["object_memory_count"] = int(len(object_memory.nodes))
        row["goal_candidate_count"] = int(goal_candidate_count)
        row["sgnav_decision_mode"] = last_decision_mode
        row["sgnav_decision_reason"] = last_decision_reason
        row["scenegraph_backend"] = "original" if scenegraph.scenegraph is not None else "fallback"
        row["sgnav_state"] = getattr(decision_policy, "state", last_decision_mode)
        decision_metadata = dict(getattr(last_nav_decision, "metadata", {}) or {})
        row["sgnav_decision_metadata"] = decision_metadata
        row["target_cells_count"] = int(len(getattr(last_nav_decision, "target_cells", []) or []))
        row["selected_frontier_index"] = (
            int(last_nav_decision.frontier_decision.selected_index)
            if last_nav_decision is not None
            and last_nav_decision.frontier_decision is not None
            and last_nav_decision.frontier_decision.selected_index is not None
            else None
        )
        row["candidate_credibility"] = decision_metadata.get("candidate_credibility")
        row["candidate_reperception_steps"] = decision_metadata.get("candidate_reperception_steps")
        row["candidate_rejected"] = bool(decision_metadata.get("candidate_rejected", False))
        row["candidate_accepted"] = bool(decision_metadata.get("candidate_accepted", False))
        row["detector_confidence_threshold"] = float(args.detector_conf)
        row["min_valid_detection_confidence"] = float(args.min_valid_detection_confidence)
        row["candidate_start_min_confidence"] = float(args.candidate_start_min_confidence)
        row["candidate_start_min_hits"] = int(args.candidate_start_min_hits)
        row["candidate_standoff_max_cells"] = int(args.candidate_standoff_max_cells)
        row["detected_category_counts"] = {
            str(key): int(value)
            for key, value in sorted(detection_category_counts.items(), key=lambda item: (-item[1], item[0]))
        }
        row["goal_detection_history"] = list(goal_detection_history)
        object_memory_gnn_snapshot = fused_instance_registry.gnn_snapshot()
        row["object_memory_gnn_snapshot"] = object_memory_gnn_snapshot
        registry_raw_detections = list(object_memory_gnn_snapshot.get("raw_detections") or [])
        pre_registry_rejections = [item for item in raw_detection_debug_log if item.get("reject_reason")]
        row["raw_detection_log"] = (
            pre_registry_rejections + registry_raw_detections
            if registry_raw_detections
            else list(raw_detection_debug_log)
        )
        row["raw_rejected_detections_count"] = int(
            len([item for item in row["raw_detection_log"] if item.get("reject_reason")])
        )
        row["object_memory_tracks"] = object_memory.to_dicts()
        row["selected_candidate"] = last_selected_candidate.to_dict() if last_selected_candidate is not None else None
        row["object_memory_goal_candidates"] = [
            node.to_dict()
            for node in object_memory.nodes
            if category_matches_goal(node.category)
        ]
        row["candidate_pair_distances_m"] = goal_candidate_pair_distances(object_memory, episode["goal_category"])
        row["candidate_duplicate_warning"] = any(float(pair["dist"]) < 0.75 for pair in row["candidate_pair_distances_m"])
        row["panorama_frames"] = int(panorama_frames)
        row["requested_perception_every_steps"] = int(requested_perception_every)
        row["effective_perception_every_steps"] = int(perception_every)
        row["open_vocab_detector_every_frame"] = bool(str(getattr(args, "detector", "")).strip().lower() in {"yolo_world", "grounding_dino"} and perception_every == 1)
        row["yolo_world_every_frame"] = bool(str(getattr(args, "detector", "")).strip().lower() == "yolo_world" and perception_every == 1)
        row["graph_object_nodes"] = int(len(getattr(scenegraph, "runtime_nodes", {})))
        row["graph_group_nodes"] = int(len(getattr(scenegraph, "runtime_groups", [])))
        row["graph_room_nodes"] = int(len(getattr(scenegraph, "runtime_rooms", {})))
        row["graph_edges"] = int(len(getattr(scenegraph, "runtime_edges", [])))
        paper_graph = getattr(scenegraph, "paper_graph", None)
        row["sgnav_mode"] = str(getattr(args, "sgnav_mode", "legacy"))
        row["long_term_goal"] = {
            "mode": str(long_term_goal.mode),
            "target_cells_count": int(len(long_term_goal.target_cells)),
            "center_grid": list(long_term_goal.center_grid) if long_term_goal.center_grid is not None else None,
            "selected_step": int(long_term_goal.selected_step),
            "invalid_reason": str(long_term_goal.invalid_reason),
            "reached": bool(long_term_goal.reached),
        }
        row["frontier_target_mode"] = frontier_target_mode
        row["frontier_center_grid"] = frontier_center_grid
        row["frontier_actual_target_grid"] = frontier_actual_target_grid
        row["frontier_unreachable_recovery"] = bool(frontier_unreachable_recovery)
        row["frontier_unreachable_reason"] = frontier_unreachable_reason
        row["frontier_refresh_pending"] = bool(frontier_arrival_update_state.refresh_pending)
        row["frontier_refresh_reason"] = str(frontier_arrival_update_state.refresh_reason)
        row["frontier_refresh_consumed_step"] = int(frontier_arrival_update_state.last_update_step)
        row["frontier_reselect_blocked_until_refresh"] = bool(frontier_arrival_update_state.block_reselect_until_refresh)
        row["frontier_refresh_requested_step"] = int(frontier_arrival_update_state.refresh_requested_step)
        row["frontier_refresh_request_count"] = int(frontier_arrival_update_state.refresh_request_count)
        row["frontier_duplicate_refresh_requests"] = int(frontier_arrival_update_state.duplicate_refresh_requests)
        row["frontier_last_duplicate_refresh_step"] = int(frontier_arrival_update_state.last_duplicate_refresh_step)
        row["frontier_direct_target_unreachable_count"] = int(frontier_direct_no_path_count)
        row["frontier_recovery"] = dict(last_frontier_recovery_metadata)
        row["frontier_execution_debug"] = committed_frontier_execution.debug_metadata(
            step=int(evaluator.num_steps),
            current_grid=tuple(int(v) for v in current_grid) if current_grid is not None else None,
            resolution_m=float(dynamic_map_info.resolution_m),
        )
        frontier_terminal_debug = frontier_terminal_registry.debug_metadata()
        row["exploration_complete"] = bool(exploration_complete)
        row["exploration_complete_reason"] = str(exploration_complete_reason)
        row["exploration_stop_reason_detailed"] = str(exploration_stop_reason_detailed or exploration_complete_reason)
        row["frontier_exhaustion_audit"] = last_frontier_exhaustion_audit.to_dict()
        row["frontier_raw_count"] = int(last_frontier_raw_cells)
        row["frontier_real_count"] = int(last_frontier_real_count)
        row["frontier_near_fallback_count"] = int(last_frontier_near_fallback_count)
        row["frontier_terminal_suppressed_count"] = int(last_frontier_terminal_suppressed_count)
        row["frontier_filtered_count"] = int(last_frontier_filtered_count)
        row["frontier_terminal_registry_count"] = int(frontier_terminal_debug.get("frontier_terminal_registry_size", 0))
        row["frontier_terminal_reached_count"] = int(frontier_terminal_debug.get("frontier_terminal_reached_count", 0))
        row["frontier_terminal_partial_reached_count"] = int(
            frontier_terminal_debug.get("frontier_terminal_partial_reached_count", 0)
        )
        row["frontier_terminal_failed_count"] = int(frontier_terminal_debug.get("frontier_terminal_failed_count", 0))
        row["frontier_terminal_near_fallback_count"] = int(
            frontier_terminal_debug.get("frontier_terminal_near_fallback_count", 0)
        )
        row["frontier_terminal_suppressed_total"] = int(frontier_terminal_debug.get("frontier_terminal_suppressed_total", 0))
        row["frontier_terminal_duplicate_mark_count"] = int(
            frontier_terminal_debug.get("frontier_terminal_duplicate_mark_count", 0)
        )
        row["frontier_refresh_loop_prevented_count"] = int(frontier_refresh_loop_prevented_count)
        row["frontier_repeated_same_target_refresh_count"] = int(frontier_repeated_same_target_refresh_count)
        row["frontier_stop_at_current_grid"] = frontier_stop_at_current_grid
        row["frontier_blacklisted"] = bool(frontier_blacklisted)
        row["active_long_term_goal_mode"] = str(long_term_goal.mode)
        row["active_long_term_goal_age"] = (
            max(0, int(evaluator.num_steps) - int(long_term_goal.selected_step))
            if int(long_term_goal.selected_step) >= 0
            else None
        )
        row["paper_num_object_nodes"] = int(len(getattr(paper_graph, "object_nodes", {}) or {}))
        row["paper_num_room_nodes"] = int(len(getattr(paper_graph, "room_nodes", {}) or {}))
        row["paper_num_group_nodes"] = int(len(getattr(paper_graph, "group_nodes", {}) or {}))
        row["paper_num_object_edges"] = int(len(getattr(paper_graph, "object_edges", []) or []))
        row["paper_frontier_interpolation"] = dict(getattr(scenegraph, "last_score_debug", {}) or {})
        row["vllm_frontier_scoring"] = bool(getattr(args, "vllm_frontier_scoring", False))
        row["vllm_image_scoring"] = bool(getattr(args, "vllm_image_scoring", True))
        row["score_frontiers_before_candidate"] = bool(getattr(args, "score_frontiers_before_candidate", False))
        vllm_scorer = getattr(scenegraph, "vllm_scorer", None)
        row["vllm_last_used_image"] = bool(getattr(vllm_scorer, "last_used_image", False))
        row["vllm_num_requests"] = int(getattr(vllm_scorer, "request_count", 0))
        row["vllm_cache_hits"] = int(getattr(vllm_scorer, "cache_hit_count", 0))
        row["vllm_last_skip_reason"] = getattr(vllm_scorer, "last_skip_reason", None)
        row["vllm_last_request_frontiers"] = int(getattr(vllm_scorer, "last_request_frontiers", 0))
        row["vllm_last_response_chars"] = int(getattr(vllm_scorer, "last_response_chars", 0))
        row["vllm_disabled_reason"] = getattr(vllm_scorer, "disabled_reason", None)
        paper_llm_client = getattr(scenegraph, "paper_llm_client", None)
        hcot_scorer = getattr(scenegraph, "hcot_scorer", None)
        row["paper_llm_enabled"] = paper_llm_client is not None
        row["paper_llm_requests"] = int(getattr(paper_llm_client, "request_count", 0))
        row["hcot_llm_enabled"] = getattr(hcot_scorer, "llm_client", None) is not None
        row["hcot_llm_attempts"] = int(getattr(hcot_scorer, "llm_request_count", 0))
        row["hcot_llm_failures"] = int(getattr(hcot_scorer, "llm_failure_count", 0))
        row["hcot_llm_fallbacks"] = int(getattr(hcot_scorer, "llm_fallback_count", 0))
        row["hcot_llm_fallback_count"] = int(getattr(hcot_scorer, "llm_fallback_count", 0))
        row["hcot_llm_last_error"] = getattr(hcot_scorer, "last_error", None)
        row["hc_p_num_subgraphs_total"] = row["paper_frontier_interpolation"].get("num_subgraphs_total")
        row["hc_p_num_subgraphs_scored"] = row["paper_frontier_interpolation"].get("num_subgraphs_scored")
        row["detection_localization"] = detection_localization
        row["read_depth"] = bool(getattr(args, "read_depth", False))
        row["mapping_source"] = (
            "depth_ray_online+static_nearfield"
            if bool(getattr(args, "static_nearfield_map", False))
            else "depth_ray_online"
        )
        row["online_clearance_policy_version"] = "v38_online_only_clearance_targeting"
        row["configured_robot_radius_m"] = float(getattr(args, "configured_robot_radius_m", args.robot_radius_m))
        row["runtime_robot_radius_m"] = float(args.robot_radius_m)
        row["tiny_robot_radius_debug"] = bool(getattr(args, "tiny_robot_radius_debug", False))
        row["frontier_target_resolution_failures"] = int(frontier_target_resolution_failures)
        row["frontier_target_no_clearance_safe_count"] = int(frontier_target_no_clearance_safe_count)
        row["frontier_commit_target_mismatch_count"] = int(frontier_commit_target_mismatch_count)
        row["guard_low_clearance_blocked_steps"] = int(guard_low_clearance_blocked_steps)
        row["agent_off_static_metric_map_online_diagnostic"] = bool(int(metric_pose_invalid_steps) > 0)
        row["last_guard_debug"] = dict(last_guard_debug)
        row["online_map_resolution_m"] = float(dynamic_map_info.resolution_m)
        row["robot_radius_m"] = float(args.robot_radius_m)
        row["robot_width_m"] = float(args.robot_radius_m) * 2.0
        row["online_effective_obstacle_inflation_m"] = float(args.robot_radius_m)
        row["online_extra_inflation_radius_m_requested"] = float(args.online_inflation_radius_m)
        row["online_observed_cells"] = int(np.count_nonzero(last_dynamic_observed))
        row["online_raw_free_cells"] = int(np.count_nonzero(last_dynamic_free))
        row["online_free_cells"] = int(np.count_nonzero(last_dynamic_navigable))
        row["online_astar_extra_clearance_m"] = float(args.runtime_planning_clearance_m)
        row["online_astar_clearance_source"] = "raw_occupied"
        row["online_astar_clearance_cost_enabled"] = bool(getattr(args, "astar_clearance_cost_enabled", False))
        row["online_astar_clearance_desired_m"] = float(getattr(args, "astar_clearance_desired_m", 0.25))
        row["online_astar_goal_min_clearance_m"] = float(getattr(args, "astar_goal_min_clearance_m", 0.18))
        row["online_lookahead_min_clearance_m_configured"] = float(getattr(args, "lookahead_min_clearance_m", 0.14))
        row["online_lookahead_min_clearance_m_effective"] = float(
            getattr(args, "lookahead_effective_min_clearance_m", getattr(args, "lookahead_min_clearance_m", 0.14))
        )
        row["online_guard_min_clearance_m_configured"] = float(getattr(args, "guard_min_clearance_m", 0.14))
        row["online_guard_min_clearance_m_effective"] = float(
            getattr(args, "guard_effective_min_clearance_m", getattr(args, "guard_min_clearance_m", 0.14))
        )
        row["online_astar_path_min_clearance_m"] = map_state.get("astar_path_min_clearance_m") if isinstance(locals().get("map_state"), dict) else None
        row["online_astar_path_mean_clearance_m"] = map_state.get("astar_path_mean_clearance_m") if isinstance(locals().get("map_state"), dict) else None
        row["online_astar_path_too_close_cells"] = map_state.get("astar_path_too_close_cells") if isinstance(locals().get("map_state"), dict) else None
        row["online_astar_free_cells"] = int(np.count_nonzero(last_dynamic_astar_navigable))
        row["online_occupied_cells"] = int(np.count_nonzero(last_dynamic_occupancy))
        row["online_mapper_debug"] = dict(getattr(mapper, "last_debug_stats", {}))
        row["voxel_perf_recent_path"] = str(voxel_perf_recent_path)
        row["voxel_perf_recent_count"] = int(len(voxel_perf_recent_rows))
        row["runtime_perf_path"] = str(runtime_perf_path)
        row["runtime_decision_trace_path"] = str(runtime_decision_trace_path)
        row["frontier_initial_refresh_loop_detected"] = bool(frontier_initial_refresh_loop_detected)
        row["frontier_initial_refresh_loop_initial_grid"] = (
            [int(v) for v in initial_frontier_loop_grid] if initial_frontier_loop_grid is not None else None
        )
        row["frontier_initial_roomseg_updates_since_start"] = int(roomseg_updates_since_start)
        row["frontier_initial_selected_changes_since_start"] = int(selected_frontier_changes_since_start)
        row["frontier_initial_control_steps_with_nonzero_motion"] = int(control_steps_with_nonzero_motion)
        row["nearfield_depth"] = bool(getattr(args, "nearfield_depth", False))
        row["nearfield_mapper_debug"] = dict(getattr(mapper, "last_nearfield_debug_stats", {}))
        row["static_nearfield_map"] = bool(getattr(args, "static_nearfield_map", False))
        row["static_nearfield_radius_m"] = float(getattr(args, "static_nearfield_radius_m", 1.0))
        row["static_nearfield_mapper_debug"] = dict(getattr(mapper, "last_static_nearfield_debug_stats", {}))
        row["frontier_raw_cells"] = int(last_frontier_raw_cells)
        row["frontier_clusters"] = int(last_frontier_clusters)
        row["frontier_unknown_source"] = str(args.frontier_unknown_source)
        row["frontier_cluster_distance_mode"] = str(args.frontier_cluster_distance_mode)
        row["frontier_allow_near_fallback"] = bool(args.frontier_allow_near_fallback)
        row["frontier_min_distance_m"] = float(args.frontier_min_distance_m)
        row["frontier_selectable_failure_reason"] = decision_metadata.get("frontier_selectable_failure_reason")
        row["frontier_debug_dump"] = bool(args.frontier_debug_dump)
        row["frontier_debug_dir"] = str(args.frontier_debug_dir)
        row["frontier_commitment_enabled"] = bool(args.frontier_commitment_enabled)
        row["frontier_commitment_reason"] = str(last_frontier_commitment_reason)
        row["frontier_commitment"] = dict(last_frontier_commitment_metadata)
        row.update(
            committed_frontier_execution.debug_metadata(
                step=int(evaluator.num_steps),
                current_grid=tuple(int(v) for v in current_grid) if current_grid is not None else None,
                resolution_m=float(dynamic_map_info.resolution_m),
            )
        )
        row["active_frontier_id"] = last_frontier_commitment_metadata.get("active_frontier_id")
        row["active_frontier_age"] = last_frontier_commitment_metadata.get("active_frontier_age")
        row["active_frontier_distance_m"] = last_frontier_commitment_metadata.get("active_frontier_distance_m")
        row["frontier_scenegraph_score_norm"] = str(args.frontier_scenegraph_score_norm)
        row["room_map_mode"] = str(room_map_mode)
        row["roomseg_debug_only"] = bool(roomseg_debug_only)
        room_context_row = dict(last_room_context_metadata)
        row["room_context"] = room_context_row
        for key in (
            "room_context_source",
            "room_update_invoked_for_frontier_scoring",
            "room_segmentation_ran",
            "room_labeling_ran",
            "room_context_cache_hit",
            "room_label_count",
            "room_label_requests",
            "room_label_cache_hits",
            "room_call_order_trace",
            "room_segmentation_called_for",
            "room_segmentation_algorithm",
            "room_segmentation_step_index",
            "room_vlm_called",
            "scenegraph_updated_after_room_context",
            "frontier_scoring_after_room_context",
        ):
            row[key] = room_context_row.get(key)
        row["room_segmentation"] = _compact_json_debug(dict(last_room_segmentation_debug))
        row["room_semantics"] = _compact_json_debug(dict(last_room_semantics_debug))
        row["room_mask_count"] = int(room_context_row.get("room_mask_count", len([room for room in last_room_masks if not getattr(room, "stale", False)])) or 0)
        row["room_vlm_backend"] = str(getattr(room_labeler, "backend", "unavailable") if room_labeler is not None else "unavailable")
        row["room_vlm_requests"] = int(getattr(room_labeler, "request_count", 0) if room_labeler is not None else 0)
        row["room_vlm_failures"] = int(getattr(room_labeler, "failure_count", 0) if room_labeler is not None else 0)
        row["room_vlm_invalid_json"] = bool(getattr(room_labeler, "failure_count", 0) if room_labeler is not None else 0)
        row["reperception_state"] = build_reperception_state_payload(decision_metadata)
        row["stop_state"] = build_stop_state_payload(last_nav_decision, row)
        row["debug_graph_dump"] = bool(args.debug_graph_dump)
        row["debug_graph_dump_dir"] = str(args.debug_graph_dump_dir)
        row["segmenter"] = str(getattr(args, "segmenter", "none") or "none")
        row["camera_annotator_device"] = camera_annotator_device
        row["detector_cuda_rgb"] = bool(detector_cuda_rgb)
        row["isaac_visual_robot_proxy"] = bool(obs.get("visual_robot_proxy", False)) if isinstance(obs, dict) else False
        row["isaac_robot_pose_sync_failures"] = int(obs.get("robot_pose_sync_failures", 0)) if isinstance(obs, dict) else 0
        row["isaac_camera_pose_sync_failures"] = int(obs.get("camera_pose_sync_failures", 0)) if isinstance(obs, dict) else 0
        row["isaac_nearfield_camera_pose_sync_failures"] = (
            int(obs.get("nearfield_camera_pose_sync_failures", 0)) if isinstance(obs, dict) else 0
        )
        row["voxroom_viz_every_steps"] = int(viz_every)
        row["max_vx_mps"] = float(args.max_vx_mps)
        row["max_vy_mps"] = float(args.max_vy_mps)
        row["max_wz_radps"] = float(args.max_wz_radps)
        row["perception_latency_ms"] = float(latency_totals_ms["perception"])
        row["mapping_latency_ms"] = float(latency_totals_ms["mapping"])
        row["graph_latency_ms"] = float(latency_totals_ms["graph"])
        row["llm_latency_ms"] = float(latency_totals_ms["llm"])
        row["planning_latency_ms"] = float(latency_totals_ms["planning"])
        row["latency_counts"] = {key: int(value) for key, value in latency_counts.items()}
        row["mapping_latency_breakdown_ms"] = {
            str(key): float(value) for key, value in sorted(mapping_breakdown_totals_ms.items())
        }
        row["mapping_latency_breakdown_counts"] = {
            str(key): int(value) for key, value in sorted(mapping_breakdown_counts.items())
        }
        row["mapping_latency_breakdown_avg_ms"] = {
            str(key): float(mapping_breakdown_totals_ms[key] / max(1, mapping_breakdown_counts.get(key, 0)))
            for key in sorted(mapping_breakdown_totals_ms)
        }
        for key, value in row["mapping_latency_breakdown_ms"].items():
            row["mapping_%s" % str(key)] = float(value)
        if args.save_debug_video or args.debug_map:
            debug_map = args.debug_map or str(Path(args.output).with_suffix(".png"))
            start = world_xy_to_grid(float(start_pose[0]), float(start_pose[1]), dynamic_map_info)
            save_map_png(debug_map, last_dynamic_occupancy, last_dynamic_navigable, start=start, goals=goal_cells, path_cells=full_path)
        loop_tick(evaluator.num_steps, "after_row_payload_build")
        row = complete_result_row(row, args)
        loop_tick(evaluator.num_steps, "after_complete_result_row")
        if roomseg_debug_only:
            row["metric_valid"] = False
            row["detector_backend"] = "none"
            row["segmenter_backend"] = "none"
            row["llm_backend"] = "disabled_roomseg_debug_only"
            row["policy_name"] = "roomseg_debug_only"
            row["sgnav_decision_mode"] = "roomseg_debug_only"
            row["sgnav_decision_reason"] = "roomseg_debug_only_no_frontier_or_policy_decision"
        row = make_jsonable(row)
        loop_tick(evaluator.num_steps, "after_make_jsonable")
        summary_row = final_log_row(row)
        JsonlEpisodeLogger(args.output).log(row)
        loop_tick(evaluator.num_steps, "after_jsonl_log")
        print(json.dumps(summary_row, ensure_ascii=False), flush=True)
        args._row_already_logged = True
        if args.hold_open:
            print("[isaac-loop] episode finished; hold-open enabled, close the Isaac window or press Ctrl+C", flush=True)
            while True:
                server.app.update()
        return row
    except BaseException:
        import traceback

        print("[voxroom-loop] exception before Isaac shutdown:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if "live_baseline_manager" in locals():
            live_baseline_manager.on_episode_end()
        detector = getattr(args, "_detector_instance", None)
        if detector is not None and hasattr(detector, "close"):
            detector.close()
            args._detector_instance = None
            args._detector_key = None
        if viz is not None:
            viz.close()
        if not args.hold_open:
            server.close()


def run_episode_map_sim(episode: dict, args) -> dict:
    scene_dir, map_info, occupancy, navigable = load_preprocessed_for_episode(episode)
    navigable = apply_episode_planning_clearance(
        scene_dir,
        map_info,
        navigable,
        episode,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    ensure_detector_loaded(args, scene_dir)
    planner = GridAStarPlanner(navigable, map_info.resolution_m, allow_diagonal=True)
    start = (int(episode["start_grid"][0]), int(episode["start_grid"][1]))
    goals = [(int(r), int(c)) for r, c in episode["goal_regions_grid"]]
    planning_started_at = time.perf_counter()
    result = planner.plan(start, goals)
    planning_latency_ms = max(0.0, (time.perf_counter() - planning_started_at) * 1000.0)
    evaluator = EpisodeEvaluator(episode, planner)
    env = MapSimHabitatLikeEnv(args.episode_file, args.episode_index)
    env.reset()

    object_memory = ObjectMemory(min_valid_confidence=float(args.min_valid_detection_confidence))
    scenegraph = SGNavSceneGraphAdapter(
        args.sgnav_repo,
        use_original=args.use_original_scenegraph,
        semantic_priors_path=getattr(args, "semantic_priors_path", None),
        llm_config={
            "enabled": bool(getattr(args, "llm_enabled", False)),
            "base_url": getattr(args, "llm_base_url", None),
            "model": getattr(args, "llm_model", None),
            "api_key": getattr(args, "llm_api_key", None),
            "timeout_s": float(getattr(args, "llm_timeout_s", 30.0)),
            "temperature": float(getattr(args, "llm_temperature", 0.0)),
            "max_tokens": int(getattr(args, "llm_max_tokens", 512)),
            "max_hcot_subgraphs_per_decision": int(getattr(args, "max_hcot_subgraphs_per_decision", 8)),
        },
        sgnav_mode=str(getattr(args, "sgnav_mode", "legacy")),
    )
    scenegraph.reset(episode["goal_category"])
    scenegraph.update(object_memory)
    decision = SGNavDecision(scenegraph)
    _ = decision  # kept to make the VoxRoom decision adapter part of the run loop surface.

    failure_reason = None
    stop_called = False
    if not result.path:
        failure_reason = "astar_no_path"
        evaluator.final_distance_to_goal = math.inf
    else:
        for cell in result.path:
            x, y = grid_to_world_xy(cell[0], cell[1], map_info)
            pose = [x, y, episode["start_pose_world"][2], episode["start_pose_world"][3]]
            env.set_pose(pose)
            evaluator.update_pose(pose, cell, collided=False)
        stop_called = True

    row = evaluator.finish(stop_called=stop_called, planner=args.planner, detector=args.detector, failure_reason=failure_reason)
    row["sim_backend"] = "map"
    row["map_source"] = "static_preprocessed"
    row["policy_name"] = "%s_static_map_baseline" % str(args.planner)
    row["object_memory_count"] = 0
    row["goal_candidate_count"] = 0
    row["frontier_count"] = 0
    row["selected_frontier"] = None
    row["detector_confidence_threshold"] = float(args.detector_conf)
    row["min_valid_detection_confidence"] = float(args.min_valid_detection_confidence)
    row["planning_latency_ms"] = float(planning_latency_ms)
    row = complete_result_row(row, args)
    if args.save_debug_video or args.debug_map:
        debug_map = args.debug_map or str(Path(args.output).with_suffix(".png"))
        save_map_png(debug_map, occupancy, navigable, start=start, goals=goals, path_cells=result.path)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/voxroom_online.yaml")
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--success-distance-m", type=float, default=None)
    parser.add_argument("--planner", default=None, choices=["astar"])
    parser.add_argument("--detector", default=None, choices=["dry_run", "none"])
    parser.add_argument("--yolo-world-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grounding-dino-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grounding-dino-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grounding-dino-device", default=None, choices=["cpu", "cuda"], help=argparse.SUPPRESS)
    parser.add_argument("--detector-conf", type=float, default=None)
    parser.add_argument("--min-valid-detection-confidence", type=float, default=None)
    parser.add_argument("--detector-iou", type=float, default=None)
    parser.add_argument("--reject-edge-touching-bboxes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bbox-edge-margin-px", type=float, default=None)
    parser.add_argument("--bbox-edge-margin-ratio", type=float, default=None)
    parser.add_argument("--headless", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--sim-backend", default=None, choices=["map", "isaac"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug-map", default=None)
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--strict-benchmark", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-strict-benchmark", dest="strict_benchmark", action="store_false")
    parser.add_argument("--allow-debug-fallbacks", action="store_true", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--ablation-name", default=None)
    parser.add_argument("--sgnav-repo", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sgnav-mode", default=None, choices=["legacy", "paper"], help=argparse.SUPPRESS)
    parser.add_argument("--use-original-scenegraph", action="store_true", default=None)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument(
        "--explore-until-no-frontiers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Debug/probe mode: ignore goal success radius and keep selecting frontiers until none remain or max-control-steps is reached.",
    )
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--replan-every-steps", type=int, default=None)
    parser.add_argument("--path-trim-max-distance-cells", "--path_trim_max_distance_cells", type=int, default=None)
    parser.add_argument("--allow-path-replan-during-roomseg-freeze", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--perception-every-steps", type=int, default=None)
    parser.add_argument("--observed-radius-m", type=float, default=None)
    parser.add_argument("--lookahead-m", type=float, default=None)
    parser.add_argument("--max-vx-mps", type=float, default=None)
    parser.add_argument("--max-vy-mps", type=float, default=None)
    parser.add_argument("--max-wz-radps", type=float, default=None)
    parser.add_argument("--render-updates-per-step", type=int, default=None)
    parser.add_argument("--isaac-width", type=int, default=None)
    parser.add_argument("--isaac-height", type=int, default=None)
    parser.add_argument("--camera-hfov-deg", type=float, default=None)
    parser.add_argument("--camera-mast-height-m", type=float, default=None)
    parser.add_argument("--camera-forward-offset-m", type=float, default=None)
    parser.add_argument("--camera-pitch-deg", type=float, default=None)
    parser.add_argument("--camera-near-m", type=float, default=None)
    parser.add_argument("--camera-far-m", type=float, default=None)
    parser.add_argument("--camera-annotator-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--read-depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nearfield-depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nearfield-width", type=int, default=None)
    parser.add_argument("--nearfield-height", type=int, default=None)
    parser.add_argument("--nearfield-hfov-deg", type=float, default=None)
    parser.add_argument("--nearfield-height-m", type=float, default=None)
    parser.add_argument("--nearfield-near-m", type=float, default=None)
    parser.add_argument("--nearfield-far-m", type=float, default=None)
    parser.add_argument("--nearfield-radius-m", type=float, default=None)
    parser.add_argument("--nearfield-ignore-radius-m", type=float, default=None)
    parser.add_argument("--nearfield-depth-stride-px", type=int, default=None)
    parser.add_argument("--nearfield-floor-tolerance-m", type=float, default=None)
    parser.add_argument("--nearfield-obstacle-min-height-m", type=float, default=None)
    parser.add_argument("--nearfield-obstacle-max-height-m", type=float, default=None)
    parser.add_argument("--nearfield-splat-point-threshold", type=int, default=None)
    parser.add_argument("--nearfield-free-splat-point-threshold", type=int, default=None)
    parser.add_argument("--static-nearfield-map", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--static-nearfield-radius-m", type=float, default=None)
    parser.add_argument("--panorama-steps", type=int, default=None)
    parser.add_argument("--panorama-wz-radps", type=float, default=None)
    parser.add_argument("--panorama-render-updates-per-step", type=int, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-stride-px", type=int, default=None)
    parser.add_argument("--online-map-size-m", type=float, default=None)
    parser.add_argument("--online-resolution-m", type=float, default=None)
    parser.add_argument("--obstacle-min-height-m", type=float, default=None)
    parser.add_argument("--obstacle-max-height-m", type=float, default=None)
    parser.add_argument("--free-min-height-m", type=float, default=None)
    parser.add_argument("--free-max-height-m", type=float, default=None)
    parser.add_argument("--splat-point-threshold", type=int, default=None)
    parser.add_argument("--free-splat-point-threshold", type=int, default=None)
    parser.add_argument("--mapping-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-min-cluster-size", type=int, default=None, help="Deprecated; frontier extraction uses per-cell frontiers without clustering.")
    parser.add_argument("--frontier-min-distance-m", type=float, default=None)
    parser.add_argument("--frontier-max-count", type=int, default=None)
    parser.add_argument("--frontier-obstacle-dilation-radius-cells", type=int, default=None)
    parser.add_argument("--frontier-unknown-dilation-radius-cells", type=int, default=None)
    parser.add_argument("--frontier-unknown-source", "--frontier_unknown_source", default=None, choices=["observed", "implicit"])
    parser.add_argument("--frontier-source", "--frontier_source", default=None, choices=["navigation", "vertical_free", "height_profile_vertical_free", "voxel_vertical_free"])
    parser.add_argument(
        "--frontier-vertical-free-require-navigation-reachable",
        "--frontier_vertical_free_require_navigation_reachable",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--frontier-cluster-distance-mode",
        "--frontier_cluster_distance_mode",
        default=None,
        choices=["mean", "min", "center"],
    )
    parser.add_argument("--frontier-allow-near-fallback", "--frontier_allow_near_fallback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--frontier-near-fallback-counts-as-exploration-frontier",
        "--frontier_near_fallback_counts_as_exploration_frontier",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--frontier-terminal-match-radius-m", "--frontier_terminal_match_radius_m", type=float, default=None)
    parser.add_argument("--frontier-terminal-target-match-radius-m", "--frontier_terminal_target_match_radius_m", type=float, default=None)
    parser.add_argument("--frontier-terminal-suppress-steps", "--frontier_terminal_suppress_steps", type=int, default=None)
    parser.add_argument("--frontier-terminal-reached-suppress-steps", "--frontier_terminal_reached_suppress_steps", type=int, default=None)
    parser.add_argument("--frontier-terminal-partial-reached-suppress-steps", "--frontier_terminal_partial_reached_suppress_steps", type=int, default=None)
    parser.add_argument("--frontier-terminal-failed-unreachable-suppress-steps", "--frontier_terminal_failed_unreachable_suppress_steps", type=int, default=None)
    parser.add_argument("--frontier-terminal-stale-near-fallback-suppress-steps", "--frontier_terminal_stale_near_fallback_suppress_steps", type=int, default=None)
    parser.add_argument("--frontier-terminal-require-target-match-when-record-has-target", "--frontier_terminal_require_target_match_when_record_has_target", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-terminal-allow-center-only-match-for-targetless-records", "--frontier_terminal_allow_center_only_match_for_targetless_records", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-terminal-ignore-targetless-query-for-targeted-record", "--frontier_terminal_ignore_targetless_query_for_targeted_record", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-terminal-unsuppress-partial-when-all-filtered", "--frontier_terminal_unsuppress_partial_when_all_filtered", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-terminal-unsuppress-failed-when-all-filtered", "--frontier_terminal_unsuppress_failed_when_all_filtered", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-terminal-unsuppress-if-raw-frontier-still-exists", "--frontier_terminal_unsuppress_if_raw_frontier_still_exists", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-terminal-raw-alive-radius-m", "--frontier_terminal_raw_alive_radius_m", type=float, default=None)
    parser.add_argument("--roomseg-snapshot-failure-loop-min-interval", "--roomseg_snapshot_failure_loop_min_interval", type=int, default=None)
    parser.add_argument("--frontier-debug-dump", "--frontier_debug_dump", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-debug-dir", "--frontier_debug_dir", default=None)
    parser.add_argument("--frontier-commitment-enabled", "--frontier_commitment_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-commit-match-radius-m", "--frontier_commit_match_radius_m", type=float, default=None)
    parser.add_argument("--frontier-commit-reached-radius-m", "--frontier_commit_reached_radius_m", type=float, default=None)
    parser.add_argument("--frontier-commit-min-steps", "--frontier_commit_min_steps", type=int, default=None)
    parser.add_argument("--frontier-commit-max-steps", "--frontier_commit_max_steps", type=int, default=None)
    parser.add_argument("--frontier-commit-switch-margin", "--frontier_commit_switch_margin", type=float, default=None)
    parser.add_argument("--frontier-commit-switch-ratio", "--frontier_commit_switch_ratio", type=float, default=None)
    parser.add_argument("--frontier-commit-no-progress-steps", "--frontier_commit_no_progress_steps", type=int, default=None)
    parser.add_argument("--frontier-commit-progress-min-delta-m", "--frontier_commit_progress_min_delta_m", type=float, default=None)
    parser.add_argument("--frontier-blacklist-ttl-steps", "--frontier_blacklist_ttl_steps", type=int, default=None)
    parser.add_argument("--frontier-initial-refresh-loop-raise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--robot-radius-m", type=float, default=None)
    parser.add_argument("--allow-tiny-robot-radius-debug", action="store_true", default=False)
    parser.add_argument(
        "--online-inflation-radius-m",
        type=float,
        default=None,
        help="Deprecated compatibility option; online traversal inflation is footprint-only.",
    )
    parser.add_argument("--room-map-mode", default=None)
    parser.add_argument(
        "--roomseg-debug-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run only online mapping plus the current room segmentation backend; skip detector, scene graph, frontier scoring, and local planning.",
    )
    parser.add_argument("--debug-roomseg-layers", action="store_true", default=None)
    parser.add_argument("--roomseg-debug-layers", dest="debug_roomseg_layers", action="store_true", default=None)
    parser.add_argument(
        "--save-roomseg-snapshots",
        action="store_true",
        default=None,
        help="Save lightweight per-roomseg npz, summary, and unlabeled navigation mask images without enabling debug layer dumps.",
    )
    parser.add_argument("--save-cached-roomseg-snapshots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--save-roomseg-voxel-evidence",
        action="store_true",
        default=False,
        help="When saving roomseg snapshots, also store the full 3D voxel occupancy state and z metadata in each snapshot npz.",
    )
    parser.add_argument("--debug-roomseg-dir", default=None)
    parser.add_argument("--roomseg-debug-dir", dest="debug_roomseg_dir", default=None)
    parser.add_argument("--roomseg-snapshot-dir", default=None)
    parser.add_argument("--debug-roomseg-max-saves", type=int, default=None)
    parser.add_argument("--roomseg-snapshot-max-saves", type=int, default=None)
    parser.add_argument("--live-roomseg-baseline", choices=["none", "topology_visual_active", "tvars_original_isaac"], default="none")
    parser.add_argument("--live-baseline-output-dir", type=str, default=None)
    parser.add_argument("--live-baseline-save-stream", action="store_true", default=False)
    parser.add_argument("--live-baseline-save-every-snapshot", action="store_true", default=False)
    parser.add_argument(
        "--live-baseline-door-detector",
        choices=["original_detr", "disabled", "isaac_semantic_debug"],
        default="original_detr",
    )
    parser.add_argument(
        "--live-baseline-policy-control",
        choices=["never"],
        default="never",
        help="For fair room segmentation experiments, live baselines must not control exploration.",
    )
    parser.add_argument("--live-baseline-panorama-views", type=int, default=12)
    parser.add_argument("--roomseg-roomseg-depth-stride-px", type=int, default=None)
    parser.add_argument("--roomseg-disable-corridor-cuts", action="store_true", default=False)
    parser.add_argument("--roomseg-disable-doorway-cuts", action="store_true", default=False)
    parser.add_argument("--roomseg-disable-wall-completion", action="store_true", default=False)
    parser.add_argument(
        "--roomseg-backend",
        default=None,
        choices=[
            VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
            *sorted(VOXEL_OCCUPANCY_ROOMSEG_LEGACY_BACKENDS),
        ],
    )
    parser.add_argument(
        "--roomseg-finalization-mode",
        default=None,
        choices=[
            "no_merge",
            "proposal_only",
            "premerge_proposals",
            "doorway_constrained_merge",
            "no_merge_until_source_backend_verified",
            "no_merge_until_geometry_verified",
        ],
    )
    parser.add_argument("--enable-roomseg-nav-free-overlay", dest="roomseg_nav_free_overlay", action="store_true", default=None)
    parser.add_argument("--disable-roomseg-nav-free-overlay", dest="roomseg_nav_free_overlay", action="store_false")
    parser.add_argument("--enable-frontier-room-known-free-side", dest="frontier_room_known_free_side", action="store_true", default=None)
    parser.add_argument("--disable-frontier-room-known-free-side", dest="frontier_room_known_free_side", action="store_false")
    parser.add_argument("--roomseg-wall-gating-fix", action="store_true", default=None)
    parser.add_argument("--room-label-backend", default=None, choices=["vlm", "deterministic_debug", "unavailable"])
    parser.add_argument("--room-label-min-confidence", type=float, default=None)
    parser.add_argument("--room-label-ambiguity-margin", type=float, default=None)
    parser.add_argument("--room-label-min-reliable-objects", type=int, default=None)
    parser.add_argument("--room-label-unknown-category", default=None)
    parser.add_argument("--max-room-objects-in-prompt", type=int, default=None)
    parser.add_argument("--detection-localization", default=None, choices=["static_map_ray", "map_ray", "rgb_map_ray", "depth", "none"])
    parser.add_argument("--min-depth-points-per-detection", type=int, default=None)
    parser.add_argument("--segmenter", default=None, choices=["none"])
    parser.add_argument("--sam2-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sam2-model-cfg", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sam2-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-detections-per-frame", type=int, default=None)
    parser.add_argument("--object-merge-radius-m", type=float, default=None)
    parser.add_argument("--instance-merge-distance-m", type=float, default=None)
    parser.add_argument("--instance-merge-iou-3d", type=float, default=None)
    parser.add_argument("--partial-class-weight", type=float, default=None)
    parser.add_argument("--min-geometry-confidence", type=float, default=None)
    parser.add_argument("--partial-stability-min-observations", type=int, default=None)
    parser.add_argument("--mask-iou-association-threshold", type=float, default=None)
    parser.add_argument("--mask-containment-track-match-threshold", type=float, default=None)
    parser.add_argument("--footprint-iou-association-threshold", type=float, default=None)
    parser.add_argument("--child-containment-threshold", type=float, default=None)
    parser.add_argument("--child-area-ratio-threshold", type=float, default=None)
    parser.add_argument("--frontier-distance-weight", type=float, default=None)
    parser.add_argument("--frontier-scenegraph-score-norm", "--frontier_scenegraph_score_norm", default=None, choices=["none", "minmax", "zscore"])
    parser.add_argument("--frontier-selection-mode", "--frontier_selection_mode", default=None, choices=["nearest", "random"])
    parser.add_argument("--frontier-random-seed", "--frontier_random_seed", type=int, default=None)
    parser.add_argument("--semantic-priors-path", "--semantic_priors_path", default=None)
    parser.add_argument("--debug-graph-dump", "--debug_graph_dump", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-graph-dump-dir", "--debug_graph_dump_dir", default=None)
    parser.add_argument("--runtime-planning-clearance-m", type=float, default=None)
    parser.add_argument("--astar-clearance-cost-enabled", "--astar_clearance_cost_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--astar-clearance-desired-m", "--astar_clearance_desired_m", type=float, default=None)
    parser.add_argument("--astar-clearance-hard-min-m", "--astar_clearance_hard_min_m", type=float, default=None)
    parser.add_argument("--astar-clearance-weight", "--astar_clearance_weight", type=float, default=None)
    parser.add_argument("--astar-clearance-power", "--astar_clearance_power", type=float, default=None)
    parser.add_argument("--astar-goal-min-clearance-m", "--astar_goal_min_clearance_m", type=float, default=None)
    parser.add_argument("--astar-goal-search-radius-m", "--astar_goal_search_radius_m", type=float, default=None)
    parser.add_argument("--guard-min-clearance-m", "--guard_min_clearance_m", type=float, default=None)
    parser.add_argument("--collision-checked-smoothing-enabled", "--collision_checked_smoothing_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoothing-min-clearance-m", "--smoothing_min_clearance_m", type=float, default=None)
    parser.add_argument("--smoothing-max-skip-cells", "--smoothing_max_skip_cells", type=int, default=None)
    parser.add_argument("--lookahead-collision-check-enabled", "--lookahead_collision_check_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lookahead-min-clearance-m", "--lookahead_min_clearance_m", type=float, default=None)
    parser.add_argument("--candidate-min-detector-hits", type=int, default=None)
    parser.add_argument("--candidate-start-min-confidence", type=float, default=None)
    parser.add_argument("--candidate-start-min-hits", "--candidate_start_min_hits", type=int, default=None)
    parser.add_argument("--candidate-recent-max-age-steps", "--candidate_recent_max_age_steps", type=int, default=None)
    parser.add_argument("--candidate-match-substring", "--candidate_match_substring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--candidate-accept-requires-reperception",
        "--candidate_accept_requires_reperception",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--candidate-reject-ttl-steps", "--candidate_reject_ttl_steps", type=int, default=None)
    parser.add_argument("--candidate-accept-threshold", "--candidate_accept_threshold", type=float, default=None)
    parser.add_argument("--candidate-stop-distance-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-min-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-max-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-max-cells", "--candidate_standoff_max_cells", type=int, default=None)
    parser.add_argument("--candidate-standoff-ideal-m", "--candidate_standoff_ideal_m", type=float, default=None)
    parser.add_argument("--reperception-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reperception-min-observations", type=int, default=None)
    parser.add_argument("--reperception-max-steps", type=int, default=None)
    parser.add_argument("--reperception-same-goal-radius-m", type=float, default=None)
    parser.add_argument("--reperception-turn-wz-radps", type=float, default=None)
    parser.add_argument("--stop-verification-steps", type=int, default=None)
    parser.add_argument("--stop-verification-min-hits", type=int, default=None)
    parser.add_argument("--found-goal-stop-distance-m", type=float, default=None)
    parser.add_argument("--require-sgnav-stop", action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--llm-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout-s", type=float, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--max-hcot-subgraphs-per-decision", type=int, default=None)
    parser.add_argument("--vllm-frontier-scoring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--vllm-timeout-s", type=float, default=None)
    parser.add_argument("--vllm-temperature", type=float, default=None)
    parser.add_argument("--vllm-max-frontiers", type=int, default=None)
    parser.add_argument("--vllm-image-scoring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-image-max-width", type=int, default=None)
    parser.add_argument("--vllm-image-jpeg-quality", type=int, default=None)
    parser.add_argument("--score-frontiers-before-candidate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed-gt-object-memory", action="store_true", default=None)
    parser.add_argument("--no-seed-gt-object-memory", dest="seed_gt_object_memory", action="store_false")
    parser.add_argument("--allow-gt-goal-fallback", action="store_true", default=None)
    parser.add_argument("--no-allow-gt-goal-fallback", dest="allow_gt_goal_fallback", action="store_false")
    parser.add_argument("--voxroom-viz", dest="sgnav_viz", action="store_true", default=None)
    parser.add_argument("--no-voxroom-viz", dest="sgnav_viz", action="store_false")
    parser.add_argument("--voxroom-viz-every-steps", dest="voxroom_viz_every_steps", type=int, default=None, metavar="STEPS")
    parser.add_argument("--voxroom-viz-save-dir", dest="voxroom_viz_save_dir", default=None, metavar="DIR")
    parser.add_argument("--voxroom-viz-save-every-steps", dest="voxroom_viz_save_every_steps", type=int, default=None, metavar="STEPS")
    parser.add_argument("--voxroom-viz-width", dest="voxroom_viz_width", type=int, default=None, metavar="PX")
    parser.add_argument("--voxroom-viz-height", dest="voxroom_viz_height", type=int, default=None, metavar="PX")
    parser.add_argument("--voxroom-viz-jpeg-quality", dest="voxroom_viz_jpeg_quality", type=int, default=None, metavar="Q")
    parser.add_argument("--debug-overlay-layers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-overlay-layer-metadata", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-gt-goal-cells", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-room-proposals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-room-masks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-room-labels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--voxroom-viz-map-base-layer", dest="voxroom_viz_map_base_layer", default=None, choices=["default", "vertical_free"])
    parser.add_argument("--show-frontier-member-cells", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-object-nodes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-candidate-markers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-green-like-primitives-before-warning", type=int, default=None)
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    args.planner = args.planner or get_nested(cfg, "repo.planner", "astar")
    args.detector = args.detector or get_nested(cfg, "repo.detector", "dry_run")
    args.yolo_world_model = args.yolo_world_model or get_nested(cfg, "paths.yolo_world_model", get_nested(cfg, "perception.yolo_world_model", "data/models/yolov8l-worldv2.pt"))
    args.grounding_dino_checkpoint = args.grounding_dino_checkpoint or os.environ.get("GROUNDING_DINO_CHECKPOINT") or get_nested(
        cfg,
        "paths.grounding_dino_checkpoint",
        get_nested(cfg, "perception.grounding_dino.checkpoint", "data/models/groundingdino_swinb_cogcoor.pth"),
    )
    args.grounding_dino_config = args.grounding_dino_config or os.environ.get("GROUNDING_DINO_CONFIG") or get_nested(
        cfg,
        "paths.grounding_dino_config",
        get_nested(cfg, "perception.grounding_dino.config", ""),
    )
    args.grounding_dino_text_threshold = float(
        args.grounding_dino_text_threshold
        if args.grounding_dino_text_threshold is not None
        else get_nested(cfg, "perception.grounding_dino.text_threshold", 0.25)
    )
    args.grounding_dino_device = str(
        args.grounding_dino_device
        if args.grounding_dino_device is not None
        else get_nested(cfg, "perception.grounding_dino.device", "cuda")
    )
    args.min_valid_detection_confidence = float(
        args.min_valid_detection_confidence
        if args.min_valid_detection_confidence is not None
        else get_nested(cfg, "perception.min_valid_detection_confidence", MIN_VALID_DETECTION_CONFIDENCE)
    )
    args.detector_conf = max(
        float(args.detector_conf if args.detector_conf is not None else get_nested(cfg, "perception.confidence_threshold", 0.7)),
        float(args.min_valid_detection_confidence),
    )
    args.detector_iou = float(args.detector_iou if args.detector_iou is not None else get_nested(cfg, "perception.nms_iou_threshold", 0.5))
    args.reject_edge_touching_bboxes = bool(
        args.reject_edge_touching_bboxes
        if args.reject_edge_touching_bboxes is not None
        else get_nested(cfg, "perception.yolo_world.reject_edge_touching_bboxes", True)
    )
    args.bbox_edge_margin_px = float(
        args.bbox_edge_margin_px
        if args.bbox_edge_margin_px is not None
        else get_nested(cfg, "perception.yolo_world.bbox_edge_margin_px", 2)
    )
    args.bbox_edge_margin_ratio = float(
        args.bbox_edge_margin_ratio
        if args.bbox_edge_margin_ratio is not None
        else get_nested(cfg, "perception.yolo_world.bbox_edge_margin_ratio", 0.0)
    )
    args.partial_class_weight = float(
        args.partial_class_weight
        if args.partial_class_weight is not None
        else get_nested(cfg, "object_memory.partial_class_weight", 0.25)
    )
    args.min_geometry_confidence = float(
        args.min_geometry_confidence
        if args.min_geometry_confidence is not None
        else get_nested(cfg, "object_memory.min_geometry_confidence", 0.50)
    )
    args.partial_stability_min_observations = int(
        args.partial_stability_min_observations
        if args.partial_stability_min_observations is not None
        else get_nested(cfg, "object_memory.partial_stability_min_observations", 3)
    )
    args.mask_iou_association_threshold = float(
        args.mask_iou_association_threshold
        if args.mask_iou_association_threshold is not None
        else get_nested(cfg, "object_memory.mask_iou_association_threshold", get_nested(cfg, "object_memory.mask_iou_track_match_threshold", 0.25))
    )
    args.mask_containment_track_match_threshold = float(
        args.mask_containment_track_match_threshold
        if args.mask_containment_track_match_threshold is not None
        else get_nested(cfg, "object_memory.mask_containment_track_match_threshold", 0.60)
    )
    args.footprint_iou_association_threshold = float(
        args.footprint_iou_association_threshold
        if args.footprint_iou_association_threshold is not None
        else get_nested(cfg, "object_memory.footprint_iou_association_threshold", get_nested(cfg, "object_memory.footprint_iou_track_match_threshold", 0.20))
    )
    args.child_containment_threshold = float(
        args.child_containment_threshold
        if args.child_containment_threshold is not None
        else get_nested(cfg, "object_memory.child_containment_threshold", 0.70)
    )
    args.child_area_ratio_threshold = float(
        args.child_area_ratio_threshold
        if args.child_area_ratio_threshold is not None
        else get_nested(cfg, "object_memory.child_area_ratio_threshold", get_nested(cfg, "object_memory.child_object_area_ratio_max", 0.35))
    )
    args.headless = bool(get_nested(cfg, "isaac.headless", True) if args.headless is None else args.headless)
    viz_cfg = get_nested(cfg, "visualization.voxroom_popup", get_nested(cfg, "visualization.voxroom_popup", "auto"))
    if args.sgnav_viz is None:
        if str(viz_cfg).strip().lower() == "auto":
            args.sgnav_viz = not args.headless
        else:
            args.sgnav_viz = bool(str_to_bool(viz_cfg))
    args.voxroom_viz_every_steps = int(args.voxroom_viz_every_steps or get_nested(cfg, "visualization.voxroom_popup_every_steps", get_nested(cfg, "visualization.voxroom_popup_every_steps", 20)))
    args.voxroom_viz_save_dir = args.voxroom_viz_save_dir or get_nested(cfg, "visualization.voxroom_popup_save_dir", get_nested(cfg, "visualization.voxroom_popup_save_dir", None))
    args.voxroom_viz_save_every_steps = int(args.voxroom_viz_save_every_steps or get_nested(cfg, "visualization.voxroom_popup_save_every_steps", get_nested(cfg, "visualization.voxroom_popup_save_every_steps", 10)))
    args.voxroom_viz_width = int(args.voxroom_viz_width or get_nested(cfg, "visualization.voxroom_popup_width", get_nested(cfg, "visualization.voxroom_popup_width", 1440)))
    args.voxroom_viz_height = int(args.voxroom_viz_height or get_nested(cfg, "visualization.voxroom_popup_height", get_nested(cfg, "visualization.voxroom_popup_height", 900)))
    args.voxroom_viz_jpeg_quality = int(args.voxroom_viz_jpeg_quality or get_nested(cfg, "visualization.voxroom_popup_jpeg_quality", get_nested(cfg, "visualization.voxroom_popup_jpeg_quality", 75)))
    args.debug_overlay_layers = bool(
        args.debug_overlay_layers
        if args.debug_overlay_layers is not None
        else get_nested(cfg, "visualization.debug_overlay_layers", True)
    )
    args.save_overlay_layer_metadata = bool(
        args.save_overlay_layer_metadata
        if args.save_overlay_layer_metadata is not None
        else get_nested(cfg, "visualization.save_overlay_layer_metadata", True)
    )
    args.show_gt_goal_cells = bool(
        args.show_gt_goal_cells
        if args.show_gt_goal_cells is not None
        else get_nested(cfg, "visualization.show_gt_goal_cells", False)
    )
    args.show_room_proposals = bool(
        args.show_room_proposals
        if args.show_room_proposals is not None
        else get_nested(cfg, "visualization.show_room_proposals", True)
    )
    args.show_room_masks = bool(
        args.show_room_masks
        if args.show_room_masks is not None
        else get_nested(cfg, "visualization.show_room_masks", True)
    )
    args.show_room_labels = bool(
        args.show_room_labels
        if args.show_room_labels is not None
        else get_nested(cfg, "visualization.show_room_labels", True)
    )
    args.show_frontier_member_cells = bool(
        args.show_frontier_member_cells
        if args.show_frontier_member_cells is not None
        else get_nested(cfg, "visualization.show_frontier_member_cells", True)
    )
    args.show_object_nodes = bool(
        args.show_object_nodes
        if args.show_object_nodes is not None
        else get_nested(cfg, "visualization.show_object_nodes", True)
    )
    args.show_candidate_markers = bool(
        args.show_candidate_markers
        if args.show_candidate_markers is not None
        else get_nested(cfg, "visualization.show_candidate_markers", True)
    )
    args.max_green_like_primitives_before_warning = int(
        args.max_green_like_primitives_before_warning
        if args.max_green_like_primitives_before_warning is not None
        else get_nested(cfg, "visualization.max_green_like_primitives_before_warning", 200)
    )
    args.sgnav_mode = str(args.sgnav_mode or get_nested(cfg, "sgnav.mode", "legacy")).strip().lower()
    args.strict_benchmark = bool(
        args.strict_benchmark
        if args.strict_benchmark is not None
        else get_nested(cfg, "benchmark.strict_benchmark", True)
    )
    args.allow_debug_fallbacks = bool(
        args.allow_debug_fallbacks
        if args.allow_debug_fallbacks is not None
        else get_nested(cfg, "benchmark.allow_debug_fallbacks", False)
    )
    args.policy = args.policy or get_nested(cfg, "benchmark.policy", None)
    args.ablation_name = args.ablation_name or get_nested(cfg, "benchmark.ablation_name", None)
    args.sim_backend = args.sim_backend or get_nested(cfg, "repo.backend", "map")
    args.output = args.output or str(Path(get_nested(cfg, "project.output_dir", "data/isaac_bench_runs")) / "run_one_episode" / "results.jsonl")
    args.sgnav_repo = args.sgnav_repo or get_nested(cfg, "paths.sgnav_repo", "")
    if args.use_original_scenegraph is None:
        args.use_original_scenegraph = bool(get_nested(cfg, "repo.use_original_scenegraph", False))
    args.max_control_steps = int(args.max_control_steps or get_nested(cfg, "episodes.max_steps", 500))
    args.control_dt = float(args.control_dt or get_nested(cfg, "isaac.control_dt", 0.2))
    args.replan_every_steps = int(args.replan_every_steps or get_nested(cfg, "astar.replan_every_steps", 5))
    args.path_trim_max_distance_cells = int(
        args.path_trim_max_distance_cells
        if args.path_trim_max_distance_cells is not None
        else get_nested(cfg, "astar.path_trim_max_distance_cells", get_nested(cfg, "navigation.path_trim_max_distance_cells", 3))
    )
    args.allow_path_replan_during_roomseg_freeze = bool(
        args.allow_path_replan_during_roomseg_freeze
        if args.allow_path_replan_during_roomseg_freeze is not None
        else get_nested(cfg, "astar.allow_path_replan_during_roomseg_freeze", True)
    )
    args.perception_every_steps = int(args.perception_every_steps or get_nested(cfg, "isaac.perception_every_steps", 10))
    args.observed_radius_m = float(args.observed_radius_m or get_nested(cfg, "isaac.observed_radius_m", 3.0))
    args.lookahead_m = float(args.lookahead_m or get_nested(cfg, "astar.waypoint_lookahead_m", 0.15))
    args.max_vx_mps = float(args.max_vx_mps if args.max_vx_mps is not None else get_nested(cfg, "robot.max_vx_mps", 0.15))
    args.max_vy_mps = float(args.max_vy_mps if args.max_vy_mps is not None else get_nested(cfg, "robot.max_vy_mps", 0.15))
    args.max_wz_radps = float(args.max_wz_radps if args.max_wz_radps is not None else get_nested(cfg, "robot.max_wz_radps", 0.35))
    args.render_updates_per_step = int(args.render_updates_per_step or get_nested(cfg, "isaac.render_updates_per_step", 2))
    args.isaac_width = int(args.isaac_width or get_nested(cfg, "camera.width", 640))
    args.isaac_height = int(args.isaac_height or get_nested(cfg, "camera.height", 480))
    args.camera_hfov_deg = float(args.camera_hfov_deg or get_nested(cfg, "camera.hfov_deg", 110.0))
    args.camera_mast_height_m = float(args.camera_mast_height_m or get_nested(cfg, "camera.mast_height_m", 1.35))
    args.camera_forward_offset_m = float(args.camera_forward_offset_m if args.camera_forward_offset_m is not None else get_nested(cfg, "camera.forward_offset_m", 0.0))
    args.camera_pitch_deg = float(args.camera_pitch_deg if args.camera_pitch_deg is not None else get_nested(cfg, "camera.pitch_deg", 0.0))
    args.camera_near_m = float(args.camera_near_m if args.camera_near_m is not None else get_nested(cfg, "camera.near_m", 0.02))
    args.camera_far_m = float(args.camera_far_m if args.camera_far_m is not None else get_nested(cfg, "camera.far_m", 10.0))
    args.camera_annotator_device = str(args.camera_annotator_device or get_nested(cfg, "isaac.camera_annotator_device", "cuda")).strip().lower()
    args.live_roomseg_baseline = str(getattr(args, "live_roomseg_baseline", "none") or "none").strip().lower()
    args.live_baseline_policy_control = str(getattr(args, "live_baseline_policy_control", "never") or "never").strip().lower()
    if args.live_roomseg_baseline != "none" and args.live_baseline_policy_control != "never":
        raise ValueError("live roomseg baseline policy control must be 'never'")
    args.live_baseline_panorama_views = int(getattr(args, "live_baseline_panorama_views", 12) or 12)
    args.read_depth = bool(args.read_depth if args.read_depth is not None else get_nested(cfg, "isaac.read_depth", False))
    if args.live_roomseg_baseline != "none":
        args.read_depth = True
    args.nearfield_depth = bool(args.nearfield_depth if args.nearfield_depth is not None else get_nested(cfg, "nearfield_depth.enabled", False))
    args.nearfield_width = int(args.nearfield_width if args.nearfield_width is not None else get_nested(cfg, "nearfield_depth.width", 192))
    args.nearfield_height = int(args.nearfield_height if args.nearfield_height is not None else get_nested(cfg, "nearfield_depth.height", 192))
    args.nearfield_hfov_deg = float(args.nearfield_hfov_deg if args.nearfield_hfov_deg is not None else get_nested(cfg, "nearfield_depth.hfov_deg", 115.0))
    args.nearfield_height_m = float(args.nearfield_height_m if args.nearfield_height_m is not None else get_nested(cfg, "nearfield_depth.height_m", 1.15))
    args.nearfield_near_m = float(args.nearfield_near_m if args.nearfield_near_m is not None else get_nested(cfg, "nearfield_depth.near_m", 0.02))
    args.nearfield_far_m = float(args.nearfield_far_m if args.nearfield_far_m is not None else get_nested(cfg, "nearfield_depth.far_m", 1.8))
    args.nearfield_radius_m = float(args.nearfield_radius_m if args.nearfield_radius_m is not None else get_nested(cfg, "nearfield_depth.radius_m", 1.2))
    args.nearfield_ignore_radius_m = float(args.nearfield_ignore_radius_m if args.nearfield_ignore_radius_m is not None else get_nested(cfg, "nearfield_depth.ignore_radius_m", 0.14))
    args.nearfield_depth_stride_px = int(args.nearfield_depth_stride_px if args.nearfield_depth_stride_px is not None else get_nested(cfg, "nearfield_depth.depth_stride_px", 3))
    args.nearfield_floor_tolerance_m = float(args.nearfield_floor_tolerance_m if args.nearfield_floor_tolerance_m is not None else get_nested(cfg, "nearfield_depth.floor_tolerance_m", 0.12))
    args.nearfield_obstacle_min_height_m = float(args.nearfield_obstacle_min_height_m if args.nearfield_obstacle_min_height_m is not None else get_nested(cfg, "nearfield_depth.obstacle_min_height_m", 0.18))
    args.nearfield_obstacle_max_height_m = float(args.nearfield_obstacle_max_height_m if args.nearfield_obstacle_max_height_m is not None else get_nested(cfg, "nearfield_depth.obstacle_max_height_m", 0.90))
    args.nearfield_splat_point_threshold = int(args.nearfield_splat_point_threshold if args.nearfield_splat_point_threshold is not None else get_nested(cfg, "nearfield_depth.splat_point_threshold", 2))
    args.nearfield_free_splat_point_threshold = int(args.nearfield_free_splat_point_threshold if args.nearfield_free_splat_point_threshold is not None else get_nested(cfg, "nearfield_depth.free_splat_point_threshold", 1))
    args.static_nearfield_map = bool(
        args.static_nearfield_map
        if args.static_nearfield_map is not None
        else get_nested(cfg, "nearfield_static_map.enabled", False)
    )
    args.static_nearfield_radius_m = float(
        args.static_nearfield_radius_m
        if args.static_nearfield_radius_m is not None
        else get_nested(cfg, "nearfield_static_map.radius_m", 1.0)
    )
    args.panorama_steps = int(args.panorama_steps if args.panorama_steps is not None else get_nested(cfg, "sgnav.panorama_steps", 8))
    if args.live_roomseg_baseline == "topology_visual_active" and args.panorama_steps < int(args.live_baseline_panorama_views):
        args.panorama_steps = int(args.live_baseline_panorama_views)
    args.panorama_wz_radps = float(args.panorama_wz_radps if args.panorama_wz_radps is not None else get_nested(cfg, "sgnav.panorama_wz_radps", 0.8))
    args.panorama_render_updates_per_step = int(args.panorama_render_updates_per_step if args.panorama_render_updates_per_step is not None else get_nested(cfg, "sgnav.panorama_render_updates_per_step", get_nested(cfg, "isaac.render_updates_per_step", 1)))
    args.depth_max_m = float(args.depth_max_m if args.depth_max_m is not None else get_nested(cfg, "mapping.depth_max_m", 5.0))
    args.depth_min_m = float(args.depth_min_m if args.depth_min_m is not None else get_nested(cfg, "mapping.depth_min_m", 0.20))
    args.depth_stride_px = int(args.depth_stride_px if args.depth_stride_px is not None else get_nested(cfg, "mapping.depth_stride_px", 8))
    args.online_map_size_m = float(args.online_map_size_m if args.online_map_size_m is not None else get_nested(cfg, "mapping.map_size_m", 40.0))
    args.online_resolution_m = float(args.online_resolution_m if args.online_resolution_m is not None else get_nested(cfg, "mapping.online_resolution_m", get_nested(cfg, "scene_preprocess.map_resolution_m", 0.05)))
    args.obstacle_min_height_m = float(args.obstacle_min_height_m if args.obstacle_min_height_m is not None else get_nested(cfg, "mapping.obstacle_min_height_m", 0.20))
    args.obstacle_max_height_m = float(args.obstacle_max_height_m if args.obstacle_max_height_m is not None else get_nested(cfg, "mapping.obstacle_max_height_m", 1.50))
    args.free_min_height_m = float(args.free_min_height_m if args.free_min_height_m is not None else get_nested(cfg, "mapping.free_min_height_m", -1.50))
    args.free_max_height_m = float(args.free_max_height_m if args.free_max_height_m is not None else get_nested(cfg, "mapping.free_max_height_m", 0.10))
    args.splat_point_threshold = int(args.splat_point_threshold if args.splat_point_threshold is not None else get_nested(cfg, "mapping.splat_point_threshold", 6))
    args.free_splat_point_threshold = int(
        args.free_splat_point_threshold
        if args.free_splat_point_threshold is not None
        else get_nested(cfg, "mapping.free_splat_point_threshold", 1)
    )
    args.mapping_debug = bool(args.mapping_debug if args.mapping_debug is not None else get_nested(cfg, "mapping.debug", False))
    args.frontier_min_cluster_size = int(args.frontier_min_cluster_size if args.frontier_min_cluster_size is not None else get_nested(cfg, "mapping.frontier_min_cluster_size", 1))
    args.frontier_min_distance_m = float(args.frontier_min_distance_m if args.frontier_min_distance_m is not None else get_nested(cfg, "mapping.frontier_min_distance_m", 1.0))
    args.frontier_max_count = int(args.frontier_max_count if args.frontier_max_count is not None else get_nested(cfg, "mapping.frontier_max_count", 0))
    args.frontier_obstacle_dilation_radius_cells = int(
        args.frontier_obstacle_dilation_radius_cells
        if args.frontier_obstacle_dilation_radius_cells is not None
        else get_nested(cfg, "mapping.frontier_obstacle_dilation_radius_cells", 4)
    )
    args.frontier_unknown_dilation_radius_cells = int(
        args.frontier_unknown_dilation_radius_cells
        if args.frontier_unknown_dilation_radius_cells is not None
        else get_nested(cfg, "mapping.frontier_unknown_dilation_radius_cells", 1)
    )
    args.frontier_unknown_source = str(args.frontier_unknown_source or get_nested(cfg, "mapping.frontier_unknown_source", "observed"))
    if args.frontier_unknown_source not in {"observed", "implicit"}:
        raise ValueError("mapping.frontier_unknown_source must be 'observed' or 'implicit'")
    args.frontier_source = str(args.frontier_source or get_nested(cfg, "mapping.frontier_source", "navigation")).strip().lower()
    if args.frontier_source not in {"navigation", "vertical_free", "height_profile_vertical_free", "voxel_vertical_free"}:
        raise ValueError("mapping.frontier_source must be 'navigation', 'vertical_free', 'height_profile_vertical_free', or 'voxel_vertical_free'")
    args.frontier_vertical_free_require_navigation_reachable = bool(
        args.frontier_vertical_free_require_navigation_reachable
        if args.frontier_vertical_free_require_navigation_reachable is not None
        else get_nested(cfg, "mapping.frontier_vertical_free_require_navigation_reachable", True)
    )
    args.frontier_cluster_distance_mode = str(
        args.frontier_cluster_distance_mode or get_nested(cfg, "mapping.frontier_cluster_distance_mode", "mean")
    )
    if args.frontier_cluster_distance_mode not in {"mean", "min", "center"}:
        raise ValueError("mapping.frontier_cluster_distance_mode must be 'mean', 'min', or 'center'")
    args.frontier_allow_near_fallback = bool(
        args.frontier_allow_near_fallback
        if args.frontier_allow_near_fallback is not None
        else get_nested(cfg, "mapping.frontier_allow_near_fallback", False)
    )
    args.frontier_near_fallback_counts_as_exploration_frontier = bool(
        args.frontier_near_fallback_counts_as_exploration_frontier
        if args.frontier_near_fallback_counts_as_exploration_frontier is not None
        else get_nested(cfg, "sgnav.frontier_near_fallback_counts_as_exploration_frontier", False)
    )
    args.frontier_terminal_match_radius_m = float(
        args.frontier_terminal_match_radius_m
        if args.frontier_terminal_match_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_terminal_match_radius_m", 0.25)
    )
    args.frontier_terminal_target_match_radius_m = float(
        args.frontier_terminal_target_match_radius_m
        if args.frontier_terminal_target_match_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_terminal_target_match_radius_m", 0.10)
    )
    args.frontier_terminal_suppress_steps = int(
        args.frontier_terminal_suppress_steps
        if args.frontier_terminal_suppress_steps is not None
        else get_nested(cfg, "sgnav.frontier_terminal_suppress_steps", 1000000000)
    )
    args.frontier_terminal_reached_suppress_steps = int(
        args.frontier_terminal_reached_suppress_steps
        if args.frontier_terminal_reached_suppress_steps is not None
        else get_nested(cfg, "sgnav.frontier_terminal_reached_suppress_steps", 240)
    )
    args.frontier_terminal_partial_reached_suppress_steps = int(
        args.frontier_terminal_partial_reached_suppress_steps
        if args.frontier_terminal_partial_reached_suppress_steps is not None
        else get_nested(cfg, "sgnav.frontier_terminal_partial_reached_suppress_steps", 30)
    )
    args.frontier_terminal_failed_unreachable_suppress_steps = int(
        args.frontier_terminal_failed_unreachable_suppress_steps
        if args.frontier_terminal_failed_unreachable_suppress_steps is not None
        else get_nested(cfg, "sgnav.frontier_terminal_failed_unreachable_suppress_steps", 90)
    )
    args.frontier_terminal_stale_near_fallback_suppress_steps = int(
        args.frontier_terminal_stale_near_fallback_suppress_steps
        if args.frontier_terminal_stale_near_fallback_suppress_steps is not None
        else get_nested(cfg, "sgnav.frontier_terminal_stale_near_fallback_suppress_steps", 60)
    )
    args.frontier_terminal_require_target_match_when_record_has_target = bool(
        args.frontier_terminal_require_target_match_when_record_has_target
        if args.frontier_terminal_require_target_match_when_record_has_target is not None
        else get_nested(cfg, "sgnav.frontier_terminal_require_target_match_when_record_has_target", True)
    )
    args.frontier_terminal_allow_center_only_match_for_targetless_records = bool(
        args.frontier_terminal_allow_center_only_match_for_targetless_records
        if args.frontier_terminal_allow_center_only_match_for_targetless_records is not None
        else get_nested(cfg, "sgnav.frontier_terminal_allow_center_only_match_for_targetless_records", True)
    )
    args.frontier_terminal_ignore_targetless_query_for_targeted_record = bool(
        args.frontier_terminal_ignore_targetless_query_for_targeted_record
        if args.frontier_terminal_ignore_targetless_query_for_targeted_record is not None
        else get_nested(cfg, "sgnav.frontier_terminal_ignore_targetless_query_for_targeted_record", True)
    )
    args.frontier_terminal_unsuppress_partial_when_all_filtered = bool(
        args.frontier_terminal_unsuppress_partial_when_all_filtered
        if args.frontier_terminal_unsuppress_partial_when_all_filtered is not None
        else get_nested(cfg, "sgnav.frontier_terminal_unsuppress_partial_when_all_filtered", True)
    )
    args.frontier_terminal_unsuppress_failed_when_all_filtered = bool(
        args.frontier_terminal_unsuppress_failed_when_all_filtered
        if args.frontier_terminal_unsuppress_failed_when_all_filtered is not None
        else get_nested(cfg, "sgnav.frontier_terminal_unsuppress_failed_when_all_filtered", True)
    )
    args.frontier_terminal_unsuppress_if_raw_frontier_still_exists = bool(
        args.frontier_terminal_unsuppress_if_raw_frontier_still_exists
        if args.frontier_terminal_unsuppress_if_raw_frontier_still_exists is not None
        else get_nested(cfg, "sgnav.frontier_terminal_unsuppress_if_raw_frontier_still_exists", True)
    )
    args.frontier_terminal_raw_alive_radius_m = float(
        args.frontier_terminal_raw_alive_radius_m
        if args.frontier_terminal_raw_alive_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_terminal_raw_alive_radius_m", 0.30)
    )
    args.roomseg_snapshot_failure_loop_min_interval = int(
        args.roomseg_snapshot_failure_loop_min_interval
        if args.roomseg_snapshot_failure_loop_min_interval is not None
        else get_nested(cfg, "sgnav.roomseg_snapshot_failure_loop_min_interval", 50)
    )
    args.frontier_debug_dump = bool(
        args.frontier_debug_dump
        if args.frontier_debug_dump is not None
        else get_nested(cfg, "mapping.frontier_debug_dump", False)
    )
    args.frontier_debug_dir = str(args.frontier_debug_dir or get_nested(cfg, "mapping.frontier_debug_dir", "data/debug_frontier"))
    args.frontier_commitment_enabled = bool(
        args.frontier_commitment_enabled
        if args.frontier_commitment_enabled is not None
        else get_nested(cfg, "sgnav.frontier_commitment_enabled", True)
    )
    args.frontier_commit_match_radius_m = float(
        args.frontier_commit_match_radius_m
        if args.frontier_commit_match_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_commit_match_radius_m", 0.75)
    )
    args.frontier_commit_reached_radius_m = float(
        args.frontier_commit_reached_radius_m
        if args.frontier_commit_reached_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_commit_reached_radius_m", 0.20)
    )
    args.frontier_commit_min_steps = int(
        args.frontier_commit_min_steps
        if args.frontier_commit_min_steps is not None
        else get_nested(cfg, "sgnav.frontier_commit_min_steps", 12)
    )
    args.frontier_commit_max_steps = int(
        args.frontier_commit_max_steps
        if args.frontier_commit_max_steps is not None
        else get_nested(cfg, "sgnav.frontier_commit_max_steps", 120)
    )
    args.frontier_commit_switch_margin = float(
        args.frontier_commit_switch_margin
        if args.frontier_commit_switch_margin is not None
        else get_nested(cfg, "sgnav.frontier_commit_switch_margin", 0.25)
    )
    args.frontier_commit_switch_ratio = float(
        args.frontier_commit_switch_ratio
        if args.frontier_commit_switch_ratio is not None
        else get_nested(cfg, "sgnav.frontier_commit_switch_ratio", 1.15)
    )
    args.frontier_commit_no_progress_steps = int(
        args.frontier_commit_no_progress_steps
        if args.frontier_commit_no_progress_steps is not None
        else get_nested(cfg, "sgnav.frontier_commit_no_progress_steps", 25)
    )
    args.frontier_commit_progress_min_delta_m = float(
        args.frontier_commit_progress_min_delta_m
        if args.frontier_commit_progress_min_delta_m is not None
        else get_nested(cfg, "sgnav.frontier_commit_progress_min_delta_m", 0.10)
    )
    args.frontier_blacklist_ttl_steps = int(
        args.frontier_blacklist_ttl_steps
        if args.frontier_blacklist_ttl_steps is not None
        else get_nested(cfg, "sgnav.frontier_blacklist_ttl_steps", 100)
    )
    args.frontier_execution_state_enabled = bool(get_nested(cfg, "sgnav.frontier_execution_state_enabled", True))
    args.frontier_arrival_min_steps_since_selection = int(
        get_nested(cfg, "sgnav.frontier_arrival_min_steps_since_selection", 6)
    )
    args.frontier_arrival_confirm_steps = int(get_nested(cfg, "sgnav.frontier_arrival_confirm_steps", 2))
    args.frontier_same_target_max_no_path_replans = int(
        get_nested(cfg, "sgnav.frontier_same_target_max_no_path_replans", 3)
    )
    args.frontier_guard_blocked_confirm_steps = int(
        get_nested(cfg, "sgnav.frontier_guard_blocked_confirm_steps", 5)
    )
    args.frontier_arrival_update_stop_steps = int(get_nested(cfg, "sgnav.frontier_arrival_update_stop_steps", 1))
    args.frontier_arrival_blacklist_on_reached = bool(
        get_nested(cfg, "sgnav.frontier_arrival_blacklist_on_reached", False)
    )
    args.frontier_roomseg_update_assert_no_bypass = bool(
        get_nested(cfg, "sgnav.frontier_roomseg_update_assert_no_bypass", True)
    )
    recovery_cfg = dict(get_nested(cfg, "sgnav.frontier_unreachable_recovery", {}) or {})
    args.frontier_unreachable_recovery_enabled = bool(recovery_cfg.get("enabled", True))
    args.frontier_unreachable_recovery_local_search_radius_m = float(recovery_cfg.get("local_search_radius_m", 1.50))
    args.frontier_unreachable_recovery_global_search_enabled = bool(recovery_cfg.get("global_search_enabled", True))
    args.frontier_unreachable_recovery_global_max_frontier_distance_m = float(
        recovery_cfg.get("global_max_frontier_distance_m", 3.00)
    )
    args.frontier_unreachable_recovery_min_clearance_m = float(
        recovery_cfg.get("min_clearance_m", get_nested(cfg, "frontier_targeting.min_goal_clearance_m", 0.18))
    )
    args.frontier_unreachable_recovery_min_approach_improvement_m = float(
        recovery_cfg.get("min_approach_improvement_m", 0.20)
    )
    args.frontier_unreachable_recovery_allow_current_as_partial_arrival = bool(
        recovery_cfg.get("allow_current_as_partial_arrival", False)
    )
    args.frontier_unreachable_recovery_current_partial_arrival_radius_m = float(
        recovery_cfg.get("current_partial_arrival_radius_m", 0.35)
    )
    args.frontier_unreachable_recovery_current_partial_arrival_min_motion_m = float(
        recovery_cfg.get("current_partial_arrival_min_motion_m", 0.30)
    )
    args.frontier_unreachable_recovery_current_partial_arrival_min_steps_since_selection = int(
        recovery_cfg.get("current_partial_arrival_min_steps_since_selection", 6)
    )
    args.frontier_unreachable_recovery_max_attempts_per_frontier = int(
        recovery_cfg.get("max_recovery_attempts_per_frontier", 1)
    )
    args.frontier_unreachable_recovery_max_debug_candidates = int(recovery_cfg.get("max_debug_candidates", 64))
    args.frontier_switch_requires_refresh = bool(get_nested(cfg, "sgnav.frontier_switch_requires_refresh", True))
    args.frontier_failure_requires_refresh = bool(get_nested(cfg, "sgnav.frontier_failure_requires_refresh", True))
    args.frontier_partial_arrival_requires_refresh = bool(
        get_nested(cfg, "sgnav.frontier_partial_arrival_requires_refresh", True)
    )
    args.frontier_refresh_blocks_reselect = bool(get_nested(cfg, "sgnav.frontier_refresh_blocks_reselect", True))
    args.frontier_initial_refresh_loop_raise = bool(
        args.frontier_initial_refresh_loop_raise
        if args.frontier_initial_refresh_loop_raise is not None
        else get_nested(cfg, "sgnav.frontier_initial_refresh_loop_raise", False)
    )
    default_robot_radius_m = get_nested(cfg, "robot.footprint_radius_m", None)
    if default_robot_radius_m is None:
        default_robot_radius_m = 0.5 * float(get_nested(cfg, "robot.footprint_width_m", 0.28))
    configured_robot_radius_m = float(default_robot_radius_m)
    planning_robot_radius_cfg = get_nested(cfg, "astar.planning_robot_radius_m", None)
    default_planning_robot_radius_m = (
        configured_robot_radius_m
        if planning_robot_radius_cfg is None
        else float(planning_robot_radius_cfg)
    )
    args.robot_radius_m = float(args.robot_radius_m if args.robot_radius_m is not None else default_planning_robot_radius_m)
    args.configured_robot_radius_m = float(configured_robot_radius_m)
    args.allow_tiny_robot_radius_debug = bool(
        bool(getattr(args, "allow_tiny_robot_radius_debug", False))
        or bool(get_nested(cfg, "astar.allow_tiny_robot_radius_debug", False))
    )
    args.tiny_robot_radius_debug = bool(float(args.robot_radius_m) + 1e-6 < float(configured_robot_radius_m))
    if (
        bool(get_nested(cfg, "astar.reject_runtime_robot_radius_smaller_than_config", True))
        and bool(args.tiny_robot_radius_debug)
        and not bool(args.allow_tiny_robot_radius_debug)
    ):
        raise ValueError(
            "runtime robot_radius_m %.3f is smaller than configured robot.footprint_radius_m %.3f. "
            "This makes online navigation too permissive. Remove --robot-radius-m override or set "
            "--allow-tiny-robot-radius-debug for explicit debug-only runs."
            % (float(args.robot_radius_m), float(configured_robot_radius_m))
        )
    args.online_inflation_radius_m = float(args.online_inflation_radius_m if args.online_inflation_radius_m is not None else get_nested(cfg, "mapping.inflation_radius_m", 0.0))
    args.room_map_mode = str(args.room_map_mode or get_nested(cfg, "mapping.room_map_mode", VOXEL_OCCUPANCY_ROOMSEG_CONTEXT))
    args.voxel_runtime_config = dict(get_nested(cfg, "mapping.voxel_runtime", {}) or {})
    args.runtime_debug_full_arrays = bool(args.voxel_runtime_config.get("runtime_debug_full_arrays", False))
    args.runtime_debug_full_arrays_on_viz = bool(args.voxel_runtime_config.get("runtime_debug_full_arrays_on_viz", True))
    args.runtime_debug_full_arrays_on_snapshot = bool(args.voxel_runtime_config.get("runtime_debug_full_arrays_on_snapshot", True))
    args.room_segmentation_config, args.voxel_cpu_numba_thread_override = apply_voxel_cpu_numba_thread_override(
        dict(get_nested(cfg, "mapping.room_segmentation", {}) or {})
    )
    roomseg_debug_layers_cfg = dict(args.room_segmentation_config.get("debug_layers", {}) or {})
    roomseg_overlay_cfg = dict(args.room_segmentation_config.get("navigation_free_context_overlay", {}) or {})
    roomseg_frontier_context_cfg = dict(args.room_segmentation_config.get("frontier_room_context", {}) or {})
    roomseg_frontier_update_cfg = dict(args.room_segmentation_config.get("frontier_update_gate", {}) or {})
    roomseg_wall_gating_fix_cfg = dict(args.room_segmentation_config.get("wall_gating_fix", {}) or {})
    if args.roomseg_backend is not None:
        args.room_segmentation_config["backend"] = str(args.roomseg_backend)
    if args.roomseg_finalization_mode is not None:
        args.room_segmentation_config["finalization_mode"] = str(args.roomseg_finalization_mode)
    if args.debug_roomseg_layers is not None:
        roomseg_debug_layers_cfg["enabled"] = bool(args.debug_roomseg_layers)
    if args.debug_roomseg_dir is not None:
        roomseg_debug_layers_cfg["output_dir"] = str(args.debug_roomseg_dir)
    if args.debug_roomseg_max_saves is not None:
        roomseg_debug_layers_cfg["max_saves"] = int(args.debug_roomseg_max_saves)
    if args.roomseg_nav_free_overlay is not None:
        roomseg_overlay_cfg["enabled"] = bool(args.roomseg_nav_free_overlay)
    if args.frontier_room_known_free_side is not None:
        roomseg_frontier_context_cfg["use_known_free_side"] = bool(args.frontier_room_known_free_side)
        roomseg_frontier_context_cfg["enabled"] = bool(args.frontier_room_known_free_side)
    if args.roomseg_wall_gating_fix is not None:
        roomseg_wall_gating_fix_cfg["enabled"] = bool(args.roomseg_wall_gating_fix)
    if args.roomseg_roomseg_depth_stride_px is not None:
        top_depth_cfg = dict(args.room_segmentation_config.get("depth", {}) or {})
        top_depth_cfg["roomseg_depth_stride_px"] = int(args.roomseg_roomseg_depth_stride_px)
        args.room_segmentation_config["depth"] = top_depth_cfg
    if args.roomseg_disable_corridor_cuts:
        args.room_segmentation_config["enable_step2_corridor_separators"] = False
    if args.roomseg_disable_doorway_cuts:
        args.room_segmentation_config["enable_step1_door_separators"] = False
    if args.roomseg_disable_wall_completion:
        args.room_segmentation_config["enable_projected_wall_completion"] = False
    args.room_segmentation_config["debug_layers"] = roomseg_debug_layers_cfg
    args.room_segmentation_config["navigation_free_context_overlay"] = roomseg_overlay_cfg
    args.room_segmentation_config["frontier_room_context"] = roomseg_frontier_context_cfg
    args.room_segmentation_config["frontier_update_gate"] = roomseg_frontier_update_cfg
    args.room_segmentation_config["wall_gating_fix"] = roomseg_wall_gating_fix_cfg
    args.debug_roomseg_layers = bool(roomseg_debug_layers_cfg.get("enabled", False))
    args.debug_roomseg_dir = str(roomseg_debug_layers_cfg.get("output_dir", "debug/roomseg_layers"))
    args.debug_roomseg_max_saves = int(roomseg_debug_layers_cfg.get("max_saves", 50))
    args.save_roomseg_snapshots = bool(args.save_roomseg_snapshots)
    args.roomseg_snapshot_dir = str(args.roomseg_snapshot_dir or "result/roomseg_snapshots")
    args.roomseg_snapshot_max_saves = int(args.roomseg_snapshot_max_saves if args.roomseg_snapshot_max_saves is not None else 500)
    if args.debug_roomseg_layers:
        for nested_key in ("voxel_occupancy_door_wall",):
            nested_cfg = dict(args.room_segmentation_config.get(nested_key, {}) or {})
            nested_cfg["debug_dump"] = True
            nested_cfg.setdefault("debug_dir", args.debug_roomseg_dir)
            args.room_segmentation_config[nested_key] = nested_cfg
    args.roomseg_backend = str(args.room_segmentation_config.get("backend", VOXEL_OCCUPANCY_ROOMSEG_BACKEND))
    args.roomseg_finalization_mode = str(args.room_segmentation_config.get("finalization_mode", "no_merge_until_source_backend_verified"))
    args.roomseg_nav_free_overlay = bool(roomseg_overlay_cfg.get("enabled", False))
    args.frontier_room_known_free_side = bool(roomseg_frontier_context_cfg.get("enabled", True) and roomseg_frontier_context_cfg.get("use_known_free_side", True))
    args.roomseg_frontier_update_policy = str(roomseg_frontier_update_cfg.get("policy", "at_frontier_arrival"))
    args.freeze_roomseg_and_frontiers_during_navigation = bool(roomseg_frontier_update_cfg.get("freeze_during_navigation", True))
    args.force_roomseg_frontier_update = bool(roomseg_frontier_update_cfg.get("debug_force_update", False))
    args.roomseg_frontier_update_on_target_invalidated = bool(roomseg_frontier_update_cfg.get("update_on_target_invalidated", False))
    args.roomseg_frontier_update_on_no_active_path = bool(roomseg_frontier_update_cfg.get("update_on_no_active_path", False))
    args.roomseg_frontier_update_on_no_progress = bool(roomseg_frontier_update_cfg.get("update_on_no_progress", False))
    args.roomseg_frontier_arrival_update_cooldown_steps = int(
        roomseg_frontier_update_cfg.get("arrival_update_cooldown_steps", 3)
    )
    args.save_cached_roomseg_snapshots = bool(
        args.save_cached_roomseg_snapshots
        if args.save_cached_roomseg_snapshots is not None
        else roomseg_frontier_update_cfg.get(
            "save_cached_roomseg_snapshots",
            get_nested(cfg, "debug.save_cached_roomseg_snapshots", False),
        )
    )
    room_semantics_cfg = dict(get_nested(cfg, "room_semantics", {}) or {})
    for key in (
        "use_premerge_labels_for_open_plan_merge",
        "min_label_reliability_for_functional_split",
        "unknown_allows_functional_split",
        "final_label_after_merge",
    ):
        if key in room_semantics_cfg:
            args.room_segmentation_config.setdefault(key, room_semantics_cfg[key])
    room_node_cfg = dict(get_nested(cfg, "sgnav.scene_graph.room_nodes", {}) or {})
    args.room_label_backend = str(args.room_label_backend or room_node_cfg.get("room_label_backend", "vlm"))
    args.room_label_allowed_categories = list(room_node_cfg.get("allowed_room_categories", DEFAULT_ROOM_CATEGORIES) or DEFAULT_ROOM_CATEGORIES)
    args.room_label_min_confidence = float(
        args.room_label_min_confidence
        if args.room_label_min_confidence is not None
        else room_node_cfg.get("room_label_min_confidence", 0.60)
    )
    args.room_label_ambiguity_margin = float(
        args.room_label_ambiguity_margin
        if args.room_label_ambiguity_margin is not None
        else room_node_cfg.get("room_label_ambiguity_margin", 0.15)
    )
    args.room_label_min_reliable_objects = int(
        args.room_label_min_reliable_objects
        if args.room_label_min_reliable_objects is not None
        else room_node_cfg.get("room_label_min_reliable_objects", 2)
    )
    args.room_label_unknown_category = str(args.room_label_unknown_category or room_node_cfg.get("unknown_category", "unknown"))
    args.max_room_objects_in_prompt = int(
        args.max_room_objects_in_prompt
        if args.max_room_objects_in_prompt is not None
        else room_node_cfg.get("max_room_objects_in_prompt", 25)
    )
    args.detection_localization = str(args.detection_localization or get_nested(cfg, "perception.detection_localization", "static_map_ray"))
    args.min_depth_points_per_detection = int(args.min_depth_points_per_detection if args.min_depth_points_per_detection is not None else get_nested(cfg, "perception.min_depth_points_per_detection", 20))
    args.segmenter = str(args.segmenter or get_nested(cfg, "perception.segmenter", "none"))
    sam2_checkpoint_value = args.sam2_checkpoint if args.sam2_checkpoint is not None else get_nested(cfg, "perception.sam2_checkpoint", "")
    sam2_model_cfg_value = args.sam2_model_cfg if args.sam2_model_cfg is not None else get_nested(cfg, "perception.sam2_model_cfg", "facebook/sam2.1-hiera-tiny")
    args.sam2_checkpoint = "" if sam2_checkpoint_value is None else str(sam2_checkpoint_value)
    args.sam2_model_cfg = "facebook/sam2.1-hiera-tiny" if sam2_model_cfg_value is None else str(sam2_model_cfg_value)
    args.sam2_device = str(args.sam2_device or get_nested(cfg, "perception.sam2_device", "cuda"))
    args.max_detections_per_frame = int(args.max_detections_per_frame if args.max_detections_per_frame is not None else get_nested(cfg, "perception.max_detections_per_frame", 100))
    args.object_merge_radius_m = float(args.object_merge_radius_m if args.object_merge_radius_m is not None else get_nested(cfg, "perception.object_merge_radius_m", 0.5))
    args.instance_merge_distance_m = float(
        args.instance_merge_distance_m
        if args.instance_merge_distance_m is not None
        else get_nested(cfg, "sgnav.perception.instance_merge_distance_m", get_nested(cfg, "perception.object_merge_radius_m", 0.75))
    )
    args.instance_merge_iou_3d = float(
        args.instance_merge_iou_3d
        if args.instance_merge_iou_3d is not None
        else get_nested(cfg, "sgnav.perception.instance_merge_iou_3d", 0.15)
    )
    args.frontier_distance_weight = float(args.frontier_distance_weight if args.frontier_distance_weight is not None else get_nested(cfg, "sgnav.frontier_distance_weight", 0.2))
    args.frontier_scenegraph_score_norm = str(
        args.frontier_scenegraph_score_norm
        if args.frontier_scenegraph_score_norm is not None
        else get_nested(cfg, "sgnav.frontier_scenegraph_score_norm", "minmax")
    )
    args.frontier_selection_mode = str(
        args.frontier_selection_mode
        if args.frontier_selection_mode is not None
        else get_nested(cfg, "sgnav.frontier_selection_mode", "sgnav")
    )
    args.frontier_random_seed = int(
        args.frontier_random_seed
        if args.frontier_random_seed is not None
        else get_nested(cfg, "sgnav.frontier_random_seed", 0)
    )
    args.semantic_priors_path = str(args.semantic_priors_path or get_nested(cfg, "sgnav.semantic_priors_path", "voxroom_online/isaac_runtime/configs/room_semantic_priors.yaml"))
    args.debug_graph_dump = bool(
        args.debug_graph_dump
        if args.debug_graph_dump is not None
        else get_nested(cfg, "sgnav.debug_graph_dump", False)
    )
    args.debug_graph_dump_dir = str(args.debug_graph_dump_dir or get_nested(cfg, "sgnav.debug_graph_dump_dir", "debug/graphs"))
    args.runtime_planning_clearance_m = float(args.runtime_planning_clearance_m if args.runtime_planning_clearance_m is not None else get_nested(cfg, "astar.runtime_planning_clearance_m", 0.0))
    args.astar_clearance_cost_enabled = bool(
        args.astar_clearance_cost_enabled
        if args.astar_clearance_cost_enabled is not None
        else get_nested(cfg, "astar.clearance_cost_enabled", False)
    )
    args.astar_clearance_desired_m = float(
        args.astar_clearance_desired_m
        if args.astar_clearance_desired_m is not None
        else get_nested(cfg, "astar.clearance_desired_m", 0.25)
    )
    args.astar_clearance_hard_min_m = float(
        args.astar_clearance_hard_min_m
        if args.astar_clearance_hard_min_m is not None
        else get_nested(cfg, "astar.clearance_hard_min_m", 0.0)
    )
    args.astar_clearance_weight = float(
        args.astar_clearance_weight
        if args.astar_clearance_weight is not None
        else get_nested(cfg, "astar.clearance_weight", 3.0)
    )
    args.astar_clearance_power = float(
        args.astar_clearance_power
        if args.astar_clearance_power is not None
        else get_nested(cfg, "astar.clearance_power", 2.0)
    )
    args.astar_goal_min_clearance_m = float(
        args.astar_goal_min_clearance_m
        if args.astar_goal_min_clearance_m is not None
        else get_nested(cfg, "astar.goal_min_clearance_m", 0.18)
    )
    args.astar_goal_search_radius_m = float(
        args.astar_goal_search_radius_m
        if args.astar_goal_search_radius_m is not None
        else get_nested(cfg, "astar.goal_search_radius_m", 0.35)
    )
    args.guard_min_clearance_m = float(
        args.guard_min_clearance_m
        if args.guard_min_clearance_m is not None
        else get_nested(cfg, "astar.guard_min_clearance_m", 0.14)
    )
    args.frontier_targeting_config = dict(get_nested(cfg, "frontier_targeting", {}) or {})
    args.frontier_targeting_enabled = bool(args.frontier_targeting_config.get("enabled", True))
    args.frontier_target_search_radius_m = float(args.frontier_targeting_config.get("search_radius_m", 0.45))
    args.frontier_target_min_goal_clearance_m = float(
        args.frontier_targeting_config.get("min_goal_clearance_m", args.astar_goal_min_clearance_m)
    )
    args.frontier_target_require_reachable = bool(args.frontier_targeting_config.get("require_reachable", True))
    args.frontier_target_max_candidates = int(args.frontier_targeting_config.get("max_candidates", 128))
    args.collision_checked_smoothing_enabled = bool(
        args.collision_checked_smoothing_enabled
        if args.collision_checked_smoothing_enabled is not None
        else get_nested(cfg, "astar.collision_checked_smoothing_enabled", True)
    )
    args.smoothing_min_clearance_m = float(
        args.smoothing_min_clearance_m
        if args.smoothing_min_clearance_m is not None
        else get_nested(cfg, "astar.smoothing_min_clearance_m", 0.14)
    )
    args.smoothing_max_skip_cells = int(
        args.smoothing_max_skip_cells
        if args.smoothing_max_skip_cells is not None
        else get_nested(cfg, "astar.smoothing_max_skip_cells", 8)
    )
    args.lookahead_collision_check_enabled = bool(
        args.lookahead_collision_check_enabled
        if args.lookahead_collision_check_enabled is not None
        else get_nested(cfg, "astar.lookahead_collision_check_enabled", True)
    )
    args.lookahead_min_clearance_m = float(
        args.lookahead_min_clearance_m
        if args.lookahead_min_clearance_m is not None
        else get_nested(cfg, "astar.lookahead_min_clearance_m", 0.14)
    )
    args.lookahead_effective_min_clearance_m = effective_lookahead_min_clearance_m(
        float(args.lookahead_min_clearance_m),
        robot_radius_m=float(args.robot_radius_m),
        runtime_planning_clearance_m=float(args.runtime_planning_clearance_m),
        astar_clearance_hard_min_m=float(args.astar_clearance_hard_min_m),
    )
    args.guard_effective_min_clearance_m = effective_lookahead_min_clearance_m(
        float(args.guard_min_clearance_m),
        robot_radius_m=float(args.robot_radius_m),
        runtime_planning_clearance_m=float(args.runtime_planning_clearance_m),
        astar_clearance_hard_min_m=float(args.astar_clearance_hard_min_m),
    )
    args.candidate_min_detector_hits = int(args.candidate_min_detector_hits if args.candidate_min_detector_hits is not None else get_nested(cfg, "sgnav.candidate_min_detector_hits", 2))
    args.candidate_start_min_confidence = max(
        float(args.candidate_start_min_confidence if args.candidate_start_min_confidence is not None else get_nested(cfg, "sgnav.candidate_start_min_confidence", MIN_VALID_DETECTION_CONFIDENCE)),
        float(args.min_valid_detection_confidence),
    )
    args.candidate_start_min_hits = int(args.candidate_start_min_hits if args.candidate_start_min_hits is not None else get_nested(cfg, "sgnav.candidate_start_min_hits", 2))
    args.candidate_recent_max_age_steps = int(
        args.candidate_recent_max_age_steps
        if args.candidate_recent_max_age_steps is not None
        else get_nested(cfg, "sgnav.candidate_recent_max_age_steps", 30)
    )
    args.candidate_match_substring = bool(
        args.candidate_match_substring
        if args.candidate_match_substring is not None
        else get_nested(cfg, "sgnav.candidate_match_substring", False)
    )
    args.candidate_accept_requires_reperception = bool(
        args.candidate_accept_requires_reperception
        if args.candidate_accept_requires_reperception is not None
        else get_nested(cfg, "sgnav.candidate_accept_requires_reperception", True)
    )
    args.candidate_reject_ttl_steps = int(
        args.candidate_reject_ttl_steps
        if args.candidate_reject_ttl_steps is not None
        else get_nested(cfg, "sgnav.candidate_reject_ttl_steps", 80)
    )
    args.candidate_accept_threshold = float(
        args.candidate_accept_threshold
        if args.candidate_accept_threshold is not None
        else get_nested(cfg, "sgnav.candidate_accept_threshold", 0.65)
    )
    args.candidate_stop_distance_m = float(args.candidate_stop_distance_m if args.candidate_stop_distance_m is not None else get_nested(cfg, "sgnav.candidate_stop_distance_m", get_nested(cfg, "episodes.success_distance_m", 1.0)))
    args.candidate_standoff_min_m = float(args.candidate_standoff_min_m if args.candidate_standoff_min_m is not None else get_nested(cfg, "sgnav.candidate_standoff_min_m", 0.65))
    args.candidate_standoff_max_m = float(args.candidate_standoff_max_m if args.candidate_standoff_max_m is not None else get_nested(cfg, "sgnav.candidate_standoff_max_m", 1.80))
    args.candidate_standoff_max_cells = int(
        args.candidate_standoff_max_cells
        if args.candidate_standoff_max_cells is not None
        else get_nested(cfg, "sgnav.candidate_standoff_max_cells", 16)
    )
    args.candidate_standoff_ideal_m = float(
        args.candidate_standoff_ideal_m
        if args.candidate_standoff_ideal_m is not None
        else get_nested(cfg, "sgnav.candidate_standoff_ideal_m", 1.0)
    )
    args.reperception_enabled = bool(args.reperception_enabled if args.reperception_enabled is not None else get_nested(cfg, "sgnav.reperception_enabled", True))
    args.reperception_min_observations = int(args.reperception_min_observations if args.reperception_min_observations is not None else get_nested(cfg, "sgnav.reperception_min_observations", 3))
    args.reperception_max_steps = int(args.reperception_max_steps if args.reperception_max_steps is not None else get_nested(cfg, "sgnav.reperception_max_steps", 10))
    args.reperception_same_goal_radius_m = float(args.reperception_same_goal_radius_m if args.reperception_same_goal_radius_m is not None else get_nested(cfg, "sgnav.reperception_same_goal_radius_m", 0.8))
    args.reperception_turn_wz_radps = float(args.reperception_turn_wz_radps if args.reperception_turn_wz_radps is not None else get_nested(cfg, "sgnav.reperception_turn_wz_radps", 0.5))
    args.stop_verification_steps = int(args.stop_verification_steps if args.stop_verification_steps is not None else get_nested(cfg, "sgnav.stop_verification_steps", 4))
    args.stop_verification_min_hits = int(args.stop_verification_min_hits if args.stop_verification_min_hits is not None else get_nested(cfg, "sgnav.stop_verification_min_hits", 2))
    args.found_goal_stop_distance_m = float(args.found_goal_stop_distance_m if args.found_goal_stop_distance_m is not None else get_nested(cfg, "sgnav.found_goal_stop_distance_m", 0.35))
    args.require_sgnav_stop = bool(args.require_sgnav_stop if args.require_sgnav_stop is not None else get_nested(cfg, "episodes.success_requires_stop", True))
    args.llm_enabled = bool(args.llm_enabled if args.llm_enabled is not None else get_nested(cfg, "llm.enabled", True))
    args.llm_base_url = args.llm_base_url or get_nested(cfg, "llm.base_url", "http://127.0.0.1:8000/v1")
    args.llm_model = args.llm_model or get_nested(cfg, "llm.model", "qwen3-vl-8b-instruct")
    args.llm_api_key = args.llm_api_key or get_nested(cfg, "llm.api_key", "EMPTY")
    args.llm_timeout_s = float(args.llm_timeout_s if args.llm_timeout_s is not None else get_nested(cfg, "llm.timeout_s", 30.0))
    args.llm_temperature = float(args.llm_temperature if args.llm_temperature is not None else get_nested(cfg, "llm.temperature", 0.0))
    args.llm_max_tokens = int(args.llm_max_tokens if args.llm_max_tokens is not None else get_nested(cfg, "llm.max_tokens", 512))
    args.max_hcot_subgraphs_per_decision = int(
        args.max_hcot_subgraphs_per_decision
        if args.max_hcot_subgraphs_per_decision is not None
        else get_nested(cfg, "llm.max_hcot_subgraphs_per_decision", 8)
    )
    args.vllm_frontier_scoring = bool(args.vllm_frontier_scoring if args.vllm_frontier_scoring is not None else get_nested(cfg, "vllm.frontier_scoring", False))
    args.vllm_base_url = str(args.vllm_base_url or get_nested(cfg, "vllm.base_url", "http://127.0.0.1:8000/v1"))
    args.vllm_model = str(args.vllm_model or get_nested(cfg, "vllm.model", "qwen3-vl-8b-instruct"))
    args.vllm_timeout_s = float(args.vllm_timeout_s if args.vllm_timeout_s is not None else get_nested(cfg, "vllm.timeout_s", 8.0))
    args.vllm_temperature = float(args.vllm_temperature if args.vllm_temperature is not None else get_nested(cfg, "vllm.temperature", 0.0))
    args.vllm_max_frontiers = int(args.vllm_max_frontiers if args.vllm_max_frontiers is not None else get_nested(cfg, "vllm.max_frontiers", 32))
    args.vllm_image_scoring = bool(args.vllm_image_scoring if args.vllm_image_scoring is not None else get_nested(cfg, "vllm.image_scoring", True))
    args.vllm_image_max_width = int(args.vllm_image_max_width if args.vllm_image_max_width is not None else get_nested(cfg, "vllm.image_max_width", 640))
    args.vllm_image_jpeg_quality = int(args.vllm_image_jpeg_quality if args.vllm_image_jpeg_quality is not None else get_nested(cfg, "vllm.image_jpeg_quality", 75))
    args.score_frontiers_before_candidate = bool(
        args.score_frontiers_before_candidate
        if args.score_frontiers_before_candidate is not None
        else get_nested(cfg, "vllm.score_frontiers_before_candidate", args.vllm_frontier_scoring)
    )
    args.seed_gt_object_memory = bool(get_nested(cfg, "sgnav.seed_gt_object_memory", False) if args.seed_gt_object_memory is None else args.seed_gt_object_memory)
    args.allow_gt_goal_fallback = bool(get_nested(cfg, "sgnav.allow_gt_goal_fallback", False) if args.allow_gt_goal_fallback is None else args.allow_gt_goal_fallback)
    configured_success_distance = get_nested(cfg, "episodes.success_distance_m", None)
    args.success_distance_m = (
        float(args.success_distance_m)
        if args.success_distance_m is not None
        else (float(configured_success_distance) if configured_success_distance is not None else None)
    )
    args.roomseg_debug_only = bool(
        args.roomseg_debug_only
        if args.roomseg_debug_only is not None
        else get_nested(cfg, "debug.roomseg_debug_only", False)
    )
    requested_explore_until_no_frontiers = args.explore_until_no_frontiers
    policy_name = str(args.policy or "")
    debug_frontier_policy = policy_name in {"random_frontier_mask_probe", "nearest_frontier_mask_probe"}
    args.explore_until_no_frontiers = bool(
        requested_explore_until_no_frontiers
        if requested_explore_until_no_frontiers is not None
        else (
            debug_frontier_policy
            or bool(get_nested(cfg, "debug.explore_until_no_frontiers", False))
        )
    )
    if debug_frontier_policy:
        args.strict_benchmark = False
        args.allow_debug_fallbacks = True
        args.llm_enabled = False
        args.vllm_frontier_scoring = False
        args.vllm_image_scoring = False
        args.score_frontiers_before_candidate = False
        args.require_sgnav_stop = False
        args.frontier_allow_near_fallback = True
        if str(getattr(args, "detector", "none") or "none") == "none":
            args.segmenter = "none"
        if str(getattr(args, "room_label_backend", "unavailable") or "unavailable") == "unavailable":
            args.room_label_backend = "unavailable"
    if args.explore_until_no_frontiers and args.frontier_allow_near_fallback is None:
        args.frontier_allow_near_fallback = bool(get_nested(cfg, "mapping.frontier_allow_near_fallback", False))
    if args.roomseg_debug_only:
        args.strict_benchmark = False
        args.allow_debug_fallbacks = True
        args.detector = "none"
        args.segmenter = "none"
        args.llm_enabled = False
        args.vllm_frontier_scoring = False
        args.vllm_image_scoring = False
        args.score_frontiers_before_candidate = False
        args.seed_gt_object_memory = False
        args.allow_gt_goal_fallback = False
        args.require_sgnav_stop = False
        args.room_label_backend = "unavailable"
        args.voxroom_viz_every_steps = 1
        args.debug_roomseg_layers = True
        debug_layers_cfg = dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {})
        debug_layers_cfg["enabled"] = True
        debug_layers_cfg.setdefault("output_dir", str(getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")))
        args.room_segmentation_config["debug_layers"] = debug_layers_cfg
        args.debug_roomseg_dir = str(debug_layers_cfg.get("output_dir", getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")))
        for nested_key in ("voxel_occupancy_door_wall",):
            nested_cfg = dict(args.room_segmentation_config.get(nested_key, {}) or {})
            nested_cfg["debug_dump"] = True
            nested_cfg.setdefault("debug_dir", args.debug_roomseg_dir)
            args.room_segmentation_config[nested_key] = nested_cfg

    if args.planner == "nav2":
        from voxroom_online.isaac_runtime.navigation.nav2_client import Nav2NavigateToPoseClient

        Nav2NavigateToPoseClient()
    try:
        validate_strict_benchmark_assets(args)
    except BenchmarkAssetError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 2
    episodes = read_jsonl(args.episode_file)
    episode = apply_success_distance_override(episodes[int(args.episode_index)], args)
    if args.sim_backend == "isaac":
        row = run_episode_isaac_closed_loop(episode, args)
    else:
        row = run_episode_map_sim(episode, args)
    if not getattr(args, "_row_already_logged", False):
        row = make_jsonable(row)
        summary_row = final_log_row(row)
        logger = JsonlEpisodeLogger(args.output)
        logger.log(row)
        print(json.dumps(summary_row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
