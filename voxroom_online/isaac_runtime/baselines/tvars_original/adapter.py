from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from ..data_contract import MapInfo, resolve_map_info
from ..mask_io import build_metric_domain_from_source, relabel_consecutive, save_baseline_snapshot_npz
from ..topology_active.depth_projection import project_door_mask_to_grid_rc_with_status
from ..topology_active.detector import (
    CameraIntrinsics,
    OriginalDetrDoorDetector,
    camera_intrinsics_from_mapping,
)
from ...evaluation.roomseg_coverage_milestones import (
    partition_free_space_by_door_lines,
)


BASELINE_NAME = "tvars_original_isaac"
ORIGINAL_POLICY_CONTROL = "original_topology"
ORIGINAL_DOOR_MASK_PIXEL_STRIDE = 2
ORIGINAL_VISION_RANGE_M = 3.0
ORIGINAL_DOOR_RELATIVE_Z_MIN_M = 0.25
ORIGINAL_DOOR_RELATIVE_Z_MAX_M = 1.50
ORIGINAL_FRONTIER_REACH_RADIUS_CELLS = 10
RUNTIME_GLOBAL_FRONTIER_REACH_RADIUS_CELLS = 2
RUNTIME_GLOBAL_FRONTIER_REACHED_MATCH_RADIUS_CELLS = 2
TVARS_MIN_ROOM_AREA_M2 = 0.5
ORIGINAL_ROOM_ENTRY_RETURN_STEP_LIMIT = 20
ORIGINAL_TOPOLOGY_EXIT_STEP_LIMIT = 100


@dataclass
class _OriginalModules:
    repo_dir: Path
    convert_2_laser: Any
    DoorDetection: Any
    FrontierDetection: Any
    TopomapConstruction: Any
    FMMPlanner: Any
    skfmm: Any
    map_size_cells: int
    resolution_m: float


@dataclass(frozen=True)
class OriginalPolicyDecision:
    target_rc: tuple[int, int] | None
    stop: bool
    reason: str
    phase: str
    target_kind: str | None
    reach_radius_m: float
    decision_index: int
    metadata: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "target_rc": None
            if self.target_rc is None
            else [int(self.target_rc[0]), int(self.target_rc[1])],
            "stop": bool(self.stop),
            "reason": str(self.reason),
            "phase": str(self.phase),
            "target_kind": self.target_kind,
            "reach_radius_m": float(self.reach_radius_m),
            "decision_index": int(self.decision_index),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class OriginalFMMNavigationStep:
    target_rc: tuple[int, int]
    short_term_goal_rc: tuple[float, float]
    short_term_goal_cell: tuple[int, int]
    short_term_distance_m: float
    target_distance_m: float
    target_already_reached: bool
    replan_requested: bool
    reachable_boundary_reached: bool
    room_door_lines_applied: bool
    door_line_cells: int
    collision_map_cells: int
    visited_map_cells: int
    obstacle_inflation_m: float
    obstacle_inflation_cells: int
    preferred_clearance_m: float
    preferred_clearance_cells: int
    clearance_soft_band_cells: int
    clearance_speed_floor: float
    collision_inflation_m: float
    collision_inflation_cells: int
    plan_index: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path_planner_backend": "tvars_original_fmm",
            "fmm_source": "env/utils/fmm_planner.py:FMMPlanner",
            "fmm_target_rc": [int(self.target_rc[0]), int(self.target_rc[1])],
            "fmm_short_term_goal_rc": [
                float(self.short_term_goal_rc[0]),
                float(self.short_term_goal_rc[1]),
            ],
            "fmm_short_term_goal_cell": [
                int(self.short_term_goal_cell[0]),
                int(self.short_term_goal_cell[1]),
            ],
            "fmm_short_term_distance_m": float(self.short_term_distance_m),
            "fmm_target_distance_m": float(self.target_distance_m),
            "fmm_target_already_reached": bool(self.target_already_reached),
            "fmm_replan_requested": bool(self.replan_requested),
            "fmm_reachable_boundary_reached": bool(
                self.reachable_boundary_reached
            ),
            "fmm_replan_semantics": "upstream_returns_current_cell",
            "fmm_room_door_lines_applied": bool(
                self.room_door_lines_applied
            ),
            "fmm_door_line_cells": int(self.door_line_cells),
            "fmm_collision_map_cells": int(self.collision_map_cells),
            "fmm_visited_map_cells": int(self.visited_map_cells),
            "fmm_collision_feedback_semantics": "upstream_collision_map",
            "fmm_obstacle_boundary_m": float(self.obstacle_inflation_m),
            "fmm_obstacle_inflation_cells": int(self.obstacle_inflation_cells),
            "fmm_preferred_obstacle_clearance_m": float(
                self.preferred_clearance_m
            ),
            "fmm_preferred_obstacle_clearance_cells": int(
                self.preferred_clearance_cells
            ),
            "fmm_clearance_soft_band_cells": int(
                self.clearance_soft_band_cells
            ),
            "fmm_clearance_speed_floor": float(self.clearance_speed_floor),
            "fmm_clearance_aware_travel_time": bool(
                self.preferred_clearance_cells > self.obstacle_inflation_cells
            ),
            "fmm_distance_field_semantics": (
                "skfmm_travel_time_with_clearance_speed"
                if self.preferred_clearance_cells > self.obstacle_inflation_cells
                else "upstream_skfmm_distance"
            ),
            "fmm_collision_inflation_m": float(self.collision_inflation_m),
            "fmm_collision_inflation_cells": int(self.collision_inflation_cells),
            "fmm_obstacle_clearance_semantics": (
                "robot_radius_plus_runtime_margin_nearest_grid_radius"
            ),
            "fmm_collision_feedback_boundary_semantics": (
                "execution_guard_clearance_boundary_no_double_inflation"
            ),
            "fmm_step_size_cells": 5,
            "fmm_short_goal_iterations": 1,
            "fmm_plan_index": int(self.plan_index),
            "astar_path_planner_executed": False,
            "planner_fallback_used": False,
        }


@dataclass(frozen=True)
class _ScanFrame:
    obs: Mapping[str, Any]
    camera_intrinsics: Any
    map_info: MapInfo
    floor_z: float


class TVARSOriginalIsaacBaseline:
    """Run TVARS room segmentation modules on Isaac observations.

    Habitat is intentionally not launched here. Isaac provides RGB-D, robot pose,
    and online maps; the door/frontier/topomap update calls come from the
    original Active_room_segmentation source checkout.
    """

    baseline_name = BASELINE_NAME

    def __init__(
        self,
        *,
        output_dir: Path,
        repo_dir: Path | str | None = None,
        detector: OriginalDetrDoorDetector | None = None,
        policy_control: str = "never",
        panorama_views: int = 12,
        save_stream: bool = False,
        save_every_snapshot: bool = True,
        fmm_obstacle_inflation_m: float = 0.05,
        fmm_preferred_clearance_m: float | None = None,
        fmm_clearance_speed_floor: float = 0.25,
        fmm_collision_inflation_m: float = 0.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.repo_dir = _resolve_repo_dir(repo_dir)
        expected_commit = str(os.environ.get("ACTIVE_ROOM_SEG_EXPECTED_COMMIT", "") or "").strip()
        if not expected_commit:
            raise RuntimeError("ACTIVE_ROOM_SEG_EXPECTED_COMMIT is required for the strict original baseline")
        actual_commit = _git_head(self.repo_dir)
        if actual_commit != expected_commit:
            raise RuntimeError(
                "Active Room source commit mismatch: "
                f"expected {expected_commit}, found {actual_commit}"
            )
        dirty = subprocess.check_output(
            ["git", "-C", str(self.repo_dir), "status", "--porcelain"],
            text=True,
            timeout=30,
        ).strip()
        if dirty:
            raise RuntimeError("the strict original Active Room source tree is dirty")
        self.detector = detector or OriginalDetrDoorDetector(repo_dir=self.repo_dir)
        if not bool(getattr(self.detector, "available", False)):
            raise RuntimeError("the original Active Room DETR detector is unavailable")
        if not bool(getattr(self.detector, "detector_adapter_verified", False)):
            raise RuntimeError("the original Active Room DETR detector adapter is not verified")
        normalized_policy_control = str(policy_control).strip().lower()
        if normalized_policy_control not in {"never", ORIGINAL_POLICY_CONTROL}:
            raise ValueError(
                "TVARS original policy_control must be 'never' or '%s'"
                % ORIGINAL_POLICY_CONTROL
            )
        if normalized_policy_control == ORIGINAL_POLICY_CONTROL and not callable(
            getattr(self.detector, "detect_batch", None)
        ):
            raise RuntimeError(
                "strict original topology control requires batched original DETR inference"
            )
        if int(panorama_views) != 12:
            raise ValueError("the original Active Room scan contract requires exactly 12 views")
        self.policy_control = normalized_policy_control
        self.fmm_obstacle_inflation_m = float(fmm_obstacle_inflation_m)
        self.fmm_preferred_clearance_m = float(
            self.fmm_obstacle_inflation_m
            if fmm_preferred_clearance_m is None
            else fmm_preferred_clearance_m
        )
        self.fmm_clearance_speed_floor = float(fmm_clearance_speed_floor)
        self.fmm_collision_inflation_m = float(fmm_collision_inflation_m)
        if self.fmm_obstacle_inflation_m < 0.05:
            raise ValueError("TVARS FMM obstacle inflation must be at least 0.05 m")
        if self.fmm_preferred_clearance_m < self.fmm_obstacle_inflation_m:
            raise ValueError(
                "TVARS FMM preferred clearance must be at least the hard obstacle inflation"
            )
        if not 0.0 < self.fmm_clearance_speed_floor <= 1.0:
            raise ValueError(
                "TVARS FMM clearance speed floor must be in (0, 1]"
            )
        if self.fmm_collision_inflation_m < 0.0:
            raise ValueError("TVARS FMM collision inflation must be non-negative")
        self.panorama_views = int(panorama_views)
        self.save_stream = bool(save_stream)
        self.save_every_snapshot = bool(save_every_snapshot)
        self._modules: _OriginalModules | None = None
        self._door_detect: Any | None = None
        self._frontier_detector: Any | None = None
        self._topomap: Any | None = None
        self._detected_door_list: list[dict[str, Any]] = []
        self._raw_detect_list: list[list[int]] = []
        self.latest_label_map: np.ndarray | None = None
        self.latest_preview_rgb: np.ndarray | None = None
        self.latest_debug_arrays: dict[str, np.ndarray] = {}
        self.latest_metadata: dict[str, Any] = {}
        self._vision_evidence_map: np.ndarray | None = None
        self._scan_frames: list[_ScanFrame] = []
        self._completed_scan_count = 0
        self._scan_required = self.policy_control == ORIGINAL_POLICY_CONTROL
        self._policy_update_required = self.policy_control == ORIGINAL_POLICY_CONTROL
        self._active_policy_target_rc: tuple[int, int] | None = None
        self._active_policy_target_kind: str | None = None
        self._active_policy_reach_radius_m = 0.0
        self._transition_waypoints_rc: list[tuple[int, int]] = []
        self._transition_trajectories_xy: list[list[list[float]]] = []
        self._active_transition_segment_index: int | None = None
        self._transition_reached_exit_count = 0
        self._transition_failed_exit_count = 0
        self._transition_entry_scan_pending = False
        self._last_transition_advance: dict[str, Any] | None = None
        self._last_frontier_goal_rc: tuple[int, int] | None = None
        self._astar_unreachable_room_frontier_targets: set[
            tuple[int, int]
        ] = set()
        self._astar_unreachable_global_frontier_targets: set[
            tuple[int, int]
        ] = set()
        self._reached_global_frontier_targets: set[tuple[int, int]] = set()
        self._policy_target_unreachable_count = 0
        self._active_policy_no_path_step_count = 0
        self._policy_decision_index = 0
        self._policy_target_reached_count = 0
        self._fmm_plan_count = 0
        self._policy_collision_map: np.ndarray | None = None
        self._policy_visited_map: np.ndarray | None = None
        self._policy_collision_update_count = 0
        self._policy_collision_width_cells = 1
        self._last_collision_current_rc: tuple[int, int] | None = None
        self._policy_resolution_m: float | None = None
        self._last_update_step: int | None = None
        self._latest_policy_decision = OriginalPolicyDecision(
            target_rc=None,
            stop=False,
            reason=(
                "original_initial_scan_required"
                if self.policy_control == ORIGINAL_POLICY_CONTROL
                else "policy_control_disabled"
            ),
            phase=(
                "scan_required"
                if self.policy_control == ORIGINAL_POLICY_CONTROL
                else "segmentation_only"
            ),
            target_kind=None,
            reach_radius_m=0.0,
            decision_index=0,
            metadata={},
        )
        self._stream_manifest_path = self.output_dir.parent.parent / "tvars_original_stream" / "stream_manifest.jsonl"
        self._policy_trace_path = self.output_dir / "policy_trace.jsonl"
        self._last_stream_step: int | None = None

    def on_episode_start(self, episode_metadata: Mapping[str, Any]) -> None:
        self._policy_collision_map = None
        self._policy_visited_map = None
        self._policy_collision_update_count = 0
        self._policy_collision_width_cells = 1
        self._last_collision_current_rc = None
        self._policy_resolution_m = None
        self._astar_unreachable_room_frontier_targets.clear()
        self._astar_unreachable_global_frontier_targets.clear()
        self._reached_global_frontier_targets.clear()
        self._policy_target_unreachable_count = 0
        self._active_policy_no_path_step_count = 0
        if self.policy_control == ORIGINAL_POLICY_CONTROL:
            self._policy_trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._policy_trace_path.write_text("", encoding="utf-8")
        if not self.save_stream:
            return
        self._stream_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._stream_manifest_path.exists():
            self._stream_manifest_path.write_text("", encoding="utf-8")
        meta_path = self._stream_manifest_path.parent / "episode_metadata.json"
        meta_path.write_text(json.dumps(_json_ready(dict(episode_metadata)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @property
    def scan_required(self) -> bool:
        return bool(self._scan_required)

    @property
    def scan_views_collected(self) -> int:
        return int(len(self._scan_frames))

    @property
    def scan_views_remaining(self) -> int:
        return max(0, int(self.panorama_views) - int(len(self._scan_frames)))

    @property
    def policy_update_required(self) -> bool:
        return bool(self._policy_update_required)

    @property
    def policy_trace_path(self) -> Path:
        return Path(self._policy_trace_path)

    @property
    def fmm_plan_count(self) -> int:
        return int(self._fmm_plan_count)

    @property
    def collision_update_count(self) -> int:
        return int(self._policy_collision_update_count)

    @property
    def terminal_frontier_target_count(self) -> int:
        return 0

    @property
    def reached_global_frontier_target_count(self) -> int:
        return int(len(self._reached_global_frontier_targets))

    def policy_decision(self) -> OriginalPolicyDecision:
        return self._latest_policy_decision

    def plan_fmm_short_term_goal(
        self,
        *,
        map_state: Mapping[str, Any],
        mapper: Any | None = None,
    ) -> OriginalFMMNavigationStep:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError(
                "FMM navigation requires original topology policy control"
            )
        if self._active_policy_target_rc is None:
            raise RuntimeError("original topology FMM has no active target")

        arrays = _arrays_from_map_state(map_state)
        map_info = resolve_map_info(
            map_state=map_state,
            mapper=mapper,
            snapshot_arrays=arrays,
        )
        shape = tuple(int(v) for v in arrays["occupancy_map"].shape)
        self._ensure_policy_navigation_memory(shape)
        assert self._policy_collision_map is not None
        assert self._policy_visited_map is not None
        modules = self._ensure_original_modules(
            shape=shape,
            resolution_m=float(map_info.resolution_m),
        )
        if self._policy_resolution_m is None:
            self._policy_resolution_m = float(map_info.resolution_m)
        elif not np.isclose(
            self._policy_resolution_m,
            float(map_info.resolution_m),
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError("original topology navigation resolution changed")
        current_rc = tuple(
            int(v)
            for v in np.asarray(map_state["current_grid"]).reshape(-1)[:2]
        )
        self._policy_visited_map[current_rc] = True
        target_rc = tuple(int(v) for v in self._active_policy_target_rc)
        for name, cell in (("current", current_rc), ("target", target_rc)):
            if not (
                0 <= int(cell[0]) < shape[0]
                and 0 <= int(cell[1]) < shape[1]
            ):
                raise RuntimeError(
                    "original topology FMM %s cell is outside the map: %s"
                    % (name, cell)
                )

        target_distance_m = float(
            np.hypot(
                float(target_rc[0] - current_rc[0]),
                float(target_rc[1] - current_rc[1]),
            )
            * float(modules.resolution_m)
        )
        reach_radius_m = float(self._active_policy_reach_radius_m)
        room_door_lines_applied = bool(
            self._latest_policy_decision.phase == "room_search_navigation"
        )
        obstacle_map = np.asarray(
            arrays["obstacle_mask"], dtype=np.float32
        ).copy()
        explored_map = (~np.asarray(arrays["unknown_mask"], dtype=bool)).astype(
            np.float32
        )
        door_line_map = np.zeros(shape, dtype=bool)
        if room_door_lines_applied:
            door_line_map = _doors_to_line_map(
                self._detected_door_list,
                shape=shape,
            )
            obstacle_map[door_line_map] = 1.0
            explored_map[door_line_map] = 0.0

        target_already_reached = bool(target_distance_m <= reach_radius_m)
        replan_requested = False
        if target_already_reached:
            short_term_goal_rc = np.asarray(
                [float(current_rc[0]) + 0.5, float(current_rc[1]) + 0.5],
                dtype=np.float64,
            )
        else:
            short_term_goal_rc, replan_requested = _original_fmm_short_term_goal(
                modules=modules,
                grid=obstacle_map,
                explored=explored_map,
                start=np.asarray(current_rc, dtype=np.float64),
                goal=np.asarray(target_rc, dtype=np.int32),
                collision_map=self._policy_collision_map,
                visited=self._policy_visited_map,
                obstacle_inflation_m=self.fmm_obstacle_inflation_m,
                preferred_clearance_m=self.fmm_preferred_clearance_m,
                clearance_speed_floor=self.fmm_clearance_speed_floor,
                return_replan=True,
            )
        reachable_boundary_reached = bool(
            replan_requested and not target_already_reached
        )

        if not np.all(np.isfinite(short_term_goal_rc)):
            raise RuntimeError(
                "original topology FMM produced a non-finite short-term goal"
            )
        short_term_goal_cell = (
            int(np.floor(float(short_term_goal_rc[0]))),
            int(np.floor(float(short_term_goal_rc[1]))),
        )
        if not (
            0 <= short_term_goal_cell[0] < shape[0]
            and 0 <= short_term_goal_cell[1] < shape[1]
        ):
            raise RuntimeError(
                "original topology FMM short-term goal is outside the map: %s"
                % (short_term_goal_cell,)
            )
        short_term_distance_m = float(
            np.linalg.norm(
                short_term_goal_rc
                - np.asarray(current_rc, dtype=np.float64)
            )
            * float(modules.resolution_m)
        )
        if (
            not target_already_reached
            and not reachable_boundary_reached
            and short_term_goal_cell == current_rc
        ):
            raise RuntimeError(
                "original topology FMM made no progress toward an unreached target"
            )

        self._fmm_plan_count += 1
        return OriginalFMMNavigationStep(
            target_rc=target_rc,
            short_term_goal_rc=(
                float(short_term_goal_rc[0]),
                float(short_term_goal_rc[1]),
            ),
            short_term_goal_cell=short_term_goal_cell,
            short_term_distance_m=short_term_distance_m,
            target_distance_m=target_distance_m,
            target_already_reached=target_already_reached,
            replan_requested=bool(replan_requested),
            reachable_boundary_reached=reachable_boundary_reached,
            room_door_lines_applied=room_door_lines_applied,
            door_line_cells=int(np.count_nonzero(door_line_map)),
            collision_map_cells=int(np.count_nonzero(self._policy_collision_map)),
            visited_map_cells=int(np.count_nonzero(self._policy_visited_map)),
            obstacle_inflation_m=float(self.fmm_obstacle_inflation_m),
            obstacle_inflation_cells=_clearance_radius_cells(
                self.fmm_obstacle_inflation_m,
                float(modules.resolution_m),
                minimum_cells=1,
            ),
            preferred_clearance_m=float(self.fmm_preferred_clearance_m),
            preferred_clearance_cells=_clearance_radius_cells(
                self.fmm_preferred_clearance_m,
                float(modules.resolution_m),
                minimum_cells=1,
            ),
            clearance_soft_band_cells=max(
                0,
                _clearance_radius_cells(
                    self.fmm_preferred_clearance_m,
                    float(modules.resolution_m),
                    minimum_cells=1,
                )
                - _clearance_radius_cells(
                    self.fmm_obstacle_inflation_m,
                    float(modules.resolution_m),
                    minimum_cells=1,
                ),
            ),
            clearance_speed_floor=float(self.fmm_clearance_speed_floor),
            collision_inflation_m=float(self.fmm_collision_inflation_m),
            collision_inflation_cells=_clearance_radius_cells(
                self.fmm_collision_inflation_m,
                float(modules.resolution_m),
                minimum_cells=0,
            ),
            plan_index=int(self._fmm_plan_count),
        )

    def _ensure_policy_navigation_memory(self, shape: tuple[int, int]) -> None:
        expected = tuple(int(v) for v in shape)
        if self._policy_collision_map is None:
            self._policy_collision_map = np.zeros(expected, dtype=bool)
            self._policy_visited_map = np.zeros(expected, dtype=bool)
            return
        if self._policy_collision_map.shape != expected:
            raise RuntimeError(
                "original topology collision map shape changed: expected=%s actual=%s"
                % (self._policy_collision_map.shape, expected)
            )
        if self._policy_visited_map is None or self._policy_visited_map.shape != expected:
            raise RuntimeError("original topology visited map shape changed")

    def record_policy_collision(
        self,
        *,
        step: int,
        current_rc: Sequence[int],
        collision_rc: Sequence[int],
        intended_rc: Sequence[int],
    ) -> None:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError(
                "policy collision feedback requires original topology control"
            )
        if self._policy_collision_map is None or self._policy_visited_map is None:
            raise RuntimeError("original topology collision feedback arrived before FMM initialization")
        shape = self._policy_collision_map.shape
        current = tuple(int(v) for v in np.asarray(current_rc).reshape(-1)[:2])
        collision = tuple(int(v) for v in np.asarray(collision_rc).reshape(-1)[:2])
        intended = tuple(int(v) for v in np.asarray(intended_rc).reshape(-1)[:2])
        for name, cell in (
            ("current", current),
            ("collision", collision),
            ("intended", intended),
        ):
            if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
                raise RuntimeError(
                    "original topology %s collision-feedback cell is outside the map: %s"
                    % (name, cell)
                )
        self._policy_visited_map[current] = True
        if self._last_collision_current_rc is not None and float(
            np.hypot(
                current[0] - self._last_collision_current_rc[0],
                current[1] - self._last_collision_current_rc[1],
            )
        ) < 1.0:
            self._policy_collision_width_cells = min(
                9, int(self._policy_collision_width_cells) + 2
            )
        else:
            self._policy_collision_width_cells = 1
        self._last_collision_current_rc = current

        direction = np.asarray(
            [intended[0] - current[0], intended[1] - current[1]],
            dtype=np.float64,
        )
        if float(np.linalg.norm(direction)) <= 1e-9:
            direction = np.asarray(
                [collision[0] - current[0], collision[1] - current[1]],
                dtype=np.float64,
            )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise RuntimeError("original topology collision feedback has no direction")
        direction /= norm
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        written_cells: list[tuple[int, int]] = []
        width = int(self._policy_collision_width_cells)
        for forward_offset in (3.0, 4.0):
            for lateral_offset in range(-(width // 2), width // 2 + 1):
                point = (
                    np.asarray(current, dtype=np.float64)
                    + direction * forward_offset
                    + lateral * float(lateral_offset)
                )
                row = min(max(0, int(round(float(point[0])))), shape[0] - 1)
                col = min(max(0, int(round(float(point[1])))), shape[1] - 1)
                written_cells.append((row, col))
        collision_seed = np.zeros(shape, dtype=bool)
        for row, col in written_cells:
            collision_seed[row, col] = True
        if self._policy_resolution_m is None:
            raise RuntimeError("original topology collision feedback has no map resolution")
        collision_inflation_cells = _clearance_radius_cells(
            self.fmm_collision_inflation_m,
            self._policy_resolution_m,
            minimum_cells=0,
        )
        if collision_inflation_cells > 0:
            import skimage.morphology

            collision_seed = skimage.morphology.binary_dilation(
                collision_seed,
                skimage.morphology.disk(collision_inflation_cells),
            )
            expanded_rows, expanded_cols = np.nonzero(collision_seed)
            forward_projection = (
                (expanded_rows.astype(np.float64) - float(current[0]))
                * float(direction[0])
                + (expanded_cols.astype(np.float64) - float(current[1]))
                * float(direction[1])
            )
            remove = forward_projection < 2.0 - 1e-9
            collision_seed[expanded_rows[remove], expanded_cols[remove]] = False
        self._policy_collision_map |= collision_seed
        self._policy_collision_update_count += 1
        self._record_policy_event(
            {
                "event": "collision_map_updated",
                "step": int(step),
                "current_rc": [int(current[0]), int(current[1])],
                "collision_rc": [int(collision[0]), int(collision[1])],
                "intended_rc": [int(intended[0]), int(intended[1])],
                "collision_band_width_cells": int(width),
                "collision_band_forward_offsets_cells": [3, 4],
                "collision_band_cells": [
                    [int(cell[0]), int(cell[1])] for cell in written_cells
                ],
                "collision_inflation_m": float(self.fmm_collision_inflation_m),
                "collision_inflation_cells": int(collision_inflation_cells),
                "collision_escape_halfspace_min_forward_cells": 2,
                "collision_feedback_cells_written": int(
                    np.count_nonzero(collision_seed)
                ),
                "collision_map_cells": int(np.count_nonzero(self._policy_collision_map)),
                "collision_update_count": int(self._policy_collision_update_count),
                "collision_feedback_semantics": "upstream_collision_map",
            }
        )

    def refresh_snapshot_partition(
        self,
        *,
        step: int,
        map_state: Mapping[str, Any],
    ) -> None:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError(
                "snapshot partition refresh requires original topology control"
            )
        if self.latest_label_map is None:
            raise RuntimeError(
                "original topology has no completed result to refresh"
            )
        arrays = _arrays_from_map_state(map_state)
        shape = tuple(int(v) for v in arrays["occupancy_map"].shape)
        accepted_door_line_map = _doors_to_line_map(
            self._detected_door_list,
            shape=shape,
        )
        vertical_label_map, partition_debug = partition_free_space_by_door_lines(
            np.asarray(arrays["vertical_free_room_domain"], dtype=bool),
            accepted_door_line_map,
        )
        vertical_label_map, label_map, projection_debug = (
            _filter_and_project_partition_labels(
                vertical_label_map,
                navigation_free=np.asarray(
                    arrays["navigation_free_room_domain"], dtype=bool
                ),
                resolution_m=float(self._modules.resolution_m),
                min_room_area_m2=TVARS_MIN_ROOM_AREA_M2,
            )
        )
        partition_debug = {**dict(partition_debug), **projection_debug}
        self.latest_label_map = label_map
        self.latest_debug_arrays = {
            **dict(self.latest_debug_arrays),
            "tvars_original_door_line_map": accepted_door_line_map,
            "tvars_original_vertical_free_partition_label_map": (
                vertical_label_map.astype(np.int32)
            ),
            "tvars_original_navigation_projected_label_map": (
                label_map.astype(np.int32)
            ),
            "tvars_original_policy_target_map": _policy_target_map(
                self._latest_policy_decision.target_rc,
                shape=shape,
            ),
        }
        self.latest_preview_rgb = _render_colored_room_preview(
            labels=label_map,
            occupancy=np.asarray(arrays["occupancy_map"], dtype=bool),
            unknown=np.asarray(arrays["unknown_mask"], dtype=bool),
            hough=np.asarray(
                self.latest_debug_arrays[
                    "tvars_original_hough_door_seed_map"
                ],
                dtype=bool,
            ),
            filtered=np.asarray(
                self.latest_debug_arrays[
                    "tvars_original_filtered_hough_door_seed_map"
                ],
                dtype=bool,
            ),
            accepted=accepted_door_line_map,
        )
        self.latest_metadata = {
            **dict(self.latest_metadata),
            "snapshot_step": int(step),
            "evaluation_label_source": (
                "accepted_door_lines_partition_vertical_free_"
                "filter_lt_0p5m2_then_project_navigation_free"
            ),
            "door_partition": dict(partition_debug),
            "snapshot_partition_refresh": True,
            "snapshot_partition_refresh_source": (
                "vertical_free_partition_then_navigation_free_projection"
            ),
            "snapshot_partition_room_count": int(
                partition_debug.get("room_count", 0)
            ),
            "policy_state_mutated_by_snapshot_refresh": False,
        }

    def observe_policy_pose(
        self,
        *,
        step: int,
        current_rc: Sequence[int],
    ) -> None:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError(
                "policy pose observation is only valid under original topology control"
            )
        if self._policy_visited_map is not None:
            current = tuple(int(v) for v in np.asarray(current_rc).reshape(-1)[:2])
            if not (
                0 <= current[0] < self._policy_visited_map.shape[0]
                and 0 <= current[1] < self._policy_visited_map.shape[1]
            ):
                raise RuntimeError("observed original topology pose is outside the visited map")
            self._policy_visited_map[current] = True
        _ = step
        if self._active_policy_target_kind != "topology_exit_waypoint":
            return
        segment_index = self._active_transition_segment_index
        if segment_index is None:
            raise RuntimeError("topology exit target has no active transition segment")
        if not 0 <= int(segment_index) < len(self._transition_trajectories_xy):
            raise RuntimeError("active transition segment index is out of range")
        point_xy = [float(current_rc[1]), float(current_rc[0])]
        trace = self._transition_trajectories_xy[int(segment_index)]
        if not trace or trace[-1] != point_xy:
            trace.append(point_xy)

    def observe_scan_frame(
        self,
        *,
        obs: Mapping[str, Any],
        map_state: Mapping[str, Any],
        mapper: Any | None,
        camera_intrinsics: Any,
    ) -> None:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError("scan frames are only valid under original topology control")
        if not self._scan_required:
            raise RuntimeError("original topology scan frame received while no scan is pending")
        if len(self._scan_frames) >= int(self.panorama_views):
            raise RuntimeError("original topology scan already contains 12 views")
        arrays = _arrays_from_map_state(map_state)
        map_info = resolve_map_info(
            map_state=map_state,
            mapper=mapper,
            snapshot_arrays=arrays,
        )
        copied_obs = _copy_strict_rgbd_observation(obs)
        floor_z = _floor_z_from_map_state(map_state)
        self._scan_frames.append(
            _ScanFrame(
                obs=copied_obs,
                camera_intrinsics=camera_intrinsics,
                map_info=map_info,
                floor_z=float(floor_z),
            )
        )
        if len(self._scan_frames) == int(self.panorama_views):
            self._scan_required = False
            self._policy_update_required = True

    def mark_policy_target_reached(
        self,
        *,
        step: int,
        current_rc: Sequence[int],
        reached_via: str = "radius",
    ) -> None:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError("policy target arrival is only valid under original topology control")
        if self._active_policy_target_rc is None:
            raise RuntimeError("original topology has no active target to mark reached")
        reached_via = str(reached_via).strip().lower()
        if reached_via not in {
            "radius",
            "fmm_reachable_boundary",
            "astar_collision_boundary",
        }:
            raise ValueError(
                "unsupported original topology target completion mode: %s"
                % reached_via
            )
        reached_target = self._active_policy_target_rc
        reached_kind = self._active_policy_target_kind
        reached_radius_m = float(self._active_policy_reach_radius_m)
        reached_global_frontier_already_registered = False
        if reached_kind == "global_voxroom_frontier" and reached_via == "radius":
            reached_global_frontier_already_registered = (
                reached_target in self._reached_global_frontier_targets
            )
            self._reached_global_frontier_targets.add(reached_target)
            # Arrival is audit history, not a terminal blacklist.  If the
            # same Vertical Free frontier survives the arrival scan it stays
            # eligible, but the original one-step last-goal suppression keeps
            # it from being selected again immediately when alternatives exist.
        if reached_kind == "topology_exit_waypoint":
            self.observe_policy_pose(step=int(step), current_rc=current_rc)
            self._transition_reached_exit_count += 1
        self._policy_target_reached_count += 1
        self._active_policy_target_rc = None
        self._active_policy_target_kind = None
        self._active_policy_reach_radius_m = 0.0
        if self._transition_waypoints_rc:
            self._scan_required = False
            next_target = self._transition_waypoints_rc.pop(0)
            if reached_kind != "topology_exit_waypoint":
                raise RuntimeError(
                    "queued topology waypoints followed a non-transition target"
                )
            assert self._active_transition_segment_index is not None
            self._active_transition_segment_index += 1
            next_trace = self._transition_trajectories_xy[
                self._active_transition_segment_index
            ]
            next_trace.append([float(current_rc[1]), float(current_rc[0])])
            self._activate_policy_target(
                step=int(step),
                target_rc=next_target,
                target_kind="topology_exit_waypoint",
                phase="room_transition_navigation",
                reason="original_choose_door_next_waypoint",
                reach_radius_m=reached_radius_m,
                metadata={
                    "queued_transition_waypoints": int(
                        len(self._transition_waypoints_rc)
                    )
                },
            )
            self._policy_update_required = False
        else:
            if reached_kind == "topology_exit_waypoint":
                self._active_transition_segment_index = None
                self._transition_entry_scan_pending = True
            self._scan_frames.clear()
            self._scan_required = True
            self._policy_update_required = True
        self._record_policy_event(
            {
                "event": "target_reached",
                "step": int(step),
                "current_rc": [int(current_rc[0]), int(current_rc[1])],
                "target_rc": [int(reached_target[0]), int(reached_target[1])],
                "target_kind": reached_kind,
                "reached_via": reached_via,
                "reachable_boundary_reached": bool(
                    reached_via
                    in {
                        "fmm_reachable_boundary",
                        "astar_collision_boundary",
                    }
                ),
                "target_reached_count": int(self._policy_target_reached_count),
                "next_scan_required": bool(self._scan_required),
                "queued_transition_waypoints": int(len(self._transition_waypoints_rc)),
                "transition_reached_exit_count": int(
                    self._transition_reached_exit_count
                ),
                "transition_entry_scan_pending": bool(
                    self._transition_entry_scan_pending
                ),
                "terminal_frontier_target_count": 0,
                "reached_global_frontier_registered": bool(
                    reached_kind == "global_voxroom_frontier"
                    and reached_via == "radius"
                ),
                "reached_global_frontier_already_registered": bool(
                    reached_global_frontier_already_registered
                ),
                "reached_global_frontier_target_count": int(
                    len(self._reached_global_frontier_targets)
                ),
            }
        )

    def mark_policy_target_unreachable(
        self,
        *,
        step: int,
        current_rc: Sequence[int],
        failure_reason: str,
    ) -> str:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            raise RuntimeError(
                "policy target rejection is only valid under original topology control"
            )
        if self._active_policy_target_rc is None:
            raise RuntimeError("original topology has no active target to reject")
        target_kind = self._active_policy_target_kind
        if target_kind in {"room_entry_return", "topology_exit_waypoint"}:
            return self._record_original_navigation_no_path(
                step=int(step),
                current_rc=current_rc,
                failure_reason=str(failure_reason),
            )
        if target_kind == "global_voxroom_frontier":
            target_rc = tuple(int(v) for v in self._active_policy_target_rc)
            already_rejected = (
                target_rc in self._astar_unreachable_global_frontier_targets
            )
            self._astar_unreachable_global_frontier_targets.add(target_rc)
            self._policy_target_unreachable_count += 1
            self._active_policy_target_rc = None
            self._active_policy_target_kind = None
            self._active_policy_reach_radius_m = 0.0
            self._active_policy_no_path_step_count = 0
            self._scan_frames.clear()
            self._scan_required = True
            self._policy_update_required = True
            event_metadata = {
                "current_rc": [int(current_rc[0]), int(current_rc[1])],
                "target_rc": [int(target_rc[0]), int(target_rc[1])],
                "target_kind": target_kind,
                "failure_reason": str(failure_reason),
                "already_rejected": bool(already_rejected),
                "policy_target_unreachable_count": int(
                    self._policy_target_unreachable_count
                ),
                "rejected_global_frontier_target_count": int(
                    len(self._astar_unreachable_global_frontier_targets)
                ),
                "next_scan_required": True,
                "planner_fallback_used": False,
            }
            self._record_policy_event(
                {
                    "event": "target_unreachable",
                    "step": int(step),
                    **event_metadata,
                }
            )
            self._set_policy_decision(
                step=int(step),
                target_rc=None,
                stop=False,
                reason=(
                    "original_global_frontier_reselection_after_astar_unreachable"
                ),
                phase="scan_required",
                target_kind=None,
                reach_radius_m=0.0,
                metadata=event_metadata,
            )
            return "global_voxroom_frontier_rejected"
        if target_kind != "room_frontier":
            raise RuntimeError(
                "strict original topology cannot reject a non-frontier target as "
                f"unreachable: {target_kind}"
            )
        target_rc = tuple(int(v) for v in self._active_policy_target_rc)
        rejected_targets = self._astar_unreachable_room_frontier_targets
        already_rejected = target_rc in rejected_targets
        rejected_targets.add(target_rc)
        self._policy_target_unreachable_count += 1
        self._active_policy_target_rc = None
        self._active_policy_target_kind = None
        self._active_policy_reach_radius_m = 0.0
        self._active_policy_no_path_step_count = 0
        self._scan_frames.clear()
        self._scan_required = True
        self._policy_update_required = True
        event_metadata = {
            "current_rc": [int(current_rc[0]), int(current_rc[1])],
            "target_rc": [int(target_rc[0]), int(target_rc[1])],
            "target_kind": target_kind,
            "failure_reason": str(failure_reason),
            "already_rejected": bool(already_rejected),
            "policy_target_unreachable_count": int(
                self._policy_target_unreachable_count
            ),
            "rejected_target_count_for_kind": int(len(rejected_targets)),
            "rejected_target_count_total": int(len(rejected_targets)),
            "next_scan_required": True,
            "planner_fallback_used": False,
        }
        self._record_policy_event(
            {
                "event": "target_unreachable",
                "step": int(step),
                **event_metadata,
            }
        )
        self._set_policy_decision(
            step=int(step),
            target_rc=None,
            stop=False,
            reason="original_frontier_reselection_after_astar_unreachable",
            phase="scan_required",
            target_kind=None,
            reach_radius_m=0.0,
            metadata=event_metadata,
        )
        return "room_frontier_rejected"

    def _record_original_navigation_no_path(
        self,
        *,
        step: int,
        current_rc: Sequence[int],
        failure_reason: str,
    ) -> str:
        target_kind = str(self._active_policy_target_kind)
        target_rc = tuple(int(v) for v in self._active_policy_target_rc or ())
        if len(target_rc) != 2:
            raise RuntimeError("original topology navigation target is invalid")
        limit = (
            ORIGINAL_TOPOLOGY_EXIT_STEP_LIMIT
            if target_kind == "topology_exit_waypoint"
            else ORIGINAL_ROOM_ENTRY_RETURN_STEP_LIMIT
        )
        self._active_policy_no_path_step_count = 1
        attempt_count = 1
        if target_kind == "topology_exit_waypoint":
            self.observe_policy_pose(step=int(step), current_rc=current_rc)
        event_metadata = {
            "current_rc": [int(current_rc[0]), int(current_rc[1])],
            "target_rc": [int(target_rc[0]), int(target_rc[1])],
            "target_kind": target_kind,
            "failure_reason": str(failure_reason),
            "no_path_step_count": attempt_count,
            "source_step_limit": int(limit),
            "source_zero_motion_attempts_skipped": int(limit),
            "conclusive_astar_no_path": True,
            "planner_fallback_used": False,
        }

        reach_radius_m = float(self._active_policy_reach_radius_m)
        self._policy_target_unreachable_count += 1
        self._active_policy_target_rc = None
        self._active_policy_target_kind = None
        self._active_policy_reach_radius_m = 0.0
        self._active_policy_no_path_step_count = 0
        outcome = f"{target_kind}_exhausted"
        if target_kind == "topology_exit_waypoint":
            self._transition_failed_exit_count += 1
            if self._transition_waypoints_rc:
                self._scan_required = False
                next_target = self._transition_waypoints_rc.pop(0)
                assert self._active_transition_segment_index is not None
                self._active_transition_segment_index += 1
                next_trace = self._transition_trajectories_xy[
                    self._active_transition_segment_index
                ]
                next_trace.append([float(current_rc[1]), float(current_rc[0])])
                self._activate_policy_target(
                    step=int(step),
                    target_rc=next_target,
                    target_kind="topology_exit_waypoint",
                    phase="room_transition_navigation",
                    reason="original_choose_door_next_waypoint_after_step_limit",
                    reach_radius_m=reach_radius_m,
                    metadata={
                        **event_metadata,
                        "queued_transition_waypoints": int(
                            len(self._transition_waypoints_rc)
                        ),
                    },
                )
                self._policy_update_required = False
            else:
                self._active_transition_segment_index = None
                self._transition_entry_scan_pending = True
                self._scan_frames.clear()
                self._scan_required = True
                self._policy_update_required = True
                self._set_policy_decision(
                    step=int(step),
                    target_rc=None,
                    stop=False,
                    reason="original_topology_exit_step_limit_reached",
                    phase="scan_required",
                    target_kind=None,
                    reach_radius_m=0.0,
                    metadata=event_metadata,
                )
        else:
            self._scan_frames.clear()
            self._scan_required = True
            self._policy_update_required = True
            self._set_policy_decision(
                step=int(step),
                target_rc=None,
                stop=False,
                reason="original_room_entry_return_step_limit_reached",
                phase="scan_required",
                target_kind=None,
                reach_radius_m=0.0,
                metadata=event_metadata,
            )
        self._record_policy_event(
            {
                "event": "target_unreachable",
                "step": int(step),
                **event_metadata,
                "outcome": outcome,
                "next_scan_required": bool(self._scan_required),
                "queued_transition_waypoints": int(
                    len(self._transition_waypoints_rc)
                ),
                "transition_reached_exit_count": int(
                    self._transition_reached_exit_count
                ),
                "transition_failed_exit_count": int(
                    self._transition_failed_exit_count
                ),
                "policy_target_unreachable_count": int(
                    self._policy_target_unreachable_count
                ),
            }
        )
        return outcome

    def update(
        self,
        *,
        step: int,
        obs: Mapping[str, Any] | None,
        sgnav_obs: Mapping[str, Any] | None = None,
        map_state: Mapping[str, Any],
        mapper: Any | None = None,
        room_segmenter: Any | None = None,
        frontier_map: np.ndarray | None = None,
        selected_frontier_center_rc: tuple[int, int] | None = None,
        camera_intrinsics: Any | None = None,
    ) -> None:
        if self._last_update_step == int(step):
            return
        if (
            self.policy_control == ORIGINAL_POLICY_CONTROL
            and not self._policy_update_required
        ):
            return
        if (
            self.policy_control == ORIGINAL_POLICY_CONTROL
            and self._policy_update_required
            and self._scan_required
        ):
            raise RuntimeError(
                "original topology policy update requires a completed 12-view scan"
            )
        _ = sgnav_obs, room_segmenter, selected_frontier_center_rc
        arrays = _arrays_from_map_state(map_state)
        map_info = resolve_map_info(map_state=map_state, mapper=mapper, snapshot_arrays=arrays)
        shape = tuple(int(v) for v in arrays["occupancy_map"].shape)
        global_frontier_map = np.zeros(shape, dtype=bool)
        global_frontier_list: list[np.ndarray] = []
        global_frontier_info_gain_list: list[float] = []
        global_frontier_metadata: dict[str, Any] = {
            "runtime_global_frontier_raw_cells": 0,
            "runtime_global_frontier_component_count": 0,
        }
        if frontier_map is not None:
            global_frontier_map = np.asarray(frontier_map, dtype=bool)
            if global_frontier_map.shape != shape:
                raise RuntimeError(
                    "runtime global frontier map shape differs from the TVARS map: "
                    f"{global_frontier_map.shape} vs {shape}"
                )
            (
                global_frontier_list,
                global_frontier_info_gain_list,
                global_frontier_metadata,
            ) = _global_frontier_map_to_original_lists(
                global_frontier_map,
                avoid_targets_rc=tuple(self._reached_global_frontier_targets),
            )
        elif (
            self.policy_control == ORIGINAL_POLICY_CONTROL
            and self._policy_update_required
        ):
            raise RuntimeError(
                "original topology policy update requires the online global frontier map"
            )
        modules = self._ensure_original_modules(shape=shape, resolution_m=float(map_info.resolution_m))
        agent_rc = tuple(int(v) for v in np.asarray(map_state["current_grid"]).reshape(-1)[:2])
        floor_z = _floor_z_from_map_state(map_state)

        if self.save_stream:
            self._save_stream_frame(step=int(step), arrays=arrays, obs=obs or {}, map_info=map_info)

        metadata: dict[str, Any] = {
            "method": BASELINE_NAME,
            "runner_type": "original_tvars_modules_isaac_adapter",
            "original_repo": "FreeformRobotics/Active_room_segmentation",
            "original_repo_path": str(self.repo_dir),
            "original_repo_commit": _git_head(self.repo_dir),
            "environment_adapter": "isaac_no_habitat",
            "habitat_runtime_launched": False,
            "uses_rgb": True,
            "uses_depth": True,
            "uses_occupancy": True,
            "uses_oracle_semantics": False,
            "policy_control": str(self.policy_control),
            "baseline_policy_control": str(self.policy_control),
            "snapshot_step": int(step),
            "map_info": map_info.to_metadata(),
            "tvars_original_map_size_cells": int(modules.map_size_cells),
            "tvars_original_resolution_m": float(modules.resolution_m),
            "panorama_views": int(self.panorama_views),
            "scan_views_consumed": 0,
        }

        # The upstream Habitat MapBuilder supplies float32 maps. Keeping that
        # dtype is required because check_topomap applies np.rint() before
        # passing the map into OpenCV morphology.
        obs_map = np.asarray(arrays["obstacle_mask"], dtype=np.float32)
        exp_map = (~np.asarray(arrays["unknown_mask"], dtype=bool)).astype(
            np.float32
        )
        free_mask = np.asarray(arrays["navigation_free_room_domain"], dtype=bool)
        vertical_free_mask = np.asarray(
            arrays["vertical_free_room_domain"], dtype=bool
        )
        h, w = obs_map.shape
        agent_r = int(np.clip(agent_rc[0], 0, h - 1))
        agent_c = int(np.clip(agent_rc[1], 0, w - 1))
        yaw_deg = _yaw_deg_from_map_state(map_state)
        bot_pose_m = np.asarray([agent_r * modules.resolution_m, agent_c * modules.resolution_m, yaw_deg], dtype=np.float32)
        lmb = np.asarray([0, h, 0, w], dtype=np.int32)

        with _suppress_original_stdout():
            hough_door_list, laser_list = modules.convert_2_laser(obs_map.copy(), exp_map.copy(), bot_pose_m.copy())
        hough_door_list = _clip_xy_points(hough_door_list, shape=shape, margin=22)
        vision_candidates, vision_map, vision_meta = self._resolve_visual_door_evidence(
            obs=obs or {},
            camera_intrinsics=camera_intrinsics,
            map_info=map_info,
            shape=shape,
            floor_z=float(floor_z),
        )
        metadata["scan_views_consumed"] = int(
            vision_meta.get("scan_views_consumed", 0)
        )
        filtered_hough = _filter_hough_with_vision(
            hough_door_list,
            vision_map=vision_map,
            radius_cells=2,
        )

        scan_routes = _build_original_scan_routes(
            modules=modules,
            obstacle_map=obs_map,
            explored_map=exp_map,
            agent_rc=(agent_r, agent_c),
            yaw_deg=yaw_deg,
        )
        if self._door_detect is None:
            first_x, first_y = scan_routes[0]
            self._door_detect = modules.DoorDetection(first_x, first_y)
        else:
            first_x, first_y = scan_routes[0]
            self._door_detect.reset(first_x, first_y)
        for route_x, route_y in scan_routes[1:]:
            self._door_detect.new_list(route_x, route_y)
        if self._frontier_detector is None:
            self._frontier_detector = modules.FrontierDetection(int(h))
        if self._topomap is None:
            self._topomap = modules.TopomapConstruction(
                map_size=int(h),
                vision_range=60,
            )

        door_list: list[dict[str, Any]] = []
        raw_list: list[list[int]] = []
        room_exp_list: list[list[int]] = []
        door_grid: list[list[int]] = []
        close_door_list: list[dict[str, Any]] = []
        door_filter_debug_list: list[dict[str, Any]] = []
        checked_doors: list[dict[str, Any]] = []
        door_remove_list: list[dict[str, Any]] = []
        f_list: list[Any] = []
        info_gain_list: list[Any] = []
        in_point: Any = None
        return_flag = False
        with _suppress_original_stdout():
            door_list, raw_list = self._door_detect.door_filter(
                obs_map.copy(),
                obs_map.copy(),
                exp_map.copy(),
                [agent_c, agent_r],
                [],
                self._detected_door_list,
                use_12point=False,
                external_door_point=filtered_hough,
            )
            close_door_list = list(door_list)
            door_filter_debug_list = [dict(door) for door in door_list]
            self._detected_door_list.extend(door_list)
            self._raw_detect_list.extend(raw_list)
            f_list, info_gain_list, show_map, room_exp_list, door_grid = self._frontier_detector.frontier_detection(
                np.asarray([agent_r, agent_c], dtype=np.int32),
                np.asarray([agent_r, agent_c], dtype=np.int32),
                obs_map.copy(),
                exp_map.copy(),
                lmb.copy(),
                self._detected_door_list,
                laser_list,
            )
            self._detected_door_list = self._topomap.same_node_check(room_exp_list, self._detected_door_list)
            checked_doors, door_remove_list = self._topomap.check_topomap(
                close_door_list,
                self._detected_door_list,
                room_exp_list,
                [agent_c, agent_r],
                obs_map.copy(),
                exp_map.copy(),
                lmb.copy(),
                door_grid,
                0,
                "isaac",
                laser_list,
            )
            in_point, return_flag = self._topomap.add_room(
                checked_doors,
                [agent_c, agent_r],
                obs_map.copy(),
                exp_map.copy(),
                lmb.copy(),
            )
        for door in list(door_remove_list):
            if door in self._detected_door_list:
                self._detected_door_list.remove(door)

        # TVARS keeps accepted doors across scans.  Use that persistent door
        # memory to close the current room before the shared Vertical Free
        # frontier is handed to the original distance/information-gain policy.
        accepted_door_line_map = _doors_to_line_map(
            self._detected_door_list,
            shape=shape,
        )
        (
            current_room_frontier_map,
            current_room_vertical_free_mask,
            current_room_frontier_metadata,
        ) = _restrict_vertical_frontiers_to_current_tvars_room(
            global_frontier_map=global_frontier_map,
            vertical_free_mask=vertical_free_mask,
            persistent_door_line_map=accepted_door_line_map,
            agent_rc=(agent_r, agent_c),
        )
        (
            current_room_frontier_list,
            current_room_frontier_info_gain_list,
            current_room_component_metadata,
        ) = _global_frontier_map_to_original_lists(
            current_room_frontier_map,
        )
        current_room_frontier_metadata = {
            **dict(current_room_frontier_metadata),
            **{
                str(key).replace("runtime_global_", "runtime_room_", 1): value
                for key, value in current_room_component_metadata.items()
            },
            "original_internal_frontier_used_for_policy": False,
            "original_internal_frontier_ignored_count": int(len(f_list)),
        }

        if self._transition_entry_scan_pending:
            self._complete_pending_transition_entry_scan(step=int(step))

        if (
            self.policy_control == ORIGINAL_POLICY_CONTROL
            and self._policy_update_required
        ):
            self._update_original_policy(
                step=int(step),
                agent_rc=(agent_r, agent_c),
                frontier_list=current_room_frontier_list,
                info_gain_list=current_room_frontier_info_gain_list,
                room_frontier_metadata=current_room_frontier_metadata,
                global_frontier_list=global_frontier_list,
                global_info_gain_list=global_frontier_info_gain_list,
                global_frontier_metadata=global_frontier_metadata,
                in_point=in_point,
                return_flag=bool(return_flag),
                resolution_m=float(modules.resolution_m),
            )
            self._policy_update_required = bool(self._scan_required)

        original_topomap_label_map = self._label_map_from_original_topomap(
            shape=shape,
            domain=free_mask,
        )
        frontier_room_exp_map = _rc_points_to_map(room_exp_list, shape=shape)
        frontier_door_grid_map = _rc_points_to_map(door_grid, shape=shape)
        door_filter_line_map = _doors_to_line_map(door_filter_debug_list, shape=shape)
        vertical_label_map, partition_debug = partition_free_space_by_door_lines(
            vertical_free_mask,
            accepted_door_line_map,
        )
        vertical_label_map, label_map, projection_debug = (
            _filter_and_project_partition_labels(
                vertical_label_map,
                navigation_free=free_mask,
                resolution_m=float(modules.resolution_m),
                min_room_area_m2=TVARS_MIN_ROOM_AREA_M2,
            )
        )
        partition_debug = {**dict(partition_debug), **projection_debug}
        self.latest_label_map = label_map
        self.latest_debug_arrays = {
            "tvars_original_hough_door_seed_map": _xy_points_to_map(hough_door_list, shape=shape),
            "tvars_original_filtered_hough_door_seed_map": _xy_points_to_map(filtered_hough, shape=shape),
            "tvars_original_vision_door_seed_map": vision_map.astype(bool),
            "tvars_original_door_filter_line_map": door_filter_line_map,
            "tvars_original_door_line_map": accepted_door_line_map,
            "tvars_original_frontier_room_exp_map": frontier_room_exp_map,
            "tvars_original_frontier_door_grid_map": frontier_door_grid_map,
            "tvars_original_frontier_show_map": np.asarray(show_map, dtype=np.float32),
            "tvars_original_runtime_global_frontier_map": global_frontier_map,
            "tvars_original_current_room_vertical_free_mask": (
                current_room_vertical_free_mask
            ),
            "tvars_original_current_room_frontier_map": (
                current_room_frontier_map
            ),
            "tvars_original_topomap_label_map": original_topomap_label_map.astype(np.int32),
            "tvars_original_door_partition_label_map": label_map.astype(np.int32),
            "tvars_original_vertical_free_partition_label_map": (
                vertical_label_map.astype(np.int32)
            ),
            "tvars_original_navigation_projected_label_map": (
                label_map.astype(np.int32)
            ),
            "tvars_original_policy_target_map": _policy_target_map(
                self._latest_policy_decision.target_rc,
                shape=shape,
            ),
        }
        self.latest_preview_rgb = _render_colored_room_preview(
            labels=label_map,
            occupancy=np.asarray(arrays["occupancy_map"], dtype=bool),
            unknown=np.asarray(arrays["unknown_mask"], dtype=bool),
            hough=self.latest_debug_arrays["tvars_original_hough_door_seed_map"],
            filtered=self.latest_debug_arrays[
                "tvars_original_filtered_hough_door_seed_map"
            ],
            accepted=self.latest_debug_arrays["tvars_original_door_line_map"],
        )
        metadata.update(
            {
                "detector_name": str(getattr(self.detector, "name", "unknown")),
                "detector_available": bool(getattr(self.detector, "available", False)),
                "detector_adapter_verified": bool(getattr(self.detector, "detector_adapter_verified", False)),
                "checkpoint_path": getattr(self.detector, "checkpoint_path", None),
                "checkpoint_sha256": getattr(self.detector, "checkpoint_sha256", None),
                "vision_num_detections": int(vision_meta.get("num_detections", 0)),
                "vision_num_candidates": int(len(vision_candidates)),
                "vision_projection_attempted": bool(vision_meta.get("projection_attempted", False)),
                "vision_projection_missing_inputs": list(vision_meta.get("missing_inputs", [])),
                "vision_projection_status_counts": dict(vision_meta.get("projection_status_counts", {})),
                "vision_projection_mode": "original_mask_depth_map",
                "vision_projected_mask_pixels": int(vision_meta.get("projected_mask_pixels", 0)),
                "vision_projected_grid_cells": int(vision_meta.get("projected_grid_cells", 0)),
                "vision_mask_pixel_stride": int(ORIGINAL_DOOR_MASK_PIXEL_STRIDE),
                "vision_depth_max_m": float(ORIGINAL_VISION_RANGE_M),
                "vision_relative_z_band_m": [
                    float(ORIGINAL_DOOR_RELATIVE_Z_MIN_M),
                    float(ORIGINAL_DOOR_RELATIVE_Z_MAX_M),
                ],
                "vision_floor_reference": "map_state_base_pose_z",
                "vision_floor_z_m": float(floor_z),
                "vision_evidence_scope": "latest_completed_scan",
                "vision_rgb_square_crop_applied": bool(vision_meta.get("rgb_square_crop_applied", False)),
                "vision_rgb_square_crop_xyxy": vision_meta.get("rgb_square_crop_xyxy"),
                "vision_rgb_original_shape_hw": vision_meta.get("rgb_original_shape_hw"),
                "vision_rgb_detector_shape_hw": vision_meta.get("rgb_detector_shape_hw"),
                "vision_depth_square_crop_applied": bool(vision_meta.get("depth_square_crop_applied", False)),
                "hough_num_candidates": int(len(hough_door_list)),
                "hough_num_rgb_filtered_candidates": int(len(filtered_hough)),
                "hough_rgb_filter_radius_cells": 2,
                "door_filter_num_raw": int(len(raw_list)),
                "door_filter_num_accepted_this_step": int(len(door_list)),
                "topomap_num_checked_doors": int(len(checked_doors)),
                "topomap_num_removed_doors": int(len(door_remove_list)),
                "door_memory_num_detected": int(len(self._detected_door_list)),
                **dict(current_room_frontier_metadata),
                "topomap_num_vertices": int(_topomap_vertex_count(self._topomap)),
                "original_topomap_label_count": int(np.max(original_topomap_label_map))
                if original_topomap_label_map.size
                else 0,
                "evaluation_label_source": (
                    "accepted_door_lines_partition_vertical_free_"
                    "filter_lt_0p5m2_then_project_navigation_free"
                ),
                "door_partition": dict(partition_debug),
                "isaac_adapter_frontier_room_exp_cells": int(np.count_nonzero(frontier_room_exp_map)),
                "return_flag": bool(return_flag),
                "original_frontier_count": int(len(f_list)),
                "original_frontier_info_gain_count": int(len(info_gain_list)),
                "original_internal_frontier_used_for_policy": False,
                "policy_frontier_source": "voxel_vertical_free",
                **dict(global_frontier_metadata),
                "original_scan_route_count": int(len(scan_routes)),
                "original_scan_route_lengths": [
                    int(len(route_x)) for route_x, _ in scan_routes
                ],
                "original_policy": self._latest_policy_decision.to_metadata(),
                "original_policy_target_reached_count": int(
                    self._policy_target_reached_count
                ),
                "original_policy_scan_required": bool(self._scan_required),
                "original_policy_scan_views_collected": int(len(self._scan_frames)),
                "original_policy_transition_waypoints_queued": int(
                    len(self._transition_waypoints_rc)
                ),
                "original_policy_transition_entry_scan_pending": bool(
                    self._transition_entry_scan_pending
                ),
                "original_policy_last_transition_advance": (
                    self._last_transition_advance
                ),
                "main_experiment_allowed": True,
                "strict_original_core_no_fallback": True,
                "full_original_habitat_sensor_pipeline": False,
                "approximation_note": (
                    "Original TVARS door/frontier/topomap modules are used, but Habitat is not launched; Isaac supplies RGB-D and maps."
                ),
            }
        )
        self.latest_metadata = metadata
        self._last_update_step = int(step)

    def save_snapshot_like_voxroom(
        self,
        *,
        source_snapshot_npz: Path | Mapping[str, Any],
        step: int,
        source_summary_json: Path | None = None,
        output_dir: Path | None = None,
        excluded_source_keys: tuple[str, ...] | None = None,
    ) -> Path:
        source_snapshot_npz = _snapshot_npz_path(source_snapshot_npz)
        with np.load(source_snapshot_npz, allow_pickle=False) as data:
            occupancy = np.asarray(data["occupancy_map"])
        label_map = self.latest_label_map
        if label_map is None:
            raise RuntimeError(
                "strict original Active Room has no completed topology result or "
                "accepted-door-line partition for this snapshot"
            )
        if label_map.shape != occupancy.shape:
            raise RuntimeError(
                f"strict original Active Room label shape {label_map.shape} "
                f"does not match source occupancy {occupancy.shape}"
            )
        snapshot_root = self.output_dir if output_dir is None else Path(output_dir)
        output_npz = snapshot_root / "roomseg_snapshots" / source_snapshot_npz.name
        preview_path = output_npz.with_suffix(".png")
        metadata = {
            **dict(self.latest_metadata),
            "source_snapshot": str(source_snapshot_npz),
            "source_summary_json": None if source_summary_json is None else str(source_summary_json),
            "snapshot_step": int(step),
            "save_every_snapshot": bool(self.save_every_snapshot),
            "colored_room_preview": str(preview_path),
        }
        if excluded_source_keys:
            metadata["shared_voxel_snapshot"] = str(source_snapshot_npz)
            metadata["excluded_shared_source_arrays"] = list(
                excluded_source_keys
            )
        save_baseline_snapshot_npz(
            source_npz_path=source_snapshot_npz,
            output_npz_path=output_npz,
            baseline_label_map=label_map,
            baseline_name=BASELINE_NAME,
            metadata=metadata,
            debug_arrays=self.latest_debug_arrays,
            excluded_source_keys=excluded_source_keys,
        )
        _save_colored_room_preview(output_npz, preview_path)
        _write_summary_json(
            output_npz.with_suffix(".summary.json"),
            output_npz=output_npz,
            source_npz=source_snapshot_npz,
            metadata=metadata,
            label_map=label_map,
        )
        return output_npz

    def on_episode_end(self) -> None:
        return None

    def _ensure_original_modules(self, *, shape: tuple[int, int], resolution_m: float) -> _OriginalModules:
        if self._modules is not None:
            return self._modules
        if shape[0] != shape[1]:
            raise ValueError("TVARS original map expects square maps, got %s" % (shape,))
        if abs(float(resolution_m) - 0.05) > 1e-4:
            raise ValueError("TVARS original hard-codes 0.05 m cells, got %.6f" % float(resolution_m))
        self._modules = _load_original_modules(
            repo_dir=self.repo_dir,
            map_size_cells=int(shape[0]),
            resolution_m=float(resolution_m),
        )
        return self._modules

    def _resolve_visual_door_evidence(
        self,
        *,
        obs: Mapping[str, Any],
        camera_intrinsics: Any | None,
        map_info: MapInfo,
        shape: tuple[int, int],
        floor_z: float,
    ) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            candidates, vision_map, meta = self._project_visual_door_candidates(
                obs=obs,
                camera_intrinsics=camera_intrinsics,
                map_info=map_info,
                shape=shape,
                floor_z=float(floor_z),
            )
            meta["scan_views_consumed"] = 1
            return candidates, vision_map, meta

        if self._vision_evidence_map is None:
            self._vision_evidence_map = np.zeros(shape, dtype=bool)
        elif self._vision_evidence_map.shape != shape:
            raise RuntimeError(
                "original topology visual evidence shape changed from %s to %s"
                % (self._vision_evidence_map.shape, shape)
            )

        if not self._scan_frames:
            if self._completed_scan_count <= 0:
                raise RuntimeError(
                    "original topology has no visual door evidence from a completed scan"
                )
            return [], self._vision_evidence_map.copy(), {
                "num_detections": 0,
                "projection_attempted": False,
                "missing_inputs": [],
                "projection_status_counts": {},
                "projected_mask_pixels": 0,
                "projected_grid_cells": 0,
                "rgb_square_crop_applied": False,
                "depth_square_crop_applied": False,
                "scan_views_consumed": 0,
                "vision_evidence_reused": True,
            }

        if len(self._scan_frames) != int(self.panorama_views):
            raise RuntimeError(
                "original topology policy requires 12 scan frames, found %d"
                % len(self._scan_frames)
            )
        rgb_frames = [np.asarray(frame.obs["rgb"]) for frame in self._scan_frames]
        detections_by_frame = self.detector.detect_batch(rgb_frames)
        if len(detections_by_frame) != len(self._scan_frames):
            raise RuntimeError(
                "original DETR returned %d frame results for %d scan frames"
                % (len(detections_by_frame), len(self._scan_frames))
            )

        candidates: list[Any] = []
        scan_map = np.zeros(shape, dtype=bool)
        merged_meta: dict[str, Any] = {
            "num_detections": 0,
            "projection_attempted": True,
            "missing_inputs": [],
            "projection_status_counts": {},
            "projected_mask_pixels": 0,
            "projected_grid_cells": 0,
            "rgb_square_crop_applied": False,
            "depth_square_crop_applied": False,
            "scan_views_consumed": int(len(self._scan_frames)),
            "vision_evidence_reused": False,
        }
        for frame, detections in zip(self._scan_frames, detections_by_frame):
            frame_candidates, frame_map, frame_meta = (
                self._project_visual_door_candidates(
                    obs=frame.obs,
                    camera_intrinsics=frame.camera_intrinsics,
                    map_info=frame.map_info,
                    shape=shape,
                    floor_z=float(frame.floor_z),
                    detections=detections,
                )
            )
            candidates.extend(frame_candidates)
            scan_map |= frame_map
            merged_meta["num_detections"] = int(
                merged_meta["num_detections"]
            ) + int(frame_meta.get("num_detections", 0))
            merged_counts = merged_meta["projection_status_counts"]
            for key, value in dict(
                frame_meta.get("projection_status_counts", {})
            ).items():
                merged_counts[str(key)] = int(merged_counts.get(str(key), 0)) + int(
                    value
                )
            merged_meta["projected_mask_pixels"] = int(
                merged_meta["projected_mask_pixels"]
            ) + int(frame_meta.get("projected_mask_pixels", 0))
            merged_meta["projected_grid_cells"] = int(
                merged_meta["projected_grid_cells"]
            ) + int(frame_meta.get("projected_grid_cells", 0))
        merged_meta["projected_grid_cells"] = int(np.count_nonzero(scan_map))
        self._vision_evidence_map = scan_map.copy()
        self._scan_frames.clear()
        self._completed_scan_count += 1
        merged_meta["completed_scan_count"] = int(self._completed_scan_count)
        return candidates, self._vision_evidence_map.copy(), merged_meta

    def _update_original_policy(
        self,
        *,
        step: int,
        agent_rc: tuple[int, int],
        frontier_list: Sequence[Any],
        info_gain_list: Sequence[Any],
        room_frontier_metadata: Mapping[str, Any],
        global_frontier_list: Sequence[Any],
        global_info_gain_list: Sequence[Any],
        global_frontier_metadata: Mapping[str, Any],
        in_point: Any,
        return_flag: bool,
        resolution_m: float,
    ) -> None:
        if self._active_policy_target_rc is not None:
            return

        if self._transition_waypoints_rc:
            target_rc = self._transition_waypoints_rc.pop(0)
            self._activate_policy_target(
                step=step,
                target_rc=target_rc,
                target_kind="topology_exit_waypoint",
                phase="room_transition_navigation",
                reason="original_choose_door_next_waypoint",
                reach_radius_m=2.0 * float(resolution_m),
                metadata={
                    "queued_transition_waypoints": int(
                        len(self._transition_waypoints_rc)
                    )
                },
            )
            return

        if bool(return_flag):
            if in_point is None or len(in_point) < 2:
                raise RuntimeError(
                    "original add_room requested return navigation without an entry point"
                )
            target_rc = (int(round(float(in_point[1]))), int(round(float(in_point[0]))))
            self._activate_policy_target(
                step=step,
                target_rc=target_rc,
                target_kind="room_entry_return",
                phase="room_search_navigation",
                reason="original_add_room_return_flag",
                reach_radius_m=10.0 * float(resolution_m),
                metadata={"return_flag": True},
            )
            return

        # Preserve TVARS' source ordering: explore frontiers in the current
        # room first, and only ask the topological map for a door transition
        # after that room has no selectable frontier.  The frontier geometry
        # is shared Vertical Free, while the room constraint comes from
        # TVARS' own persistent accepted-door list.
        target_rc, frontier_meta = _select_original_frontier_target(
            frontier_list=frontier_list,
            info_gain_list=info_gain_list,
            agent_rc=agent_rc,
            last_goal_rc=self._last_frontier_goal_rc,
            excluded_targets_rc=tuple(
                self._astar_unreachable_room_frontier_targets
            ),
        )
        frontier_meta = {
            **dict(room_frontier_metadata),
            **dict(frontier_meta),
            "policy_frontier_source": "voxel_vertical_free_current_room",
            "runtime_terminal_frontier_registry_enabled": False,
            "runtime_original_astar_unreachable_registry_enabled": True,
        }
        if target_rc is not None:
            self._last_frontier_goal_rc = target_rc
            self._activate_policy_target(
                step=step,
                target_rc=target_rc,
                target_kind="room_frontier",
                phase="room_search_navigation",
                reason="unified_vertical_free_current_room_frontier",
                reach_radius_m=10.0 * float(resolution_m),
                metadata=frontier_meta,
            )
            return

        with _suppress_original_stdout():
            exit_waypoints_xy = self._topomap.choose_door(
                [int(agent_rc[0]), int(agent_rc[1])]
            )
        self._begin_transition(
            agent_rc=agent_rc,
            exit_waypoints_xy=list(exit_waypoints_xy or []),
        )
        if self._transition_waypoints_rc:
            target_rc = self._transition_waypoints_rc.pop(0)
            self._activate_policy_target(
                step=step,
                target_rc=target_rc,
                target_kind="topology_exit_waypoint",
                phase="room_transition_navigation",
                reason="original_choose_door",
                reach_radius_m=2.0 * float(resolution_m),
                metadata={
                    **frontier_meta,
                    "queued_transition_waypoints": int(
                        len(self._transition_waypoints_rc)
                    ),
                },
            )
            return

        if bool(self._topomap.stop_exp()):
            global_frontiers = list(global_frontier_list)
            global_gains = list(global_info_gain_list)
            if len(global_frontiers) != len(global_gains):
                raise RuntimeError(
                    "global frontier and information-gain counts differ: %d vs %d"
                    % (len(global_frontiers), len(global_gains))
                )
            selectable_global_frontiers: list[Any] = []
            selectable_global_gains: list[Any] = []
            fresh_global_frontiers: list[Any] = []
            fresh_global_gains: list[Any] = []
            reached_history_matches_rc: list[tuple[int, int]] = []
            astar_unreachable_excluded_rc: list[tuple[int, int]] = []
            for frontier, gain in zip(global_frontiers, global_gains):
                values = np.asarray(frontier).reshape(-1)
                if values.size < 2:
                    raise RuntimeError(
                        "global frontier point has fewer than two coordinates"
                    )
                frontier_rc = (
                    int(round(float(values[0]))),
                    int(round(float(values[1]))),
                )
                matched_reached_history = _target_matches_any(
                    frontier_rc,
                    tuple(self._reached_global_frontier_targets),
                    radius_cells=RUNTIME_GLOBAL_FRONTIER_REACHED_MATCH_RADIUS_CELLS,
                )
                if matched_reached_history:
                    reached_history_matches_rc.append(frontier_rc)
                if _target_matches_any(
                    frontier_rc,
                    tuple(self._astar_unreachable_global_frontier_targets),
                    radius_cells=RUNTIME_GLOBAL_FRONTIER_REACHED_MATCH_RADIUS_CELLS,
                ):
                    astar_unreachable_excluded_rc.append(frontier_rc)
                    continue
                selectable_global_frontiers.append(frontier)
                selectable_global_gains.append(gain)
                if not matched_reached_history:
                    fresh_global_frontiers.append(frontier)
                    fresh_global_gains.append(gain)
            selection_global_frontiers = (
                fresh_global_frontiers
                if fresh_global_frontiers
                else selectable_global_frontiers
            )
            selection_global_gains = (
                fresh_global_gains
                if fresh_global_frontiers
                else selectable_global_gains
            )
            reached_global_meta = {
                "runtime_reached_global_frontier_registry_enabled": True,
                "runtime_reached_global_frontier_registry_used_for_exclusion": False,
                "runtime_reached_global_frontier_match_radius_cells": int(
                    RUNTIME_GLOBAL_FRONTIER_REACHED_MATCH_RADIUS_CELLS
                ),
                "runtime_reached_global_frontier_target_count": int(
                    len(self._reached_global_frontier_targets)
                ),
                "runtime_reached_global_frontier_excluded_count": int(
                    0
                ),
                "runtime_reached_global_frontier_excluded_rc": [],
                "runtime_global_frontier_near_reached_history_count": int(
                    len(reached_history_matches_rc)
                ),
                "runtime_global_frontier_near_reached_history_rc": [
                    [int(target[0]), int(target[1])]
                    for target in reached_history_matches_rc
                ],
                "runtime_global_frontier_selectable_after_reached_count": int(
                    len(selectable_global_frontiers)
                ),
                "runtime_global_frontier_fresh_count": int(
                    len(fresh_global_frontiers)
                ),
                "runtime_global_frontier_reached_history_fallback_used": bool(
                    selectable_global_frontiers and not fresh_global_frontiers
                ),
                "runtime_global_frontier_selection_candidate_count": int(
                    len(selection_global_frontiers)
                ),
                "runtime_global_frontier_astar_unreachable_registry_enabled": True,
                "runtime_global_frontier_astar_unreachable_target_count": int(
                    len(self._astar_unreachable_global_frontier_targets)
                ),
                "runtime_global_frontier_astar_unreachable_excluded_count": int(
                    len(astar_unreachable_excluded_rc)
                ),
                "runtime_global_frontier_astar_unreachable_excluded_rc": [
                    [int(target[0]), int(target[1])]
                    for target in astar_unreachable_excluded_rc
                ],
            }
            global_target_rc, global_selection_meta = (
                _select_original_frontier_target(
                    frontier_list=selection_global_frontiers,
                    info_gain_list=selection_global_gains,
                    agent_rc=agent_rc,
                    last_goal_rc=self._last_frontier_goal_rc,
                )
            )
            global_policy_meta = {
                **dict(global_frontier_metadata),
                **{
                    "runtime_global_%s" % str(key): value
                    for key, value in global_selection_meta.items()
                },
                **reached_global_meta,
                "runtime_global_frontier_unresolved_count": int(
                    len(selectable_global_frontiers)
                ),
            }
            if global_target_rc is not None:
                self._last_frontier_goal_rc = global_target_rc
                self._activate_policy_target(
                    step=step,
                    target_rc=global_target_rc,
                    target_kind="global_voxroom_frontier",
                    phase="global_frontier_navigation",
                    reason="original_topomap_complete_global_voxroom_frontier",
                    reach_radius_m=(
                        float(RUNTIME_GLOBAL_FRONTIER_REACH_RADIUS_CELLS)
                        * float(resolution_m)
                    ),
                    metadata={
                        **frontier_meta,
                        **global_policy_meta,
                        "target_source": "voxroom_online_global_frontier",
                        "uses_reference_map": False,
                    },
                )
                return
            if selectable_global_frontiers:
                self._last_frontier_goal_rc = None
                self._scan_frames.clear()
                self._scan_required = True
                self._set_policy_decision(
                    step=step,
                    target_rc=None,
                    stop=False,
                    reason="original_topomap_stop_blocked_by_global_voxroom_frontiers",
                    phase="scan_required",
                    target_kind=None,
                    reach_radius_m=0.0,
                    metadata={**frontier_meta, **global_policy_meta},
                )
                return
            if astar_unreachable_excluded_rc:
                self._set_policy_decision(
                    step=step,
                    target_rc=None,
                    stop=True,
                    reason=(
                        "original_topomap_all_remaining_global_frontiers_"
                        "astar_unreachable"
                    ),
                    phase="complete",
                    target_kind=None,
                    reach_radius_m=0.0,
                    metadata={**frontier_meta, **global_policy_meta},
                )
                return
            self._set_policy_decision(
                step=step,
                target_rc=None,
                stop=True,
                reason="original_topomap_all_rooms_explored",
                phase="complete",
                target_kind=None,
                reach_radius_m=0.0,
                metadata={**frontier_meta, **global_policy_meta},
            )
            return

        self._scan_frames.clear()
        self._scan_required = True
        self._set_policy_decision(
            step=step,
            target_rc=None,
            stop=False,
            reason="original_topomap_rescan_required",
            phase="scan_required",
            target_kind=None,
            reach_radius_m=0.0,
            metadata=frontier_meta,
        )

    def _begin_transition(
        self,
        *,
        agent_rc: tuple[int, int],
        exit_waypoints_xy: Sequence[Any],
    ) -> None:
        if self._transition_entry_scan_pending:
            raise RuntimeError(
                "cannot select another topology exit before the prior entry scan"
            )
        parsed_rc: list[tuple[int, int]] = []
        for point in exit_waypoints_xy:
            values = np.asarray(point).reshape(-1)
            if values.size < 2:
                raise RuntimeError(
                    "original topology exit waypoint has fewer than two coordinates"
                )
            parsed_rc.append(
                (
                    int(round(float(values[1]))),
                    int(round(float(values[0]))),
                )
            )
        self._transition_waypoints_rc = parsed_rc
        self._transition_reached_exit_count = 0
        self._transition_failed_exit_count = 0
        self._active_transition_segment_index = None
        self._transition_trajectories_xy = [
            [] for _ in self._transition_waypoints_rc
        ]
        if self._transition_waypoints_rc:
            self._active_transition_segment_index = 0
            self._transition_trajectories_xy[0].append(
                [float(agent_rc[1]), float(agent_rc[0])]
            )

    def _complete_pending_transition_entry_scan(self, *, step: int) -> None:
        if not self._transition_entry_scan_pending:
            raise RuntimeError("no original topology transition entry scan is pending")
        self._last_transition_advance = {
            "exit_goal_count": int(len(self._transition_trajectories_xy)),
            "reached_exit_count": int(
                self._transition_reached_exit_count
            ),
            "failed_exit_count": int(self._transition_failed_exit_count),
            "current_node_id": int(self._topomap.current_node_id),
            "transition_method": "upstream_source_semantics",
            "entry_scan_step": int(step),
        }
        self._record_policy_event(
            {
                "event": "room_transition_advanced",
                "step": int(step),
                **self._last_transition_advance,
            }
        )
        self._transition_entry_scan_pending = False
        self._transition_trajectories_xy = []
        self._transition_reached_exit_count = 0
        self._transition_failed_exit_count = 0

    def _activate_policy_target(
        self,
        *,
        step: int,
        target_rc: tuple[int, int],
        target_kind: str,
        phase: str,
        reason: str,
        reach_radius_m: float,
        metadata: Mapping[str, Any],
    ) -> None:
        self._active_policy_target_rc = (
            int(target_rc[0]),
            int(target_rc[1]),
        )
        self._active_policy_target_kind = str(target_kind)
        self._active_policy_reach_radius_m = float(reach_radius_m)
        self._active_policy_no_path_step_count = 0
        self._set_policy_decision(
            step=step,
            target_rc=self._active_policy_target_rc,
            stop=False,
            reason=reason,
            phase=phase,
            target_kind=target_kind,
            reach_radius_m=reach_radius_m,
            metadata=metadata,
        )

    def _set_policy_decision(
        self,
        *,
        step: int,
        target_rc: tuple[int, int] | None,
        stop: bool,
        reason: str,
        phase: str,
        target_kind: str | None,
        reach_radius_m: float,
        metadata: Mapping[str, Any],
    ) -> None:
        self._policy_decision_index += 1
        decision = OriginalPolicyDecision(
            target_rc=target_rc,
            stop=bool(stop),
            reason=str(reason),
            phase=str(phase),
            target_kind=target_kind,
            reach_radius_m=float(reach_radius_m),
            decision_index=int(self._policy_decision_index),
            metadata=dict(metadata),
        )
        self._latest_policy_decision = decision
        self._record_policy_event(
            {
                "event": "policy_decision",
                "step": int(step),
                **decision.to_metadata(),
            }
        )

    def _record_policy_event(self, payload: Mapping[str, Any]) -> None:
        if self.policy_control != ORIGINAL_POLICY_CONTROL:
            return
        self._policy_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._policy_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_ready(dict(payload)), ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    def _project_visual_door_candidates(
        self,
        *,
        obs: Mapping[str, Any],
        camera_intrinsics: Any | None,
        map_info: MapInfo,
        shape: tuple[int, int],
        floor_z: float,
        detections: Sequence[Any] | None = None,
    ) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
        vision_map = np.zeros(shape, dtype=bool)
        meta = {
            "num_detections": 0,
            "projection_attempted": False,
            "missing_inputs": [],
            "projection_status_counts": {},
            "projected_mask_pixels": 0,
            "projected_grid_cells": 0,
            "rgb_square_crop_applied": False,
            "depth_square_crop_applied": False,
        }
        if not bool(getattr(self.detector, "available", False)):
            raise RuntimeError("the original Active Room DETR detector is unavailable")
        if not (obs.get("has_rgb") and "rgb" in obs):
            meta["missing_inputs"].append("rgb")
        depth = obs.get("depth")
        camera_pose_world = obs.get("camera_pose_world")
        if depth is None:
            meta["missing_inputs"].append("depth")
        if camera_intrinsics is None:
            meta["missing_inputs"].append("camera_intrinsics")
        if camera_pose_world is None:
            meta["missing_inputs"].append("camera_pose_world")
        rgb = np.asarray(obs.get("rgb"))
        if meta["missing_inputs"]:
            raise RuntimeError(
                "strict original Active Room observation is incomplete: "
                + ", ".join(meta["missing_inputs"])
            )
        depth_for_projection = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"strict original Active Room RGB must be HxWx3, got {rgb.shape}")
        if depth_for_projection.ndim == 3 and depth_for_projection.shape[2] == 1:
            depth_for_projection = depth_for_projection[:, :, 0]
        if depth_for_projection.ndim != 2:
            raise ValueError(
                f"strict original Active Room depth must be HxW, got {depth_for_projection.shape}"
            )
        if rgb.shape[:2] != depth_for_projection.shape:
            raise ValueError(
                "strict original Active Room RGB/depth shapes differ: "
                f"{rgb.shape[:2]} vs {depth_for_projection.shape}"
            )
        if rgb.shape[0] != rgb.shape[1]:
            raise ValueError(
                "strict original Active Room requires square 256x256 RGB-D frames, "
                f"got {rgb.shape[:2]}"
            )
        if rgb.shape[:2] != (256, 256):
            raise ValueError(
                f"strict original Active Room requires 256x256 RGB-D frames, got {rgb.shape[:2]}"
            )
        intr_for_projection = _camera_intrinsics_duck(camera_intrinsics)
        if intr_for_projection is None:
            raise ValueError("strict original Active Room camera intrinsics are invalid")
        meta["rgb_original_shape_hw"] = [256, 256]
        meta["rgb_detector_shape_hw"] = [256, 256]
        if detections is None:
            detections = self.detector.detect(np.asarray(rgb[:, :, :3]))
        detections = list(detections)
        meta["num_detections"] = int(len(detections))
        meta["projection_attempted"] = True
        candidates = []
        for detection in detections:
            if getattr(detection, "mask", None) is None:
                raise RuntimeError(
                    "strict original Active Room DETR output is missing its binary door mask"
                )
            attempt = project_door_mask_to_grid_rc_with_status(
                detection=detection,
                depth=np.asarray(depth_for_projection, dtype=np.float32),
                camera_intrinsics=intr_for_projection,
                camera_pose_world=np.asarray(camera_pose_world, dtype=np.float32),
                map_info=map_info,
                floor_z=float(floor_z),
                pixel_stride=int(ORIGINAL_DOOR_MASK_PIXEL_STRIDE),
                depth_max_m=float(ORIGINAL_VISION_RANGE_M),
                relative_z_min_m=float(ORIGINAL_DOOR_RELATIVE_Z_MIN_M),
                relative_z_max_m=float(ORIGINAL_DOOR_RELATIVE_Z_MAX_M),
            )
            candidate = attempt.result
            status = str(attempt.status)
            status_counts = meta.setdefault("projection_status_counts", {})
            status_counts[str(status)] = int(status_counts.get(str(status), 0)) + 1
            if candidate is None:
                continue
            rc = np.asarray(candidate.rc, dtype=np.int32).reshape(-1, 2)
            if rc.size == 0:
                raise RuntimeError("strict original mask projection returned no grid cells")
            vision_map[rc[:, 0], rc[:, 1]] = True
            meta["projected_mask_pixels"] = int(meta["projected_mask_pixels"]) + int(
                candidate.sampled_mask_pixels
            )
            meta["projected_grid_cells"] = int(meta["projected_grid_cells"]) + int(
                rc.shape[0]
            )
            candidates.append(candidate)
        meta["projected_grid_cells"] = int(np.count_nonzero(vision_map))
        return candidates, vision_map, meta

    def _label_map_from_original_topomap(
        self,
        *,
        shape: tuple[int, int],
        domain: np.ndarray,
    ) -> np.ndarray:
        labels = np.zeros(shape, dtype=np.int32)
        graph = getattr(self._topomap, "g", None)
        if graph is None:
            raise RuntimeError("the original Active Room topological graph is unavailable")
        for vertex_idx in range(int(graph.vcount())):
            room_exp = graph.vs[vertex_idx]["room_exp"]
            for rc in room_exp or []:
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    labels[r, c] = int(vertex_idx) + 1
        labels[~np.asarray(domain, dtype=bool)] = 0
        return relabel_consecutive(labels)

    def _save_stream_frame(
        self,
        *,
        step: int,
        arrays: Mapping[str, np.ndarray],
        obs: Mapping[str, Any],
        map_info: MapInfo,
    ) -> None:
        if self._last_stream_step == int(step):
            return
        self._last_stream_step = int(step)
        self._stream_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = self._stream_manifest_path.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_npz = frames_dir / ("frame_%06d.npz" % int(step))
        frame_arrays: dict[str, Any] = {
            "occupancy_map": np.asarray(arrays["occupancy_map"]),
            "observed_free_mask": np.asarray(arrays["observed_free_mask"]),
            "obstacle_mask": np.asarray(arrays["obstacle_mask"]),
            "unknown_mask": np.asarray(arrays["unknown_mask"]),
            "agent_rc": np.asarray(arrays["agent_rc"], dtype=np.int32),
            "map_resolution_m": np.asarray(float(map_info.resolution_m), dtype=np.float32),
            "control_step": np.asarray(int(step), dtype=np.int64),
        }
        if obs.get("has_rgb") and "rgb" in obs:
            frame_arrays["rgb"] = np.asarray(obs["rgb"])
        if obs.get("has_depth") and "depth" in obs:
            frame_arrays["depth"] = np.asarray(obs["depth"], dtype=np.float32)
        if "camera_pose_world" in obs:
            frame_arrays["camera_pose_world"] = np.asarray(obs["camera_pose_world"], dtype=np.float32)
        np.savez_compressed(frame_npz, **frame_arrays)
        with self._stream_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), "frame_npz": str(frame_npz)}, sort_keys=True) + "\n")


def _load_original_modules(*, repo_dir: Path, map_size_cells: int, resolution_m: float) -> _OriginalModules:
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Active_room_segmentation repo not found: {repo_dir}")
    map_resolution_cm = int(round(float(resolution_m) * 100.0))
    map_size_cm = int(map_size_cells * map_resolution_cm)
    argv = [
        "tvars_original_isaac",
        "--no_cuda",
        "--visualize",
        "0",
        "--map_size_cm",
        str(map_size_cm),
        "--map_resolution",
        str(map_resolution_cm),
    ]
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    try:
        sys.argv = argv
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        _patch_matplotlib_for_original()
        door_module = importlib.import_module("door_detection")
        frontier_module = importlib.import_module("frontier_detection")
        topo_module = importlib.import_module("topomap_construction")
        fmm_module = _load_fmm_module(repo_dir)
        hough_module = _load_hough_module(repo_dir)
        return _OriginalModules(
            repo_dir=repo_dir,
            convert_2_laser=hough_module.convert_2_laser,
            DoorDetection=door_module.Door_detection,
            FrontierDetection=frontier_module.Frontier_detection,
            TopomapConstruction=topo_module.Topomap_construction,
            FMMPlanner=fmm_module.FMMPlanner,
            skfmm=fmm_module.skfmm,
            map_size_cells=int(map_size_cells),
            resolution_m=float(resolution_m),
        )
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


def _load_hough_module(repo_dir: Path) -> Any:
    path = repo_dir / "env" / "habitat" / "hough_door_detection.py"
    spec = importlib.util.spec_from_file_location("_tvars_original_hough_door_detection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fmm_module(repo_dir: Path) -> Any:
    path = repo_dir / "env" / "utils" / "fmm_planner.py"
    spec = importlib.util.spec_from_file_location(
        "_tvars_original_fmm_planner",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_matplotlib_for_original() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.ion = lambda *args, **kwargs: None
    plt.ioff = lambda *args, **kwargs: None
    plt.show = lambda *args, **kwargs: None
    plt.pause = lambda *args, **kwargs: None


def _resolve_repo_dir(repo_dir: Path | str | None) -> Path:
    if repo_dir is not None:
        return Path(repo_dir).expanduser().resolve()
    raw = os.environ.get("ACTIVE_ROOM_SEG_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "external_baselines" / "Active_room_segmentation").resolve()


def _floor_z_from_map_state(map_state: Mapping[str, Any]) -> float:
    if "pose" not in map_state:
        raise RuntimeError("strict original Active Room map state is missing pose")
    pose = np.asarray(map_state["pose"], dtype=np.float64)
    if pose.shape == (4, 4):
        floor_z = float(pose[2, 3])
    else:
        flat = pose.reshape(-1)
        if flat.size < 3:
            raise RuntimeError("strict original Active Room map pose has no floor height")
        floor_z = float(flat[2])
    if not np.isfinite(floor_z):
        raise RuntimeError("strict original Active Room floor height is non-finite")
    return floor_z


def _copy_strict_rgbd_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    if not bool(obs.get("has_rgb")) or "rgb" not in obs:
        missing.append("rgb")
    if obs.get("depth") is None:
        missing.append("depth")
    if obs.get("camera_pose_world") is None:
        missing.append("camera_pose_world")
    if missing:
        raise RuntimeError(
            "original topology scan observation is incomplete: " + ", ".join(missing)
        )
    return {
        "has_rgb": True,
        "has_depth": True,
        "rgb": np.asarray(obs["rgb"]).copy(),
        "depth": np.asarray(obs["depth"], dtype=np.float32).copy(),
        "camera_pose_world": np.asarray(
            obs["camera_pose_world"], dtype=np.float32
        ).copy(),
    }


def _build_original_scan_routes(
    *,
    modules: _OriginalModules,
    obstacle_map: np.ndarray,
    explored_map: np.ndarray,
    agent_rc: tuple[int, int],
    yaw_deg: float,
) -> list[tuple[list[float], list[float]]]:
    """Reproduce the source policy's 12 radial FMM route probes."""

    grid = np.rint(np.asarray(obstacle_map)).astype(np.uint8)
    explored = np.rint(np.asarray(explored_map)).astype(np.uint8)
    if grid.ndim != 2 or explored.shape != grid.shape:
        raise RuntimeError("original scan route maps must be same-shaped 2D arrays")

    start = np.asarray(
        [
            int(np.clip(agent_rc[0], 0, grid.shape[0] - 1)),
            int(np.clip(agent_rc[1], 0, grid.shape[1] - 1)),
        ],
        dtype=np.int32,
    )
    radius_cells = 3.0 / float(modules.resolution_m)
    goals: list[np.ndarray] = []
    for index in range(12):
        angle = np.deg2rad(float(yaw_deg) + float(index * 30))
        goals.append(
            np.asarray(
                [
                    int(float(start[0]) + radius_cells * np.cos(angle)),
                    int(float(start[1]) + radius_cells * np.sin(angle)),
                ],
                dtype=np.int32,
            )
        )

    routes: list[tuple[list[float], list[float]]] = []
    for raw_goal in goals:
        goal = np.asarray(
            [
                int(np.clip(raw_goal[0], 0, grid.shape[0] - 1)),
                int(np.clip(raw_goal[1], 0, grid.shape[1] - 1)),
            ],
            dtype=np.int32,
        )
        current = start.astype(np.float64)
        total_dist_m = 0.0
        route_x: list[float] = []
        route_y: list[float] = []
        for _ in range(150):
            stg = _original_fmm_short_term_goal(
                modules=modules,
                grid=grid,
                explored=explored,
                start=current,
                goal=goal,
            )
            relative_dist_m = float(np.linalg.norm(stg - current)) * float(
                modules.resolution_m
            )
            total_dist_m += relative_dist_m
            route_x.append(float(stg[0]))
            route_y.append(float(stg[1]))
            dist_to_goal_cells = float(np.linalg.norm(stg - goal))
            if total_dist_m > 3.15 and dist_to_goal_cells > 3.0:
                break
            if dist_to_goal_cells < 1.0:
                break
            current = stg
        if len(route_x) < 2:
            raise RuntimeError(
                "original 12-direction route probe produced fewer than two points"
            )
        routes.append((route_x, route_y))

    if len(routes) != 12:
        raise RuntimeError("original scan route planner did not produce 12 routes")
    return routes


def _clearance_radius_cells(
    clearance_m: float,
    resolution_m: float,
    *,
    minimum_cells: int,
) -> int:
    if float(resolution_m) <= 0.0:
        raise ValueError("FMM map resolution must be positive")
    if float(clearance_m) < 0.0:
        raise ValueError("FMM clearance must be non-negative")
    return max(
        int(minimum_cells),
        int(np.floor(float(clearance_m) / float(resolution_m) + 0.5 + 1e-9)),
    )


def _set_clearance_aware_fmm_goal(
    *,
    planner: Any,
    traversible: np.ndarray,
    goal_xy: tuple[int, int],
    clearance_soft_band_cells: int,
    clearance_speed_floor: float,
    skfmm_module: Any,
) -> np.ndarray:
    """Set an FMM travel-time field that remains passable near walls."""

    domain = np.asarray(traversible, dtype=bool)
    goal_x, goal_y = (int(goal_xy[0]), int(goal_xy[1]))
    if not (
        0 <= goal_y < domain.shape[0]
        and 0 <= goal_x < domain.shape[1]
        and bool(domain[goal_y, goal_x])
    ):
        raise RuntimeError("clearance-aware FMM goal is outside the traversible domain")
    if int(clearance_soft_band_cells) <= 0:
        raise ValueError("clearance-aware FMM requires a positive soft band")
    if not 0.0 < float(clearance_speed_floor) <= 1.0:
        raise ValueError("clearance-aware FMM speed floor must be in (0, 1]")
    if not callable(getattr(skfmm_module, "travel_time", None)):
        raise RuntimeError("strict clearance-aware FMM requires skfmm.travel_time")

    phi = np.ma.masked_array(
        np.ones(domain.shape, dtype=np.float64),
        mask=~domain,
    )
    phi[goal_y, goal_x] = 0.0
    clearance_cells = ndimage.distance_transform_edt(domain)
    clearance_fraction = np.clip(
        clearance_cells / float(clearance_soft_band_cells),
        0.0,
        1.0,
    )
    speed = float(clearance_speed_floor) + (
        1.0 - float(clearance_speed_floor)
    ) * np.square(clearance_fraction)
    speed_ma = np.ma.masked_array(speed, mask=~domain)
    travel_time = skfmm_module.travel_time(phi, speed_ma, dx=1.0)
    travel_values = np.asarray(
        np.ma.filled(travel_time, np.nan),
        dtype=np.float64,
    )
    reachable = np.isfinite(travel_values)
    if not bool(reachable[goal_y, goal_x]):
        raise RuntimeError("clearance-aware FMM did not reach its goal cell")
    finite_values = travel_values[reachable]
    unreachable_value = float(np.max(finite_values)) + float(
        max(domain.shape) ** 2
    )
    planner.fmm_dist = np.where(
        reachable,
        travel_values,
        unreachable_value,
    )
    return reachable


def _original_fmm_short_term_goal(
    *,
    modules: _OriginalModules,
    grid: np.ndarray,
    explored: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    collision_map: np.ndarray | None = None,
    visited: np.ndarray | None = None,
    obstacle_inflation_m: float = 0.05,
    preferred_clearance_m: float | None = None,
    clearance_speed_floor: float = 0.25,
    strict_no_replan: bool = False,
    return_replan: bool = False,
) -> np.ndarray | tuple[np.ndarray, bool]:
    """Port Exploration_Env._get_stg for source route generation."""

    import skimage.morphology

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.int32).copy()
    x1 = min(float(start[0]), float(goal[0]))
    x2 = max(float(start[0]), float(goal[0]))
    y1 = min(float(start[1]), float(goal[1]))
    y2 = max(float(start[1]), float(goal[1]))
    dist = float(np.linalg.norm(start - goal))
    buf = max(20.0, dist)
    x1 = max(1, int(x1 - buf))
    x2 = min(grid.shape[0] - 1, int(x2 + buf))
    y1 = max(1, int(y1 - buf))
    y2 = min(grid.shape[1] - 1, int(y2 + buf))

    rows = explored.sum(axis=1).copy()
    rows[rows > 0] = 1
    ex1 = int(np.argmax(rows))
    ex2 = int(len(rows) - np.argmax(np.flip(rows)))
    cols = explored.sum(axis=0).copy()
    cols[cols > 0] = 1
    ey1 = int(np.argmax(cols))
    ey2 = int(len(cols) - np.argmax(np.flip(cols)))

    ex1 = min(int(start[0]) - 2, ex1)
    ex2 = max(int(start[0]) + 2, ex2)
    ey1 = min(int(start[1]) - 2, ey1)
    ey2 = max(int(start[1]) + 2, ey2)
    x1 = max(x1, ex1)
    x2 = min(x2, ex2)
    y1 = max(y1, ey1)
    y2 = min(y2, ey2)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("original FMM planning crop is empty")

    obstacle_radius_cells = _clearance_radius_cells(
        obstacle_inflation_m,
        float(modules.resolution_m),
        minimum_cells=1,
    )
    preferred_clearance_m = float(
        obstacle_inflation_m
        if preferred_clearance_m is None
        else preferred_clearance_m
    )
    if preferred_clearance_m < float(obstacle_inflation_m):
        raise ValueError(
            "FMM preferred clearance must be at least the hard obstacle inflation"
        )
    preferred_radius_cells = _clearance_radius_cells(
        preferred_clearance_m,
        float(modules.resolution_m),
        minimum_cells=1,
    )
    clearance_soft_band_cells = max(
        0,
        int(preferred_radius_cells) - int(obstacle_radius_cells),
    )
    footprint = skimage.morphology.disk(obstacle_radius_cells)
    traversible = ~skimage.morphology.binary_dilation(
        grid[x1:x2, y1:y2].astype(bool),
        footprint,
    )
    if collision_map is not None:
        collision = np.asarray(collision_map, dtype=bool)
        if collision.shape != grid.shape:
            raise ValueError("original FMM collision map shape does not match grid")
        traversible[collision[x1:x2, y1:y2]] = False
    if visited is not None:
        visited_map = np.asarray(visited, dtype=bool)
        if visited_map.shape != grid.shape:
            raise ValueError("original FMM visited map shape does not match grid")
        traversible[visited_map[x1:x2, y1:y2]] = True

    start_r = int(start[0] - x1)
    start_c = int(start[1] - y1)
    traversible[start_r - 1 : start_r + 2, start_c - 1 : start_c + 2] = True
    if (
        goal[0] - 2 > x1
        and goal[0] + 3 < x2
        and goal[1] - 2 > y1
        and goal[1] + 3 < y2
    ):
        goal_r = int(goal[0] - x1)
        goal_c = int(goal[1] - y1)
        traversible[goal_r - 2 : goal_r + 3, goal_c - 2 : goal_c + 3] = True
    else:
        goal[0] = min(max(x1, int(goal[0])), x2)
        goal[1] = min(max(y1, int(goal[1])), y2)

    bounded = np.ones(
        (traversible.shape[0] + 2, traversible.shape[1] + 2),
        dtype=np.uint8,
    )
    bounded[1:-1, 1:-1] = traversible.astype(np.uint8)
    planner = modules.FMMPlanner(bounded, 360 // 10)
    goal_xy = (
        int(goal[1] - y1 + 1),
        int(goal[0] - x1 + 1),
    )
    if clearance_soft_band_cells > 0:
        reachable = _set_clearance_aware_fmm_goal(
            planner=planner,
            traversible=bounded,
            goal_xy=goal_xy,
            clearance_soft_band_cells=clearance_soft_band_cells,
            clearance_speed_floor=clearance_speed_floor,
            skfmm_module=modules.skfmm,
        )
    else:
        reachable = planner.set_goal(list(goal_xy))
    stg_x = float(start[0] - x1 + 1)
    stg_y = float(start[1] - y1 + 1)
    start_reachable = bool(
        np.asarray(reachable, dtype=bool)[int(stg_x), int(stg_y)]
    )
    if strict_no_replan and not start_reachable:
        raise RuntimeError(
            "original topology FMM target is unreachable from the current cell"
        )
    if not start_reachable:
        result = start.copy()
        return (result, True) if return_replan else result
    stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
    if bool(replan):
        if strict_no_replan:
            raise RuntimeError(
                "original topology FMM requested replan; A* fallback is disabled"
            )
        result = start.copy()
        return (result, True) if return_replan else result
    result = np.asarray(
        [float(stg_x + x1 - 1), float(stg_y + y1 - 1)],
        dtype=np.float64,
    )
    return (result, False) if return_replan else result


def _global_frontier_map_to_original_lists(
    frontier_map: np.ndarray,
    *,
    avoid_targets_rc: Iterable[tuple[int, int]] = (),
) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    frontier = np.asarray(frontier_map, dtype=bool)
    if frontier.ndim != 2:
        raise RuntimeError(
            "runtime global frontier map must be two-dimensional, got "
            f"shape={frontier.shape}"
        )
    labels, component_count = ndimage.label(
        frontier,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    frontier_list: list[np.ndarray] = []
    info_gain_list: list[float] = []
    component_sizes: list[int] = []
    component_fresh_sizes: list[int] = []
    component_centers_rc: list[list[int]] = []
    avoid_targets = np.asarray(
        [
            [int(target[0]), int(target[1])]
            for target in avoid_targets_rc
        ],
        dtype=np.int32,
    ).reshape(-1, 2)
    avoid_radius_sq = int(
        RUNTIME_GLOBAL_FRONTIER_REACHED_MATCH_RADIUS_CELLS
        * RUNTIME_GLOBAL_FRONTIER_REACHED_MATCH_RADIUS_CELLS
    )
    for label_id in range(1, int(component_count) + 1):
        members = np.argwhere(labels == label_id)
        if members.size == 0:
            raise RuntimeError(
                "runtime global frontier labeling produced an empty component"
            )
        centroid = np.mean(members.astype(np.float64), axis=0)
        fresh_mask = np.ones((members.shape[0],), dtype=bool)
        min_avoid_distance_sq = np.full(
            (members.shape[0],), np.iinfo(np.int32).max, dtype=np.int64
        )
        if avoid_targets.size:
            delta = (
                members[:, None, :].astype(np.int64)
                - avoid_targets[None, :, :].astype(np.int64)
            )
            min_avoid_distance_sq = np.min(
                np.sum(delta * delta, axis=2), axis=1
            )
            fresh_mask = min_avoid_distance_sq > avoid_radius_sq
        fresh_size = int(np.count_nonzero(fresh_mask))
        if fresh_size > 0 and avoid_targets.size:
            farthest_distance_sq = int(np.max(min_avoid_distance_sq[fresh_mask]))
            representative_candidates = np.flatnonzero(
                fresh_mask & (min_avoid_distance_sq == farthest_distance_sq)
            )
            representative_index = int(
                representative_candidates[
                    np.argmin(
                        np.sum(
                            (
                                members[representative_candidates]
                                - centroid[None, :]
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                ]
            )
        else:
            representative_index = int(
                np.argmin(np.sum((members - centroid[None, :]) ** 2, axis=1))
            )
        representative = members[representative_index].astype(np.int32)
        size = int(members.shape[0])
        frontier_list.append(representative)
        info_gain_list.append(float(fresh_size if fresh_size > 0 else size))
        component_sizes.append(size)
        component_fresh_sizes.append(fresh_size)
        component_centers_rc.append(
            [int(representative[0]), int(representative[1])]
        )
    return frontier_list, info_gain_list, {
        "runtime_global_frontier_raw_cells": int(np.count_nonzero(frontier)),
        "runtime_global_frontier_reachable_cells": int(
            np.count_nonzero(frontier)
        ),
        "runtime_global_frontier_component_count": int(component_count),
        "runtime_global_frontier_component_sizes": component_sizes,
        "runtime_global_frontier_component_fresh_sizes": component_fresh_sizes,
        "runtime_global_frontier_fresh_cells": int(sum(component_fresh_sizes)),
        "runtime_global_frontier_representative_semantics": (
            "farthest_surviving_cell_from_reached_history_then_component_centroid"
        ),
        "runtime_global_frontier_component_centers_rc": component_centers_rc,
        "runtime_global_frontier_component_connectivity": 8,
        "runtime_global_frontier_uses_reference_map": False,
        "runtime_global_frontier_fallback_used": False,
    }


def _restrict_vertical_frontiers_to_current_tvars_room(
    *,
    global_frontier_map: np.ndarray,
    vertical_free_mask: np.ndarray,
    persistent_door_line_map: np.ndarray,
    agent_rc: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply TVARS' persistent door memory before choosing room frontiers."""

    frontier = np.asarray(global_frontier_map, dtype=bool)
    vertical_free = np.asarray(vertical_free_mask, dtype=bool)
    door_lines = np.asarray(persistent_door_line_map, dtype=bool)
    if frontier.ndim != 2 or vertical_free.shape != frontier.shape:
        raise ValueError(
            "global frontier and Vertical Free masks must be matching 2D arrays"
        )
    if door_lines.shape != frontier.shape:
        raise ValueError(
            "persistent TVARS door lines must match the Vertical Free map"
        )

    partition_labels, partition_debug = partition_free_space_by_door_lines(
        vertical_free,
        door_lines,
    )
    agent_row, agent_col = int(agent_rc[0]), int(agent_rc[1])
    if not (
        0 <= agent_row < frontier.shape[0]
        and 0 <= agent_col < frontier.shape[1]
    ):
        raise ValueError("TVARS current-room agent cell lies outside the map")

    current_label = int(partition_labels[agent_row, agent_col])
    seed_rc = (agent_row, agent_col)
    seed_fallback = False
    if current_label <= 0:
        labeled_cells = np.argwhere(partition_labels > 0)
        if labeled_cells.size == 0:
            raise RuntimeError("TVARS Vertical Free partition contains no room")
        delta = labeled_cells.astype(np.int64) - np.asarray(
            [agent_row, agent_col], dtype=np.int64
        )[None, :]
        nearest_index = int(np.argmin(np.sum(delta * delta, axis=1)))
        nearest = labeled_cells[nearest_index]
        seed_rc = (int(nearest[0]), int(nearest[1]))
        current_label = int(partition_labels[seed_rc])
        seed_fallback = True

    current_room_mask = partition_labels == int(current_label)
    current_room_frontier = frontier & current_room_mask
    return current_room_frontier, current_room_mask, {
        "tvars_room_frontier_source": "voxel_vertical_free",
        "tvars_room_constraint_source": "persistent_original_door_lines",
        "tvars_room_selection_order": (
            "current_room_frontiers_then_original_choose_door"
        ),
        "tvars_persistent_door_line_cells": int(np.count_nonzero(door_lines)),
        "tvars_current_room_partition_count": int(
            partition_debug["room_count"]
        ),
        "tvars_current_room_label": int(current_label),
        "tvars_current_room_seed_rc": [int(seed_rc[0]), int(seed_rc[1])],
        "tvars_current_room_seed_fallback": bool(seed_fallback),
        "tvars_current_room_vertical_free_cells": int(
            np.count_nonzero(current_room_mask)
        ),
        "tvars_current_room_frontier_cells": int(
            np.count_nonzero(current_room_frontier)
        ),
    }


def _select_original_frontier_target(
    *,
    frontier_list: Sequence[Any],
    info_gain_list: Sequence[Any],
    agent_rc: tuple[int, int],
    last_goal_rc: tuple[int, int] | None,
    excluded_targets_rc: Iterable[tuple[int, int]] = (),
) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    frontiers = list(frontier_list)
    gains = list(info_gain_list)
    if len(frontiers) != len(gains):
        raise RuntimeError(
            "original frontier and information-gain counts differ: %d vs %d"
            % (len(frontiers), len(gains))
        )
    if not frontiers:
        return None, {
            "frontier_count": 0,
            "frontier_selection_reason": "original_frontier_list_empty",
        }

    parsed: list[tuple[int, int]] = []
    for point in frontiers:
        values = np.asarray(point).reshape(-1)
        if values.size < 2:
            raise RuntimeError("original frontier point has fewer than two coordinates")
        parsed.append(
            (int(round(float(values[0]))), int(round(float(values[1]))))
        )

    excluded = {
        (int(target[0]), int(target[1])) for target in excluded_targets_rc
    }
    excluded_present = [target for target in parsed if target in excluded]
    selection_metadata = {
        "frontier_astar_unreachable_excluded_count": int(len(excluded_present)),
        "frontier_astar_unreachable_excluded_rc": [
            [int(target[0]), int(target[1])] for target in excluded_present
        ],
        "frontier_selectable_count": int(
            sum(target not in excluded for target in parsed)
        ),
    }

    if len(parsed) == 1:
        target = parsed[0]
        if target in excluded:
            return None, {
                "frontier_count": 1,
                **selection_metadata,
                "frontier_selection_reason": (
                    "original_frontier_astar_unreachable_excluded"
                ),
            }
        if last_goal_rc is not None and target == tuple(last_goal_rc):
            return None, {
                "frontier_count": 1,
                **selection_metadata,
                "frontier_selection_reason": "original_last_goal_repeat_suppressed",
                "repeated_frontier_rc": [int(target[0]), int(target[1])],
            }
        return target, {
            "frontier_count": 1,
            **selection_metadata,
            "frontier_selected_index": 0,
            "frontier_selected_rc": [int(target[0]), int(target[1])],
            "frontier_selected_distance_cells": float(
                np.hypot(target[0] - agent_rc[0], target[1] - agent_rc[1])
            ),
            "frontier_selected_info_gain": float(gains[0]),
            "frontier_selection_cost": float(
                np.hypot(target[0] - agent_rc[0], target[1] - agent_rc[1])
                - 2.0 * float(gains[0])
            ),
            "frontier_selection_reason": "original_single_frontier",
        }

    selected_index: int | None = None
    selected_cost = float("inf")
    selected_distance = float("inf")
    for index, (target, gain) in enumerate(zip(parsed, gains)):
        if target in excluded:
            continue
        if last_goal_rc is not None and target == tuple(last_goal_rc):
            continue
        distance = float(
            np.hypot(target[0] - agent_rc[0], target[1] - agent_rc[1])
        )
        cost = float(distance - 2.0 * float(gain))
        if cost < selected_cost:
            selected_index = int(index)
            selected_cost = cost
            selected_distance = distance
    if selected_index is None:
        if excluded_present and len(excluded_present) == len(parsed):
            return None, {
                "frontier_count": int(len(parsed)),
                **selection_metadata,
                "frontier_selection_reason": (
                    "original_all_frontiers_astar_unreachable_excluded"
                ),
            }
        return None, {
            "frontier_count": int(len(parsed)),
            **selection_metadata,
            "frontier_selection_reason": "original_all_frontiers_repeat_last_goal",
            "repeated_frontier_rc": None
            if last_goal_rc is None
            else [int(last_goal_rc[0]), int(last_goal_rc[1])],
        }
    target = parsed[selected_index]
    return target, {
        "frontier_count": int(len(parsed)),
        **selection_metadata,
        "frontier_selected_index": int(selected_index),
        "frontier_selected_rc": [int(target[0]), int(target[1])],
        "frontier_selected_distance_cells": float(selected_distance),
        "frontier_selected_info_gain": float(gains[selected_index]),
        "frontier_selection_cost": float(selected_cost),
        "frontier_selection_weights": {
            "distance": 1.0,
            "information_gain": 2.0,
        },
        "frontier_selection_reason": "original_distance_minus_information_gain",
    }


def _target_matches_any(
    target_rc: tuple[int, int],
    candidates_rc: Sequence[tuple[int, int]],
    *,
    radius_cells: int,
) -> bool:
    radius = max(0, int(radius_cells))
    target_r, target_c = int(target_rc[0]), int(target_rc[1])
    radius_sq = radius * radius
    return any(
        (target_r - int(candidate[0])) ** 2
        + (target_c - int(candidate[1])) ** 2
        <= radius_sq
        for candidate in candidates_rc
    )


def _policy_target_map(
    target_rc: tuple[int, int] | None,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if target_rc is None:
        return out
    row, col = int(target_rc[0]), int(target_rc[1])
    if 0 <= row < shape[0] and 0 <= col < shape[1]:
        out[row, col] = True
    return out


def _filter_and_project_partition_labels(
    label_map: np.ndarray,
    *,
    navigation_free: np.ndarray,
    resolution_m: float,
    min_room_area_m2: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Filter in the partition domain, then project labels to navigation free."""

    source = np.asarray(label_map, dtype=np.int32).copy()
    nav_free = np.asarray(navigation_free, dtype=bool)
    if source.ndim != 2 or nav_free.shape != source.shape:
        raise ValueError(
            "partition labels and navigation-free projection must be matching 2D arrays"
        )
    cell_area_m2 = max(float(resolution_m) ** 2, 1.0e-12)
    threshold_m2 = max(0.0, float(min_room_area_m2))
    removed: list[dict[str, Any]] = []
    for label in sorted(int(value) for value in np.unique(source) if int(value) > 0):
        area_cells = int(np.count_nonzero(source == label))
        area_m2 = float(area_cells) * cell_area_m2
        if area_m2 + 1.0e-12 >= threshold_m2:
            continue
        source[source == label] = 0
        removed.append(
            {
                "label": int(label),
                "area_cells": int(area_cells),
                "area_m2": float(area_m2),
            }
        )
    source = relabel_consecutive(source)
    projected = source.copy()
    projected[~nav_free] = 0
    projected = relabel_consecutive(projected)
    return source, projected, {
        "partition_source": "vertical_free_room_domain",
        "projection_target": "navigation_free_room_domain",
        "min_room_area_m2": float(threshold_m2),
        "small_room_filter_semantics": "remove_area_strictly_less_than_threshold",
        "small_room_removed_count": int(len(removed)),
        "small_room_removed": removed,
        "vertical_partition_room_count": int(np.max(source)) if source.size else 0,
        "vertical_partition_labeled_cells": int(np.count_nonzero(source)),
        "navigation_projected_room_count": (
            int(np.max(projected)) if projected.size else 0
        ),
        "navigation_projected_labeled_cells": int(np.count_nonzero(projected)),
        "labels_removed_by_navigation_projection_cells": int(
            np.count_nonzero((source > 0) & ~nav_free)
        ),
    }


def _arrays_from_map_state(map_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    occupancy = np.asarray(map_state["occupancy"], dtype=bool)
    free = np.asarray(map_state["free"], dtype=bool)
    observed = np.asarray(map_state["observed"], dtype=bool)
    vertical_free = np.asarray(
        map_state.get("vertical_free_room_domain", free), dtype=bool
    )
    navigation_free = np.asarray(
        map_state.get("navigation_free_room_domain", free), dtype=bool
    )
    if vertical_free.shape != free.shape or navigation_free.shape != free.shape:
        raise ValueError(
            "TVARS vertical-free and navigation-free room domains must match the map"
        )
    return {
        "occupancy_map": occupancy,
        "observed_free_mask": free,
        "obstacle_mask": occupancy,
        "unknown_mask": ~observed,
        "vertical_free_room_domain": vertical_free,
        "navigation_free_room_domain": navigation_free,
        "agent_rc": np.asarray(map_state["current_grid"], dtype=np.int32),
    }


def _yaw_deg_from_map_state(map_state: Mapping[str, Any]) -> float:
    pose = map_state.get("pose")
    if pose is None:
        return 0.0
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape == (4, 4):
        return float(
            np.degrees(
                np.arctan2(float(matrix[1, 0]), float(matrix[0, 0]))
            )
        )
    arr = matrix.reshape(-1)
    if arr.size >= 4:
        return float(np.degrees(float(arr[3])))
    if arr.size >= 3:
        return float(np.degrees(float(arr[2])))
    return 0.0


def _clip_xy_points(points: Any, *, shape: tuple[int, int], margin: int = 0) -> list[list[int]]:
    out: list[list[int]] = []
    h, w = int(shape[0]), int(shape[1])
    for point in list(points or []):
        if len(point) < 2:
            continue
        x, y = int(point[0]), int(point[1])
        if margin <= x < w - margin and margin <= y < h - margin:
            out.append([x, y])
    return out


def _filter_hough_with_vision(points: list[list[int]], *, vision_map: np.ndarray, radius_cells: int) -> list[list[int]]:
    if not bool(np.any(vision_map)):
        return []
    h, w = vision_map.shape
    radius = int(radius_cells)
    out: list[list[int]] = []
    for x, y in points:
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        if bool(np.any(vision_map[y0:y1, x0:x1])):
            out.append([int(x), int(y)])
    return out


def _xy_points_to_map(points: list[list[int]], *, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for x, y in points:
        if 0 <= int(y) < shape[0] and 0 <= int(x) < shape[1]:
            out[int(y), int(x)] = True
    return out


def _rc_points_to_map(points: list[list[int]], *, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for point in points or []:
        if len(point) < 2:
            continue
        r, c = int(point[0]), int(point[1])
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            out[r, c] = True
    return out


def _doors_to_line_map(doors: list[dict[str, Any]], *, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for door in doors:
        start = tuple(int(v) for v in door["start"])
        end = tuple(int(v) for v in door["end"])
        cv2.line(out, start, end, 1, thickness=3)
    return out.astype(bool)


def _camera_intrinsics_duck(value: Any) -> CameraIntrinsics | None:
    if value is not None and any(hasattr(value, name) for name in ("fx", "f_x")):
        fx = getattr(value, "fx") if hasattr(value, "fx") else getattr(value, "f_x")
        fy = getattr(value, "fy") if hasattr(value, "fy") else getattr(value, "f_y")
        cx = getattr(value, "cx") if hasattr(value, "cx") else getattr(value, "c_x")
        cy = getattr(value, "cy") if hasattr(value, "cy") else getattr(value, "c_y")
        return CameraIntrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=None if getattr(value, "width", None) is None else int(getattr(value, "width")),
            height=None if getattr(value, "height", None) is None else int(getattr(value, "height")),
        )
    return camera_intrinsics_from_mapping(value)


def _topomap_vertex_count(topo: Any) -> int:
    return int(topo.g.vcount())


@contextlib.contextmanager
def _suppress_original_stdout():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _snapshot_npz_path(value: Path | Mapping[str, Any]) -> Path:
    if isinstance(value, Mapping):
        paths = value.get("paths")
        if isinstance(paths, Mapping) and paths.get("npz") is not None:
            return Path(str(paths["npz"]))
        if value.get("npz") is not None:
            return Path(str(value["npz"]))
    return Path(value)


def _write_summary_json(
    path: Path,
    *,
    output_npz: Path,
    source_npz: Path,
    metadata: Mapping[str, Any],
    label_map: np.ndarray,
) -> None:
    labels = np.asarray(label_map, dtype=np.int32)
    payload = {
        "step": int(metadata.get("snapshot_step", metadata.get("step", -1))),
        "method": BASELINE_NAME,
        "output_npz": str(output_npz),
        "source_npz": str(source_npz),
        "metadata": dict(metadata),
        "shape": [int(v) for v in labels.shape],
        "counts": {
            "final_labeled": int(np.count_nonzero(labels > 0)),
            "positive_labels": int(len([v for v in np.unique(labels) if int(v) > 0])),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_colored_room_preview(snapshot_path: Path, output_path: Path) -> None:
    with np.load(snapshot_path, allow_pickle=False) as data:
        labels = np.asarray(data["final_room_label_map"], dtype=np.int32)
        occupancy = np.asarray(data["occupancy_map"], dtype=bool)
        unknown = np.asarray(data["unknown_mask"], dtype=bool)
        hough = np.asarray(data["tvars_original_hough_door_seed_map"], dtype=bool)
        filtered = np.asarray(data["tvars_original_filtered_hough_door_seed_map"], dtype=bool)
        accepted = np.asarray(data["tvars_original_door_line_map"], dtype=bool)
    image = _render_colored_room_preview(
        labels=labels,
        occupancy=occupancy,
        unknown=unknown,
        hough=hough,
        filtered=filtered,
        accepted=accepted,
    )
    visible = (~unknown) | occupancy | (labels > 0) | hough | filtered | accepted
    points = np.argwhere(visible)
    if points.size == 0:
        raise RuntimeError("strict original Active Room preview has no visible cells")
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    margin = max(8, int(round(max(r1 - r0 + 1, c1 - c0 + 1) * 0.06)))
    r0, c0 = max(0, int(r0) - margin), max(0, int(c0) - margin)
    r1 = min(labels.shape[0], int(r1) + margin + 1)
    c1 = min(labels.shape[1], int(c1) + margin + 1)
    cropped = image[r0:r1, c0:c1]
    scale = max(1, int(np.ceil(1200.0 / max(cropped.shape[:2]))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cropped, mode="RGB").resize(
        (cropped.shape[1] * scale, cropped.shape[0] * scale),
        Image.Resampling.NEAREST,
    ).save(output_path)


def _render_colored_room_preview(
    *,
    labels: np.ndarray,
    occupancy: np.ndarray,
    unknown: np.ndarray,
    hough: np.ndarray,
    filtered: np.ndarray,
    accepted: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    occupancy = np.asarray(occupancy, dtype=bool)
    unknown = np.asarray(unknown, dtype=bool)
    hough = np.asarray(hough, dtype=bool)
    filtered = np.asarray(filtered, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    shape = labels.shape
    for name, value in (
        ("occupancy", occupancy),
        ("unknown", unknown),
        ("hough", hough),
        ("filtered", filtered),
        ("accepted", accepted),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} does not match labels {shape}")
    image = np.full(shape + (3,), (238, 238, 238), dtype=np.uint8)
    image[unknown] = (255, 255, 255)
    for label in np.unique(labels):
        if label <= 0:
            continue
        image[labels == label] = _room_color(int(label))
    image[occupancy] = (35, 35, 35)
    image[hough] = (255, 152, 0)
    image[filtered] = (0, 188, 212)
    image[accepted] = (216, 27, 96)
    return image


def _room_color(label: int) -> tuple[int, int, int]:
    palette = (
        (68, 199, 183),
        (232, 62, 157),
        (104, 174, 232),
        (201, 178, 124),
        (98, 185, 143),
        (245, 132, 31),
        (145, 104, 207),
        (232, 111, 81),
    )
    return palette[(int(label) - 1) % len(palette)]


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.PIPE,
        timeout=30,
    ).strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
